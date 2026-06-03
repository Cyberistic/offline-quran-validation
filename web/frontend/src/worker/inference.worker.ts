/// <reference lib="webworker" />

import * as ort from "onnxruntime-web/wasm";
import { downloadModel, getCachedModel } from "../lib/model-cache";
import { QuranDB, type QuranCandidate, type QuranMatch, type QuranVerse } from "../lib/quran-db";
import { normalizeArabic } from "../lib/normalizer";
import type { InferenceResult, WorkerInbound, WorkerOutbound } from "../lib/types";

let session: ort.InferenceSession | null = null;
let db: QuranDB | null = null;
let vocab: Map<number, string> | null = null;
let ctcTokens: Record<string, number[]> | null = null;

const BLANK_ID = 1024;
const FALLBACK_THRESHOLD = 0.8;
const SPAN_PENALTY = 0.5;

function post(message: WorkerOutbound) {
  self.postMessage(message);
}

function decodeGreedy(logits: Float32Array, timeSteps: number, vocabSize: number): string {
  const blankId = vocabSize - 1;
  const ids: number[] = [];
  for (let t = 0; t < timeSteps; t++) {
    let bestIndex = 0;
    let bestValue = -Infinity;
    const offset = t * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      const value = logits[offset + v];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = v;
      }
    }
    ids.push(bestIndex);
  }

  const tokens: string[] = [];
  let prev = -1;
  for (const id of ids) {
    if (id !== prev && id !== blankId) {
      tokens.push(vocab?.get(id) ?? "");
    }
    prev = id;
  }

  return normalizeArabic(tokens.join("").replace(/▁/g, " ").trim());
}

async function ensureResources(modelUrl: string) {
  if (!db || !vocab || !ctcTokens) {
    const [quranResponse, vocabResponse, ctcTokenResponse] = await Promise.all([
      fetch("/quran.json"),
      fetch("/vocab.json"),
      fetch("/quran_ctc_tokens.json"),
    ]);
    const quran = (await quranResponse.json()) as QuranVerse[];
    const vocabJson = (await vocabResponse.json()) as Record<string, string>;
    ctcTokens = (await ctcTokenResponse.json()) as Record<string, number[]>;
    db = new QuranDB(quran);
    vocab = new Map(Object.entries(vocabJson).map(([key, value]) => [Number(key), value]));
  }

  if (!session) {
    post({ type: "loading", phase: "Checking local model cache" });
    let model = await getCachedModel(modelUrl);
    if (!model) {
      model = await downloadModel(modelUrl, (loaded, total) => {
        post({ type: "loading", phase: "Downloading model from R2", loaded, total });
      });
    }
    post({ type: "loading", phase: "Creating ONNX Runtime session" });
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
    session = await ort.InferenceSession.create(model.bytes.slice(0), { executionProviders: ["wasm"] });
  }
}

async function run(audio: Float32Array): Promise<InferenceResult> {
  if (!session || !db || !vocab || !ctcTokens) throw new Error("Worker not initialized");

  const totalStart = performance.now();
  const input = new ort.Tensor("float32", audio, [1, audio.length]);
  const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(audio.length)]), [1]);

  const inferenceStart = performance.now();
  const outputs = await session.run({ audio_signal: input, length });
  const inferenceMs = performance.now() - inferenceStart;

  const output = outputs.log_probs;
  const dims = output.dims as number[];
  const timeSteps = dims[1];
  const vocabSize = dims[2];

  const decodeStart = performance.now();
  const transcript = decodeGreedy(output.data as Float32Array, timeSteps, vocabSize);
  const decodeMs = performance.now() - decodeStart;

  const matchStart = performance.now();
  const { base, candidates } = db.buildCandidates(transcript);

  const logProbs = output.data as Float32Array;
  const ctcRanked = rerankCandidates(logProbs, timeSteps, vocabSize, candidates, ctcTokens)
    .sort((a, b) => b.finalScore - a.finalScore);

  const useCtc = !base || base.score < FALLBACK_THRESHOLD;
  const chosen = useCtc && ctcRanked.length > 0 ? ctcRanked[0].candidate : base;
  const topMatches: QuranMatch[] = [];
  if (chosen) {
    const verse = db.getVerse(chosen.surah, chosen.ayah);
    if (verse) {
      topMatches.push({ ...verse, score: useCtc && ctcRanked[0] ? Math.exp(-ctcRanked[0].normLoss) : chosen.score });
    }
  }
  for (const item of (useCtc ? ctcRanked.slice(0, 5).map((entry) => entry.candidate) : candidates.slice(0, 5))) {
    const verse = db.getVerse(item.surah, item.ayah);
    if (!verse) continue;
    if (topMatches.some((match) => match.surah === verse.surah && match.ayah === verse.ayah)) continue;
    topMatches.push({ ...verse, score: item.score });
  }
  const matchMs = performance.now() - matchStart;

  return {
    transcript,
    match: topMatches[0] ?? null,
    topMatches,
    source: useCtc && ctcRanked.length > 0 ? "ctc" : "text",
    rerankedCandidates: ctcRanked.slice(0, 5).map((entry) => entry.candidate),
    audioDurationMs: (audio.length / 16000) * 1000,
    timings: {
      totalMs: performance.now() - totalStart,
      inferenceMs,
      decodeMs,
      matchMs,
    },
  };
}

function rerankCandidates(
  logProbs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  candidates: QuranCandidate[],
  tokenMap: Record<string, number[]>,
) {
  const ranked: Array<{ candidate: QuranCandidate; normLoss: number; finalScore: number }> = [];
  for (const candidate of candidates) {
    const key = `${candidate.surah}:${candidate.ayah}:${candidate.ayah_end}`;
    const target = tokenMap[key];
    if (!target || target.length === 0 || target.length * 2 + 1 > timeSteps) continue;
    const normLoss = ctcLoss(logProbs, timeSteps, vocabSize, target);
    if (!Number.isFinite(normLoss)) continue;
    const spanPenalty = SPAN_PENALTY * (candidate.ayah_end - candidate.ayah);
    ranked.push({
      candidate,
      normLoss,
      finalScore: -normLoss - spanPenalty,
    });
  }
  return ranked;
}

function logAddExp(a: number, b: number) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

function ctcLoss(logProbs: Float32Array, timeSteps: number, vocabSize: number, target: number[]) {
  const ext: number[] = [];
  for (const token of target) {
    ext.push(BLANK_ID, token);
  }
  ext.push(BLANK_ID);

  let prev = new Float64Array(ext.length).fill(-Infinity);
  let next = new Float64Array(ext.length).fill(-Infinity);
  prev[0] = logProbs[BLANK_ID];
  if (ext.length > 1) prev[1] = logProbs[target[0]];

  for (let t = 1; t < timeSteps; t++) {
    next.fill(-Infinity);
    const offset = t * vocabSize;
    for (let s = 0; s < ext.length; s++) {
      const token = ext[s];
      let value = prev[s];
      if (s > 0) value = logAddExp(value, prev[s - 1]);
      if (s > 1 && token !== BLANK_ID && token !== ext[s - 2]) {
        value = logAddExp(value, prev[s - 2]);
      }
      next[s] = value + logProbs[offset + token];
    }
    [prev, next] = [next, prev];
  }

  const logLike = logAddExp(prev[ext.length - 1], prev[ext.length - 2] ?? -Infinity);
  return -logLike / target.length;
}

self.onmessage = async (event: MessageEvent<WorkerInbound>) => {
  try {
    if (event.data.type === "init") {
      await ensureResources(event.data.modelUrl);
      post({ type: "ready" });
      return;
    }
    if (event.data.type === "run") {
      const result = await run(event.data.audio);
      post({ type: "done", result });
    }
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : "Worker failed" });
  }
};
