import type { QuranCandidate, QuranMatch } from "./quran-db";

export interface RuntimeConfig {
  modelUrl: string;
  modelFileName: string;
}

export interface TimingBreakdown {
  totalMs: number;
  decodeMs: number;
  matchMs: number;
  inferenceMs: number;
}

export interface InferenceResult {
  transcript: string;
  match: QuranMatch | null;
  topMatches: QuranMatch[];
  source: "ctc" | "text";
  rerankedCandidates: QuranCandidate[];
  timings: TimingBreakdown;
  audioDurationMs: number;
}

export type WorkerInbound =
  | { type: "init"; modelUrl: string }
  | { type: "run"; audio: Float32Array };

export type WorkerOutbound =
  | { type: "ready" }
  | { type: "loading"; phase: string; loaded?: number; total?: number }
  | { type: "done"; result: InferenceResult }
  | { type: "error"; message: string };
