import { useEffect, useMemo, useRef, useState } from "react";
import { Download, LoaderCircle, Mic, TimerReset, Trash2, Upload } from "lucide-react";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Progress } from "./components/ui/progress";
import { clearCachedModel, getCachedModel } from "./lib/model-cache";
import type { InferenceResult, RuntimeConfig, WorkerOutbound } from "./lib/types";
import { formatBytes, formatDuration } from "./lib/utils";

type ModelState = "idle" | "downloading" | "ready" | "error";

async function fetchConfig(): Promise<RuntimeConfig> {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("Failed to load config");
  return response.json();
}

export default function App() {
  const workerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const modelTotalRef = useRef<number | null>(null);
  const recordingMimeTypeRef = useRef("audio/webm");
  const liveRunActiveRef = useRef(false);
  const liveRunQueuedRef = useRef(false);
  const liveRunChunkCountRef = useRef(0);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelStatus, setModelStatus] = useState("Waiting for model download");
  const [modelLoaded, setModelLoaded] = useState(0);
  const [modelTotal, setModelTotal] = useState<number | null>(null);
  const [cachedSize, setCachedSize] = useState<number | null>(null);
  const [workerReady, setWorkerReady] = useState(false);
  const [audioFileName, setAudioFileName] = useState<string>("");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState("Mic idle");

  useEffect(() => {
    let cancelled = false;

    fetchConfig()
      .then(async (cfg) => {
        if (cancelled) return;
        setConfig(cfg);
        const cached = await getCachedModel(cfg.modelUrl);
        if (cached && !cancelled) {
          modelTotalRef.current = cached.size;
          setModelLoaded(cached.size);
          setModelTotal(cached.size);
          setModelProgress(100);
          setCachedSize(cached.size);
          setModelStatus(`Model cached on this device (${formatBytes(cached.size)})`);
        }

        setWorkerReady(false);
        workerRef.current?.postMessage({ type: "init", modelUrl: cfg.modelUrl });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load app config");
      });

    const worker = new Worker(new URL("./worker/inference.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<WorkerOutbound>) => {
      const message = event.data;
      if (message.type === "loading") {
        setWorkerReady(false);
        setModelState("downloading");
        setModelStatus(message.phase);
        setModelLoaded(message.loaded ?? 0);
        if (message.total) {
          modelTotalRef.current = message.total;
          setModelTotal(message.total);
          setModelProgress(Math.round(((message.loaded ?? 0) / message.total) * 100));
        }
      } else if (message.type === "ready") {
        setWorkerReady(true);
        setModelState("ready");
        setModelProgress(100);
        setModelLoaded(modelTotalRef.current ?? modelLoaded);
        setModelTotal(modelTotalRef.current);
        if (modelTotalRef.current) {
          setCachedSize(modelTotalRef.current);
        }
        setModelStatus("Model ready for on-device inference");
      } else if (message.type === "done") {
        setResult(message.result);
        liveRunActiveRef.current = false;
        if (liveRunQueuedRef.current) {
          liveRunQueuedRef.current = false;
          void runLiveInference();
          return;
        }
        setRunning(false);
        if (recording) {
          setRecordingStatus("Listening for more audio");
        }
      } else if (message.type === "error") {
        setWorkerReady(false);
        setRunning(false);
        setModelState("error");
        liveRunActiveRef.current = false;
        liveRunQueuedRef.current = false;
        setError(message.message);
      }
    };

    return () => {
      cancelled = true;
      mediaRecorderRef.current?.stop();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      worker.terminate();
    };
  }, []);

  const topMatch = result?.match ?? null;
  const topMatches = result?.topMatches ?? [];
  const canRun = modelState === "ready" && workerReady && !!workerRef.current && !running;
  const downloadDetails =
    modelTotal && modelTotal > 0 ? `${formatBytes(modelLoaded)} / ${formatBytes(modelTotal)} (${modelProgress}%)` : null;

  function ensureModel() {
    if (!config || !workerRef.current) return;
    setError(null);
    setWorkerReady(false);
    workerRef.current.postMessage({ type: "init", modelUrl: config.modelUrl });
  }

  async function handleResetCache() {
    await clearCachedModel();
    modelTotalRef.current = null;
    setCachedSize(null);
    setWorkerReady(false);
    setModelState("idle");
    setModelLoaded(0);
    setModelTotal(null);
    setModelProgress(0);
    setModelStatus("Local model cache cleared");
  }

  async function handleFile(file: File) {
    if (!canRun) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setAudioFileName(file.name);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const channel = new Float32Array(decoded.getChannelData(0));
      workerRef.current?.postMessage({ type: "run", audio: channel }, [channel.buffer]);
      void audioContext.close();
    } catch (err) {
      setRunning(false);
      setError(err instanceof Error ? err.message : "Failed to read audio file");
    }
  }

  async function decodeAudioToMono16k(source: Blob | File) {
    const arrayBuffer = await source.arrayBuffer();
    const audioContext = new AudioContext();
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));

    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const buffer = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    const mono = buffer.getChannelData(0);
    const left = decoded.getChannelData(0);
    const right = decoded.numberOfChannels > 1 ? decoded.getChannelData(1) : null;
    for (let i = 0; i < decoded.length; i++) {
      mono[i] = right ? (left[i] + right[i]) * 0.5 : left[i];
    }
    const sourceNode = offline.createBufferSource();
    sourceNode.buffer = buffer;
    sourceNode.connect(offline.destination);
    sourceNode.start();
    const rendered = await offline.startRendering();
    await audioContext.close();
    return new Float32Array(rendered.getChannelData(0));
  }

  async function runLiveInference() {
    if (!workerRef.current || liveRunActiveRef.current) return;
    const chunkCount = recordedChunksRef.current.length;
    if (chunkCount === 0 || chunkCount === liveRunChunkCountRef.current) return;

    liveRunActiveRef.current = true;
    liveRunChunkCountRef.current = chunkCount;
    setRunning(true);
    setError(null);
    setRecordingStatus(recording ? "Streaming microphone result" : "Processing final microphone audio");

    try {
      const blob = new Blob(recordedChunksRef.current, { type: recordingMimeTypeRef.current });
      const audio = await decodeAudioToMono16k(blob);
      workerRef.current.postMessage({ type: "run", audio }, [audio.buffer]);
    } catch (err) {
      liveRunActiveRef.current = false;
      setRunning(false);
      setError(err instanceof Error ? err.message : "Failed to decode microphone audio");
      setRecordingStatus("Mic decode failed");
    }
  }

  function queueLiveInference() {
    if (liveRunActiveRef.current) {
      liveRunQueuedRef.current = true;
      return;
    }
    void runLiveInference();
  }

  async function startRecording() {
    if ((!canRun && !recording) || recording) return;
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          noiseSuppression: true,
          echoCancellation: true,
        },
      });
      mediaStreamRef.current = stream;
      recordedChunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      recordingMimeTypeRef.current = mimeType;
      liveRunActiveRef.current = false;
      liveRunQueuedRef.current = false;
      liveRunChunkCountRef.current = 0;
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
          queueLiveInference();
        }
      };
      recorder.onstart = () => {
        setRecording(true);
        setRecordingStatus("Recording from microphone");
        setAudioFileName("Live microphone recording");
        setResult(null);
      };
      recorder.onerror = () => {
        setRecording(false);
        setRecordingStatus("Mic error");
        setError("Microphone recording failed");
      };
      recorder.onstop = () => {
        setRecording(false);
        setRecordingStatus("Processing final microphone audio");
        queueLiveInference();
        mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
      };
      recorder.start(2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to access microphone");
      setRecordingStatus("Mic permission denied");
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }

  const metrics = useMemo(() => {
    if (!result) return [];
    return [
      ["Inference", formatDuration(result.timings.inferenceMs)],
      ["Decode", formatDuration(result.timings.decodeMs)],
      ["Match", formatDuration(result.timings.matchMs)],
      ["Total", formatDuration(result.timings.totalMs)],
      ["Audio", formatDuration(result.audioDurationMs)],
    ] as const;
  }, [result]);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <section className="grid gap-6">
        <Card className="overflow-hidden border-sky-200/10 bg-slate-950/70">
          <CardHeader>
            <div className="flex flex-wrap items-center gap-3">
              <Badge>On-device</Badge>
              <Badge variant="secondary">c2c-direct-mixed</Badge>
              <Badge variant={modelState === "ready" ? "success" : "warning"}>{modelState === "ready" ? "Cached" : "Not ready"}</Badge>
            </div>
            <CardTitle className="text-3xl">Offline Tarteel</CardTitle>
            <CardDescription>
              Download the mixed FastConformer once, cache it in the browser, and run Quran verse recognition fully on the device.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <div className="mb-3 flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm font-medium text-white">Model download</div>
                  <div className="text-sm text-slate-400">{modelStatus}</div>
                  {downloadDetails ? <div className="text-sm text-slate-500">{downloadDetails}</div> : null}
                </div>
                {cachedSize ? <Badge variant="success">{formatBytes(cachedSize)}</Badge> : null}
              </div>
              <Progress value={modelProgress} />
              <div className="mt-4 flex flex-wrap gap-3">
                <Button onClick={ensureModel} disabled={!config || modelState === "downloading"}>
                  {modelState === "downloading" ? <LoaderCircle className="size-4 animate-spin" /> : <Download className="size-4" />}
                  {modelState === "ready" ? "Recheck model" : "Download model"}
                </Button>
                <Button variant="secondary" onClick={handleResetCache} disabled={modelState === "downloading"}>
                  <Trash2 className="size-4" />
                  Clear cache
                </Button>
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <div className="mb-3 flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm font-medium text-white">Run inference</div>
                  <div className="text-sm text-slate-400">Upload a clip or record from the microphone. Inference stays in-browser.</div>
                </div>
                <Badge variant="secondary">{audioFileName || "No file"}</Badge>
              </div>
              <div className="flex flex-wrap gap-3">
                <Button variant="secondary" onClick={recording ? stopRecording : startRecording} disabled={!canRun && !recording}>
                  {recording ? <LoaderCircle className="size-4 animate-spin" /> : <Mic className="size-4" />}
                  {recording ? "Stop recording" : "Use microphone"}
                </Button>
                <Button onClick={() => fileInputRef.current?.click()} disabled={!canRun}>
                  {running ? <LoaderCircle className="size-4 animate-spin" /> : <Upload className="size-4" />}
                  {running ? "Running" : "Choose audio"}
                </Button>
                <Button variant="secondary" onClick={() => setResult(null)} disabled={!result || running}>
                  <TimerReset className="size-4" />
                  Clear result
                </Button>
              </div>
              <div className="mt-3 text-sm text-slate-400">{recordingStatus}</div>
              <input
                ref={fileInputRef}
                className="hidden"
                type="file"
                accept="audio/*"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleFile(file);
                  event.currentTarget.value = "";
                }}
              />
            </div>

            {error ? <div className="rounded-2xl border border-rose-300/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">{error}</div> : null}
          </CardContent>
        </Card>

      </section>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card className="bg-slate-950/60">
          <CardHeader>
            <CardTitle>Best Match</CardTitle>
            <CardDescription>Top result from the locally cached mixed model.</CardDescription>
          </CardHeader>
          <CardContent>
            {topMatch ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-3">
                  <Badge>{`${topMatch.surah}:${topMatch.ayah}`}</Badge>
                  <Badge variant="success">{(topMatch.score * 100).toFixed(1)}%</Badge>
                  <Badge variant="secondary">{topMatch.surah_name_en}</Badge>
                  {result ? <Badge variant="secondary">{result.source.toUpperCase()}</Badge> : null}
                </div>
                <div className="text-right font-[var(--font-arabic)] text-3xl leading-loose text-white" dir="rtl" lang="ar">
                  {topMatch.text_uthmani}
                </div>
                <div className="text-sm text-slate-300">{topMatch.surah_name}</div>
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-white/10 px-4 py-10 text-center text-sm text-slate-400">No result yet.</div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-slate-950/60">
          <CardHeader>
            <CardTitle>Timings</CardTitle>
            <CardDescription>Expose inference timing so the device behavior is obvious.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              {metrics.length > 0
                ? metrics.map(([label, value]) => (
                    <div key={label} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</div>
                      <div className="mt-2 text-lg font-semibold text-white">{value}</div>
                    </div>
                  ))
                : <div className="col-span-full rounded-2xl border border-dashed border-white/10 px-4 py-10 text-center text-sm text-slate-400">Run an audio clip to see timing breakdown.</div>}
            </div>

            {result ? (
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-2 flex items-center gap-2 text-white"><Mic className="size-4" />Transcript</div>
                <div className="text-right font-[var(--font-arabic)] text-xl leading-loose text-slate-100" dir="rtl" lang="ar">
                  {result.transcript || "-"}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="bg-slate-950/60">
          <CardHeader>
            <CardTitle>Top Candidates</CardTitle>
            <CardDescription>Top local candidates after `c2c-direct` style retrieval and CTC rerank.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {topMatches.length > 0
              ? topMatches.map((match) => (
                  <div key={`${match.surah}:${match.ayah}`} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <div className="mb-3 flex flex-wrap items-center gap-3">
                      <Badge>{`${match.surah}:${match.ayah}`}</Badge>
                      <Badge variant="secondary">{match.surah_name_en}</Badge>
                      <div className="text-sm text-slate-400">{(match.score * 100).toFixed(1)}%</div>
                    </div>
                    <div className="text-right font-[var(--font-arabic)] text-2xl leading-loose text-slate-100" dir="rtl" lang="ar">
                      {match.text_uthmani}
                    </div>
                  </div>
                ))
              : <div className="rounded-2xl border border-dashed border-white/10 px-4 py-10 text-center text-sm text-slate-400">Results appear here after inference.</div>}
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
