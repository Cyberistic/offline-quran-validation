import alchemy from "alchemy";
import { R2Bucket, Vite } from "alchemy/cloudflare";

const app = await alchemy("offline-tarteel");

export const modelBucket = await R2Bucket("models", {
  name: "offline-tarteel-models",
  adopt: true,
  devDomain: true,
  cors: [
    {
      allowed: {
        origins: ["https://quran-ml.sudokw.com"],
        methods: ["GET", "HEAD"],
        headers: ["*"],
      },
      exposeHeaders: ["Content-Length", "Content-Range", "Accept-Ranges", "ETag"],
      maxAgeSeconds: 86400,
    },
  ],
});

export const worker = await Vite("website", {
  entrypoint: "src/worker.ts",
  domains: [{ domainName: "quran-ml.sudokw.com", adopt: true }],
  bindings: {
    MODEL_URL: process.env.MODEL_URL ?? "https://pub-PLACEHOLDER.r2.dev/fastconformer_full_mixed.onnx",
    MODEL_BUCKET: modelBucket,
  },
});

console.log({
  url: worker.url,
  bucket: modelBucket.name,
  bucketDevDomain: modelBucket.devDomain,
});

await app.finalize();
