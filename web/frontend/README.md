# Offline Tarteel Web

React + Vite SPA for on-device Quran verse recognition.

## What it does

- Downloads `fastconformer_full_mixed.onnx` into the browser on first use
- Caches the model locally with IndexedDB
- Runs ONNX inference fully on-device in a web worker
- Shows timing breakdown for inference, decode, match, and total runtime
- Deploys to Cloudflare with Alchemy

## Deploy

1. Log in once:

```bash
npx alchemy login
```

2. Set the public model URL. This should point to your R2-hosted mixed ONNX:

```bash
export MODEL_URL="https://<your-r2-public-domain>/fastconformer_full_mixed.onnx"
```

3. Deploy:

```bash
npm run deploy
```

The Alchemy config binds the site to `quran-ml.sudokw.com`.

## Local dev

```bash
npm run dev
```

If you only want to run a plain Vite production build check, use:

```bash
npm run build
```

## Notes

- `alchemy.run.ts` provisions:
  - the Cloudflare website worker
  - an R2 bucket named `offline-tarteel-models`
- The app currently uses the real mixed ONNX for inference, then greedy transcript matching against `quran.json` in-browser.
- `MODEL_URL` is served at runtime via `/api/config`, so the SPA can stay static.
