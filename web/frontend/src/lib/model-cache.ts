const DB_NAME = "offline-tarteel";
const STORE_NAME = "assets";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
  });
}

async function getValue<T>(key: string): Promise<T | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).get(key);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve((request.result as T | undefined) ?? null);
  });
}

async function setValue(key: string, value: unknown): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE_NAME).put(value, key);
  });
}

export interface CachedModel {
  url: string;
  bytes: ArrayBuffer;
  size: number;
  updatedAt: number;
}

export async function getCachedModel(url: string): Promise<CachedModel | null> {
  const model = await getValue<CachedModel>("model");
  if (!model || model.url !== url) return null;
  return model;
}

export async function cacheModel(model: CachedModel): Promise<void> {
  await setValue("model", model);
}

export async function clearCachedModel(): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(STORE_NAME).delete("model");
  });
}

export async function downloadModel(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<CachedModel> {
  const response = await fetch(url, { mode: "cors" });
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download model: ${response.status}`);
  }

  const total = Number(response.headers.get("content-length") || "0");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  const model = { url, bytes: bytes.buffer, size: loaded, updatedAt: Date.now() };
  await cacheModel(model);
  return model;
}
