import type { worker } from "../alchemy.run";

export type CloudflareEnv = typeof worker.Env;

declare global {
  type Env = CloudflareEnv;
}

declare module "cloudflare:workers" {
  namespace Cloudflare {
    interface Env extends CloudflareEnv {}
  }
}
