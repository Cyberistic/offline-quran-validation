import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import alchemy from "alchemy/cloudflare/vite";
import { existsSync } from "node:fs";
import { defineConfig } from "vite";

const hasAlchemyConfig = existsSync(".alchemy/local/wrangler.jsonc");

export default defineConfig({
  plugins: [react(), tailwindcss(), ...(hasAlchemyConfig ? [alchemy()] : [])],
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
