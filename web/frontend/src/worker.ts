export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/config") {
      const modelUrl = env.MODEL_URL || "https://pub-PLACEHOLDER.r2.dev/fastconformer_full_mixed.onnx";
      return Response.json({
        modelUrl,
        modelFileName: modelUrl.split("/").pop() || "fastconformer_full_mixed.onnx",
      });
    }
    return new Response("Not Found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;
