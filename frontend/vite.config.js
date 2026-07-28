import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies to the FastAPI backend on :8000 so the frontend can
// call /api, load /media crop images, and open the /ws ingest-progress socket
// with no CORS fuss.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 127.0.0.1 (not "localhost") matches uvicorn's IPv4 bind, and disabling
      // the proxy timeouts prevents large video uploads from being dropped.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, timeout: 0, proxyTimeout: 0 },
      '/media': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
