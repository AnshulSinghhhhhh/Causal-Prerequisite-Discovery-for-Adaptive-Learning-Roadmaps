import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/graph': 'http://127.0.0.1:8000',
      '/quiz': 'http://127.0.0.1:8000',
      '/remediation': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});