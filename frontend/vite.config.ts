import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Build output goes to /static/app so Flask can serve it
// Base URL is /app/ so asset paths resolve correctly in production
export default defineConfig({
  plugins: [react()],
  base: '/app/',
  build: {
    outDir: '../static/app',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/zerodha': 'http://127.0.0.1:5000',
    },
  },
});
