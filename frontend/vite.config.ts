import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// GitHub Pages serves the site under /<repo>/; the dev server stays at root.
// The dev proxy forwards API calls to the local backend; in production the
// frontend calls the HF Space directly via VITE_API_BASE (see ClassifierView).
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/skin-lesion-classifier/' : '/',
  plugins: [vue()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/explain': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
}))
