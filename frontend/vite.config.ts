import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/explain': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
