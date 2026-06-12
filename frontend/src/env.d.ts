/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend API origin. Empty in dev (Vite proxy); the HF Space URL in prod. */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
