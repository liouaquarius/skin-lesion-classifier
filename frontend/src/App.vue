<script setup lang="ts">
import { computed, ref } from 'vue'

interface PredictionResponse {
  predicted_class: string
  confidence: number
  probabilities: Record<string, number>
}

const CLASS_LABELS: Record<string, string> = {
  akiec: 'Actinic Keratosis / IEC',
  bcc: 'Basal Cell Carcinoma',
  bkl: 'Benign Keratosis',
  df: 'Dermatofibroma',
  mel: 'Melanoma',
  nv: 'Melanocytic Nevi',
  vasc: 'Vascular Lesion',
}

const fileInput = ref<HTMLInputElement>()
const file = ref<File | null>(null)
const preview = ref<string | null>(null)
const loading = ref(false)
const result = ref<PredictionResponse | null>(null)
const error = ref<string | null>(null)
const dragActive = ref(false)

function handleFileInput(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (f) loadFile(f)
}

function handleDrop(e: DragEvent) {
  dragActive.value = false
  const f = e.dataTransfer?.files[0]
  if (f) loadFile(f)
}

function loadFile(f: File) {
  if (preview.value) URL.revokeObjectURL(preview.value)
  file.value = f
  preview.value = URL.createObjectURL(f)
  result.value = null
  error.value = null
}

async function runPrediction() {
  if (!file.value) return
  loading.value = true
  error.value = null
  result.value = null
  try {
    const body = new FormData()
    body.append('file', file.value)
    const resp = await fetch('/predict', { method: 'POST', body })
    if (!resp.ok) {
      const detail = await resp.json().catch(() => null)
      throw new Error((detail as { detail?: string })?.detail ?? `HTTP ${resp.status}`)
    }
    result.value = (await resp.json()) as PredictionResponse
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Prediction failed.'
  } finally {
    loading.value = false
  }
}

const sortedProbs = computed(() =>
  result.value
    ? Object.entries(result.value.probabilities).sort(([, a], [, b]) => b - a)
    : [],
)
</script>

<template>
  <div class="page">
    <header class="site-header">
      <h1 class="site-title">Skin Lesion Classifier</h1>
      <p class="site-subtitle">HAM10000 · 7 classes · CNN vs Vision Transformer</p>
    </header>

    <main class="container">
      <div role="note" class="disclaimer">
        <strong>Research and educational use only.</strong>
        This tool is not a medical device and must not be used for clinical diagnosis.
      </div>

      <div class="card">
        <!-- Drop zone -->
        <div
          class="drop-zone"
          :class="{ 'drop-zone--active': dragActive, 'drop-zone--filled': !!preview }"
          role="button"
          tabindex="0"
          aria-label="Upload dermoscopy image — click or drag and drop"
          @click="fileInput?.click()"
          @keydown.enter.prevent="fileInput?.click()"
          @keydown.space.prevent="fileInput?.click()"
          @dragover.prevent="dragActive = true"
          @dragleave.prevent="dragActive = false"
          @drop.prevent="handleDrop"
        >
          <img v-if="preview" :src="preview" class="preview-img" alt="Selected lesion image" />
          <div v-else class="drop-prompt">
            <svg
              class="upload-icon"
              xmlns="http://www.w3.org/2000/svg"
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p class="drop-title">Drop a dermoscopy image</p>
            <p class="drop-hint">or click to select · JPEG / PNG</p>
          </div>
        </div>
        <input
          ref="fileInput"
          type="file"
          accept="image/jpeg,image/png"
          style="display: none"
          @change="handleFileInput"
        />

        <button class="predict-btn" :disabled="!file || loading" @click="runPrediction">
          <span v-if="loading" class="spinner" aria-hidden="true" />
          {{ loading ? 'Classifying…' : 'Classify Lesion' }}
        </button>
      </div>

      <!-- Error -->
      <div v-if="error" role="alert" class="error-box">{{ error }}</div>

      <!-- Results -->
      <div v-if="result" class="card result-card">
        <div class="result-header">
          <div>
            <div class="result-class">{{ CLASS_LABELS[result.predicted_class] }}</div>
            <div class="result-code">{{ result.predicted_class }}</div>
          </div>
          <div class="confidence-badge">
            {{ (result.confidence * 100).toFixed(1) }}%
          </div>
        </div>

        <div class="prob-list">
          <div v-for="[cls, p] in sortedProbs" :key="cls" class="prob-row">
            <span class="prob-name">{{ CLASS_LABELS[cls] ?? cls }}</span>
            <div class="bar-track" role="progressbar" :aria-valuenow="+(p * 100).toFixed(1)" aria-valuemin="0" aria-valuemax="100">
              <div
                class="bar-fill"
                :class="{ 'bar-fill--top': cls === result!.predicted_class }"
                :style="{ width: `${(p * 100).toFixed(1)}%` }"
              />
            </div>
            <span class="prob-pct">{{ (p * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
/* ── Layout ──────────────────────────────────────────────────────────────── */
.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.site-header {
  background: #0f172a;
  color: #f8fafc;
  padding: 1.5rem 1rem;
  text-align: center;
}

.site-title {
  margin: 0 0 0.25rem;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.site-subtitle {
  margin: 0;
  font-size: 0.875rem;
  color: #94a3b8;
}

.container {
  max-width: 600px;
  width: 100%;
  margin: 2rem auto;
  padding: 0 1rem;
  flex: 1;
}

/* ── Disclaimer ──────────────────────────────────────────────────────────── */
.disclaimer {
  background: #fef9c3;
  border: 1px solid #fde047;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  font-size: 0.8125rem;
  color: #713f12;
  margin-bottom: 1.25rem;
}

/* ── Card ────────────────────────────────────────────────────────────────── */
.card {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 1px 4px rgb(0 0 0 / 0.08);
  padding: 1.5rem;
  margin-bottom: 1.25rem;
}

/* ── Drop zone ───────────────────────────────────────────────────────────── */
.drop-zone {
  border: 2px dashed #cbd5e1;
  border-radius: 10px;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  overflow: hidden;
  outline: none;
}

.drop-zone:focus-visible {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgb(59 130 246 / 0.2);
}

.drop-zone:hover,
.drop-zone--active {
  border-color: #3b82f6;
  background: #eff6ff;
}

.drop-zone--filled {
  border-style: solid;
  border-color: #e2e8f0;
}

.drop-prompt {
  text-align: center;
  color: #64748b;
  padding: 2rem;
}

.upload-icon {
  color: #94a3b8;
  margin-bottom: 0.75rem;
}

.drop-title {
  margin: 0 0 0.25rem;
  font-weight: 500;
  color: #475569;
}

.drop-hint {
  margin: 0;
  font-size: 0.8125rem;
  color: #94a3b8;
}

.preview-img {
  width: 100%;
  height: 100%;
  max-height: 320px;
  object-fit: contain;
  display: block;
}

/* ── Button ──────────────────────────────────────────────────────────────── */
.predict-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  width: 100%;
  margin-top: 1rem;
  padding: 0.75rem 1.5rem;
  background: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}

.predict-btn:hover:not(:disabled) {
  background: #2563eb;
}

.predict-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgb(255 255 255 / 0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.65s linear infinite;
  flex-shrink: 0;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ── Error ───────────────────────────────────────────────────────────────── */
.error-box {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  color: #991b1b;
  font-size: 0.875rem;
  margin-bottom: 1.25rem;
}

/* ── Result card ─────────────────────────────────────────────────────────── */
.result-card {
  animation: fade-in 0.25s ease;
}

@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
}

.result-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.result-class {
  font-size: 1.125rem;
  font-weight: 700;
  color: #0f172a;
}

.result-code {
  font-size: 0.8125rem;
  color: #64748b;
  font-family: ui-monospace, monospace;
  margin-top: 0.125rem;
}

.confidence-badge {
  background: #dbeafe;
  color: #1d4ed8;
  font-weight: 700;
  font-size: 1rem;
  padding: 0.375rem 0.875rem;
  border-radius: 20px;
  white-space: nowrap;
  flex-shrink: 0;
}

/* ── Probability chart ───────────────────────────────────────────────────── */
.prob-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.prob-row {
  display: grid;
  grid-template-columns: 11rem 1fr 3.5rem;
  align-items: center;
  gap: 0.625rem;
  font-size: 0.8125rem;
}

.prob-name {
  color: #475569;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.bar-track {
  background: #f1f5f9;
  border-radius: 4px;
  height: 10px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: #93c5fd;
  border-radius: 4px;
  transition: width 0.4s ease;
}

.bar-fill--top {
  background: #3b82f6;
}

.prob-pct {
  text-align: right;
  color: #64748b;
  font-variant-numeric: tabular-nums;
}
</style>
