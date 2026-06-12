<script setup lang="ts">
import { computed, ref } from 'vue'

interface PredictionResponse {
  predicted_class: string
  confidence: number
  probabilities: Record<string, number>
}

interface ExplainResponse extends PredictionResponse {
  grad_cam_image: string | null
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
const explaining = ref(false)
const result = ref<PredictionResponse | null>(null)
const gradCamImage = ref<string | null>(null)
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
  gradCamImage.value = null
  error.value = null
}

async function postImage(endpoint: string): Promise<Response> {
  const body = new FormData()
  body.append('file', file.value!)
  const resp = await fetch(endpoint, { method: 'POST', body })
  if (!resp.ok) {
    const detail = await resp.json().catch(() => null)
    throw new Error((detail as { detail?: string })?.detail ?? `HTTP ${resp.status}`)
  }
  return resp
}

async function runPrediction() {
  if (!file.value) return
  loading.value = true
  error.value = null
  result.value = null
  gradCamImage.value = null
  try {
    const resp = await postImage('/predict')
    result.value = (await resp.json()) as PredictionResponse
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Prediction failed.'
  } finally {
    loading.value = false
  }
}

async function runExplain() {
  if (!file.value) return
  explaining.value = true
  error.value = null
  gradCamImage.value = null
  try {
    const resp = await postImage('/explain')
    const data = (await resp.json()) as ExplainResponse
    result.value = data
    gradCamImage.value = data.grad_cam_image
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Explanation failed.'
  } finally {
    explaining.value = false
  }
}

const sortedProbs = computed(() =>
  result.value
    ? Object.entries(result.value.probabilities).sort(([, a], [, b]) => b - a)
    : [],
)
</script>

<template>
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

    <div class="btn-row">
      <button class="predict-btn" :disabled="!file || loading || explaining" @click="runPrediction">
        <span v-if="loading" class="spinner" aria-hidden="true" />
        {{ loading ? 'Classifying…' : 'Classify' }}
      </button>
      <button
        class="explain-btn"
        :disabled="!file || loading || explaining"
        @click="runExplain"
      >
        <span v-if="explaining" class="spinner spinner--dark" aria-hidden="true" />
        {{ explaining ? 'Generating…' : 'Grad-CAM' }}
      </button>
    </div>
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
        <div
          class="bar-track"
          role="progressbar"
          :aria-valuenow="+(p * 100).toFixed(1)"
          aria-valuemin="0"
          aria-valuemax="100"
        >
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

  <!-- Grad-CAM heatmap -->
  <div v-if="gradCamImage" class="card gradcam-card">
    <p class="gradcam-title">Grad-CAM · activation heatmap</p>
    <img :src="gradCamImage" class="gradcam-img" alt="Grad-CAM activation heatmap" />
    <p class="gradcam-caption">
      Red regions indicate the areas most influential for the predicted class.
    </p>
  </div>
</template>

<style scoped>
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

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.btn-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.75rem;
  margin-top: 1rem;
}

.predict-btn,
.explain-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}

.predict-btn {
  background: #3b82f6;
  color: #fff;
}

.predict-btn:hover:not(:disabled) {
  background: #2563eb;
}

.explain-btn {
  background: #f1f5f9;
  color: #334155;
  border: 1px solid #e2e8f0;
}

.explain-btn:hover:not(:disabled) {
  background: #e2e8f0;
}

.predict-btn:disabled,
.explain-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
.spinner {
  display: inline-block;
  width: 15px;
  height: 15px;
  border: 2px solid rgb(255 255 255 / 0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.65s linear infinite;
  flex-shrink: 0;
}

.spinner--dark {
  border-color: rgb(51 65 85 / 0.25);
  border-top-color: #334155;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Result card ─────────────────────────────────────────────────────────── */
.result-card {
  animation: fade-in 0.25s ease;
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

/* ── Grad-CAM card ───────────────────────────────────────────────────────── */
.gradcam-card {
  animation: fade-in 0.3s ease;
}

.gradcam-title {
  margin: 0 0 0.75rem;
  font-size: 0.8125rem;
  font-weight: 600;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.gradcam-img {
  width: 100%;
  border-radius: 8px;
  display: block;
}

.gradcam-caption {
  margin: 0.625rem 0 0;
  font-size: 0.75rem;
  color: #94a3b8;
}
</style>
