<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

interface ModelRow {
  run_name: string
  model: string
  loss: string
  accuracy: number
  macro_f1: number
  macro_auc: number
  per_class_sensitivity: Record<string, number>
  confusion_png: string
  sensitivity_png: string
}

interface Summary {
  classes: string[]
  deployed: string
  aggregate_figures: string[]
  models: ModelRow[]
}

const MODEL_LABEL: Record<string, string> = {
  resnet18: 'ResNet18',
  efficientnet_b0: 'EfficientNet-B0',
  vit_tiny: 'ViT-Tiny',
}
const LOSS_LABEL: Record<string, string> = {
  ce: 'cross-entropy',
  wce: 'weighted-CE',
  focal: 'focal',
}

const asset = (p: string) => `${import.meta.env.BASE_URL}${p}`

const summary = ref<Summary | null>(null)
const error = ref<string | null>(null)
const selectedRun = ref<string>('')
const figView = ref<'confusion' | 'sensitivity'>('confusion')

onMounted(async () => {
  try {
    const resp = await fetch(asset('results_summary.json'))
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    summary.value = (await resp.json()) as Summary
    selectedRun.value = summary.value.deployed
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load results.'
  }
})

// Highlight the best (max) value per metric column.
const best = computed(() => {
  const m = summary.value?.models ?? []
  const maxBy = (f: (r: ModelRow) => number) => (m.length ? Math.max(...m.map(f)) : NaN)
  return {
    accuracy: maxBy((r) => r.accuracy),
    macro_f1: maxBy((r) => r.macro_f1),
    macro_auc: maxBy((r) => r.macro_auc),
    mel: maxBy((r) => r.per_class_sensitivity.mel),
  }
})

const selected = computed(
  () => summary.value?.models.find((r) => r.run_name === selectedRun.value) ?? null,
)

const fmt = (x: number) => x.toFixed(3)
const label = (r: ModelRow) => `${MODEL_LABEL[r.model]} · ${LOSS_LABEL[r.loss]}`
</script>

<template>
  <div v-if="error" role="alert" class="error-box">無法載入結果：{{ error }}</div>

  <template v-if="summary">
    <!-- Key findings -->
    <div class="card">
      <h2 class="section-title">實驗結果總覽</h2>
      <p class="lead">
        3 架構 × 3 loss 共 9 組，於留出測試集（病灶感知切割，seed 42）的表現。
        <code>sens(mel)</code> 為黑色素瘤敏感度——臨床上最不容漏判的類別。
      </p>
      <ul class="findings">
        <li><strong>ViT-Tiny 整體領先</strong>：accuracy / macro-F1 / macro-AUC 最佳值皆由其取得。</li>
        <li>
          <strong>準確率 ↔ 少數類敏感度的取捨</strong>：cross-entropy 準確率最高但 melanoma
          敏感度最低；weighted-CE / focal 犧牲數個百分點換取更高的 melanoma 敏感度。
        </li>
        <li>部署模型為 <strong>ViT-Tiny + weighted-CE</strong>（macro-F1 最佳、兼顧不平衡）。</li>
      </ul>
    </div>

    <!-- Comparison table -->
    <div class="card">
      <h2 class="section-title">模型比較表</h2>
      <div class="table-wrap">
        <table class="cmp">
          <thead>
            <tr>
              <th class="left">模型</th>
              <th>accuracy</th>
              <th>macro F1</th>
              <th>macro AUC</th>
              <th>sens(mel)</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="r in summary.models"
              :key="r.run_name"
              :class="{ deployed: r.run_name === summary.deployed }"
            >
              <td class="left">
                {{ label(r) }}
                <span v-if="r.run_name === summary.deployed" class="badge">deployed</span>
              </td>
              <td :class="{ top: r.accuracy === best.accuracy }">{{ fmt(r.accuracy) }}</td>
              <td :class="{ top: r.macro_f1 === best.macro_f1 }">{{ fmt(r.macro_f1) }}</td>
              <td :class="{ top: r.macro_auc === best.macro_auc }">{{ fmt(r.macro_auc) }}</td>
              <td :class="{ top: r.per_class_sensitivity.mel === best.mel }">
                {{ fmt(r.per_class_sensitivity.mel) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="note">粗體＝該指標最佳值。</p>
    </div>

    <!-- Aggregate scatter -->
    <div class="card">
      <h2 class="section-title">準確率 vs melanoma 敏感度</h2>
      <img :src="asset('results/model_comparison.png')" class="fig" alt="Accuracy vs melanoma sensitivity" />
    </div>

    <!-- Per-model confusion matrix / sensitivity -->
    <div class="card">
      <div class="picker">
        <h2 class="section-title">逐模型細項</h2>
        <select v-model="selectedRun" class="select" aria-label="選擇模型">
          <option v-for="r in summary.models" :key="r.run_name" :value="r.run_name">
            {{ label(r) }}
          </option>
        </select>
      </div>
      <div class="toggle">
        <button :class="{ on: figView === 'confusion' }" @click="figView = 'confusion'">混淆矩陣</button>
        <button :class="{ on: figView === 'sensitivity' }" @click="figView = 'sensitivity'">per-class 敏感度</button>
      </div>
      <img
        v-if="selected"
        :src="asset(figView === 'confusion' ? selected.confusion_png : selected.sensitivity_png)"
        class="fig"
        :alt="`${label(selected)} ${figView}`"
      />
    </div>

    <!-- Grad-CAM gallery -->
    <div class="card">
      <h2 class="section-title">Grad-CAM：成功 vs 誤判</h2>
      <p class="note">
        上排為高信心正確預測，下排為高信心誤判——其中 melanoma 被以接近 1.0 的信心誤判為良性痣，
        正是不可臨床使用的核心理由。
      </p>
      <img :src="asset('results/grad_cam_gallery.png')" class="fig" alt="Grad-CAM gallery" />
    </div>
  </template>

  <div v-else-if="!error" class="card">載入結果中…</div>
</template>

<style scoped>
.section-title {
  margin: 0 0 0.75rem;
  font-size: 1.05rem;
  font-weight: 700;
  color: #0f172a;
}

.lead {
  margin: 0 0 0.75rem;
  font-size: 0.9rem;
  color: #475569;
}

.findings {
  margin: 0;
  padding-left: 1.1rem;
  font-size: 0.875rem;
  color: #334155;
}

.findings li {
  margin-bottom: 0.4rem;
}

code {
  background: #f1f5f9;
  padding: 0.05rem 0.3rem;
  border-radius: 4px;
  font-size: 0.85em;
}

/* ── Comparison table ────────────────────────────────────────────────────── */
.table-wrap {
  overflow-x: auto;
}

.cmp {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  font-variant-numeric: tabular-nums;
}

.cmp th,
.cmp td {
  padding: 0.5rem 0.625rem;
  text-align: right;
  border-bottom: 1px solid #e2e8f0;
  white-space: nowrap;
}

.cmp th.left,
.cmp td.left {
  text-align: left;
}

.cmp thead th {
  color: #64748b;
  font-weight: 600;
}

.cmp tr.deployed {
  background: #eff6ff;
}

.cmp td.top {
  font-weight: 700;
  color: #1d4ed8;
}

.badge {
  display: inline-block;
  margin-left: 0.4rem;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #1d4ed8;
  background: #dbeafe;
  padding: 0.1rem 0.4rem;
  border-radius: 10px;
  vertical-align: middle;
}

.note {
  margin: 0.625rem 0 0;
  font-size: 0.78rem;
  color: #94a3b8;
}

/* ── Figures ─────────────────────────────────────────────────────────────── */
.fig {
  width: 100%;
  border-radius: 8px;
  display: block;
}

/* ── Per-model picker ────────────────────────────────────────────────────── */
.picker {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
}

.picker .section-title {
  margin: 0;
}

.select {
  padding: 0.4rem 0.625rem;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  font-size: 0.875rem;
  background: #fff;
  color: #1e293b;
  cursor: pointer;
}

.toggle {
  display: flex;
  gap: 0.4rem;
  margin-bottom: 0.875rem;
}

.toggle button {
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  color: #475569;
  font-size: 0.8125rem;
  font-weight: 600;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}

.toggle button.on {
  background: #3b82f6;
  border-color: #3b82f6;
  color: #fff;
}
</style>
