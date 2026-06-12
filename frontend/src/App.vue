<script setup lang="ts">
import { ref } from 'vue'

import ClassifierView from './views/ClassifierView.vue'
import ResultsView from './views/ResultsView.vue'

type Tab = 'classify' | 'results'
const tab = ref<Tab>('classify')
</script>

<template>
  <div class="page">
    <header class="site-header">
      <h1 class="site-title">Skin Lesion Classifier</h1>
      <p class="site-subtitle">HAM10000 · 7 classes · CNN vs Vision Transformer</p>
      <nav class="tabs" aria-label="Sections">
        <button class="tab" :class="{ 'tab--active': tab === 'classify' }" @click="tab = 'classify'">
          Classifier
        </button>
        <button class="tab" :class="{ 'tab--active': tab === 'results' }" @click="tab = 'results'">
          Results
        </button>
      </nav>
    </header>

    <main class="container" :class="{ 'container--wide': tab === 'results' }">
      <div role="note" class="disclaimer">
        <strong>Research and educational use only.</strong>
        This tool is not a medical device and must not be used for clinical diagnosis.
      </div>

      <ClassifierView v-if="tab === 'classify'" />
      <ResultsView v-else />
    </main>
  </div>
</template>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.site-header {
  background: #0f172a;
  color: #f8fafc;
  padding: 1.5rem 1rem 0;
  text-align: center;
}

.site-title {
  margin: 0 0 0.25rem;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.site-subtitle {
  margin: 0 0 1rem;
  font-size: 0.875rem;
  color: #94a3b8;
}

.tabs {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
}

.tab {
  background: transparent;
  border: none;
  color: #94a3b8;
  font-size: 0.9375rem;
  font-weight: 600;
  padding: 0.625rem 1.25rem;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.tab:hover {
  color: #e2e8f0;
}

.tab--active {
  color: #fff;
  border-bottom-color: #3b82f6;
}
</style>
