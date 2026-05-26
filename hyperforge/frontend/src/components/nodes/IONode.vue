<template>
  <div class="io-node" :class="data.icon === '▶' ? 'io-start' : 'io-end'">
    <Handle v-if="data.icon !== '▶'" type="target" :position="Position.Left" />
    <div class="io-icon">{{ data.icon }}</div>
    <div class="io-label">{{ data.label }}</div>
    <Handle v-if="data.icon === '▶'" type="source" :position="Position.Right" />
    <Handle v-if="data.icon !== '▶'" type="source" :position="Position.Right" />
  </div>
</template>

<script setup lang="ts">
import { Handle, Position } from '@vue-flow/core'

defineProps<{
  data: { label: string; icon: string }
}>()
</script>

<style scoped>
.io-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 72px;
  height: 72px;
  border-radius: 50%;
  border: 2px solid #444;
  background: #1e1e30;
  cursor: default;
}

.io-start {
  border-color: #4caf50;
}

.io-end {
  border-color: #888;
}

.io-icon {
  font-size: 18px;
  line-height: 1;
}

.io-label {
  font-size: 10px;
  color: #888;
  margin-top: 2px;
}
</style>
