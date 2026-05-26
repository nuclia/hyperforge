<template>
  <div class="gate-node">
    <Handle type="target" :position="Position.Left" />
    <div class="gate-icon">◆</div>
    <div class="gate-label">{{ data.label }}</div>
    <Handle type="source" :position="Position.Right" />
  </div>
</template>

<script setup lang="ts">
import { Handle, Position } from '@vue-flow/core'

defineProps<{
  data: { label: string }
}>()
</script>

<style scoped>
.gate-node {
  background: #252538;
  border: 2px solid #444;
  border-radius: 8px;
  padding: 8px 14px;
  min-width: 80px;
  text-align: center;
  cursor: default;
}

.gate-icon {
  font-size: 16px;
  color: #888;
  line-height: 1;
}

.gate-label {
  font-size: 10px;
  color: #777;
  white-space: pre-line;
  line-height: 1.3;
  margin-top: 4px;
}
</style>
