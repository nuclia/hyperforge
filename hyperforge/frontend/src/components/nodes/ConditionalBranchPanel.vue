<template>
  <div class="panel-node" :style="{ background: branchColor, borderColor: borderColor }">
    <div class="panel-label">{{ data.label }}</div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  data: { label: string; branch: 'then' | 'else_' }
}>()

const branchColor = computed(() =>
  props.data.branch === 'then'
    ? 'rgba(76, 175, 80, 0.06)'
    : 'rgba(255, 152, 0, 0.06)',
)
const borderColor = computed(() =>
  props.data.branch === 'then' ? 'rgba(76, 175, 80, 0.3)' : 'rgba(255, 152, 0, 0.3)',
)
</script>

<style scoped>
.panel-node {
  border: 1px dashed;
  border-radius: 10px;
  pointer-events: none;
  width: 100%;
  height: 100%;
}
.panel-label {
  position: absolute;
  top: 6px;
  left: 10px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  opacity: 0.6;
  pointer-events: none;
}
</style>
