<template>
  <div class="kv-field">
    <div v-if="pairs.length > 0" class="kv-rows">
      <div v-for="(pair, i) in pairs" :key="i" class="kv-row">
        <input
          class="kv-input kv-key"
          :value="pair.k"
          placeholder="key"
          @input="updateKey(i, ($event.target as HTMLInputElement).value)"
        />
        <span class="kv-sep">:</span>
        <input
          class="kv-input kv-val"
          :value="pair.v"
          placeholder="value"
          @input="updateVal(i, ($event.target as HTMLInputElement).value)"
        />
        <button class="kv-remove" type="button" @click="remove(i)">×</button>
      </div>
    </div>
    <button class="kv-add" type="button" @click="add">+ Add entry</button>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  modelValue: Record<string, string>
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', val: Record<string, string>): void
}>()

interface Pair { k: string; v: string }

const pairs = computed((): Pair[] =>
  Object.entries(props.modelValue ?? {}).map(([k, v]) => ({ k, v: String(v) })),
)

function toDictionary(ps: Pair[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (const { k, v } of ps) if (k) out[k] = v
  return out
}

function add() {
  emit('update:modelValue', toDictionary([...pairs.value, { k: '', v: '' }]))
}

function remove(idx: number) {
  const ps = [...pairs.value]
  ps.splice(idx, 1)
  emit('update:modelValue', toDictionary(ps))
}

function updateKey(idx: number, k: string) {
  const ps = pairs.value.map((p, i) => (i === idx ? { ...p, k } : p))
  emit('update:modelValue', toDictionary(ps))
}

function updateVal(idx: number, v: string) {
  const ps = pairs.value.map((p, i) => (i === idx ? { ...p, v } : p))
  emit('update:modelValue', toDictionary(ps))
}
</script>

<style scoped>
.kv-field { display: flex; flex-direction: column; gap: 6px; }

.kv-rows { display: flex; flex-direction: column; gap: 4px; }

.kv-row {
  display: flex; align-items: center; gap: 4px;
}

.kv-input {
  background: #13131f; border: 1px solid #333; border-radius: 5px;
  padding: 5px 8px; color: #e0e0e0; font-size: 12px; outline: none;
  transition: border-color 0.15s; box-sizing: border-box; flex: 1;
}
.kv-input:focus { border-color: #7c5cfc; }
.kv-key { flex: 0 0 38%; font-family: monospace; }

.kv-sep { color: #555; font-size: 13px; flex-shrink: 0; }

.kv-remove {
  background: none; border: none; cursor: pointer;
  color: #555; font-size: 16px; line-height: 1; padding: 2px 4px;
  transition: color 0.1s; flex-shrink: 0;
}
.kv-remove:hover { color: #ff5555; }

.kv-add {
  background: none; border: 1px dashed #333; border-radius: 5px;
  color: #666; font-size: 12px; padding: 4px 10px; cursor: pointer;
  transition: border-color 0.15s, color 0.15s; align-self: flex-start;
}
.kv-add:hover { border-color: #7c5cfc; color: #7c5cfc; }
</style>
