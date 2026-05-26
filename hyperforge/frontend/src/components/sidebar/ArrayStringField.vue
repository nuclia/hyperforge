<template>
  <div class="array-string-field">
    <div class="tags">
      <span v-for="(tag, i) in modelValue" :key="i" class="tag">
        {{ tag }}
        <button class="tag-remove" type="button" @click="remove(i)">×</button>
      </span>
      <span v-if="modelValue.length === 0" class="empty-hint">No items yet</span>
    </div>
    <input
      ref="inputRef"
      class="tag-input"
      :placeholder="placeholder ?? 'Type and press Enter…'"
      @keydown.enter.prevent="add"
      @keydown.tab.prevent="add"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  modelValue: string[]
  placeholder?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', val: string[]): void
}>()

const inputRef = ref<HTMLInputElement | null>(null)

function add() {
  const val = inputRef.value?.value.trim()
  if (!val) return
  emit('update:modelValue', [...props.modelValue, val])
  if (inputRef.value) inputRef.value.value = ''
}

function remove(idx: number) {
  const arr = [...props.modelValue]
  arr.splice(idx, 1)
  emit('update:modelValue', arr)
}
</script>

<style scoped>
.array-string-field { display: flex; flex-direction: column; gap: 6px; }

.tags { display: flex; flex-wrap: wrap; gap: 6px; min-height: 24px; align-items: center; }

.tag {
  display: inline-flex; align-items: center; gap: 4px;
  background: #1e2a3a; color: #7eb8f7;
  padding: 3px 8px; border-radius: 99px;
  font-size: 12px;
}

.tag-remove {
  background: none; border: none; cursor: pointer;
  color: #5b8bb5; font-size: 14px; line-height: 1; padding: 0;
  transition: color 0.1s;
}
.tag-remove:hover { color: #ff5555; }

.empty-hint { font-size: 12px; color: #444; font-style: italic; }

.tag-input {
  background: #13131f; border: 1px solid #333; border-radius: 6px;
  padding: 5px 8px; color: #e0e0e0; font-size: 12px; outline: none;
  transition: border-color 0.15s; width: 100%; box-sizing: border-box;
}
.tag-input:focus { border-color: #7c5cfc; }
</style>
