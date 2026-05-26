<template>
  <div class="feedback-form">
    <!-- If there's a response_schema, render it as a form -->
    <div v-if="hasSchema" class="schema-fields">
      <div v-for="(fieldSchema, key) in schemaProperties" :key="key" class="field-row">
        <label class="field-label">{{ fieldSchema.title || key }}</label>
        <p v-if="fieldSchema.description" class="field-desc">{{ fieldSchema.description }}</p>

        <!-- Enum → select -->
        <select v-if="fieldSchema.enum" v-model="formData[key]" class="field-input">
          <option v-for="opt in fieldSchema.enum" :key="String(opt)" :value="opt">
            {{ String(opt) }}
          </option>
        </select>

        <!-- Boolean → checkbox -->
        <label v-else-if="resolvedType(fieldSchema) === 'boolean'" class="checkbox-label">
          <input type="checkbox" v-model="formData[key]" />
          <span>{{ fieldSchema.title || key }}</span>
        </label>

        <!-- Number -->
        <input
          v-else-if="resolvedType(fieldSchema) === 'number' || resolvedType(fieldSchema) === 'integer'"
          type="number"
          v-model.number="formData[key]"
          class="field-input"
          :placeholder="String(fieldSchema.default ?? '')"
        />

        <!-- Default: text -->
        <input
          v-else
          type="text"
          v-model="formData[key]"
          class="field-input"
          :placeholder="String(fieldSchema.default ?? '')"
        />
      </div>
    </div>

    <!-- Fallback: plain textarea -->
    <textarea
      v-else
      v-model="freeText"
      class="feedback-textarea"
      :placeholder="feedback.question"
      rows="3"
    />

    <div class="feedback-actions">
      <button class="submit-btn" :disabled="!canSubmit" @click="onSubmit">Submit response</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive } from 'vue'
import type { Feedback, JsonSchema } from '@/types/arag'

const props = defineProps<{ feedback: Feedback }>()
const emit = defineEmits<{ (e: 'submit', response: string): void }>()

const freeText = ref('')
const formData = reactive<Record<string, unknown>>({})

const hasSchema = computed(() => {
  const s = props.feedback.response_schema
  return s != null && typeof s === 'object' && Object.keys(s.properties ?? {}).length > 0
})

const schemaProperties = computed((): Record<string, JsonSchema> => {
  return props.feedback.response_schema?.properties ?? {}
})

function resolvedType(schema: JsonSchema): string | undefined {
  if (schema.type && typeof schema.type === 'string') return schema.type
  const anyOf = schema.anyOf ?? schema.oneOf
  if (anyOf) {
    const nonNull = anyOf.find((s) => s.type !== 'null')
    if (nonNull?.type && typeof nonNull.type === 'string') return nonNull.type
  }
  return undefined
}

const canSubmit = computed(() => {
  if (hasSchema.value) {
    // Allow submit if at least one field filled, or if no required fields
    return true
  }
  return freeText.value.trim().length > 0
})

function onSubmit() {
  let response: string
  if (hasSchema.value) {
    response = JSON.stringify(formData)
  } else {
    response = freeText.value.trim()
  }
  emit('submit', response)
  freeText.value = ''
  Object.keys(formData).forEach((k) => delete formData[k])
}
</script>

<style scoped>
.feedback-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.schema-fields {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.field-row {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.field-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #6ab0ff;
}

.field-desc {
  font-size: 11px;
  color: #666;
  margin: 0;
}

.field-input {
  background: #13131f;
  border: 1px solid #1a3a60;
  border-radius: 6px;
  padding: 7px 10px;
  color: #e0e0e0;
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}
.field-input:focus { border-color: #2196f3; }

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #e0e0e0;
  cursor: pointer;
}

.feedback-textarea {
  background: #13131f;
  border: 1px solid #1a3a60;
  border-radius: 8px;
  padding: 10px 12px;
  color: #e0e0e0;
  font-size: 13px;
  font-family: inherit;
  resize: vertical;
  outline: none;
  min-height: 80px;
  transition: border-color 0.15s;
}
.feedback-textarea:focus { border-color: #2196f3; }

.feedback-actions { display: flex; justify-content: flex-end; }

.submit-btn {
  background: #2196f3;
  color: #fff;
  border: none;
  padding: 8px 18px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}
.submit-btn:hover:not(:disabled) { background: #1976d2; }
.submit-btn:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
