<template>
  <div class="drivers-page">
    <div class="page-header">
      <div class="page-header-left">
        <h1 class="page-title">Sources</h1>
        <span class="page-subtitle">Drivers shared across all agents</span>
      </div>
      <button class="btn-primary" @click="openAdd">+ Add driver</button>
    </div>

    <!-- Empty state -->
    <div v-if="workflowStore.allDrivers.length === 0" class="empty-state">
      <div class="empty-icon">⛁</div>
      <p>No drivers configured yet.</p>
      <p class="empty-hint">Drivers provide data sources (search engines, databases, APIs) that agents can query.</p>
      <button class="btn-primary" @click="openAdd">Add your first driver</button>
    </div>

    <!-- Drivers table -->
    <div v-else class="table-wrap">
      <table class="drivers-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Identifier</th>
            <th>Provider</th>
            <th>Agent</th>
            <th class="th-actions"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="d in workflowStore.allDrivers"
            :key="`${d.agentId}:${d.identifier}`"
            class="driver-row"
            @click="openEdit(d)"
          >
            <td class="td-name">{{ d.name || d.identifier }}</td>
            <td class="td-identifier"><code>{{ d.identifier }}</code></td>
            <td class="td-provider">
              <span class="provider-badge">{{ d.provider }}</span>
            </td>
            <td class="td-agent"><code>{{ d.agentId }}</code></td>
            <td class="td-actions" @click.stop>
              <button class="btn-edit" @click="openEdit(d)" title="Edit">✎</button>
              <button class="btn-delete" @click="confirmDelete(d)" title="Delete">✕</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Edit / Add panel -->
    <DriverEditPanel
      v-if="panelOpen"
      :driver="editingDriver"
      :agent-id="editingAgentId"
      @save="onSave"
      @close="panelOpen = false"
    />

    <!-- Delete confirmation -->
    <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
      <div class="modal">
        <h3 class="modal-title">Remove driver?</h3>
        <p class="modal-body">
          Remove <strong>{{ deleteTarget.name || deleteTarget.identifier }}</strong>
          (<code>{{ deleteTarget.identifier }}</code>) from agent
          <code>{{ deleteTarget.agentId }}</code>?
        </p>
        <p class="modal-warn">Any agent config referencing this identifier will break.</p>
        <div class="modal-actions">
          <button class="btn-ghost" @click="deleteTarget = null">Cancel</button>
          <button class="btn-danger" @click="doDelete">Remove</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useWorkflowStore } from '@/stores/workflow'
import type { DriverConfig } from '@/types/arag'
import DriverEditPanel from '@/components/drivers/DriverEditPanel.vue'

const workflowStore = useWorkflowStore()

// ── Panel state ────────────────────────────────────────────────────────────
const panelOpen = ref(false)
const editingDriver = ref<DriverConfig | null>(null)
const editingAgentId = ref<string>('')

function openAdd() {
  editingDriver.value = null
  editingAgentId.value = workflowStore.activeAgentId ?? workflowStore.agentIds[0] ?? ''
  panelOpen.value = true
}

function openEdit(d: DriverConfig & { agentId: string }) {
  editingDriver.value = { ...d }
  editingAgentId.value = d.agentId
  panelOpen.value = true
}

function onSave(payload: { agentId: string; driver: DriverConfig; isNew: boolean }) {
  if (payload.isNew) {
    workflowStore.addDriver(payload.agentId, payload.driver)
  } else {
    workflowStore.updateDriver(payload.agentId, payload.driver.identifier, payload.driver)
  }
  panelOpen.value = false
  workflowStore.save()
}

// ── Delete ─────────────────────────────────────────────────────────────────
const deleteTarget = ref<(DriverConfig & { agentId: string }) | null>(null)

function confirmDelete(d: DriverConfig & { agentId: string }) {
  deleteTarget.value = d
}

function doDelete() {
  if (!deleteTarget.value) return
  workflowStore.removeDriver(deleteTarget.value.agentId, deleteTarget.value.identifier)
  deleteTarget.value = null
  workflowStore.save()
}
</script>

<style scoped>
.drivers-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  padding: 28px 32px;
  gap: 24px;
  background: #13131f;
  position: relative;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.page-header-left {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.page-title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #e0e0e0;
}

.page-subtitle {
  font-size: 13px;
  color: #666;
}

.btn-primary {
  background: #7c5cfc;
  border: none;
  color: #fff;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-primary:hover { background: #6a4de8; }

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  gap: 12px;
  color: #555;
}

.empty-icon { font-size: 56px; opacity: 0.4; }

.empty-state p {
  margin: 0;
  font-size: 14px;
  text-align: center;
}

.empty-hint { color: #444; font-size: 12px !important; max-width: 380px; }

/* Table */
.table-wrap {
  flex: 1;
  overflow-y: auto;
  border: 1px solid #1e1e30;
  border-radius: 10px;
}

.drivers-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.drivers-table th {
  text-align: left;
  padding: 10px 14px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #666;
  background: #0e0e1c;
  border-bottom: 1px solid #1e1e30;
  position: sticky;
  top: 0;
}

.th-actions { width: 72px; }

.driver-row {
  cursor: pointer;
  border-bottom: 1px solid #1a1a2e;
  transition: background 0.1s;
}
.driver-row:hover { background: #1a1a2e; }
.driver-row:last-child { border-bottom: none; }

.drivers-table td {
  padding: 10px 14px;
  color: #ccc;
  vertical-align: middle;
}

.td-name { font-weight: 600; color: #e0e0e0; }

.td-identifier code,
.td-agent code {
  font-family: monospace;
  font-size: 12px;
  background: #1e1e30;
  padding: 2px 6px;
  border-radius: 4px;
  color: #aaa;
}

.provider-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 99px;
  background: #1e2a3a;
  color: #5b9bd5;
  font-size: 11px;
  font-weight: 600;
  text-transform: lowercase;
}

.td-actions {
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: flex-end;
}

.btn-edit, .btn-delete {
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px 6px;
  border-radius: 4px;
  font-size: 14px;
  transition: background 0.1s, color 0.1s;
  line-height: 1;
}

.btn-edit { color: #666; }
.btn-edit:hover { background: #252540; color: #bbb; }

.btn-delete { color: #555; }
.btn-delete:hover { background: rgba(255, 85, 85, 0.15); color: #ff5555; }

/* Delete modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
}

.modal {
  background: #1a1a2e;
  border: 1px solid #2a2a40;
  border-radius: 12px;
  padding: 24px;
  width: 400px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.modal-title {
  margin: 0;
  font-size: 16px;
  font-weight: 700;
  color: #e0e0e0;
}

.modal-body {
  margin: 0;
  font-size: 13px;
  color: #aaa;
  line-height: 1.5;
}

.modal-body code, .modal-warn code {
  font-family: monospace;
  background: #252535;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

.modal-warn {
  margin: 0;
  font-size: 12px;
  color: #ff9800;
}

.modal-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  margin-top: 8px;
}

.btn-ghost {
  background: none;
  border: 1px solid #333;
  color: #aaa;
  padding: 7px 14px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: border-color 0.15s;
}
.btn-ghost:hover { border-color: #666; }

.btn-danger {
  background: transparent;
  border: 1px solid #ff5555;
  color: #ff5555;
  padding: 7px 14px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-danger:hover { background: rgba(255, 85, 85, 0.12); }
</style>
