<template>
  <div class="canvas-wrap">
    <VueFlow
      :nodes="initialNodes"
      :edges="initialEdges"
      :default-zoom="0.85"
      :min-zoom="0.2"
      :max-zoom="2"
      fit-view-on-init
      class="arag-flow"
      @node-click="onNodeClick"
      @node-drag-stop="onNodeDragStop"
      @pane-click="onPaneClick"
    >
      <Background pattern-color="#2a2a3e" :gap="20" />
      <Controls />
      <MiniMap node-color="#555" />

      <template #node-agent="props">
        <AgentNode v-bind="props" :selected="props.id === selectedNodeId" />
      </template>
      <template #node-stageGate="props">
        <StageGateNode v-bind="props" />
      </template>
      <template #node-io="props">
        <IONode v-bind="props" />
      </template>
      <template #node-branchPanel="props">
        <ConditionalBranchPanel v-bind="props" />
      </template>
    </VueFlow>
  </div>
</template>

<script setup lang="ts">
import { computed, watchEffect } from 'vue'
import {
  VueFlow,
  useVueFlow,
  type Node,
  type Edge,
  type NodeMouseEvent,
  type NodeDragEvent,
  Position,
  MarkerType,
} from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import { MiniMap } from '@vue-flow/minimap'
import '@vue-flow/core/dist/style.css'
import '@vue-flow/core/dist/theme-default.css'
import '@vue-flow/controls/dist/style.css'
import '@vue-flow/minimap/dist/style.css'

import AgentNode from '@/components/nodes/AgentNode.vue'
import StageGateNode from '@/components/nodes/StageGateNode.vue'
import IONode from '@/components/nodes/IONode.vue'
import ConditionalBranchPanel from '@/components/nodes/ConditionalBranchPanel.vue'

import { useWorkflowStore } from '@/stores/workflow'
import {
  STAGE_ORDER,
  CONNECTABLE_KEYS,
  type StageType,
  type AgentStepConfig,
  type ConnectableKey,
} from '@/types/arag'

const store = useWorkflowStore()
const selectedNodeId = computed(() => store.selectedNodeId)
const { setNodes, setEdges, fitView } = useVueFlow()

// ── Layout constants ───────────────────────────────────────────────────────
const COL_X: Record<string, number> = {
  input: 40,
  preprocess: 260,
  preprocess_gate: 500,
  context: 600,
  context_gate: 840,
  generation: 940,
  generation_gate: 1180,
  postprocess: 1280,
  output: 1560,
}
const AGENT_Y_START = 180
const AGENT_Y_STEP = 170
const GATE_Y = 280

// ── Per-link styling table ─────────────────────────────────────────────────
//
// `lane` keys render as a stacked column below the parent (with a background
// panel). `right` keys render to the right of the parent.
type LinkLayout = 'lane' | 'right'
interface LinkStyle {
  layout: LinkLayout
  color: string
  label: string
  dashed: boolean
  /** y offset (lane: top of lane below parent; right: y delta from parent) */
  yOffset: number
}

const LINK_STYLES: Record<ConnectableKey, LinkStyle> = {
  then:              { layout: 'lane',  color: '#4caf50', label: 'then',       dashed: true,  yOffset: 180 },
  else_:             { layout: 'lane',  color: '#ff9800', label: 'else',       dashed: true,  yOffset: 360 },
  fallback:          { layout: 'right', color: '#e57373', label: 'fallback',   dashed: true,  yOffset: 60 },
  next_agent:        { layout: 'right', color: '#64b5f6', label: 'next',       dashed: false, yOffset: 0 },
  agents:            { layout: 'right', color: '#9575cd', label: 'agent',      dashed: false, yOffset: 0 },
  registered_agents: { layout: 'right', color: '#7c5cfc', label: 'registered', dashed: false, yOffset: 0 },
}

const SUB_X_OFFSET = 240          // x indent of every child level
const SUB_LANE_X_STEP = 220       // horizontal step between siblings in a lane
const SUB_LANE_Y_STEP = 150       // vertical step within a lane
const SUB_RIGHT_Y_STEP = 130      // vertical step between right-stacked siblings

// ── Builders ───────────────────────────────────────────────────────────────

function buildNodes(): Node[] {
  const nodes: Node[] = []
  const wf = store.activeWorkflow
  const positions = (wf as Record<string, unknown> | null)?._ui as
    | Record<string, { x: number; y: number }>
    | undefined ?? {}

  // I/O terminals
  nodes.push({
    id: 'input', type: 'io',
    position: positions['input'] ?? { x: COL_X.input, y: GATE_Y },
    data: { label: 'Input', icon: '▶' },
    sourcePosition: Position.Right, targetPosition: Position.Left,
  })
  nodes.push({
    id: 'output', type: 'io',
    position: positions['output'] ?? { x: COL_X.output, y: GATE_Y },
    data: { label: 'Output', icon: '⬛' },
    sourcePosition: Position.Right, targetPosition: Position.Left,
  })

  // Stage gate nodes
  const gates = [
    { id: 'gate_preprocess', label: 'Preprocess\nDone', x: COL_X.preprocess_gate },
    { id: 'gate_context',    label: 'Context\nDone',    x: COL_X.context_gate },
    { id: 'gate_generation', label: 'Generation\nDone', x: COL_X.generation_gate },
  ]
  for (const g of gates) {
    nodes.push({
      id: g.id, type: 'stageGate',
      position: positions[g.id] ?? { x: g.x, y: GATE_Y },
      data: { label: g.label },
      sourcePosition: Position.Right, targetPosition: Position.Left,
    })
  }

  // Top-level + nested
  if (wf) {
    for (const stage of STAGE_ORDER) {
      const steps = (wf[stage] as AgentStepConfig[]) ?? []
      steps.forEach((step, idx) => {
        const nodeId = `${stage}:${idx}`
        const pos = positions[nodeId] ?? {
          x: COL_X[stage],
          y: AGENT_Y_START + idx * AGENT_Y_STEP,
        }
        nodes.push({
          id: nodeId, type: 'agent',
          position: pos,
          data: { stage, stepIndex: idx, config: step },
          sourcePosition: Position.Right, targetPosition: Position.Left,
          zIndex: 1,
        })
        _addSubagentNodes(nodes, step, nodeId, stage, pos, positions)
      })
    }
  }

  return nodes
}

function _addSubagentNodes(
  nodes: Node[],
  step: AgentStepConfig,
  parentId: string,
  parentStage: StageType,
  parentPos: { x: number; y: number },
  positions: Record<string, { x: number; y: number }>,
) {
  for (const key of CONNECTABLE_KEYS) {
    const value = step[key]
    if (!value) continue

    const style = LINK_STYLES[key]
    // Smart-agent children always render in the context stage.
    const childStage: StageType =
      key === 'registered_agents' || key === 'agents' ? 'context' : parentStage

    // Normalize to (childList, isSingle).
    const isSingle = !Array.isArray(value)
    const children = (isSingle ? [value as AgentStepConfig] : (value as AgentStepConfig[]))

    if (children.length === 0) continue

    // Lane background panel (only for then/else_).
    if (style.layout === 'lane' && !isSingle) {
      const panelId = `${parentId}:${key}:__panel`
      nodes.push({
        id: panelId,
        type: 'branchPanel',
        position: positions[panelId] ?? {
          x: parentPos.x + SUB_X_OFFSET - 12,
          y: parentPos.y + style.yOffset - 28,
        },
        data: { label: style.label, branch: key },
        style: {
          width: `${Math.max(children.length, 1) * SUB_LANE_X_STEP + 24}px`,
          height: `${children.length * SUB_LANE_Y_STEP + 16}px`,
        },
        draggable: false,
        selectable: false,
        zIndex: 0,
      })
    }

    // Track per-link sibling counts so that multiple "right"-laid
    // children of the same parent stack vertically without overlapping.
    children.forEach((child, idx) => {
      const childPath = isSingle
        ? `${parentId}:${key}:*`
        : `${parentId}:${key}:${idx}`

      let defaultPos: { x: number; y: number }
      if (style.layout === 'lane') {
        defaultPos = {
          x: parentPos.x + SUB_X_OFFSET + idx * 20,
          y: parentPos.y + style.yOffset + idx * SUB_LANE_Y_STEP,
        }
      } else {
        // right-of-parent; stack siblings vertically when there are several
        defaultPos = {
          x: parentPos.x + SUB_X_OFFSET,
          y: parentPos.y + style.yOffset + idx * SUB_RIGHT_Y_STEP,
        }
      }

      const childPos = positions[childPath] ?? defaultPos
      nodes.push({
        id: childPath, type: 'agent',
        position: childPos,
        data: {
          stage: childStage,
          stepIndex: idx,
          config: child,
          parentLinkType: key,
        },
        sourcePosition: Position.Right, targetPosition: Position.Left,
        zIndex: 2,
      })
      _addSubagentNodes(nodes, child, childPath, childStage, childPos, positions)
    })
  }
}

// ── Edges ──────────────────────────────────────────────────────────────────

function buildEdges(): Edge[] {
  const edges: Edge[] = []
  const wf = store.activeWorkflow

  const edgeStyle = { stroke: '#555', strokeWidth: 2 }
  const markerEnd = { type: MarkerType.ArrowClosed, color: '#555' }

  const link = (id: string, source: string, target: string) =>
    edges.push({ id, source, target, style: edgeStyle, markerEnd })

  // Stage pipeline edges
  const stagePairs: Array<[StageType, string, string]> = [
    ['preprocess',  'input',           'gate_preprocess'],
    ['context',     'gate_preprocess', 'gate_context'],
    ['generation',  'gate_context',    'gate_generation'],
    ['postprocess', 'gate_generation', 'output'],
  ]
  for (const [stage, sourceId, targetId] of stagePairs) {
    const steps = (wf?.[stage] as AgentStepConfig[] | undefined) ?? []
    if (steps.length === 0) {
      link(`${sourceId}→${targetId}`, sourceId, targetId)
    } else {
      steps.forEach((_, i) => {
        link(`${sourceId}→${stage}${i}`, sourceId, `${stage}:${i}`)
        link(`${stage}${i}→${targetId}`, `${stage}:${i}`, targetId)
      })
    }
  }

  // Subagent edges
  if (wf) {
    for (const stage of STAGE_ORDER) {
      const steps = (wf[stage] as AgentStepConfig[]) ?? []
      steps.forEach((step, idx) => {
        _addSubagentEdges(edges, step, `${stage}:${idx}`)
      })
    }
  }

  return edges
}

function _addSubagentEdges(edges: Edge[], step: AgentStepConfig, parentId: string) {
  for (const key of CONNECTABLE_KEYS) {
    const value = step[key]
    if (!value) continue

    const style = LINK_STYLES[key]
    const isSingle = !Array.isArray(value)
    const children = isSingle ? [value as AgentStepConfig] : (value as AgentStepConfig[])
    if (children.length === 0) continue

    const edgeStyle: Record<string, unknown> = {
      stroke: style.color,
      strokeWidth: 1.5,
    }
    if (style.dashed) edgeStyle.strokeDasharray = '5,3'
    const marker = { type: MarkerType.ArrowClosed, color: style.color }

    children.forEach((child, idx) => {
      const childPath = isSingle ? `${parentId}:${key}:*` : `${parentId}:${key}:${idx}`
      edges.push({
        id: `${parentId}→${key}${isSingle ? '' : idx}`,
        source: parentId,
        target: childPath,
        style: edgeStyle as Edge['style'],
        markerEnd: marker,
        label: idx === 0 ? style.label : undefined,
      })
      _addSubagentEdges(edges, child, childPath)
    })
  }
}

// ── Reactive ───────────────────────────────────────────────────────────────

const initialNodes = buildNodes()
const initialEdges = buildEdges()

watchEffect(() => {
  setNodes(buildNodes())
  setEdges(buildEdges())
})

// ── Event handlers ─────────────────────────────────────────────────────────

function onNodeClick(event: NodeMouseEvent) {
  if (event.node.type === 'agent') {
    store.selectNode(event.node.id)
  }
}

function onPaneClick() {
  store.selectNode(null)
}

function onNodeDragStop(event: NodeDragEvent) {
  const { id, position } = event.node
  if (id.includes('__panel')) return
  store.updateNodePosition(id, position.x, position.y)
}

watchEffect(() => {
  void store.activeWorkflowId
  void store.activeAgentId
  setTimeout(() => fitView({ padding: 0.12, duration: 300 }), 50)
})
</script>

<style scoped>
.canvas-wrap { width: 100%; height: 100%; }
.arag-flow { background: #13131f; }
</style>
