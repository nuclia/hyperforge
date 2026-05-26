import type { AragAnswer } from '@/types/arag'

/**
 * Send a question to the ARAG pipeline and yield parsed AragAnswer objects
 * from the NDJSON streaming response.
 */
export async function* streamChat(
  agentId: string,
  sessionId: string,
  question: string,
  workflowId?: string,
): AsyncGenerator<AragAnswer> {
  const url = workflowId
    ? `/api/v1/agent/${agentId}/workflow/${workflowId}/session/${sessionId}`
    : `/api/v1/agent/${agentId}/session/${sessionId}`

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed: ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        yield JSON.parse(trimmed) as AragAnswer
      } catch {
        // ignore non-JSON lines (keepalives, etc.)
      }
    }
  }
  // flush remaining buffer
  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer.trim()) as AragAnswer
    } catch {
      // ignore
    }
  }
}
