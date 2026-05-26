import type { StandaloneConfig } from '@/types/arag'

export async function fetchConfig(): Promise<StandaloneConfig> {
  const res = await fetch('/api/v1/ui/config')
  if (!res.ok) throw new Error(`Failed to fetch config: ${res.status}`)
  return res.json()
}

export async function saveConfig(config: StandaloneConfig): Promise<void> {
  const res = await fetch('/api/v1/ui/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Failed to save config: ${res.status} — ${body}`)
  }
}
