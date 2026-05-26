import type { SchemaRegistry } from '@/types/arag'

export async function fetchSchema(): Promise<SchemaRegistry> {
  const res = await fetch('/api/v1/ui/schema')
  if (!res.ok) throw new Error(`Failed to fetch schema: ${res.status}`)
  return res.json()
}
