export function safeOAuthUrl(value: string): string | undefined {
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:' || !url.hostname || url.username || url.password) {
      return undefined
    }
    return url.href
  } catch {
    return undefined
  }
}
