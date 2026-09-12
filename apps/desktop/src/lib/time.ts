/** Parse API timestamps consistently across browsers and both agent tracks.
 *
 * Older LangGraph rows were emitted as naive UTC ISO strings. JavaScript treats
 * those as local time, so an explicit UTC suffix is added only when no offset
 * is present. Offset-bearing timestamps are left untouched.
 */
export function parseApiTimestamp(value: string | null | undefined): Date | null {
  if (!value || !value.trim()) return null;
  const raw = value.trim();
  const normalized = /(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(raw) ? raw : `${raw}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function formatLocalTime(value: string | null | undefined): string {
  return parseApiTimestamp(value)?.toLocaleTimeString() || "";
}

export function formatLocalDateTime(value: string | null | undefined): string {
  return parseApiTimestamp(value)?.toLocaleString() || "";
}

export function timeAgo(value: string | null | undefined): string {
  const date = parseApiTimestamp(value);
  if (!date) return "";
  const elapsed = Math.max(0, Date.now() - date.getTime());
  const minutes = Math.floor(elapsed / 60000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d`;
  return date.toLocaleDateString();
}
