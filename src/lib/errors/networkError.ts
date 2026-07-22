/**
 * 识别浏览器 fetch / 后端 worker 抛出的网络与连接类错误，映射为统一友好文案。
 */
const NETWORK_ERROR_PATTERNS: RegExp[] = [
  /failed to fetch/i,
  /networkerror/i,
  /network request failed/i,
  /load failed/i,
  /fetch failed/i,
  /err_internet_disconnected/i,
  /err_network_changed/i,
  /err_connection_refused/i,
  /err_connection_reset/i,
  /err_connection_timed_out/i,
  /err_name_not_resolved/i,
  /connection refused/i,
  /connection reset/i,
  /connection error/i,
  /connection aborted/i,
  /econnrefused/i,
  /econnreset/i,
  /etimedout/i,
  /timed out/i,
  /timeout/i,
  /name or service not known/i,
  /getaddrinfo/i,
  /no route to host/i,
  /network is unreachable/i,
  /host unreachable/i,
  /max retries exceeded/i,
  /httpsconnectionpool/i,
  /connectionerror/i,
  /socket hang up/i,
  /network unreachable/i,
  /sslerror/i,
  /certificate verify failed/i,
];

export const DEFAULT_NETWORK_CONNECTION_MESSAGE =
  '网络连接超时，请检查网络连接或稍后重试';

export function isNetworkOrConnectionError(raw?: string | null): boolean {
  const msg = String(raw ?? '').trim();
  if (!msg) return false;
  return NETWORK_ERROR_PATTERNS.some((re) => re.test(msg));
}

/**
 * 若为网络/连接类错误则返回 friendlyMessage，否则返回原始文案（trim 后）。
 */
export function resolveNetworkConnectionMessage(
  raw?: string | null,
  friendlyMessage: string = DEFAULT_NETWORK_CONNECTION_MESSAGE
): string {
  const msg = String(raw ?? '').trim();
  if (!msg) return friendlyMessage;
  if (isNetworkOrConnectionError(msg)) return friendlyMessage;
  return msg;
}
