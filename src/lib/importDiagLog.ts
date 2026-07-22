/**
 * 浏览器直传导入链路的统一诊断日志（控制台过滤前缀：[eai-import]）。
 */

export function importDiagLog(event: string, payload?: Record<string, unknown>): void {
  if (typeof console === 'undefined' || typeof console.info !== 'function') return;
  try {
    console.info(`[eai-import] ${event}`, payload && Object.keys(payload).length ? payload : '');
  } catch {
    /* ignore */
  }
}
