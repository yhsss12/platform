/**
 * 格式化文件大小
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

function pad2(n: number): string {
  return String(n).padStart(2, '0');
}

/**
 * 列表「创建时间」等与标注任务列表一致的展示：YYYY-MM-DD HH:mm（本地 24 小时制，不显示秒）。
 * 解析规则与数据资产等模块一致：空格日期时间先归一为 ISO 式再交给 Date，避免无时区歧义。
 */
export function formatDateTimeMinute(dateStr: string | null | undefined): string {
  if (dateStr == null || String(dateStr).trim() === '') return '—';
  const s = String(dateStr).trim();
  const toParse = s.replace(' ', 'T');
  try {
    const date = new Date(toParse);
    if (Number.isNaN(date.getTime())) return s;
    const year = date.getFullYear();
    const month = pad2(date.getMonth() + 1);
    const day = pad2(date.getDate());
    const hours = pad2(date.getHours());
    const minutes = pad2(date.getMinutes());
    return `${year}-${month}-${day} ${hours}:${minutes}`;
  } catch {
    return s;
  }
}

/**
 * 与 formatDateTimeMinute 相同解析规则；日期为 YYYY/MM/DD，时分 HH:mm（本地）。
 */
export function formatDateTimeMinuteYmdSlash(dateStr: string | null | undefined): string {
  if (dateStr == null || String(dateStr).trim() === '') return '—';
  const s = String(dateStr).trim();
  const toParse = s.replace(' ', 'T');
  try {
    const date = new Date(toParse);
    if (Number.isNaN(date.getTime())) return s;
    const year = date.getFullYear();
    const month = pad2(date.getMonth() + 1);
    const day = pad2(date.getDate());
    const hours = pad2(date.getHours());
    const minutes = pad2(date.getMinutes());
    return `${year}/${month}/${day} ${hours}:${minutes}`;
  } catch {
    return s;
  }
}

/** YYYY-MM-DD → YYYY/MM/DD；非法则返回 trim 后的原串。 */
export function isoDateToYmdSlashDisplay(iso: string | null | undefined): string {
  if (iso == null || String(iso).trim() === '') return '';
  const s = String(iso).trim().slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return String(iso).trim();
  const [y, m, d] = s.split('-');
  return `${y}/${m}/${d}`;
}

/**
 * 筛选用：YYYY/MM/DD 或 YYYY-MM-DD。
 * @returns undefined 清空；false 非法；否则 YYYY-MM-DD
 */
export function parseYmdSlashOrDashToIso(raw: string): string | undefined | false {
  const norm = raw.trim().replace(/\//g, '-');
  if (!norm) return undefined;
  const head = norm.slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(head)) return false;
  const [ys, ms, ds] = head.split('-');
  const y = Number(ys);
  const mo = Number(ms);
  const d = Number(ds);
  const check = new Date(Date.UTC(y, mo - 1, d));
  if (
    check.getUTCFullYear() !== y ||
    check.getUTCMonth() !== mo - 1 ||
    check.getUTCDate() !== d
  ) {
    return false;
  }
  return head;
}

/**
 * 将后端返回的时间字符串格式化为本地时间 YYYY/MM/DD HH:mm:ss。
 * - 若字符串已带时区（Z / ±HH:mm），按标准时间解析。
 * - 若字符串无时区，不做任何手动 +8/-8，按本地时间解析（避免重复偏移）。
 */
export function formatDateTimeToLocal(dateStr: string | null | undefined): string {
  if (dateStr == null || String(dateStr).trim() === '') return '—';
  const s = String(dateStr).trim();
  const toParse = s.replace(' ', 'T');
  try {
    const date = new Date(toParse);
    if (Number.isNaN(date.getTime())) return s;
    const y = date.getFullYear();
    const m = pad2(date.getMonth() + 1);
    const d = pad2(date.getDate());
    const hh = pad2(date.getHours());
    const mm = pad2(date.getMinutes());
    const ss = pad2(date.getSeconds());
    return `${y}/${m}/${d} ${hh}:${mm}:${ss}`;
  } catch {
    return s;
  }
}























