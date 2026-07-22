export function formatRelativeTime(isoTime: string, nowMs: number = Date.now()): string {
  const t = new Date(isoTime).getTime();
  if (!Number.isFinite(t)) return '—';
  const diffMs = Math.max(0, nowMs - t);
  const diffMin = Math.floor(diffMs / (60 * 1000));
  if (diffMin <= 0) return '刚刚';
  if (diffMin < 60) return `${diffMin}分钟前`;
  const diffHour = Math.floor(diffMin / 60);
  if (diffHour < 24) return `${diffHour}小时前`;
  const diffDay = Math.floor(diffHour / 24);
  if (diffDay < 7) return `${diffDay}天前`;
  const diffWeek = Math.floor(diffDay / 7);
  return `${diffWeek}周前`;
}

type Locale = 'zh-CN' | 'en' | 'sv';
type TFunc = (path: string, vars?: Record<string, string | number>) => string;

export function formatRelativeTimeByLocale(
  isoTime: string,
  locale: Locale,
  t: TFunc,
  nowMs: number = Date.now()
): string {
  const ts = new Date(isoTime).getTime();
  if (!Number.isFinite(ts)) return '—';
  const diffMs = Math.max(0, nowMs - ts);
  const diffMin = Math.floor(diffMs / (60 * 1000));
  if (diffMin <= 0) return t('relativeTime.justNow');
  if (diffMin < 60) return t('relativeTime.minutesAgo', { n: diffMin });
  const diffHour = Math.floor(diffMin / 60);
  if (diffHour < 24) return t('relativeTime.hoursAgo', { n: diffHour });
  const diffDay = Math.floor(diffHour / 24);
  if (diffDay < 7) return t('relativeTime.daysAgo', { n: diffDay });
  const diffWeek = Math.floor(diffDay / 7);
  return t('relativeTime.weeksAgo', { n: diffWeek });
}

const localeToBCP: Record<Locale, string> = {
  'zh-CN': 'zh-CN',
  en: 'en',
  sv: 'sv',
};

export function formatDateTimeByLocale(isoTime: string, locale: Locale): string {
  try {
    const d = new Date(isoTime);
    return d.toLocaleString(localeToBCP[locale], {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoTime;
  }
}

