import type { Locale, Messages } from './types';
import { zhCNMessages } from './messages/zh-CN';
import { enMessages } from './messages/en';
import { svMessages } from './messages/sv';

export const LOCALE_STORAGE_KEY = 'epi_locale' as const;

export function getStoredLocale(): Locale | null {
  if (typeof window === 'undefined') return null;
  const v = window.localStorage.getItem(LOCALE_STORAGE_KEY);
  if (v === 'zh-CN' || v === 'en' || v === 'sv') return v;
  return null;
}

export function setStoredLocale(locale: Locale) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
}

export function detectBrowserLocale(): Locale | null {
  if (typeof navigator === 'undefined') return null;
  const lang = (navigator.language || '').toLowerCase();
  if (lang.startsWith('zh')) return 'zh-CN';
  if (lang.startsWith('en')) return 'en';
  if (lang.startsWith('sv')) return 'sv';
  return null;
}

export function resolveInitialLocale(): Locale {
  return getStoredLocale() ?? detectBrowserLocale() ?? 'zh-CN';
}

export function getMessages(locale: Locale): Messages {
  switch (locale) {
    case 'en':
      return enMessages;
    case 'sv':
      return svMessages;
    case 'zh-CN':
    default:
      return zhCNMessages;
  }
}

function formatTemplate(template: string, vars?: Record<string, string | number>) {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_, k: string) => {
    const v = vars[k];
    return v == null ? `{${k}}` : String(v);
  });
}

export function tByPath(
  messages: Messages,
  path: string,
  vars?: Record<string, string | number>
): string {
  const parts = path.split('.').filter(Boolean);
  let cur: any = messages;
  for (const p of parts) {
    cur = cur?.[p];
  }
  if (typeof cur !== 'string') return path;
  return formatTemplate(cur, vars);
}

