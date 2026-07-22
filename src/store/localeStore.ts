'use client';

import { create } from 'zustand';
import type { Locale, Messages } from '@/lib/i18n/types';
import { getMessages, resolveInitialLocale, setStoredLocale } from '@/lib/i18n';

function getInitialLocaleState(): { locale: Locale; messages: Messages; isHydrated: boolean } {
  // 必须与 SSR 首帧一致：服务端无 window 时用 zh-CN；客户端首帧也先用 zh-CN，
  // 再在 I18nProvider 的 useEffect 里 hydrateFromStorage()，否则 t() 在英文浏览器下会 hydration mismatch。
  if (typeof window === 'undefined') {
    return { locale: 'zh-CN', messages: getMessages('zh-CN'), isHydrated: false };
  }
  return { locale: 'zh-CN', messages: getMessages('zh-CN'), isHydrated: false };
}

const initial = getInitialLocaleState();

interface LocaleState {
  locale: Locale;
  messages: Messages;
  isHydrated: boolean;
  setLocale: (locale: Locale) => void;
  hydrateFromStorage: () => void;
}

export const useLocaleStore = create<LocaleState>((set, get) => ({
  locale: initial.locale,
  messages: initial.messages,
  isHydrated: initial.isHydrated,
  setLocale: (locale) => {
    setStoredLocale(locale);
    set({ locale, messages: getMessages(locale) });
  },
  hydrateFromStorage: () => {
    if (get().isHydrated) return;
    const next = resolveInitialLocale();
    setStoredLocale(next);
    set({ locale: next, messages: getMessages(next), isHydrated: true });
  },
}));

