'use client';

import React, { createContext, useContext, useEffect, useMemo } from 'react';
import type { Locale, Messages } from '@/lib/i18n/types';
import { tByPath } from '@/lib/i18n';
import { useLocaleStore } from '@/store/localeStore';

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  messages: Messages;
  t: (path: string, vars?: Record<string, string | number>) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const locale = useLocaleStore((s) => s.locale);
  const messages = useLocaleStore((s) => s.messages);
  const setLocale = useLocaleStore((s) => s.setLocale);
  const hydrateFromStorage = useLocaleStore((s) => s.hydrateFromStorage);

  useEffect(() => {
    hydrateFromStorage();
  }, [hydrateFromStorage]);

  const value = useMemo<I18nContextValue>(() => {
    return {
      locale,
      setLocale,
      messages,
      t: (path, vars) => tByPath(messages, path, vars),
    };
  }, [locale, setLocale, messages]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error('useI18n must be used within I18nProvider');
  }
  return ctx;
}

