'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import type { Locale } from '@/lib/i18n/types';
import { useI18n } from '@/components/common/I18nProvider';

const LOCALE_ORDER: Locale[] = ['zh-CN', 'en', 'sv'];

export default function LanguageSwitcher({ size = 'sm' }: { size?: 'sm' | 'md' }) {
  const { locale, setLocale, t } = useI18n();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDocMouseDown = (e: MouseEvent) => {
      const el = rootRef.current;
      if (!el) return;
      if (!el.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, []);

  const options = useMemo(() => {
    const map: Record<Locale, string> = {
      'zh-CN': t('common.chinese'),
      en: t('common.english'),
      sv: t('common.swedish'),
    };
    return LOCALE_ORDER.map((value) => ({ value, label: map[value] }));
  }, [t]);

  const current = options.find((x) => x.value === locale) ?? options[0];
  const padY = size === 'md' ? 6 : 4;
  const fontSize = size === 'md' ? 13 : 12;
  const minWidth = size === 'md' ? 116 : 108;

  return (
    <div ref={rootRef} style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', flexShrink: 0 }}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        style={{
          height: size === 'md' ? 30 : 28,
          minWidth,
          padding: `${padY}px 12px`,
          borderRadius: 8,
          border: '1px solid rgba(226,232,240,0.95)',
          backgroundColor: '#f8fafc',
          color: '#334155',
          fontSize,
          fontWeight: 500,
          letterSpacing: '-0.005em',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          boxShadow: open ? '0 10px 32px rgba(15,23,42,0.10)' : 'none',
          transition: 'background-color 120ms ease, box-shadow 120ms ease, border-color 120ms ease',
          whiteSpace: 'nowrap',
        }}
        onMouseEnter={(e) => {
          if (open) return;
          e.currentTarget.style.backgroundColor = '#ffffff';
          e.currentTarget.style.borderColor = 'rgba(203,213,225,0.95)';
        }}
        onMouseLeave={(e) => {
          if (open) return;
          e.currentTarget.style.backgroundColor = '#f8fafc';
          e.currentTarget.style.borderColor = 'rgba(226,232,240,0.95)';
        }}
      >
        <span style={{ lineHeight: 1, whiteSpace: 'nowrap' }}>{current.label}</span>
        <ChevronDown
          size={14}
          style={{ color: '#64748b', flexShrink: 0, transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 120ms ease' }}
        />
      </button>

      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute',
            right: 0,
            top: 'calc(100% + 8px)',
            minWidth: 168,
            backgroundColor: '#ffffff',
            border: '1px solid rgba(226,232,240,0.95)',
            borderRadius: 10,
            boxShadow: '0 18px 60px rgba(2,6,23,0.14)',
            padding: '6px',
            zIndex: 100,
          }}
        >
          {options.map((opt) => {
            const active = opt.value === locale;
            return (
              <button
                key={opt.value}
                type="button"
                role="menuitem"
                onClick={() => {
                  setLocale(opt.value);
                  setOpen(false);
                }}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 10,
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  backgroundColor: active ? 'rgba(37,99,235,0.08)' : 'transparent',
                  color: '#0f172a',
                  fontSize: 13,
                  cursor: 'pointer',
                  transition: 'background-color 120ms ease',
                }}
                onMouseEnter={(e) => {
                  if (active) return;
                  e.currentTarget.style.backgroundColor = '#f8fafc';
                }}
                onMouseLeave={(e) => {
                  if (active) return;
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <span style={{ color: '#334155' }}>{opt.label}</span>
                <span style={{ width: 16, display: 'inline-flex', justifyContent: 'center' }}>
                  {active ? <Check size={16} strokeWidth={2.2} style={{ color: '#2563eb' }} /> : null}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

