'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';

/** 与数据资产页 FiltersBar 输入框一致的基础样式 */
const filterInputShell: React.CSSProperties = {
  padding: '8px 12px',
  paddingRight: 34,
  backgroundColor: '#ffffff',
  border: '1px solid #d1d5db',
  borderRadius: '6px',
  color: '#111827',
  fontSize: '14px',
  outline: 'none',
  boxSizing: 'border-box',
  width: '100%',
};

/** 与筛选栏一致的弹层内字段外框（日期：展示壳 + 原生 date 覆盖） */
const modalPickShell: React.CSSProperties = {
  position: 'relative',
  width: '100%',
  minHeight: 40,
  border: '1px solid #d1d5db',
  borderRadius: 6,
  backgroundColor: '#ffffff',
  boxSizing: 'border-box',
};

const nativePickOverlay: React.CSSProperties = {
  position: 'absolute',
  left: 0,
  top: 0,
  width: '100%',
  height: '100%',
  margin: 0,
  padding: 0,
  border: 'none',
  opacity: 0,
  cursor: 'pointer',
  zIndex: 2,
  fontSize: 16,
  boxSizing: 'border-box',
  backgroundColor: 'transparent',
};

const modalTextInput: React.CSSProperties = {
  width: '100%',
  padding: '10px 12px',
  fontSize: 14,
  border: 'none',
  outline: 'none',
  borderRadius: 6,
  boxSizing: 'border-box',
  color: '#111827',
  backgroundColor: 'transparent',
};

/** 父级 `YYYY-MM-DDTHH:mm` -> 筛选条展示用 `YYYY-MM-DD HH:mm`（24 小时制） */
export function toDisplayFromDatetimeLocal(iso: string): string {
  if (!iso) return '';
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!m) return iso;
  return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
}

/** 统一为 `YYYY-MM-DDTHH:mm`（来自父级或合并结果） */
export function normalizeDatetimeLocal(raw: string): string {
  if (!raw) return '';
  const m = raw.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})/);
  return m ? m[1] : raw.slice(0, 16);
}

function splitValue(value: string): { date: string; time: string } {
  const n = normalizeDatetimeLocal(value);
  if (!n) return { date: '', time: '' };
  const [d, t] = n.split('T');
  return { date: d || '', time: t || '' };
}

/** 合法则返回 `HH:mm`，否则 `null`（24 小时制 00:00–23:59） */
export function parseValidHm24(s: string): string | null {
  const t = s.trim();
  if (!t) return null;
  const m = t.match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return null;
  const h = parseInt(m[1], 10);
  const mi = parseInt(m[2], 10);
  if (Number.isNaN(h) || Number.isNaN(mi) || h < 0 || h > 23 || mi < 0 || mi > 59) return null;
  return `${String(h).padStart(2, '0')}:${String(mi).padStart(2, '0')}`;
}

/** 最多 4 位数字 → 展示串（1～2 位仅小时；3～4 位自动插入冒号） */
function formatTimeDigitsDisplay(digitsRaw: string): string {
  const d = digitsRaw.replace(/\D/g, '').slice(0, 4);
  if (d.length === 0) return '';
  if (d.length <= 2) return d;
  if (d.length === 3) return `${d.slice(0, 2)}:${d.slice(2)}`;
  return `${d.slice(0, 2)}:${d.slice(2, 4)}`;
}

/** 从输入框当前值提取最多 4 位数字（退格删到 `09:` 等也会正确收敛） */
function parseDigitsFromTimeInput(raw: string): string {
  return raw.replace(/\D/g, '').slice(0, 4);
}

/**
 * 确定时：仅数字串 → `HH:mm`。
 * - 0 位：由调用方用默认 00:00 / 23:59
 * - 1 位：单数字小时，分 00
 * - 2 位：小时（00–23），分 00
 * - 3 位：未完成（缺分钟个位）→ null
 * - 4 位：HHMM
 */
function parseValidHmFromDigits(digitsRaw: string): string | null {
  const d = digitsRaw.replace(/\D/g, '');
  if (d.length === 0) return null;
  if (d.length === 3) return null;
  let h: number;
  let m: number;
  if (d.length === 1) {
    h = parseInt(d[0], 10);
    m = 0;
  } else if (d.length === 2) {
    h = parseInt(d.slice(0, 2), 10);
    m = 0;
  } else {
    h = parseInt(d.slice(0, 2), 10);
    m = parseInt(d.slice(2, 4), 10);
  }
  if (Number.isNaN(h) || Number.isNaN(m) || h < 0 || h > 23 || m < 0 || m > 59) return null;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

function CalendarGlyph() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden style={{ flexShrink: 0 }}>
      <rect x="3" y="5" width="18" height="16" rx="2" stroke="#9ca3af" strokeWidth="1.5" />
      <path d="M3 10h18M8 3v4M16 3v4" stroke="#9ca3af" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export interface AuditDateTimeFilterInputProps {
  /** `YYYY-MM-DDTHH:mm` 或空；仅「确定」或「清空」后由父组件更新 */
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  ariaLabel: string;
  pickAriaLabel: string;
  confirmLabel: string;
  clearLabel: string;
  width?: number;
  /**
   * 仅选日期、未填时间时，确定后补的时分。
   * `start` → 00:00；`end` → 23:59
   */
  timeRangeEnd: 'start' | 'end';
  dateFieldLabel: string;
  timeFieldLabel: string;
  pickDatePlaceholder: string;
  /** 时间手输框 placeholder（如：请输入时间（如 0930）） */
  timeInputPlaceholder: string;
  /** 时间格式非法时提示 */
  timeInvalidHint: string;
}

/**
 * 只读触发条 + 弹层：日期为「展示壳 + 原生 date」；时间为最多 4 位数字，实时显示为 HH:mm；确定/清空写回父级。
 */
export function AuditDateTimeFilterInput({
  value,
  onChange,
  placeholder,
  ariaLabel,
  pickAriaLabel,
  confirmLabel,
  clearLabel,
  width = 168,
  timeRangeEnd,
  dateFieldLabel,
  timeFieldLabel,
  pickDatePlaceholder,
  timeInputPlaceholder,
  timeInvalidHint,
}: AuditDateTimeFilterInputProps) {
  const [open, setOpen] = useState(false);
  const [draftDate, setDraftDate] = useState('');
  /** 仅存 0～4 位数字；展示由 formatTimeDigitsDisplay 实时格式化 */
  const [draftTimeDigits, setDraftTimeDigits] = useState('');
  const [timeError, setTimeError] = useState<string | null>(null);
  const dateInputRef = useRef<HTMLInputElement>(null);

  const closePanel = useCallback(() => {
    setOpen(false);
  }, []);

  const openPanel = useCallback(() => {
    const { date, time } = splitValue(value);
    const hm = time ? parseValidHm24(time) : null;
    setDraftDate(date);
    setDraftTimeDigits(hm ? hm.replace(':', '') : '');
    setTimeError(null);
    setOpen(true);
  }, [value]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closePanel();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, closePanel]);

  const defaultTimeIfEmpty = timeRangeEnd === 'end' ? '23:59' : '00:00';

  const handleConfirm = () => {
    const d = draftDate.trim();
    if (!d) {
      onChange('');
      closePanel();
      return;
    }
    const digits = draftTimeDigits.replace(/\D/g, '');
    let t: string;
    if (!digits) {
      t = defaultTimeIfEmpty;
    } else {
      const parsed = parseValidHmFromDigits(digits);
      if (!parsed) {
        setTimeError(timeInvalidHint);
        return;
      }
      t = parsed;
    }
    onChange(`${d}T${t}`);
    setTimeError(null);
    closePanel();
  };

  const handleClearPanel = () => {
    setDraftDate('');
    setDraftTimeDigits('');
    setTimeError(null);
    onChange('');
    closePanel();
  };

  const displayText = value ? toDisplayFromDatetimeLocal(value) : '';

  const btnBase: React.CSSProperties = {
    padding: '8px 16px',
    borderRadius: 6,
    fontSize: 14,
    cursor: 'pointer',
    fontWeight: 500,
    border: '1px solid #d1d5db',
  };

  return (
    <>
      <div style={{ position: 'relative', width, flex: `0 0 ${width}px`, minWidth: width }}>
        <input
          type="text"
          readOnly
          tabIndex={0}
          autoComplete="off"
          value={displayText}
          placeholder={placeholder}
          aria-label={ariaLabel}
          aria-haspopup="dialog"
          aria-expanded={open}
          onClick={openPanel}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              openPanel();
            }
          }}
          style={{
            ...filterInputShell,
            cursor: 'pointer',
            color: displayText ? '#111827' : '#9ca3af',
          }}
        />
        <button
          type="button"
          aria-label={pickAriaLabel}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            openPanel();
          }}
          style={{
            position: 'absolute',
            right: 6,
            top: '50%',
            transform: 'translateY(-50%)',
            border: 'none',
            background: 'transparent',
            padding: 4,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 4,
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
            <rect x="3" y="5" width="18" height="16" rx="2" stroke="#9ca3af" strokeWidth="1.5" />
            <path d="M3 10h18M8 3v4M16 3v4" stroke="#9ca3af" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {open ? (
        <div
          role="presentation"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 100,
            background: 'rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) {
              e.preventDefault();
              closePanel();
            }
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label={ariaLabel}
            onMouseDown={(e) => e.stopPropagation()}
            style={{
              background: '#ffffff',
              borderRadius: 8,
              padding: 16,
              width: 'min(100%, 360px)',
              boxShadow: '0 10px 40px rgba(0,0,0,0.12)',
              border: '1px solid #e5e7eb',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 10 }}>
              {ariaLabel}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <span style={{ display: 'block', marginBottom: 4, fontSize: 12, color: '#6b7280' }}>{dateFieldLabel}</span>
                <div style={modalPickShell}>
                  <div
                    aria-hidden
                    style={{
                      position: 'absolute',
                      inset: 0,
                      display: 'flex',
                      alignItems: 'center',
                      paddingLeft: 12,
                      paddingRight: 36,
                      pointerEvents: 'none',
                      zIndex: 1,
                      fontSize: 14,
                      color: draftDate ? '#111827' : '#9ca3af',
                    }}
                  >
                    <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {draftDate || pickDatePlaceholder}
                    </span>
                    <span style={{ position: 'absolute', right: 10, display: 'flex', alignItems: 'center' }}>
                      <CalendarGlyph />
                    </span>
                  </div>
                  <input
                    ref={dateInputRef}
                    type="date"
                    value={draftDate}
                    onChange={(e) => setDraftDate(e.target.value)}
                    aria-label={dateFieldLabel}
                    style={{
                      ...nativePickOverlay,
                      outline: 'none',
                    }}
                  />
                </div>
              </div>
              <div>
                <span style={{ display: 'block', marginBottom: 4, fontSize: 12, color: '#6b7280' }}>{timeFieldLabel}</span>
                <div style={{ ...modalPickShell, padding: 0, display: 'flex', flexDirection: 'column' }}>
                  <input
                    type="text"
                    inputMode="numeric"
                    autoComplete="off"
                    placeholder={timeInputPlaceholder}
                    value={formatTimeDigitsDisplay(draftTimeDigits)}
                    onChange={(e) => {
                      setDraftTimeDigits(parseDigitsFromTimeInput(e.target.value));
                      setTimeError(null);
                    }}
                    aria-label={timeFieldLabel}
                    aria-invalid={Boolean(timeError)}
                    style={{
                      ...modalTextInput,
                      color: draftTimeDigits ? '#111827' : '#9ca3af',
                    }}
                  />
                </div>
                {timeError ? (
                  <span style={{ fontSize: 12, color: '#dc2626', marginTop: 6, display: 'block' }} role="alert">
                    {timeError}
                  </span>
                ) : null}
              </div>
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'flex-end',
                gap: 10,
                marginTop: 16,
              }}
            >
              <button type="button" onClick={handleClearPanel} style={{ ...btnBase, background: '#fff', color: '#374151' }}>
                {clearLabel}
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                style={{
                  ...btnBase,
                  background: '#2563eb',
                  borderColor: '#2563eb',
                  color: '#fff',
                }}
              >
                {confirmLabel}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
