'use client';

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  workspaceFormFieldClassName,
  workspaceModalFieldLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';

export interface InitWeightsOption {
  value: string;
  titleLine: string;
  subtitleLine?: string;
  disabled?: boolean;
  title?: string;
}

const optionStyle: CSSProperties = {
  width: '100%',
  textAlign: 'left',
  border: 'none',
  background: 'transparent',
  padding: '8px 12px',
  cursor: 'pointer',
  borderBottom: '1px solid #f1f5f9',
};

const lineEllipsis: CSSProperties = {
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

export function InitWeightsSelect({
  label = '初始化权重',
  value,
  options = [],
  placeholder = '随机初始化（默认）',
  onChange,
}: {
  label?: string;
  value: string;
  options?: InitWeightsOption[];
  placeholder?: string;
  onChange: (value: string) => void;
}) {
  const safeOptions = Array.isArray(options) ? options : [];
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const selected = useMemo(
    () => safeOptions.find((item) => item.value === value) ?? null,
    [safeOptions, value]
  );

  const buttonText = selected?.titleLine || placeholder;

  useEffect(() => {
    if (!open) return;
    const onDocClick = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open]);

  return (
    <div ref={rootRef} style={{ position: 'relative', width: '100%', minWidth: 0 }}>
      <label style={workspaceModalFieldLabel}>{label}</label>
      <button
        type="button"
        title={selected?.title || buttonText}
        onClick={() => setOpen((prev) => !prev)}
        className={`${workspaceFormFieldClassName} ws-form-trigger${open ? ' ws-form-trigger-open' : ''}`}
        style={{
          ...workspaceModalSelectStyle,
          width: '100%',
          maxWidth: '100%',
          ...lineEllipsis,
        }}
      >
        {buttonText}
      </button>
      {open ? (
        <div className="ws-form-dropdown" role="listbox">
          <button
            type="button"
            style={{ ...optionStyle, color: !value ? '#1d4ed8' : '#334155' }}
            onClick={() => {
              onChange('');
              setOpen(false);
            }}
          >
            <div style={{ ...lineEllipsis, fontSize: 13, fontWeight: 500 }}>{placeholder}</div>
          </button>
          {safeOptions.map((item) => (
            <button
              key={item.value}
              type="button"
              role="option"
              aria-selected={value === item.value}
              title={item.title || `${item.titleLine}${item.subtitleLine ? `\n${item.subtitleLine}` : ''}`}
              disabled={item.disabled}
              style={{
                ...optionStyle,
                opacity: item.disabled ? 0.45 : 1,
                cursor: item.disabled ? 'not-allowed' : 'pointer',
                backgroundColor: value === item.value ? '#eff6ff' : 'transparent',
              }}
              onClick={() => {
                if (item.disabled) return;
                onChange(item.value);
                setOpen(false);
              }}
            >
              <div style={{ ...lineEllipsis, fontSize: 13, fontWeight: 500, color: '#111827' }}>
                {item.titleLine}
              </div>
              {item.subtitleLine ? (
                <div style={{ ...lineEllipsis, fontSize: 12, color: '#64748b', marginTop: 2 }}>
                  {item.subtitleLine}
                </div>
              ) : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
