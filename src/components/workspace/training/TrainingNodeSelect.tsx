'use client';

import { useEffect, useRef, useState } from 'react';
import {
  formatTrainingNodeStatusLabel,
  trainingNodeStatusBadgeStyle,
  type TrainingDeviceOption,
  type TrainingDeviceValue,
} from '@/lib/workspace/trainingDevice';
import { workspaceFormFieldClassName } from '@/components/workspace/WorkspaceCenteredModal';

function TrainingNodeStatusBadge({
  status,
  label,
}: {
  status?: TrainingDeviceOption['status'];
  label?: string;
}) {
  const style = trainingNodeStatusBadgeStyle(status);
  const text = status ? formatTrainingNodeStatusLabel(status) : label || '未知';
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 10px',
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 500,
        backgroundColor: style.bg,
        color: style.color,
        whiteSpace: 'nowrap',
        flexShrink: 0,
      }}
    >
      {text}
    </span>
  );
}

export function TrainingNodeSelect({
  options,
  value,
  onChange,
  disabled,
}: {
  options: TrainingDeviceOption[];
  value: TrainingDeviceValue | string;
  onChange: (value: TrainingDeviceValue) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const selected =
    options.find((item) => item.value === value || item.nodeId === value) ?? options[0];

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
    <div ref={rootRef} style={{ position: 'relative' }}>
      <button
        type="button"
        className={workspaceFormFieldClassName}
        disabled={disabled || !selected}
        onClick={() => setOpen((prev) => !prev)}
        aria-haspopup="listbox"
        aria-expanded={open}
        style={{
          width: '100%',
          minHeight: 40,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          padding: '8px 12px',
          textAlign: 'left',
          cursor: disabled ? 'not-allowed' : 'pointer',
          backgroundColor: disabled ? '#f9fafb' : '#fff',
        }}
      >
        <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {selected?.label ?? '选择训练节点'}
        </span>
        {selected ? <TrainingNodeStatusBadge status={selected.status} label={selected.statusLabel} /> : null}
      </button>

      {open && !disabled ? (
        <ul
          role="listbox"
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 'calc(100% + 4px)',
            margin: 0,
            padding: 6,
            listStyle: 'none',
            backgroundColor: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: 8,
            boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)',
            zIndex: 20,
            maxHeight: 240,
            overflow: 'auto',
          }}
        >
          {options.map((option) => {
            const isSelected = option.nodeId === selected?.nodeId;
            const itemDisabled = option.selectable === false;
            return (
              <li key={option.nodeId} role="option" aria-selected={isSelected}>
                <button
                  type="button"
                  disabled={itemDisabled}
                  onClick={() => {
                    if (itemDisabled) return;
                    onChange(option.value);
                    setOpen(false);
                  }}
                  style={{
                    width: '100%',
                    border: 'none',
                    background: isSelected ? '#eff6ff' : 'transparent',
                    borderRadius: 6,
                    padding: '10px 12px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'space-between',
                    gap: 12,
                    cursor: itemDisabled ? 'not-allowed' : 'pointer',
                    opacity: itemDisabled ? 0.5 : 1,
                    textAlign: 'left',
                  }}
                >
                  <span style={{ minWidth: 0 }}>
                    <span
                      style={{
                        display: 'block',
                        fontSize: 14,
                        color: '#111827',
                        lineHeight: 1.4,
                      }}
                    >
                      {option.label}
                    </span>
                    {option.status === 'busy' && option.message ? (
                      <span style={{ display: 'block', marginTop: 4, fontSize: 12, color: '#92400e' }}>
                        {option.message}
                      </span>
                    ) : null}
                  </span>
                  <TrainingNodeStatusBadge status={option.status} label={option.statusLabel} />
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
