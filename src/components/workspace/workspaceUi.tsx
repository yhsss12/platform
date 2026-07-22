'use client';

import type { CSSProperties, ReactNode } from 'react';
import type { MockRowStatus } from '@/lib/mock/workspacePagesMock';
import { mockStatusLabel } from '@/lib/mock/workspacePagesMock';

export const WS = {
  gap: 16,
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    border: '1px solid #e5e7eb',
    boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
  } satisfies CSSProperties,
  sectionTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: '#111827',
    marginBottom: 12,
  } satisfies CSSProperties,
};

const STATUS_COLORS: Record<MockRowStatus | 'idle' | 'paused', { bg: string; color: string }> = {
  draft: { bg: '#f3f4f6', color: '#374151' },
  running: { bg: '#dbeafe', color: '#1e40af' },
  completed: { bg: '#d1fae5', color: '#065f46' },
  failed: { bg: '#fee2e2', color: '#991b1b' },
  active: { bg: '#d1fae5', color: '#065f46' },
  archived: { bg: '#f3f4f6', color: '#6b7280' },
  idle: { bg: '#f3f4f6', color: '#374151' },
  paused: { bg: '#fef3c7', color: '#92400e' },
};

const SIM_STATUS_LABEL: Record<string, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  pending: '待运行',
  paused: '已暂停',
  idle: '空闲',
};

export function StatusBadge({
  status,
  label,
}: {
  status: MockRowStatus | 'running' | 'completed' | 'failed' | 'pending' | 'paused' | 'idle';
  label?: string;
}) {
  const key = status as MockRowStatus;
  const sc = STATUS_COLORS[key] ?? STATUS_COLORS.draft;
  const text = label ?? SIM_STATUS_LABEL[status] ?? mockStatusLabel(key);
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 10px',
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 500,
        backgroundColor: sc.bg,
        color: sc.color,
        whiteSpace: 'nowrap',
        flexShrink: 0,
      }}
    >
      {text}
    </span>
  );
}

export function PrimaryButton({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        padding: '8px 16px',
        fontSize: 14,
        fontWeight: 500,
        borderRadius: 8,
        border: 'none',
        backgroundColor: disabled ? '#93c5fd' : '#2563eb',
        color: '#fff',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      {children}
    </button>
  );
}

export function SecondaryButton({
  children,
  onClick,
  disabled,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      onClick={onClick}
      style={{
        padding: '8px 14px',
        fontSize: 14,
        borderRadius: 8,
        border: '1px solid #d1d5db',
        backgroundColor: disabled ? '#f3f4f6' : '#fff',
        color: disabled ? '#9ca3af' : '#374151',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      {children}
    </button>
  );
}

export function GhostButton({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: '6px 10px',
        fontSize: 13,
        borderRadius: 6,
        border: 'none',
        backgroundColor: 'transparent',
        color: disabled ? '#94a3b8' : '#2563eb',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      {children}
    </button>
  );
}

export function SectionCard({
  title,
  children,
  style,
  className,
}: {
  title?: string;
  children: ReactNode;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <div className={className} style={{ ...WS.card, padding: 16, ...style }}>
      {title ? <div style={WS.sectionTitle}>{title}</div> : null}
      {children}
    </div>
  );
}

export function MockActionHint() {
  return null;
}

export function WorkspaceQuickLinks({
  links,
}: {
  links: { label: string; href: string }[];
}) {
  if (links.length === 0) return null;
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 8,
        marginBottom: WS.gap,
      }}
    >
      {links.map((link) => (
        <a
          key={link.href + link.label}
          href={link.href}
          style={{
            fontSize: 13,
            color: '#2563eb',
            textDecoration: 'none',
            padding: '6px 12px',
            borderRadius: 6,
            border: '1px solid #e5e7eb',
            backgroundColor: '#fff',
          }}
        >
          {link.label}
        </a>
      ))}
    </div>
  );
}

export function FilterInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="search"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder ?? '搜索…'}
      style={{
        flex: '1 1 200px',
        minWidth: 180,
        padding: '8px 12px',
        borderRadius: 8,
        border: '1px solid #d1d5db',
        fontSize: 14,
      }}
    />
  );
}

export function PlaceholderPanel({
  label,
  height = 200,
}: {
  label: string;
  height?: number;
}) {
  return (
    <div
      style={{
        height,
        borderRadius: 8,
        border: '2px dashed #d1d5db',
        backgroundColor: '#f9fafb',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#6b7280',
        fontSize: 13,
      }}
    >
      {label}
    </div>
  );
}
