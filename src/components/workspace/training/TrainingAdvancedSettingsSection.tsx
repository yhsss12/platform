'use client';

import type { CSSProperties, ReactNode } from 'react';
import { ChevronDown, ChevronRight, RotateCcw } from 'lucide-react';

const headerRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 12,
  minHeight: 38,
  marginTop: 8,
};

const headerToggleStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  padding: 0,
  border: 'none',
  background: 'none',
  cursor: 'pointer',
  minHeight: 38,
  color: '#374151',
};

const bodyStyle: CSSProperties = {
  marginTop: 10,
  padding: '14px 16px',
  borderRadius: 8,
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
};

function restoreButtonStyle(active: boolean): CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    padding: '4px 8px',
    border: 'none',
    borderRadius: 6,
    background: 'none',
    fontSize: 12,
    fontWeight: 500,
    color: active ? '#2563eb' : '#9ca3af',
    cursor: 'pointer',
    transition: 'color 0.15s ease',
  };
}

export function TrainingAdvancedSettingsSection({
  expanded,
  onExpandedChange,
  onRestoreDefaults,
  restoreActive,
  children,
}: {
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  onRestoreDefaults: () => void;
  restoreActive: boolean;
  children: ReactNode;
}) {
  return (
    <div>
      <div style={headerRowStyle}>
        <button
          type="button"
          onClick={() => onExpandedChange(!expanded)}
          style={headerToggleStyle}
          aria-expanded={expanded}
        >
          {expanded ? (
            <ChevronDown size={16} strokeWidth={2} color="#6b7280" aria-hidden />
          ) : (
            <ChevronRight size={16} strokeWidth={2} color="#6b7280" aria-hidden />
          )}
          <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>高级设置</span>
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRestoreDefaults();
          }}
          style={restoreButtonStyle(restoreActive)}
          title="恢复高级设置为默认值"
        >
          <RotateCcw size={14} strokeWidth={2} aria-hidden />
          恢复默认
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateRows: expanded ? '1fr' : '0fr',
          transition: 'grid-template-rows 0.18s ease',
        }}
      >
        <div style={{ overflow: 'hidden', minHeight: 0 }}>
          <div style={{ ...bodyStyle, pointerEvents: expanded ? 'auto' : 'none' }} aria-hidden={!expanded}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

export const workspaceModalFieldErrorStyle: CSSProperties = {
  fontSize: 12,
  color: '#dc2626',
  marginTop: -10,
  marginBottom: 14,
  lineHeight: 1.4,
};

export function modalInputStyle(hasError: boolean): CSSProperties {
  return {
    width: '100%',
    padding: '8px 10px',
    fontSize: 13,
    border: `1px solid ${hasError ? '#dc2626' : '#e5e7eb'}`,
    borderRadius: 6,
    marginBottom: hasError ? 4 : 14,
    boxSizing: 'border-box',
  };
}
