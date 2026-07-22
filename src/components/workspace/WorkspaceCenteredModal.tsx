'use client';

import { useEffect, type CSSProperties, type ReactNode } from 'react';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import './workspaceModalForm.css';

export const workspaceFormFieldClassName = 'ws-form-field';

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.45)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1500,
  padding: 24,
};

function buildOverlayStyle(zIndex: number): React.CSSProperties {
  return { ...overlayStyle, zIndex };
}

export function WorkspaceCenteredModal({
  open,
  title,
  titleId,
  width = 800,
  zIndex = 1500,
  onClose,
  steps,
  step,
  children,
  footer,
}: {
  open: boolean;
  title: string;
  titleId: string;
  width?: number;
  zIndex?: number;
  onClose: () => void;
  steps?: readonly string[];
  step?: number;
  children: ReactNode;
  footer: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      style={buildOverlayStyle(zIndex)}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      aria-hidden={false}
    >
      <div
        role="dialog"
        aria-modal
        aria-labelledby={titleId}
        style={{
          width,
          maxWidth: '96vw',
          maxHeight: '90vh',
          backgroundColor: '#fff',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 24px 80px rgba(15, 23, 42, 0.18)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexShrink: 0,
          }}
        >
          <div>
            <h2 id={titleId} style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#111827' }}>
              {title}
            </h2>
            {steps && step != null ? (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
                {steps.map((s, i) => {
                  const n = i + 1;
                  const active = step === n;
                  const done = step > n;
                  return (
                    <span
                      key={s}
                      style={{
                        fontSize: 11,
                        padding: '4px 8px',
                        borderRadius: 4,
                        backgroundColor: active ? '#2563eb' : done ? '#e0e7ff' : '#f3f4f6',
                        color: active ? '#fff' : done ? '#3730a3' : '#6b7280',
                      }}
                    >
                      {n}. {s}
                    </span>
                  );
                })}
              </div>
            ) : null}
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>{children}</div>

        <div
          style={{
            padding: '16px 20px',
            borderTop: '1px solid #e5e7eb',
            flexShrink: 0,
          }}
        >
          {footer}
        </div>
      </div>
    </div>
  );
}

export const workspaceModalFieldLabel: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  marginBottom: 4,
  display: 'block',
};

export const workspaceModalSelectStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  fontSize: 13,
  lineHeight: 1.45,
  color: '#111827',
  backgroundColor: '#ffffff',
  border: '1px solid #e2e8f0',
  borderRadius: 8,
  marginBottom: 14,
  boxSizing: 'border-box',
  minHeight: 38,
};

export const workspaceModalSectionLabel: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: '#6b7280',
  marginBottom: 8,
};

/** 弹窗内两列表单栅格（与生成任务数据弹窗一致） */
export function WorkspaceModalFieldGrid({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
        gap: '0 16px',
      }}
    >
      {children}
    </div>
  );
}

export const workspaceModalAdvancedToggleStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  marginTop: 20,
  padding: 0,
  border: 'none',
  background: 'none',
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  cursor: 'pointer',
};

export const workspaceModalAdvancedPanelStyle: CSSProperties = {
  marginTop: 10,
  padding: '14px 16px',
  borderRadius: 8,
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
};

export const workspaceModalDebugToggleStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  width: '100%',
  marginTop: 16,
  padding: 0,
  border: 'none',
  background: 'none',
  fontSize: 12,
  fontWeight: 600,
  color: '#6b7280',
  cursor: 'pointer',
};

export const workspaceModalDebugPanelStyle: CSSProperties = {
  marginTop: 10,
  padding: '10px 12px',
  borderRadius: 6,
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
  fontSize: 12,
  color: '#374151',
  lineHeight: 1.6,
};
