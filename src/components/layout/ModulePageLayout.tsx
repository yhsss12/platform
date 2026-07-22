'use client';

import React from 'react';

interface ModulePageHeaderProps {
  title: string;
  /** 标题下方灰色说明文案（可选） */
  subtitle?: string;
  /** 副标题单行展示，超出省略 */
  subtitleSingleLine?: boolean;
  actions?: React.ReactNode;
}

interface ModulePageSectionProps {
  children: React.ReactNode;
}

export function ModulePageContainer({ children }: ModulePageSectionProps) {
  return (
    <div
      style={{
        padding: 24,
        backgroundColor: '#f6f7f9',
        minHeight: '100vh',
      }}
    >
      {children}
    </div>
  );
}

export function ModulePageHeader({ title, subtitle, subtitleSingleLine, actions }: ModulePageHeaderProps) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: subtitle ? 'flex-start' : 'center',
        marginBottom: 20,
        paddingBottom: 12,
        borderBottom: '1px solid #e5e7eb',
        gap: 16,
      }}
    >
      <div style={{ minWidth: 0, flex: 1 }}>
        <h2
          style={{
            fontSize: 20,
            fontWeight: 600,
            color: '#111827',
            margin: 0,
          }}
        >
          {title}
        </h2>
        {subtitle ? (
          <p
            style={{
              margin: '6px 0 0',
              fontSize: 13,
              color: '#6b7280',
              lineHeight: 1.4,
              ...(subtitleSingleLine
                ? {
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }
                : {}),
            }}
          >
            {subtitle}
          </p>
        ) : null}
      </div>
      {actions && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            flexShrink: 0,
            marginTop: subtitle ? 2 : 0,
          }}
        >
          {actions}
        </div>
      )}
    </div>
  );
}

export function ModulePageFilterCard({
  children,
  compact = false,
}: ModulePageSectionProps & { compact?: boolean }) {
  return (
    <div
      style={{
        marginBottom: compact ? 0 : 16,
        backgroundColor: '#ffffff',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        padding: compact ? '10px 16px' : '16px 24px',
      }}
    >
      {children}
    </div>
  );
}

export function ModulePageTableCard({ children }: ModulePageSectionProps) {
  return (
    <div
      style={{
        marginBottom: 16,
        backgroundColor: '#ffffff',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        overflow: 'hidden',
      }}
    >
      {children}
    </div>
  );
}

