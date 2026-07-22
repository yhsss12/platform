'use client';

import React from 'react';
import { useI18n } from '@/components/common/I18nProvider';

interface ConfirmDialogProps {
  open: boolean;
  title?: string;
  description?: string;
  extraContent?: React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  loading?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export default function ConfirmDialog({
  open,
  title,
  description,
  extraContent,
  confirmText,
  cancelText,
  loading = false,
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  const { t } = useI18n();
  const finalTitle = title ?? t('dialog.deleteTitle');
  const finalDescription = description ?? t('dialog.deleteDescription');
  const finalConfirm = confirmText ?? t('dialog.delete');
  const finalCancel = cancelText ?? t('dialog.cancel');

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15,23,42,0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1600,
        padding: '16px',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && !loading) onCancel();
      }}
    >
      <div
        style={{
          width: '420px',
          maxWidth: '96vw',
          backgroundColor: '#ffffff',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 24px 80px rgba(15,23,42,0.18)',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '16px 18px 8px',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          <div
            style={{
              fontSize: 16,
              fontWeight: 800,
              color: '#111827',
            }}
          >
            {finalTitle}
          </div>
        </div>

        <div
          style={{
            padding: '18px',
          }}
        >
          <div
            style={{
              fontSize: 14,
              color: '#4b5563',
              marginBottom: 20,
            }}
          >
            {finalDescription}
          </div>
          {extraContent ? <div style={{ marginBottom: 16 }}>{extraContent}</div> : null}
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: 12,
            }}
          >
            <button
              type="button"
              onClick={onCancel}
              disabled={loading}
              style={{
                height: 38,
                padding: '0 16px',
                borderRadius: 10,
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                color: '#374151',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: 14,
              }}
            >
              {finalCancel}
            </button>
            <button
              type="button"
              onClick={loading ? undefined : onConfirm}
              disabled={loading}
              style={{
                height: 38,
                padding: '0 16px',
                borderRadius: 10,
                border: 'none',
                backgroundColor: loading ? '#fca5a5' : '#dc2626',
                color: '#ffffff',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: 14,
                fontWeight: 700,
              }}
            >
              {loading ? t('dialog.deleting') : finalConfirm}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

