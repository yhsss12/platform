'use client';

import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { getExportPreview } from '@/features/data-platform/api/dataAssetsApi';
import { useTaskCenter } from '@/components/task-center';
import { useI18n } from '@/components/common/I18nProvider';

export interface ExportConfigModalProps {
  open: boolean;
  assetIds: number[];
  exportCount: number;
  /** 列表展示用格式名，如 HDF5 */
  formatLabel: string;
  /** 与 formatLabel 一致即可；预留与「混合」等扩展 */
  formatSummary?: string;
  projectName?: string;
  assetNamesPreview?: string;
  onClose: () => void;
}

const cardBorder = '1px solid #e5e7eb';
const cardRadius = 12;
const labelMuted = '#6b7280';
const textPrimary = '#111827';

export default function ExportConfigModal({
  open,
  assetIds,
  exportCount,
  formatLabel,
  formatSummary,
  projectName,
  assetNamesPreview,
  onClose,
}: ExportConfigModalProps) {
  const { t } = useI18n();
  const { addExportTask } = useTaskCenter();
  const [hasAnnotations, setHasAnnotations] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [exportError, setExportError] = useState('');

  useEffect(() => {
    if (!open) {
      setExportError('');
      return;
    }
    if (!assetIds.length) return;
    let cancelled = false;
    setSummaryLoading(true);
    getExportPreview(assetIds)
      .then((res) => {
        if (!cancelled && res.ok && res.data) setHasAnnotations(res.data.has_annotations);
      })
      .finally(() => {
        if (!cancelled) setSummaryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, assetIds]);

  const handleClose = () => {
    if (submitting) return;
    onClose();
  };

  const handleStartExport = async () => {
    setExportError('');
    if (!assetIds.length) return;
    setSubmitting(true);
    const result = await addExportTask({
      assetIds,
      exportCount,
      formatLabel,
      formatSummary: formatSummary ?? formatLabel,
      projectName: projectName?.trim() || undefined,
      assetNamesPreview: assetNamesPreview?.trim() || undefined,
    });
    setSubmitting(false);
    if (result.ok) {
      onClose();
    } else {
      const err = result.error || '';
      setExportError(err.includes('.') ? t(err) : err);
    }
  };

  const exportContentTags = [
    '原始数据文件',
    ...(hasAnnotations ? ['平台标注文件'] : []),
    '数据资产清单',
  ];

  const fmtDisplay = formatSummary ?? formatLabel;

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15,23,42,0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1600,
        padding: 20,
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) handleClose();
      }}
    >
      <div
        style={{
          width: 500,
          maxWidth: '96vw',
          backgroundColor: '#fff',
          borderRadius: 16,
          border: cardBorder,
          boxShadow: '0 24px 48px rgba(15,23,42,0.12)',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            padding: '14px 20px',
            borderBottom: cardBorder,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: 17, fontWeight: 700, color: textPrimary, letterSpacing: '-0.01em' }}>
            导出数据
          </div>
          <button
            type="button"
            onClick={handleClose}
            disabled={submitting}
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              border: 'none',
              background: 'transparent',
              cursor: submitting ? 'not-allowed' : 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: labelMuted,
              opacity: submitting ? 0.6 : 1,
            }}
            aria-label="关闭"
          >
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '20px 20px 16px' }}>
          <div style={{ borderRadius: cardRadius, border: cardBorder, backgroundColor: '#fafbfc', overflow: 'hidden' }}>
            <div style={{ padding: '14px 16px', borderBottom: cardBorder }}>
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: labelMuted,
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                  marginBottom: 12,
                }}
              >
                导出摘要
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 24px', fontSize: 13 }}>
                <div style={{ display: 'flex', gap: 8 }}>
                  <span style={{ color: labelMuted, flexShrink: 0 }}>导出数量</span>
                  <span style={{ color: textPrimary, fontWeight: 600 }}>{summaryLoading ? '—' : `${exportCount} 条`}</span>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <span style={{ color: labelMuted, flexShrink: 0 }}>数据格式</span>
                  <span style={{ color: textPrimary, fontWeight: 600 }}>{fmtDisplay}</span>
                </div>
                {projectName ? (
                  <div style={{ display: 'flex', gap: 8, gridColumn: '1 / -1' }}>
                    <span style={{ color: labelMuted, flexShrink: 0 }}>所属项目</span>
                    <span style={{ color: textPrimary, fontWeight: 500 }}>{projectName}</span>
                  </div>
                ) : null}
              </div>
              {assetNamesPreview ? (
                <div style={{ marginTop: 12, fontSize: 12, color: '#4b5563' }}>
                  <span style={{ color: labelMuted }}>名称预览 </span>
                  {assetNamesPreview}
                </div>
              ) : null}
            </div>
            <div style={{ padding: '14px 16px' }}>
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: labelMuted,
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                  marginBottom: 10,
                }}
              >
                导出内容
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {exportContentTags.map((tag) => (
                  <span
                    key={tag}
                    style={{
                      display: 'inline-block',
                      padding: '6px 12px',
                      borderRadius: 8,
                      fontSize: 12,
                      fontWeight: 500,
                      color: '#374151',
                      backgroundColor: '#fff',
                      border: cardBorder,
                      boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
          {exportError ? (
            <div style={{ marginTop: 12, fontSize: 13, color: '#dc2626', lineHeight: 1.45 }}>{exportError}</div>
          ) : null}
        </div>

        <div
          style={{
            padding: '16px 20px',
            borderTop: cardBorder,
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 10,
            backgroundColor: '#fafbfc',
          }}
        >
          <button
            type="button"
            onClick={handleClose}
            disabled={submitting}
            style={{
              height: 38,
              padding: '0 18px',
              borderRadius: 10,
              border: cardBorder,
              backgroundColor: '#fff',
              color: '#374151',
              fontSize: 13,
              fontWeight: 600,
              cursor: submitting ? 'not-allowed' : 'pointer',
            }}
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleStartExport}
            disabled={submitting || !assetIds.length}
            style={{
              height: 38,
              padding: '0 18px',
              borderRadius: 10,
              border: 'none',
              backgroundColor: submitting || !assetIds.length ? '#e5e7eb' : '#2563eb',
              color: submitting || !assetIds.length ? '#9ca3af' : '#fff',
              fontSize: 13,
              fontWeight: 600,
              cursor: submitting || !assetIds.length ? 'not-allowed' : 'pointer',
              boxShadow: submitting || !assetIds.length ? 'none' : '0 1px 3px rgba(37, 99, 235, 0.25)',
            }}
          >
            {submitting ? '创建任务…' : '开始导出'}
          </button>
        </div>
      </div>
    </div>
  );
}
