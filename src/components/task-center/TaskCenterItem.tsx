'use client';

import { useState } from 'react';
import type { BackgroundTask } from './types';
import { TASK_STATUS_LABEL_KEY, isActiveStatus } from './types';
import TaskCenterDetail from './TaskCenterDetail';
import { useI18n } from '@/components/common/I18nProvider';
import { downloadExportZipFile } from '@/features/data-platform/api/dataAssetsApi';

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—';
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

type ExportDownloadPhase = 'idle' | 'starting' | 'downloading' | 'succeeded' | 'failed';

interface TaskCenterItemProps {
  task: BackgroundTask;
  onRemove: (id: string) => void;
  onRequestDelete: (task: BackgroundTask) => void;
  onRequestCancel?: (task: BackgroundTask) => void;
  cancelLoadingTaskId?: string | null;
  onNotify?: (message: string, isError?: boolean) => void;
}

export default function TaskCenterItem({
  task,
  onRemove,
  onRequestDelete,
  onRequestCancel,
  cancelLoadingTaskId = null,
  onNotify,
}: TaskCenterItemProps) {
  const { t } = useI18n();
  const [detailOpen, setDetailOpen] = useState(false);
  const [exportDownload, setExportDownload] = useState<{
    phase: ExportDownloadPhase;
    loaded: number;
    total: number | null;
  }>({ phase: 'idle', loaded: 0, total: null });

  const statusLabel = t(TASK_STATUS_LABEL_KEY[task.status]);
  const meta = task.meta || {};
  const title =
    task.type === 'export'
      ? (task.title?.trim() ? task.title : `${t('backgroundTasks.typeExport')} ${meta.format ?? ''}`.trim())
      : task.type === 'convert'
        ? task.title?.trim()
          ? task.title
          : `${t('backgroundTasks.typeConvert')} ${meta.inputFormat || 'MCAP'} → ${meta.outputFormat || 'HDF5'}`
        : task.title;

  const stepText = (() => {
    /** 已暂停：状态由角标展示，副标题留空；原因见详情 */
    if (task.status === 'paused') return '';
    /**
     * 失败 / 已取消：具体原因在 `currentStep`、`errorMessage` 中，列表卡片不展示（避免技术报错、批量摘要占满卡片）。
     * 完整信息仅在展开「详情」后由 TaskCenterDetail 展示。
     */
    if (task.status === 'failed' || task.status === 'cancelled') return '';
    const s = task.currentStep || '';
    if (!s) return '';
    if (s === 'backgroundTasks.exportReadyClickDownload') {
      const n = meta.exportTotalAssets ?? meta.exportCompletedAssets ?? meta.count ?? 0;
      return t(s, { count: n });
    }
    if (s.includes('.') && s.startsWith('backgroundTasks.')) return t(s);
    if (s === '校验导出项') return t('backgroundTasks.currentStep');
    if (s === '导出内容已保存到指定路径') return t('backgroundTasks.statusSuccess');
    if (s === '排队中') return t('backgroundTasks.statusQueued');
    if (s.startsWith('阶段：')) return `${t('backgroundTasks.currentStep')}: ${s.replace(/^阶段：/, '')}`;
    return s;
  })();

  const errorText =
    task.errorMessage && task.errorMessage.includes('.') ? t(task.errorMessage) : task.errorMessage;

  const exportDlBusy =
    exportDownload.phase === 'starting' ||
    exportDownload.phase === 'downloading' ||
    exportDownload.phase === 'succeeded';
  const exportPct =
    exportDownload.total != null && exportDownload.total > 0
      ? Math.min(100, Math.round((exportDownload.loaded / exportDownload.total) * 100))
      : null;

  return (
    <div
      style={{
        flexShrink: 0,
        border: '1px solid #e5e7eb',
        borderRadius: 10,
        backgroundColor: '#fff',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '10px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#111827' }}>{title}</span>
            {!(task.type === 'sync' && task.status === 'success') && (
              <span
                style={{
                  fontSize: 11,
                  padding: '2px 6px',
                  borderRadius: 4,
                  whiteSpace: 'nowrap',
                  lineHeight: 1,
                  flexShrink: 0,
                  backgroundColor:
                    task.status === 'success'
                      ? '#dcfce7'
                      : task.status === 'failed' || task.status === 'cancelled'
                        ? '#fee2e2'
                        : task.status === 'paused'
                          ? '#f3f4f6'
                          : '#eff6ff',
                  color:
                    task.status === 'success'
                      ? '#166534'
                      : task.status === 'failed' || task.status === 'cancelled'
                        ? '#b91c1c'
                        : task.status === 'paused'
                          ? '#4b5563'
                          : '#1e40af',
                  fontWeight: 600,
                }}
              >
                {statusLabel}
              </span>
            )}
          </div>
          {stepText ? (
            <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>{stepText}</div>
          ) : null}
          {(task.status === 'running' || task.status === 'paused' || task.status === 'queued') && (
            <div style={{ marginTop: 8 }}>
              <div
                style={{
                  height: 4,
                  borderRadius: 2,
                  backgroundColor: '#e5e7eb',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${task.progress}%`,
                    borderRadius: 2,
                    backgroundColor: task.status === 'paused' ? '#9ca3af' : '#2563eb',
                    transition: 'width 0.2s ease',
                  }}
                />
              </div>
              <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>{task.progress}%</div>
            </div>
          )}
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            alignItems: 'stretch',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          {task.type === 'export' &&
            task.status === 'success' &&
            meta.exportDeliveryMode === 'browser_zip' &&
            task.exportJobId && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'stretch' }}>
                <button
                  type="button"
                  disabled={exportDlBusy}
                  onClick={() => {
                    if (exportDlBusy) return;
                    setExportDownload({ phase: 'starting', loaded: 0, total: null });
                    const jobId = task.exportJobId!;
                    const fallbackName = meta.outputFile || 'export.zip';
                    const startDownload = () => {
                      void (async () => {
                        const r = await downloadExportZipFile(jobId, fallbackName, (p) => {
                          setExportDownload({
                            phase: 'downloading',
                            loaded: p.loaded,
                            total: p.total,
                          });
                        });
                        if (r.cancelled) {
                          setExportDownload({ phase: 'idle', loaded: 0, total: null });
                          return;
                        }
                        if (!r.ok) {
                          setExportDownload((prev) => ({
                            phase: 'failed',
                            loaded: prev.loaded,
                            total: prev.total,
                          }));
                          const msg = r.error || t('backgroundTasks.exportDownloadStateFailed');
                          onNotify?.(msg, true);
                          return;
                        }
                        setExportDownload((prev) => ({
                          phase: 'succeeded',
                          loaded: prev.loaded,
                          total: prev.total,
                        }));
                        onNotify?.(t('backgroundTasks.exportDownloadZipDone'));
                        window.setTimeout(() => {
                          setExportDownload({ phase: 'idle', loaded: 0, total: null });
                        }, 2200);
                      })();
                    };
                    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
                      window.requestAnimationFrame(() => {
                        window.requestAnimationFrame(startDownload);
                      });
                    } else {
                      setTimeout(startDownload, 0);
                    }
                  }}
                  style={{
                    ...btnSm,
                    borderColor: '#2563eb',
                    backgroundColor: exportDlBusy ? '#f3f4f6' : '#eff6ff',
                    color: exportDlBusy ? '#9ca3af' : '#1d40af',
                    cursor: exportDlBusy ? 'not-allowed' : 'pointer',
                  }}
                >
                  {exportDownload.phase === 'starting'
                    ? t('backgroundTasks.exportDownloadStarting')
                    : exportDownload.phase === 'downloading'
                      ? exportPct != null
                        ? t('backgroundTasks.exportDownloadBtnPct', { pct: String(exportPct) })
                        : exportDownload.total != null
                          ? t('backgroundTasks.exportDownloadBtnPair', {
                              loaded: formatBytes(exportDownload.loaded),
                              total: formatBytes(exportDownload.total),
                            })
                          : t('backgroundTasks.exportDownloadBtnBytesOnly', {
                              loaded: formatBytes(exportDownload.loaded),
                            })
                      : t('backgroundTasks.downloadExport')}
                </button>
                <div style={{ fontSize: 10, color: '#6b7280', lineHeight: 1.3 }}>
                  {exportDownload.phase === 'idle' && t('backgroundTasks.exportDownloadStateIdle')}
                  {exportDownload.phase === 'starting' && t('backgroundTasks.exportDownloadStateStarting')}
                  {exportDownload.phase === 'downloading' && (
                    <>
                      {t('backgroundTasks.exportDownloadStateDownloading')}
                      {exportPct != null &&
                        ` · ${formatBytes(exportDownload.loaded)} / ${formatBytes(exportDownload.total!)} (${exportPct}%)`}
                      {exportPct == null &&
                        exportDownload.loaded > 0 &&
                        ` · ${formatBytes(exportDownload.loaded)}`}
                    </>
                  )}
                  {exportDownload.phase === 'succeeded' && t('backgroundTasks.exportDownloadStateDone')}
                  {exportDownload.phase === 'failed' && t('backgroundTasks.exportDownloadStateFailed')}
                </div>
                {(exportDownload.phase === 'starting' || exportDownload.phase === 'downloading') && (
                  <div
                    style={{
                      height: 4,
                      borderRadius: 2,
                      backgroundColor: '#e5e7eb',
                      overflow: 'hidden',
                      position: 'relative',
                    }}
                  >
                    {exportPct != null ? (
                      <div
                        style={{
                          height: '100%',
                          width: `${exportPct}%`,
                          borderRadius: 2,
                          backgroundColor: '#2563eb',
                          transition: 'width 0.15s ease-out',
                        }}
                      />
                    ) : (
                      <div
                        style={{
                          height: '100%',
                          width: '100%',
                          borderRadius: 2,
                          backgroundColor: '#60a5fa',
                          animation: 'taskCenterExportZipIndeterminate 1.2s ease-in-out infinite',
                        }}
                      />
                    )}
                  </div>
                )}
              </div>
            )}
          <div
            style={{
              display: 'flex',
              flexDirection: 'row',
              flexWrap: 'wrap',
              gap: 8,
              justifyContent: 'flex-end',
              alignItems: 'center',
            }}
          >
            <button
              type="button"
              onClick={() => setDetailOpen((o) => !o)}
              style={{ ...btnAction, ...btnNeutral }}
            >
              {detailOpen ? t('backgroundTasks.collapse') : t('backgroundTasks.details')}
            </button>
            {isActiveStatus(task.status) ? (
              <button
                type="button"
                onClick={() => onRequestCancel?.(task)}
                disabled={cancelLoadingTaskId === task.id}
                style={{
                  ...btnAction,
                  ...btnDanger,
                  opacity: cancelLoadingTaskId === task.id ? 0.6 : 1,
                  cursor: cancelLoadingTaskId === task.id ? 'not-allowed' : 'pointer',
                }}
              >
                {cancelLoadingTaskId === task.id
                  ? `${t('backgroundTasks.actionCancel')}…`
                  : t('backgroundTasks.actionCancel')}
              </button>
            ) : (
              <button type="button" onClick={() => onRequestDelete(task)} style={{ ...btnAction, ...btnNeutral }}>
                {t('backgroundTasks.delete')}
              </button>
            )}
          </div>
        </div>
      </div>
      {detailOpen && (
        <div style={{ borderTop: '1px solid #e5e7eb', padding: 12 }}>
          <TaskCenterDetail task={{ ...task, errorMessage: errorText }} />
        </div>
      )}
    </div>
  );
}

const btnSm: React.CSSProperties = {
  padding: '4px 8px',
  fontSize: 11,
  fontWeight: 600,
  borderRadius: 6,
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
  color: '#2563eb',
  cursor: 'pointer',
};

const btnAction: React.CSSProperties = {
  padding: '6px 10px',
  fontSize: 12,
  fontWeight: 600,
  lineHeight: 1.2,
  minHeight: 30,
  boxSizing: 'border-box',
  borderRadius: 6,
  whiteSpace: 'nowrap',
};

const btnNeutral: React.CSSProperties = {
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
  color: '#374151',
  cursor: 'pointer',
};

const btnDanger: React.CSSProperties = {
  border: '1px solid #fecaca',
  backgroundColor: '#fff',
  color: '#ea580c',
  cursor: 'pointer',
};
