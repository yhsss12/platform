'use client';

import { useEffect, useState } from 'react';
import type { BackgroundTask } from './types';
import { getBatchDetail, type ConversionBatchChildItem } from '@/lib/conversion/conversionApi';
import { TASK_STATUS_LABEL_KEY, TASK_TYPE_LABEL_KEY } from './types';
import { useI18n } from '@/components/common/I18nProvider';

interface TaskCenterDetailProps {
  task: BackgroundTask;
}

function childStatusZh(s: string): string {
  const v = (s || '').toLowerCase();
  if (v === 'queued') return '排队中';
  if (v === 'running') return '执行中';
  if (v === 'succeeded') return '已完成';
  if (v === 'failed') return '已失败';
  if (v === 'canceled' || v === 'cancelled') return '已取消';
  return s || '—';
}

function overallStatusZh(s: string): string {
  const u = (s || '').toUpperCase();
  if (u === 'PENDING') return '排队中';
  if (u === 'RUNNING') return '执行中';
  if (u === 'SUCCESS') return '已完成';
  if (u === 'PARTIAL_SUCCESS') return '部分成功';
  if (u === 'FAILED') return '已失败';
  if (u === 'CANCELED') return '已取消';
  return s || '—';
}

function exportResultTypeLabel(format?: string): string | null {
  if (!format) return null;
  const f = format.toLowerCase();
  if (f === 'lerobot') return 'backgroundTasks.exportResultDir';
  if (f === 'hdf5' || f === 'mcap') return 'backgroundTasks.exportResultZip';
  return 'backgroundTasks.exportResultZip';
}

/** 导入中断/失败：详情区补充说明（与 currentStep / errorMessage 的 i18n key 对应） */
function importFailureContextBodyKey(task: BackgroundTask): string {
  const step = (task.currentStep || '').trim();
  const err = (task.errorMessage || '').trim();
  if (step === 'backgroundTasks.importBrowserInterrupted' || err === 'backgroundTasks.importBrowserInterrupted') {
    return 'backgroundTasks.importBrowserInterrupted';
  }
  if (
    step === 'backgroundTasks.importSessionExpiredHint' ||
    err === 'backgroundTasks.importSessionExpiredHint'
  ) {
    return 'backgroundTasks.importPausedDetailBodyExpired';
  }
  if (step === 'backgroundTasks.importStandardImportRefreshHint') {
    return 'backgroundTasks.importPausedDetailBodyStandard';
  }
  if (
    step === 'backgroundTasks.importPausedLostLocalAfterRefresh' ||
    step === 'backgroundTasks.importSessionPendingRefreshHint'
  ) {
    return 'backgroundTasks.importPausedDetailBodyRefresh';
  }
  return 'backgroundTasks.importPausedDetailBodyGeneric';
}

function shouldShowImportFailureContextBox(task: BackgroundTask): boolean {
  if (task.type !== 'import' || task.status !== 'failed') return false;
  const step = (task.currentStep || '').trim();
  const err = (task.errorMessage || '').trim();
  const keys = [
    'backgroundTasks.importBrowserInterrupted',
    'backgroundTasks.importSessionExpiredHint',
    'backgroundTasks.importStandardImportRefreshHint',
    'backgroundTasks.importPausedLostLocalAfterRefresh',
    'backgroundTasks.importSessionPendingRefreshHint',
    'backgroundTasks.importUploadSessionFailed',
  ];
  return keys.some((k) => step === k || err === k);
}

export default function TaskCenterDetail({ task }: TaskCenterDetailProps) {
  const { t } = useI18n();
  const meta = task.meta || {};
  const [batchChildren, setBatchChildren] = useState<ConversionBatchChildItem[] | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);

  useEffect(() => {
    if (task.type !== 'convert' || !task.convertBatchId) {
      setBatchChildren(null);
      return;
    }
    let cancelled = false;
    setBatchLoading(true);
    void getBatchDetail(task.convertBatchId)
      .then((d) => {
        if (!cancelled) setBatchChildren(d.children || []);
      })
      .catch(() => {
        if (!cancelled) setBatchChildren([]);
      })
      .finally(() => {
        if (!cancelled) setBatchLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [task.type, task.convertBatchId]);
  const statusLabel = t(TASK_STATUS_LABEL_KEY[task.status]);
  const typeLabel = t(TASK_TYPE_LABEL_KEY[task.type]);
  const resultTypeLabelKey = exportResultTypeLabel(meta.format);

  return (
    <div style={{ fontSize: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ color: '#6b7280' }}>{t('backgroundTasks.taskType')}</span>
        <span style={{ color: '#111827', fontWeight: 500 }}>{typeLabel}</span>
      </div>
      {!(task.type === 'sync' && task.status === 'success') && (
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#6b7280' }}>{t('backgroundTasks.currentStatus')}</span>
          <span style={{ color: '#111827', fontWeight: 500 }}>{statusLabel}</span>
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ color: '#6b7280' }}>{t('backgroundTasks.currentStep')}</span>
        <span style={{ color: '#111827' }}>
          {(() => {
            if (task.status === 'paused') return t('backgroundTasks.statusPaused');
            const s = task.currentStep || '';
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
          })()}
        </span>
      </div>
      {shouldShowImportFailureContextBox(task) && (
        <div
          style={{
            padding: 10,
            borderRadius: 6,
            backgroundColor: '#f9fafb',
            border: '1px solid #e5e7eb',
          }}
        >
          <div style={{ color: '#6b7280', marginBottom: 6, fontWeight: 600 }}>
            {t('backgroundTasks.importPausedDetailHeading')}
          </div>
          <div style={{ color: '#374151', lineHeight: 1.55 }}>{t(importFailureContextBodyKey(task))}</div>
        </div>
      )}
      {task.type === 'export' && task.status === 'success' && (
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#6b7280' }}>{t('backgroundTasks.resultType')}</span>
          <span style={{ color: '#111827' }}>{resultTypeLabelKey ? t(resultTypeLabelKey) : '—'}</span>
        </div>
      )}
      {task.type === 'export' && (
        <>
          {meta.exportProjectName && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.exportProjectLabel')}</span>
              <span style={{ color: '#111827' }}>{meta.exportProjectName}</span>
            </div>
          )}
          {meta.exportFormatSummary && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.exportFormatSummaryLabel')}</span>
              <span style={{ color: '#111827' }}>{meta.exportFormatSummary}</span>
            </div>
          )}
          {meta.exportAssetNamesPreview && (
            <div>
              <div style={{ color: '#6b7280', marginBottom: 4 }}>{t('backgroundTasks.exportNamesPreviewLabel')}</div>
              <div style={{ color: '#111827', wordBreak: 'break-all' }}>{meta.exportAssetNamesPreview}</div>
            </div>
          )}
          {(meta.exportCompletedAssets != null || meta.exportTotalAssets != null) && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.exportProgressAssetsLabel')}</span>
              <span style={{ color: '#111827' }}>
                {meta.exportCompletedAssets ?? 0} / {meta.exportTotalAssets ?? meta.count ?? '—'}
              </span>
            </div>
          )}
        </>
      )}
      {task.type === 'import' && (
        <>
          {meta.importProjectName && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.importProjectLabel')}</span>
              <span style={{ color: '#111827' }}>{meta.importProjectName}</span>
            </div>
          )}
          {meta.importModeSummary && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.importModeLabel')}</span>
              <span style={{ color: '#111827' }}>{meta.importModeSummary}</span>
            </div>
          )}
          {meta.count != null && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.quantity')}</span>
              <span style={{ color: '#111827' }}>
                {meta.count} {t('backgroundTasks.unitItems')}
              </span>
            </div>
          )}
          {(meta.importCompletedAssets != null || meta.importFailedAssets != null) && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>{t('backgroundTasks.importSuccessFailLabel')}</span>
              <span style={{ color: '#111827' }}>
                {t('backgroundTasks.importSuccessFailDetail', {
                  ok: meta.importCompletedAssets ?? 0,
                  fail: meta.importFailedAssets ?? 0,
                })}
              </span>
            </div>
          )}
          {meta.importAssetNamesPreview && (
            <div>
              <div style={{ color: '#6b7280', marginBottom: 4 }}>{t('backgroundTasks.importNamesPreviewLabel')}</div>
              <div style={{ color: '#111827', wordBreak: 'break-all' }}>{meta.importAssetNamesPreview}</div>
            </div>
          )}
          {meta.importSuccessAssetNames && meta.importSuccessAssetNames.length > 0 && (
            <div>
              <div style={{ color: '#6b7280', marginBottom: 4 }}>{t('backgroundTasks.importSucceededAssetsTitle')}</div>
              <ul style={{ margin: 0, paddingLeft: 18, color: '#111827', wordBreak: 'break-all' }}>
                {meta.importSuccessAssetNames.slice(0, 30).map((name, idx) => (
                  <li key={`${name}-${idx}`} style={{ fontSize: 11 }}>
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {meta.importFailureEntries && meta.importFailureEntries.length > 0 && (
            <div>
              <div style={{ color: '#6b7280', marginBottom: 4 }}>{t('backgroundTasks.importFailedItemsTitle')}</div>
              <ul style={{ margin: 0, paddingLeft: 18, color: '#b91c1c', wordBreak: 'break-all' }}>
                {meta.importFailureEntries.slice(0, 40).map((row, idx) => (
                  <li key={`${row.name}-${idx}`} style={{ fontSize: 11 }}>
                    {row.name}
                    {row.reason ? `（${row.reason}）` : ''}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
      {task.type !== 'import' && task.type !== 'export' && (meta.format || meta.outputFormat) && (
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#6b7280' }}>{t('backgroundTasks.outputFormat')}</span>
          <span style={{ color: '#111827' }}>{meta.format || meta.outputFormat}</span>
        </div>
      )}
      {meta.count != null && task.type !== 'import' && task.type !== 'export' && (
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#6b7280' }}>{t('backgroundTasks.quantity')}</span>
          <span style={{ color: '#111827' }}>
            {meta.count} {t('backgroundTasks.unitItems')}
          </span>
        </div>
      )}
      {(meta.fullOutputPath || meta.outputPath) && meta.exportDeliveryMode !== 'browser_zip' && (
        <div>
          <div style={{ color: '#6b7280', marginBottom: 4 }}>{t('backgroundTasks.outputPath')}</div>
          <div style={{ color: '#111827', wordBreak: 'break-all' }}>
            {meta.fullOutputPath || meta.outputPath}
          </div>
        </div>
      )}
      {meta.outputFile && meta.exportDeliveryMode !== 'browser_zip' && (
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#6b7280' }}>{t('backgroundTasks.outputFile')}</span>
          <span style={{ color: '#111827' }}>{meta.outputFile}</span>
        </div>
      )}
      {task.type === 'convert' && task.convertBatchId && (
        <>
          {meta.convertBatchOverall && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>任务状态</span>
              <span style={{ color: '#111827' }}>{overallStatusZh(String(meta.convertBatchOverall))}</span>
            </div>
          )}
          {(meta.convertBatchSuccess != null || meta.convertBatchFailed != null) && (
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#6b7280' }}>进度统计</span>
              <span style={{ color: '#111827' }}>
                成功 {meta.convertBatchSuccess ?? 0} · 失败 {meta.convertBatchFailed ?? 0} · 处理中{' '}
                {(meta.convertBatchRunning ?? 0) + (meta.convertBatchPending ?? 0)}
              </span>
            </div>
          )}
          <div style={{ color: '#6b7280', marginTop: 8, marginBottom: 4 }}>文件明细</div>
          {batchLoading ? (
            <div style={{ fontSize: 11, color: '#9ca3af' }}>加载中…</div>
          ) : batchChildren && batchChildren.length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 220, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead style={{ background: '#f9fafb', position: 'sticky', top: 0 }}>
                  <tr>
                    <th style={{ textAlign: 'left', padding: 6 }}>源文件</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>输出</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>阶段</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>状态</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>错误</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>更新</th>
                  </tr>
                </thead>
                <tbody>
                  {batchChildren.map((c) => (
                    <tr key={c.jobId} style={{ borderTop: '1px solid #f3f4f6' }}>
                      <td style={{ padding: 6, wordBreak: 'break-all' }}>{c.sourceFileName || '—'}</td>
                      <td style={{ padding: 6, wordBreak: 'break-all' }}>{c.outputFileName || '—'}</td>
                      <td style={{ padding: 6 }}>{c.itemStage || '—'}</td>
                      <td style={{ padding: 6 }}>{childStatusZh(c.itemStatus)}</td>
                      <td style={{ padding: 6, color: '#b91c1c', wordBreak: 'break-all' }}>{c.errorMessage || '—'}</td>
                      <td style={{ padding: 6, whiteSpace: 'nowrap' }}>
                        {c.updatedAt ? new Date(c.updatedAt).toLocaleString(undefined) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ fontSize: 11, color: '#9ca3af' }}>暂无文件记录</div>
          )}
        </>
      )}
      {task.errorMessage && (
        <div
          style={{
            padding: 8,
            borderRadius: 8,
            backgroundColor: '#fef2f2',
            color: '#b91c1c',
            fontSize: 11,
          }}
        >
          {task.errorMessage.includes('.') ? t(task.errorMessage) : task.errorMessage}
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', color: '#9ca3af', fontSize: 11 }}>
        <span>{t('backgroundTasks.createdAt')}</span>
        <span>{new Date(task.createdAt).toLocaleString(undefined)}</span>
      </div>
    </div>
  );
}
