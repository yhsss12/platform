'use client';

import { useMemo, useState } from 'react';
import { GhostButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  fetchAssetPipelineFileText,
  formatAssetPipelineFileError,
  resolveAssetExportFiles,
  type AssetExportFileItem,
  type AssetPipelineJobStatus,
} from '@/lib/api/sam3dAssetPipelineClient';

export interface AssetExportDialogProps {
  open: boolean;
  onClose: () => void;
  jobId: string;
  jobStatus: AssetPipelineJobStatus | null;
  onDownload: (relPath: string, filename: string) => Promise<void>;
  downloadingFile: string | null;
}

const GROUP_LABELS: Record<AssetExportFileItem['group'], string> = {
  sam3d: 'SAM3D 重建产物',
  mujoco: 'MuJoCo 资产',
  logs: '日志与元数据',
};

function formatFileSize(bytes: number | null | undefined): string {
  if (bytes == null || bytes <= 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function ExportRow({
  item,
  sizeBytes,
  downloading,
  onDownload,
}: {
  item: AssetExportFileItem;
  sizeBytes: number | null | undefined;
  downloading: boolean;
  onDownload: () => void;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '10px 12px',
        borderRadius: 8,
        border: '1px solid #eef2f7',
        background: '#fafafa',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: '#111827' }}>{item.label}</span>
          {item.tag ? (
            <span
              style={{
                fontSize: 10,
                padding: '2px 6px',
                borderRadius: 999,
                background: '#eef2ff',
                color: '#4338ca',
              }}
            >
              {item.tag}
            </span>
          ) : null}
        </div>
        <div style={{ fontSize: 11, color: '#64748b', marginTop: 2, wordBreak: 'break-all' }}>{item.relPath}</div>
      </div>
      <div style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>
        {formatFileSize(sizeBytes)}
      </div>
      <SecondaryButton onClick={onDownload} disabled={downloading}>
        {downloading ? '下载中…' : '下载'}
      </SecondaryButton>
    </div>
  );
}

export function AssetExportDialog({
  open,
  onClose,
  jobId,
  jobStatus,
  onDownload,
  downloadingFile,
}: AssetExportDialogProps) {
  const [logPreview, setLogPreview] = useState<string | null>(null);
  const [logLoading, setLogLoading] = useState(false);
  const [logError, setLogError] = useState<string | null>(null);

  const exportFiles = useMemo(() => resolveAssetExportFiles(jobStatus), [jobStatus]);

  const grouped = useMemo(() => {
    const map: Record<AssetExportFileItem['group'], AssetExportFileItem[]> = {
      sam3d: [],
      mujoco: [],
      logs: [],
    };
    for (const item of exportFiles) map[item.group].push(item);
    return map;
  }, [exportFiles]);

  const sizeMap = useMemo(() => {
    const map = new Map<string, number | null | undefined>();
    for (const file of jobStatus?.files ?? []) {
      map.set(file.path, file.sizeBytes);
    }
    return map;
  }, [jobStatus?.files]);

  if (!open) return null;

  const handleViewLog = async (relPath: string) => {
    setLogLoading(true);
    setLogError(null);
    try {
      const text = await fetchAssetPipelineFileText(jobId, relPath);
      setLogPreview(text.split('\n').slice(-200).join('\n'));
    } catch (err) {
      setLogError(formatAssetPipelineFileError(err, '查看'));
      setLogPreview(null);
    } finally {
      setLogLoading(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="asset-export-dialog-title"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
        background: 'rgba(15, 23, 42, 0.45)',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 'min(760px, 92vw)',
          maxHeight: '75vh',
          display: 'flex',
          flexDirection: 'column',
          background: '#fff',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 20px 40px rgba(15, 23, 42, 0.18)',
          overflow: 'hidden',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #e5e7eb' }}>
          <div id="asset-export-dialog-title" style={{ fontSize: 16, fontWeight: 600, color: '#111827' }}>
            导出资产文件
          </div>
          <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
            选择需要下载的重建产物或 MuJoCo 资产包。
          </div>
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 18 }}>
          {(['sam3d', 'mujoco', 'logs'] as const).map((group) => {
            const items = grouped[group];
            if (items.length === 0) return null;
            return (
              <section key={group}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>
                  {GROUP_LABELS[group]}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {items.map((item) => (
                    <ExportRow
                      key={item.key}
                      item={item}
                      sizeBytes={sizeMap.get(item.relPath)}
                      downloading={downloadingFile === item.relPath}
                      onDownload={() => void onDownload(item.relPath, item.filename)}
                    />
                  ))}
                </div>
              </section>
            );
          })}

          {grouped.logs.some((item) => item.relPath.endsWith('.log')) ? (
            <section>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {grouped.logs
                  .filter((item) => item.relPath.endsWith('.log'))
                  .map((item) => (
                    <GhostButton key={item.key} onClick={() => void handleViewLog(item.relPath)} disabled={logLoading}>
                      {logLoading ? '加载中…' : `查看 ${item.filename}`}
                    </GhostButton>
                  ))}
              </div>
              {logError ? <div style={{ fontSize: 12, color: '#b91c1c', marginTop: 8 }}>{logError}</div> : null}
              {logPreview ? (
                <pre
                  style={{
                    marginTop: 8,
                    padding: 12,
                    background: '#111827',
                    color: '#e5e7eb',
                    borderRadius: 8,
                    fontSize: 11,
                    lineHeight: 1.5,
                    maxHeight: 220,
                    overflow: 'auto',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {logPreview}
                </pre>
              ) : null}
            </section>
          ) : null}
        </div>

        <div
          style={{
            padding: '12px 20px',
            borderTop: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'flex-end',
          }}
        >
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
        </div>
      </div>
    </div>
  );
}
