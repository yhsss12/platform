'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { AssetExportDialog } from '@/components/workspace/sim-assets/AssetExportDialog';
import { SimAssetBackLink, SimAssetToast } from '@/components/workspace/sim-assets/simAssetUi';
import {
  GhostButton,
  PrimaryButton,
  SecondaryButton,
  SectionCard,
  StatusBadge,
  WS,
} from '@/components/workspace/workspaceUi';
import {
  deleteAssetPipelineJob,
  downloadAssetPipelineFile,
  formatAssetPipelineFileError,
  getAssetPipelineJob,
  listAssetPipelineJobs,
  PIPELINE_STATUS_LABELS,
  type AssetPipelineJobStatus,
} from '@/lib/api/sam3dAssetPipelineClient';
import { simAssetTargetLabel, simAssetTypeLabel } from '@/lib/workspace/simAssetDisplay';
import type { SimAssetTabFilter } from '@/types/simAsset';

const TAB_OPTIONS: { id: SimAssetTabFilter; label: string }[] = [
  { id: 'all', label: '全部资产' },
  { id: 'scene', label: '场景资产' },
  { id: 'object', label: '操作对象' },
  { id: 'reconstruction', label: '重建任务' },
];

const EMPTY_STATE =
  '暂无仿真资产。你可以通过「导入资产」上传已有 MuJoCo / Isaac Sim 资产，或通过「图像重建」从图片生成操作对象资产。';

function formatUpdatedAt(value?: string | null): string {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function primaryFormat(job: AssetPipelineJobStatus): string {
  const hasGs = job.files?.some((file) => file.path === 'sam3d/gs.ply');
  return hasGs ? 'PLY' : '-';
}

function badgeStatus(status: string): 'completed' | 'failed' | 'pending' {
  if (status === 'reconstructed') return 'completed';
  if (status === 'failed') return 'failed';
  return 'pending';
}

function matchesTab(job: AssetPipelineJobStatus, tab: SimAssetTabFilter): boolean {
  if (tab === 'all') return true;
  if (tab === 'reconstruction') return true;
  const assetType = (job.assetType || 'object').toLowerCase();
  if (tab === 'object') return assetType === 'object';
  if (tab === 'scene') return assetType === 'scene';
  return true;
}

function DangerButton({
  children,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        padding: '6px 10px',
        fontSize: 13,
        borderRadius: 6,
        border: 'none',
        backgroundColor: 'transparent',
        color: disabled ? '#fca5a5' : '#dc2626',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      {children}
    </button>
  );
}

export default function SimAssetsPage() {
  const [activeTab, setActiveTab] = useState<SimAssetTabFilter>('all');
  const [jobs, setJobs] = useState<AssetPipelineJobStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AssetPipelineJobStatus | null>(null);
  const [deletingJobId, setDeletingJobId] = useState<string | null>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportingJobId, setExportingJobId] = useState<string | null>(null);
  const [exportingJobStatus, setExportingJobStatus] = useState<AssetPipelineJobStatus | null>(null);
  const [exportLoadingJobId, setExportLoadingJobId] = useState<string | null>(null);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const response = await listAssetPipelineJobs(50);
      setJobs(response.jobs || []);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '加载重建任务列表失败');
      setJobs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!toastMsg) return;
    const timer = window.setTimeout(() => setToastMsg(null), 6000);
    return () => window.clearTimeout(timer);
  }, [toastMsg]);

  const filtered = useMemo(
    () => jobs.filter((job) => matchesTab(job, activeTab)),
    [jobs, activeTab]
  );

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    const jobId = deleteTarget.jobId;
    setDeletingJobId(jobId);
    try {
      await deleteAssetPipelineJob(jobId);
      setJobs((prev) => prev.filter((job) => job.jobId !== jobId));
      setToastMsg('删除成功');
      setDeleteTarget(null);
    } catch (err) {
      setToastMsg(err instanceof Error ? err.message : '删除失败');
    } finally {
      setDeletingJobId(null);
    }
  };

  const handleCloseExportDialog = () => {
    setExportDialogOpen(false);
    setExportingJobId(null);
    setExportingJobStatus(null);
  };

  const handleOpenExport = async (job: AssetPipelineJobStatus) => {
    setExportLoadingJobId(job.jobId);
    try {
      const status = await getAssetPipelineJob(job.jobId);
      setExportingJobId(job.jobId);
      setExportingJobStatus(status);
      setExportDialogOpen(true);
    } catch (err) {
      setToastMsg(err instanceof Error ? err.message : '获取导出文件失败');
    } finally {
      setExportLoadingJobId(null);
    }
  };

  const handleDownloadFile = async (relPath: string, filename: string) => {
    if (!exportingJobId) return;
    setDownloadingFile(relPath);
    try {
      await downloadAssetPipelineFile(exportingJobId, relPath, filename);
      setToastMsg('下载已开始');
    } catch (err) {
      setToastMsg(formatAssetPipelineFileError(err, '下载'));
    } finally {
      setDownloadingFile(null);
    }
  };

  return (
    <ModulePageContainer>
      <SimAssetBackLink href="/workspace/resources" label="返回" />

      <ModulePageHeader
        title="仿真资产"
        actions={
          <>
            <Link href="/workspace/resources/sim-assets/import" style={{ textDecoration: 'none' }}>
              <SecondaryButton>导入资产</SecondaryButton>
            </Link>
            <Link href="/workspace/resources/sim-assets/reconstruct" style={{ textDecoration: 'none' }}>
              <PrimaryButton>图像重建</PrimaryButton>
            </Link>
          </>
        }
      />

      {loadError ? (
        <SectionCard style={{ marginBottom: WS.gap, borderColor: '#fecaca', backgroundColor: '#fef2f2' }}>
          <p style={{ margin: 0, color: '#b91c1c' }}>{loadError}</p>
          <div style={{ marginTop: 12 }}>
            <SecondaryButton onClick={() => void refresh()}>重试</SecondaryButton>
          </div>
        </SectionCard>
      ) : null}

      <SectionCard style={{ marginBottom: WS.gap }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {TAB_OPTIONS.map((tab) => {
            const selected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '8px 14px',
                  fontSize: 13,
                  fontWeight: 500,
                  borderRadius: 8,
                  border: `1px solid ${selected ? '#2563eb' : '#d1d5db'}`,
                  backgroundColor: selected ? '#eff6ff' : '#fff',
                  color: selected ? '#1d4ed8' : '#374151',
                  cursor: 'pointer',
                }}
              >
                {tab.label}
              </button>
            );
          })}
          <span style={{ marginLeft: 'auto', fontSize: 13, color: '#6b7280', alignSelf: 'center' }}>
            {loading ? '加载中…' : `共 ${filtered.length} 项`}
          </span>
        </div>
      </SectionCard>

      {!loading && filtered.length === 0 ? (
        <SectionCard>
          <p style={{ textAlign: 'center', color: '#6b7280', margin: '32px 16px', lineHeight: 1.7 }}>
            {EMPTY_STATE}
          </p>
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              gap: 12,
              marginBottom: 32,
              flexWrap: 'wrap',
            }}
          >
            <Link href="/workspace/resources/sim-assets/import" style={{ textDecoration: 'none' }}>
              <SecondaryButton>导入资产</SecondaryButton>
            </Link>
            <Link href="/workspace/resources/sim-assets/reconstruct" style={{ textDecoration: 'none' }}>
              <PrimaryButton>图像重建</PrimaryButton>
            </Link>
          </div>
        </SectionCard>
      ) : (
        <SectionCard style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                  {['名称', '类型', '来源', '目标平台', '主格式', '状态', '更新时间', '操作'].map(
                    (col) => (
                      <th
                        key={col}
                        style={{
                          textAlign: 'left',
                          padding: '12px 16px',
                          fontWeight: 600,
                          color: '#374151',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {col}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={8} style={{ padding: '24px 16px', color: '#6b7280' }}>
                      加载重建任务…
                    </td>
                  </tr>
                ) : (
                  filtered.map((job) => {
                    const isDeleting = deletingJobId === job.jobId;
                    const isExportLoading = exportLoadingJobId === job.jobId;
                    return (
                      <tr key={job.jobId} style={{ borderBottom: '1px solid #f3f4f6' }}>
                        <td style={{ padding: '12px 16px', fontWeight: 500, color: '#111827' }}>
                          {job.name || job.jobId}
                        </td>
                        <td style={{ padding: '12px 16px', color: '#4b5563' }}>
                          {simAssetTypeLabel((job.assetType as 'object') || 'object')}
                        </td>
                        <td style={{ padding: '12px 16px', color: '#4b5563' }}>重建</td>
                        <td style={{ padding: '12px 16px', color: '#4b5563' }}>
                          {simAssetTargetLabel((job.targetEngine as 'generic') || 'generic')}
                        </td>
                        <td style={{ padding: '12px 16px', color: '#4b5563' }}>{primaryFormat(job)}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <StatusBadge
                            status={badgeStatus(job.status)}
                            label={PIPELINE_STATUS_LABELS[job.status] || job.status}
                          />
                        </td>
                        <td style={{ padding: '12px 16px', color: '#6b7280', whiteSpace: 'nowrap' }}>
                          {formatUpdatedAt(job.updatedAt)}
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                            <Link
                              href={`/workspace/resources/sim-assets/reconstruct?jobId=${encodeURIComponent(job.jobId)}`}
                              style={{ textDecoration: 'none' }}
                            >
                              <GhostButton>查看结果</GhostButton>
                            </Link>
                            <GhostButton
                              disabled={isExportLoading || isDeleting}
                              onClick={() => void handleOpenExport(job)}
                            >
                              {isExportLoading ? '加载中…' : '导出'}
                            </GhostButton>
                            <DangerButton
                              disabled={isDeleting}
                              onClick={() => setDeleteTarget(job)}
                            >
                              {isDeleting ? '删除中…' : '删除'}
                            </DangerButton>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {deleteTarget ? (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1100,
            background: 'rgba(15, 23, 42, 0.45)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 24,
          }}
          onClick={() => {
            if (!deletingJobId) setDeleteTarget(null);
          }}
        >
          <div
            style={{
              width: '100%',
              maxWidth: 420,
              background: '#fff',
              borderRadius: 12,
              padding: '20px 22px',
              boxShadow: '0 20px 48px rgba(15, 23, 42, 0.18)',
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <div style={{ fontSize: 16, fontWeight: 600, color: '#111827', marginBottom: 10 }}>
              确认删除该重建资产？
            </div>
            <p style={{ margin: '0 0 18px', fontSize: 13, color: '#64748b', lineHeight: 1.7 }}>
              删除后将同时删除分割、重建、MuJoCo 导出等所有文件，无法恢复。
            </p>
            <div style={{ fontSize: 13, color: '#374151', marginBottom: 18 }}>
              {deleteTarget.name || deleteTarget.jobId}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <SecondaryButton disabled={Boolean(deletingJobId)} onClick={() => setDeleteTarget(null)}>
                取消
              </SecondaryButton>
              <button
                type="button"
                disabled={Boolean(deletingJobId)}
                onClick={() => void handleConfirmDelete()}
                style={{
                  padding: '8px 16px',
                  fontSize: 14,
                  fontWeight: 500,
                  borderRadius: 8,
                  border: 'none',
                  backgroundColor: deletingJobId ? '#fca5a5' : '#dc2626',
                  color: '#fff',
                  cursor: deletingJobId ? 'not-allowed' : 'pointer',
                }}
              >
                {deletingJobId ? '删除中…' : '确认删除'}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {exportingJobId ? (
        <AssetExportDialog
          open={exportDialogOpen}
          onClose={handleCloseExportDialog}
          jobId={exportingJobId}
          jobStatus={exportingJobStatus}
          downloadingFile={downloadingFile}
          onDownload={handleDownloadFile}
        />
      ) : null}

      <SimAssetToast message={toastMsg} />
    </ModulePageContainer>
  );
}
