'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { getTrainingJobLog } from '@/lib/api/trainingClient';
import { normalizedTrainingMetrics } from '@/lib/workspace/normalizedTrainingMetrics';
import type { TrainingMetricPoint } from '@/lib/workspace/trainingLogParser';
import {
  getWorkspaceJob,
  getWorkspaceJobArtifacts,
  listWorkspaceJobs,
  type WorkspaceArtifactItem,
  type WorkspaceJobDetail,
  type WorkspaceJobSummary,
} from '@/lib/api/workspaceJobClient';
import type { ModelAsset } from '@/types/benchmark';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import { workspaceTrainingJobToRow } from '@/lib/workspace/workspaceJobMapper';
import {
  resolveModelAssetColumnLabel,
  resolveModelAssetDatasetLabel,
  resolveModelAssetSourceLabel,
  formatModelAssetRecipeLabel,
} from '@/lib/workspace/modelAssetDisplay';
import { formatPercentValue, formatLossValue } from '@/lib/workspace/chartFormat';
import { resolveModelAssetLossLabel } from '@/lib/workspace/trainingLossDisplay';
import { useJobTrainingDurationLabel } from '@/lib/workspace/useTrainingDurationLabel';
import {
  TrainingTaskSummaryGrid,
  TrainingArtifactsInfo,
  buildTrainingArtifactDisplayItems,
  trainingSectionTitleStyle,
} from '@/components/workspace/training/trainingDetailShared';
import { buildModelEvaluationCreateFromAssetUrl } from '@/lib/workspace/evaluationCreateNavigation';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { TrainingTaskMetricsPanel } from '@/components/workspace/training/TrainingTaskMetricsPanel';

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.4)',
  zIndex: 1500,
};

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 640,
  maxWidth: '100vw',
  backgroundColor: '#ffffff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

function evalJobMatchesAsset(
  job: WorkspaceJobSummary,
  asset: ModelAsset
): boolean {
  const metrics = job.metricsSummary ?? {};
  const modelAssetId = String(metrics.modelAssetId ?? '');
  const checkpointJobId = String(
    metrics.checkpointJobId ?? metrics.sourceTrainJobId ?? ''
  );
  return modelAssetId === asset.id || checkpointJobId === asset.sourceTrainingJobId;
}

function bestLossFromSeries(series: TrainingMetricPoint[]): number | null {
  const values = series
    .map((point) => point.trainLoss)
    .filter((v): v is number => v != null && Number.isFinite(v));
  if (values.length === 0) return null;
  return Math.min(...values);
}

export function ModelAssetDetailDrawer({
  asset,
  trainingRow,
  onClose,
  onDelete,
}: {
  asset: ModelAsset | null;
  trainingRow?: TrainingTaskRow | null;
  onClose: () => void;
  onDelete?: (asset: ModelAsset) => void;
}) {
  const [jobDetail, setJobDetail] = useState<WorkspaceJobDetail | null>(null);
  const [artifacts, setArtifacts] = useState<WorkspaceArtifactItem[]>([]);
  const [evalJobs, setEvalJobs] = useState<WorkspaceJobSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [showTechnical, setShowTechnical] = useState(false);
  const [accumulatedLoss, setAccumulatedLoss] = useState<TrainingMetricPoint[]>([]);
  const [log, setLog] = useState('');

  const row = useMemo(() => {
    if (trainingRow) return trainingRow;
    if (jobDetail) return workspaceTrainingJobToRow(jobDetail);
    return null;
  }, [trainingRow, jobDetail]);

  useEffect(() => {
    if (!asset) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [asset, onClose]);

  useEffect(() => {
    if (!asset) {
      setJobDetail(null);
      setArtifacts([]);
      setEvalJobs([]);
      setLog('');
      setAccumulatedLoss([]);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setShowTechnical(false);
    setAccumulatedLoss([]);

    void Promise.all([
      getWorkspaceJob(asset.sourceTrainingJobId).catch(() => null),
      getWorkspaceJobArtifacts(asset.sourceTrainingJobId).catch(() => null),
      getTrainingJobLog(asset.sourceTrainingJobId).catch(() => ({ log: '' })),
      listWorkspaceJobs({ jobType: 'evaluation', source: 'real', limit: 100 }).catch(() => ({
        jobs: [],
        total: 0,
      })),
    ])
      .then(([detail, artifactResp, logRes, evalResp]) => {
        if (cancelled) return;
        setJobDetail(detail);
        setArtifacts(artifactResp?.artifacts ?? []);
        setLog(logRes.log || '');
        setEvalJobs(
          evalResp.jobs.filter((job) => evalJobMatchesAsset(job, asset))
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [asset]);

  useEffect(() => {
    if (!row || row.currentEpoch <= 0 || row.loss == null || !Number.isFinite(row.loss)) return;
    setAccumulatedLoss((prev) => {
      const map = new Map(prev.map((point) => [point.epoch, point]));
      const existing = map.get(row.currentEpoch);
      map.set(row.currentEpoch, {
        epoch: row.currentEpoch,
        trainLoss: row.loss ?? existing?.trainLoss,
        validLoss: existing?.validLoss,
      });
      return Array.from(map.values()).sort((a, b) => a.epoch - b.epoch);
    });
  }, [row?.currentEpoch, row?.loss]);

  const metrics = useMemo(
    () =>
      normalizedTrainingMetrics({
        log,
        row,
        metrics: {
          ...(jobDetail?.metrics ?? {}),
          lossHistory: jobDetail?.metrics?.lossHistory ?? jobDetail?.metrics?.loss_history,
          totalEpochs: row?.totalEpochs,
          epoch: row?.currentEpoch,
          loss: row?.loss,
        },
        accumulated: accumulatedLoss,
      }),
    [log, row, jobDetail?.metrics, accumulatedLoss]
  );

  const backendStatus = jobDetail?.status ?? row?.backendStatus ?? 'completed';
  const trainingDuration = useJobTrainingDurationLabel({
    status: backendStatus,
    jobDetail,
  });
  const artifactItems = useMemo(() => buildTrainingArtifactDisplayItems(artifacts), [artifacts]);
  const lossFieldLabel = resolveModelAssetLossLabel(backendStatus);
  const displayLoss = metrics.loss ?? metrics.bestLoss;

  if (!asset) return null;

  const isImportedAsset =
    asset.assetSource === 'imported' ||
    asset.checkpointKind === 'imported' ||
    asset.sourceTrainingJobId === 'model_asset_import_hub';
  const importMeta = (asset.importMetadata ?? {}) as Record<string, unknown>;
  const validationResult = (asset.validationResult ?? importMeta.validationResult ?? {}) as Record<string, unknown>;
  const structureConfig = (asset.structureConfig ?? {}) as Record<string, unknown>;
  const inputCfg = (structureConfig.input ?? {}) as Record<string, unknown>;

  const displayName = resolveModelAssetColumnLabel(asset, row);
  const recipeLabel = formatModelAssetRecipeLabel(asset);
  const datasetLabel = resolveModelAssetDatasetLabel(asset, row);
  const sourceLabel = resolveModelAssetSourceLabel(asset);
  const updatedAt = formatDateTimeMinuteYmdSlash(jobDetail?.updatedAt ?? asset.updatedAt);

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal aria-labelledby="model-asset-drawer-title">
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 12,
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <h2
              id="model-asset-drawer-title"
              style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827', lineHeight: 1.35 }}
            >
              {displayName}
            </h2>
            <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              <span
                style={{
                  display: 'inline-block',
                  padding: '2px 10px',
                  borderRadius: 999,
                  fontSize: 12,
                  fontWeight: 500,
                  backgroundColor: '#eff6ff',
                  color: '#1d4ed8',
                }}
              >
                {recipeLabel}
              </span>
              <span
                style={{
                  display: 'inline-block',
                  padding: '2px 10px',
                  borderRadius: 999,
                  fontSize: 12,
                  fontWeight: 500,
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                }}
              >
                {datasetLabel}
              </span>
            </div>
            <div style={{ marginTop: 8, fontSize: 12, color: '#6b7280' }}>
              来源 {sourceLabel} · 创建 {formatDateTimeMinuteYmdSlash(asset.createdAt)}
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
              gap: 10,
              marginBottom: 20,
            }}
          >
            {[
              {
                label: isImportedAsset ? '模型类型' : '训练耗时',
                value: isImportedAsset ? recipeLabel : trainingDuration,
              },
              {
                label: lossFieldLabel,
                value: displayLoss != null ? formatLossValue(displayLoss) : '—',
              },
              {
                label: isImportedAsset ? '校验状态' : '产物数量',
                value: isImportedAsset
                  ? validationResult.structureMatched === true || validationResult.hasStateDict
                    ? '结构校验通过'
                    : '已导入'
                  : String(artifactItems.length),
              },
              { label: '最后更新', value: updatedAt },
            ].map((item) => (
              <div
                key={item.label}
                style={{
                  padding: '10px 12px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#fff',
                }}
              >
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 4 }}>{item.label}</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#111827' }}>{item.value}</div>
              </div>
            ))}
          </div>

          {isImportedAsset ? (
            <>
              <div style={trainingSectionTitleStyle}>导入信息</div>
              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#fff',
                  fontSize: 13,
                  color: '#334155',
                  lineHeight: 1.7,
                }}
              >
                <div>适用任务：{String(importMeta.taskLabel || asset.taskType || '—')}</div>
                <div>参考数据集：{String(importMeta.referenceDatasetName || datasetLabel)}</div>
                <div>action_dim：{String((structureConfig.output as Record<string, unknown> | undefined)?.action_dim ?? validationResult.actionDim ?? '—')}</div>
                <div>image_keys：{Array.isArray(inputCfg.image_keys) ? (inputCfg.image_keys as string[]).join(', ') : '—'}</div>
                <div>low_dim_keys：{Array.isArray(inputCfg.low_dim_keys) ? (inputCfg.low_dim_keys as string[]).join(', ') : '—'}</div>
                <div>image_size：{String(inputCfg.image_size ?? '—')}</div>
                <div>normalizer：{validationResult.hasNormalizer ? '已包含' : '缺失'}</div>
                <div>train_config：{validationResult.hasTrainConfig ? '已包含' : '缺失'}</div>
                {importMeta.note ? <div>备注：{String(importMeta.note)}</div> : null}
              </div>
            </>
          ) : loading && !row ? (
            <p style={{ fontSize: 13, color: '#9ca3af' }}>加载训练详情…</p>
          ) : row ? (
            <>
              <div style={trainingSectionTitleStyle}>关联训练任务</div>
              <TrainingTaskSummaryGrid row={row} jobDetail={jobDetail} displayName={displayName} />

              <div style={{ ...trainingSectionTitleStyle, marginTop: 20 }}>训练过程性能</div>
              <TrainingTaskMetricsPanel metrics={metrics} status={backendStatus} />

              <div style={{ ...trainingSectionTitleStyle, marginTop: 20 }}>评测结果</div>
              <div
                style={{
                  padding: '12px 14px',
                  borderRadius: 8,
                  border: '1px solid #e5e7eb',
                  backgroundColor: '#fff',
                }}
              >
                {evalJobs.length === 0 ? (
                  <div style={{ fontSize: 13, color: '#9ca3af' }}>暂无关联评测记录</div>
                ) : (
                  <ul style={{ margin: 0, padding: 0, listStyle: 'none' }}>
                    {evalJobs.map((job) => {
                      const metrics = job.metricsSummary ?? {};
                      const successRate = metrics.successRate ?? metrics.success_rate;
                      const rateText =
                        successRate != null && job.status === 'completed'
                          ? formatPercentValue(Number(successRate))
                          : '—';
                      return (
                        <li
                          key={job.jobId}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: 12,
                            padding: '8px 0',
                            borderBottom: '1px solid #f3f4f6',
                            fontSize: 13,
                          }}
                        >
                          <span
                            style={{
                              minWidth: 0,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                            title={job.taskName ?? job.jobId}
                          >
                            {job.taskName ?? job.jobId}
                          </span>
                          <span style={{ color: '#6b7280', flexShrink: 0 }}>
                            {job.status === 'completed' ? rateText : job.status}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>

              <div style={{ marginTop: 20 }}>
                <TrainingArtifactsInfo items={artifactItems} />
              </div>
            </>
          ) : !isImportedAsset ? (
            <p style={{ fontSize: 13, color: '#b45309' }}>无法加载关联训练任务详情</p>
          ) : null}

          <button
            type="button"
            onClick={() => setShowTechnical((v) => !v)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              marginTop: 20,
              padding: 0,
              border: 'none',
              background: 'none',
              fontSize: 12,
              fontWeight: 600,
              color: '#6b7280',
              cursor: 'pointer',
            }}
          >
            <span style={{ fontSize: 10, color: '#9ca3af' }}>{showTechnical ? '▼' : '▶'}</span>
            技术详情
          </button>

          {showTechnical ? (
            <div
              style={{
                marginTop: 10,
                padding: '12px 14px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                backgroundColor: '#fff',
                fontSize: 12,
                color: '#475569',
                lineHeight: 1.6,
                wordBreak: 'break-all',
              }}
            >
              <div>modelAssetId：{asset.id}</div>
              <div>modelTypeId：{asset.modelTypeId ?? '—'}</div>
              <div>modelTypeName：{asset.modelTypeName ?? '—'}</div>
              <div>baseAlgorithm：{asset.baseAlgorithm ?? '—'}</div>
              <div>adapterId：{asset.adapterId ?? '—'}</div>
              <div>structureConfig：{asset.structureConfig ? JSON.stringify(asset.structureConfig) : '—'}</div>
              <div>resolvedModelParams：{asset.resolvedModelParams ? JSON.stringify(asset.resolvedModelParams) : '—'}</div>
              <div>openpiEnvironment：{asset.openpiEnvironment ? JSON.stringify(asset.openpiEnvironment) : '—'}</div>
              <div>storedName：{asset.name || '—'}</div>
              <div>displayName：{displayName}</div>
              <div>taskTemplateId：{asset.taskTemplateId ?? '—'}</div>
              <div>sourceDatasetId：{asset.sourceDatasetId ?? '—'}</div>
              <div>sourceTrainingJobId：{asset.sourceTrainingJobId}</div>
              <div>version：{asset.version}</div>
              <div>status：{asset.status}</div>
              <div>checkpointPath：{asset.checkpointPath || '—'}</div>
              <div>checkpointKind：{asset.checkpointKind ?? '—'}</div>
              <div>checkpointEpoch：{asset.checkpointEpoch ?? '—'}</div>
              <div>manifestPath：{asset.manifestPath || '—'}</div>
            </div>
          ) : null}
        </div>

        <div
          style={{
            padding: '14px 20px',
            borderTop: '1px solid #e5e7eb',
            display: 'flex',
            gap: 8,
            flexWrap: 'wrap',
          }}
        >
          <SecondaryButton onClick={onClose}>关闭</SecondaryButton>
          {onDelete ? (
            <button
              type="button"
              onClick={() => onDelete(asset)}
              style={{
                padding: '8px 14px',
                fontSize: 14,
                fontWeight: 500,
                borderRadius: 8,
                border: '1px solid #fecaca',
                backgroundColor: '#fff',
                color: '#b91c1c',
                cursor: 'pointer',
              }}
            >
              删除
            </button>
          ) : null}
          {!isImportedAsset ? (
            <Link
              href={`/workspace/training?jobId=${encodeURIComponent(asset.sourceTrainingJobId)}`}
              style={{
                padding: '8px 14px',
                fontSize: 14,
                fontWeight: 500,
                borderRadius: 8,
                border: '1px solid #d1d5db',
                backgroundColor: '#fff',
                color: '#374151',
                textDecoration: 'none',
              }}
            >
              查看训练任务
            </Link>
          ) : null}
          <Link
            href={buildModelEvaluationCreateFromAssetUrl(asset)}
            style={{
              padding: '8px 14px',
              fontSize: 14,
              fontWeight: 500,
              borderRadius: 8,
              border: 'none',
              backgroundColor: '#2563eb',
              color: '#fff',
              textDecoration: 'none',
            }}
          >
            发起评测
          </Link>
        </div>
      </aside>
    </>
  );
}
