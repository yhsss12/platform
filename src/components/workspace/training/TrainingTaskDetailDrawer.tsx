'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { getTrainingJobLog, getTrainingJobModel, getTrainingJobStatus, getTrainingCapabilities, type TrainingCapabilities, type TrainingJobStatus } from '@/lib/api/trainingClient';
import { normalizedTrainingMetrics } from '@/lib/workspace/normalizedTrainingMetrics';
import {
  getWorkspaceJob,
  getWorkspaceJobArtifacts,
  type WorkspaceArtifactItem,
  type WorkspaceJobDetail,
} from '@/lib/api/workspaceJobClient';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import {
  isTrainingJobInProgressFromSignals,
  normalizeTrainingJobStatus,
} from '@/lib/workspace/trainingStatus';
import { unavailableDetailExplanation } from '@/lib/workspace/trainingCapabilityUi';
import { resolveTrainingTaskDisplayName } from '@/lib/workspace/trainingDisplay';
import { resolveTrainingDisplayState } from '@/lib/workspace/trainingDisplayState';
import { resolvePolicySchemaDisplay } from '@/lib/workspace/trainingPolicySchemaDisplay';
import { extractStoredTrainingJobConfig } from '@/lib/workspace/trainingJobConfig';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { TrainingTaskMetricsPanel } from '@/components/workspace/training/TrainingTaskMetricsPanel';
import { TrainingJobModelAssetsPanel } from '@/components/workspace/training/TrainingJobModelAssetsPanel';
import { TrainingStatusCell } from '@/components/workspace/training/TrainingStatusCell';
import {
  TrainingTaskSummaryGrid,
  trainingSectionTitleStyle,
  resolveAttachmentSideChannelDisplay,
  formatAttachmentInputModeLabel,
  formatAttachmentControlModeLabel,
} from '@/components/workspace/training/trainingDetailShared';

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

export function TrainingTaskDetailDrawer({
  row,
  onClose,
  onRefresh,
  onDelete,
}: {
  row: TrainingTaskRow | null;
  onClose: () => void;
  onRefresh?: () => void | Promise<void>;
  onDelete?: (row: TrainingTaskRow) => void;
}) {
  const [log, setLog] = useState('');
  const [modelReady, setModelReady] = useState(false);
  const [modelAssetId, setModelAssetId] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);
  const [jobDetail, setJobDetail] = useState<WorkspaceJobDetail | null>(null);
  const [liveStatus, setLiveStatus] = useState<TrainingJobStatus | null>(null);
  const [artifacts, setArtifacts] = useState<WorkspaceArtifactItem[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [showTechnical, setShowTechnical] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [loadingLog, setLoadingLog] = useState(false);
  const prevInProgressRef = useRef(false);

  useEffect(() => {
    if (!row) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [row, onClose]);

  useEffect(() => {
    void getTrainingCapabilities().then(setCapabilities).catch(() => setCapabilities(null));
  }, [row?.trainJobId]);

  useEffect(() => {
    if (!row) {
      setJobDetail(null);
      setArtifacts([]);
      return;
    }

    let cancelled = false;
    setLoadingDetail(true);
    void Promise.all([
      getWorkspaceJob(row.trainJobId).catch(() => null),
      getWorkspaceJobArtifacts(row.trainJobId).catch(() => null),
    ])
      .then(([detail, artifactResp]) => {
        if (cancelled) return;
        setJobDetail(detail);
        setArtifacts(artifactResp?.artifacts ?? []);
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });

    return () => {
      cancelled = true;
    };
  }, [row?.trainJobId]);

  const metrics = useMemo(() => {
    const statusEpoch = liveStatus?.epoch ?? row?.currentEpoch;
    const statusTotalEpochs = liveStatus?.totalEpochs ?? row?.totalEpochs;
    const statusProgress =
      liveStatus != null ? Math.round((liveStatus.progress ?? 0) * 100) : row?.progressPercent;
    const statusBackend = liveStatus?.status ?? row?.backendStatus;
    const checkpointExists = liveStatus?.checkpointExists ?? row?.checkpointExists;

    const statusPreview = row
      ? normalizeTrainingJobStatus({
          backendStatus: statusBackend,
          status: row.status,
          currentEpoch: statusEpoch,
          totalEpochs: statusTotalEpochs,
          progressPercent: statusProgress,
          log,
          checkpointExists,
          message: liveStatus?.message ?? row.message,
        })
      : null;

    return normalizedTrainingMetrics({
      log,
      row,
      metrics: {
        ...(jobDetail?.metrics ?? {}),
        lossHistory: jobDetail?.metrics?.lossHistory ?? jobDetail?.metrics?.loss_history,
        totalEpochs: statusTotalEpochs,
        epoch: statusEpoch,
        loss: liveStatus?.loss ?? row?.loss,
        progress: liveStatus?.progress ?? jobDetail?.metrics?.progress,
        bestLoss: jobDetail?.metrics?.bestLoss,
        finalLoss: statusPreview?.completed ? jobDetail?.metrics?.finalLoss : null,
      },
    });
  }, [log, row, jobDetail?.metrics, liveStatus]);

  const resolvedRow = useMemo(() => {
    if (!row) return null;
    const backendStatus = liveStatus?.status ?? row.backendStatus;
    const checkpointExists = liveStatus?.checkpointExists ?? row.checkpointExists;
    const normalized = normalizeTrainingJobStatus({
      backendStatus,
      status: row.status,
      currentEpoch: metrics.currentEpoch,
      totalEpochs: metrics.totalEpochs,
      progressPercent: metrics.progressPercent,
      log,
      checkpointExists,
      message: liveStatus?.message ?? row.message,
    });
    return {
      ...row,
      status: normalized.displayStatus,
      backendStatus: normalized.backendStatus,
      currentEpoch: metrics.currentEpoch,
      totalEpochs: metrics.totalEpochs,
      progressPercent: metrics.progressPercent ?? row.progressPercent,
      loss: metrics.loss ?? row.loss,
      checkpointExists,
      message: liveStatus?.message ?? row.message,
    };
  }, [row, metrics, log, liveStatus]);

  useEffect(() => {
    setShowLog(false);
    prevInProgressRef.current = resolvedRow
      ? isTrainingJobInProgressFromSignals({
          backendStatus: resolvedRow.backendStatus,
          status: resolvedRow.status,
          currentEpoch: resolvedRow.currentEpoch,
          totalEpochs: resolvedRow.totalEpochs,
          progressPercent: resolvedRow.progressPercent,
          log,
          checkpointExists: resolvedRow.checkpointExists,
        })
      : false;
  }, [row?.trainJobId]);

  useEffect(() => {
    if (!row?.trainJobId) {
      setLog('');
      setLiveStatus(null);
      return;
    }

    let cancelled = false;
    let finalRefreshTimer: number | undefined;

    const refreshJobDetail = async () => {
      try {
        const detail = await getWorkspaceJob(row.trainJobId);
        if (!cancelled) setJobDetail(detail);
      } catch {
        // keep previous job detail on transient errors
      }
    };

    const refreshLiveStatus = async () => {
      try {
        const status = await getTrainingJobStatus(row.trainJobId);
        if (!cancelled) setLiveStatus(status);
      } catch {
        // keep previous live status on transient errors
      }
    };

    const loadLog = async () => {
      setLoadingLog(true);
      try {
        const logRes = await getTrainingJobLog(row.trainJobId);
        if (!cancelled && logRes.log) setLog(logRes.log);
      } catch {
        // keep previous log on transient errors
      } finally {
        if (!cancelled) setLoadingLog(false);
      }
    };

    const loadMetricsSources = async () => {
      await Promise.all([loadLog(), refreshJobDetail(), refreshLiveStatus()]);

      try {
        const modelRes = await getTrainingJobModel(row.trainJobId);
        if (!cancelled) {
          setModelReady(Boolean(modelRes.ready));
          setModelAssetId(
            (modelRes.modelManifest?.modelAssetId as string | undefined) ?? row.modelAssetId
          );
        }
      } catch {
        if (!cancelled) {
          setModelReady(Boolean(liveStatus?.checkpointExists ?? row.checkpointExists));
          setModelAssetId(row.modelAssetId);
        }
      }
    };

    const wasInProgress = prevInProgressRef.current;
    const inProgress = isTrainingJobInProgressFromSignals({
      backendStatus: liveStatus?.status ?? row.backendStatus,
      status: row.status,
      currentEpoch: liveStatus?.epoch ?? row.currentEpoch,
      totalEpochs: liveStatus?.totalEpochs ?? row.totalEpochs,
      progressPercent:
        liveStatus != null ? Math.round((liveStatus.progress ?? 0) * 100) : row.progressPercent,
      log,
      checkpointExists: liveStatus?.checkpointExists ?? row.checkpointExists,
    });

    void loadMetricsSources();

    if (wasInProgress && !inProgress) {
      finalRefreshTimer = window.setTimeout(() => {
        void loadMetricsSources();
        void onRefresh?.();
      }, 600);
    }

    prevInProgressRef.current = inProgress;

    if (!inProgress) {
      return () => {
        cancelled = true;
        if (finalRefreshTimer != null) window.clearTimeout(finalRefreshTimer);
      };
    }

    const executionMode = String(
      (jobDetail?.metadata?.trainConfig as { executionMode?: string } | undefined)?.executionMode ?? ''
    ).toLowerCase();
    const pollMs = executionMode === 'remote_ssh' ? 2000 : 4000;

    const timer = window.setInterval(() => {
      void loadMetricsSources();
      void onRefresh?.();
    }, pollMs);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
      if (finalRefreshTimer != null) window.clearTimeout(finalRefreshTimer);
    };
  }, [
    row?.trainJobId,
    row?.status,
    row?.backendStatus,
    row?.currentEpoch,
    row?.totalEpochs,
    row?.progressPercent,
    row?.checkpointExists,
    row?.modelAssetId,
    jobDetail?.metadata,
    liveStatus?.status,
    liveStatus?.epoch,
    liveStatus?.totalEpochs,
    liveStatus?.progress,
    liveStatus?.checkpointExists,
    log,
    onRefresh,
  ]);

  const displayState = useMemo(() => {
    if (!resolvedRow) return null;
    const executionMode = String(
      (jobDetail?.metadata?.trainConfig as { executionMode?: string } | undefined)?.executionMode ?? ''
    );
    return resolveTrainingDisplayState({
      backendStatus: resolvedRow.backendStatus,
      status: resolvedRow.status,
      currentEpoch: resolvedRow.currentEpoch,
      totalEpochs: resolvedRow.totalEpochs,
      progressPercent: resolvedRow.progressPercent,
      log,
      message: resolvedRow.message,
      lossSeries: metrics.lossSeries,
      executionMode,
    });
  }, [resolvedRow, log, metrics.lossSeries, jobDetail?.metadata]);

  const metadata = jobDetail?.metadata ?? {};
  const trainConfig = extractStoredTrainingJobConfig(metadata);
  const attachmentDisplay = useMemo(
    () => resolveAttachmentSideChannelDisplay(metadata),
    [metadata]
  );

  const displayName = useMemo(() => {
    if (!row) return '';
    return resolveTrainingTaskDisplayName({
      taskName: row.name,
      metaTaskName: typeof metadata.taskName === 'string' ? metadata.taskName : null,
      trainConfigTaskName:
        typeof trainConfig?.taskName === 'string' ? trainConfig.taskName : null,
      datasetName: row.datasetName ?? row.relatedTask,
      trainingBackend: row.trainingBackend,
      modelType: row.modelType,
      jobId: row.trainJobId,
    });
  }, [row, metadata.taskName, trainConfig?.taskName]);

  const policySchemaDisplay = useMemo(
    () =>
      resolvePolicySchemaDisplay({
        trainingBackend: row?.trainingBackend,
        modelType: row?.modelType,
        trainConfig: trainConfig as Record<string, unknown> | null,
      }),
    [row?.trainingBackend, row?.modelType, trainConfig]
  );

  if (!row || !resolvedRow) return null;

  const isUnavailable = resolvedRow.status === '失败' || resolvedRow.backendStatus === 'failed';
  const unavailableLines = isUnavailable
    ? unavailableDetailExplanation(row.modelType, capabilities)
    : [];

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal aria-labelledby="training-task-drawer-title">
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
              id="training-task-drawer-title"
              style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827', lineHeight: 1.35 }}
            >
              {displayName}
            </h2>
            <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280', fontFamily: 'ui-monospace, monospace' }}>
              {row.trainJobId}
            </div>
            <div style={{ marginTop: 10 }}>
              <TrainingStatusCell row={resolvedRow} />
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          <div style={trainingSectionTitleStyle}>训练任务信息</div>
          <div style={{ marginBottom: 20 }}>
            <TrainingTaskSummaryGrid
              row={resolvedRow}
              jobDetail={jobDetail}
              displayName={displayName}
            />
          </div>

          {displayState &&
          (displayState.phase === 'failed' ||
            displayState.phase === 'waiting' ||
            (displayState.progressHint &&
              (displayState.phase === 'created' || displayState.phase === 'launching'))) ? (
            <div
              style={{
                marginBottom: 16,
                padding: '10px 12px',
                borderRadius: 8,
                backgroundColor:
                  displayState.phase === 'failed' ? '#fef2f2' : '#fffbeb',
                border: `1px solid ${
                  displayState.phase === 'failed' ? '#fecaca' : '#fde68a'
                }`,
                fontSize: 13,
                color: displayState.phase === 'failed' ? '#991b1b' : '#92400e',
                lineHeight: 1.55,
              }}
            >
              {displayState.phase === 'failed'
                ? displayState.subLabel ?? resolvedRow.message
                : displayState.progressHint ?? displayState.subLabel}
            </div>
          ) : null}

          {isUnavailable ? (
            <div
              style={{
                marginBottom: 16,
                padding: '10px 12px',
                borderRadius: 8,
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                fontSize: 13,
                color: '#991b1b',
                lineHeight: 1.55,
              }}
            >
              {unavailableLines.map((line) => (
                <div key={line}>{line}</div>
              ))}
            </div>
          ) : null}

          <div style={{ ...trainingSectionTitleStyle, marginTop: 4 }}>训练过程性能</div>
          <TrainingTaskMetricsPanel
            metrics={metrics}
            status={resolvedRow.backendStatus}
            displayState={displayState ?? undefined}
          />

          <TrainingJobModelAssetsPanel
            trainJobId={row.trainJobId}
            jobStatus={resolvedRow.status}
            jobBackendStatus={resolvedRow.backendStatus}
            jobCurrentEpoch={resolvedRow.currentEpoch}
            jobTotalEpochs={resolvedRow.totalEpochs}
            jobProgressPercent={resolvedRow.progressPercent}
            onNavigateToEvaluation={onClose}
          />

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
              }}
            >
              <div>jobId：{row.trainJobId}</div>
              <div>taskType：{jobDetail?.taskType ?? row.taskType ?? '—'}</div>
              <div>runner：{jobDetail?.runner ?? row.runner ?? '—'}</div>
              <div>backend status：{jobDetail?.status ?? row.backendStatus ?? '—'}</div>
              <div>
                checkpoint：
                {row.checkpointExists ? row.checkpointPath ?? row.checkpoint ?? '已生成' : '未生成'}
              </div>
              <div>
                model asset：
                {modelReady && modelAssetId ? `${modelAssetId}（已注册）` : '—'}
              </div>
              {attachmentDisplay?.enabled ? (
                <>
                  <div style={{ marginTop: 8 }}>抓取侧信道：已启用</div>
                  <div>策略输入：{formatAttachmentInputModeLabel(attachmentDisplay.inputMode)}</div>
                  <div>
                    评测控制：{formatAttachmentControlModeLabel(attachmentDisplay.controlMode)}
                  </div>
                </>
              ) : null}
              {policySchemaDisplay ? (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontWeight: 600, color: '#374151' }}>Policy / I/O schema</div>
                  <div>Policy：{policySchemaDisplay.policyLabel}</div>
                  <div style={{ marginTop: 4 }}>Observation schema：</div>
                  <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                    {policySchemaDisplay.observationKeys.map((key) => (
                      <li key={key}>{key}</li>
                    ))}
                  </ul>
                  <div style={{ marginTop: 4 }}>Action schema：</div>
                  <div>action_key = {policySchemaDisplay.actionKey}</div>
                  <div>action_dim = {policySchemaDisplay.actionDim ?? '—'}</div>
                  <div>{policySchemaDisplay.actionDescription}</div>
                  <div style={{ marginTop: 4 }}>Controller：{policySchemaDisplay.controllerType}</div>
                  <div>Eval executor：{policySchemaDisplay.evalExecutor}</div>
                  {policySchemaDisplay.note ? (
                    <div style={{ marginTop: 6, color: '#64748b' }}>{policySchemaDisplay.note}</div>
                  ) : null}
                </div>
              ) : null}
              {trainConfig?.pretrained?.modelAssetId || trainConfig?.pretrained?.checkpointPath ? (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontWeight: 600, color: '#374151' }}>初始化权重（只读）</div>
                  {trainConfig.pretrained.modelAssetName ? (
                    <div>模型资产名称：{trainConfig.pretrained.modelAssetName}</div>
                  ) : null}
                  {trainConfig.pretrained.modelAssetId ? (
                    <div>模型资产 ID：{trainConfig.pretrained.modelAssetId}</div>
                  ) : null}
                  {trainConfig.pretrained.checkpointPath ? (
                    <div>checkpoint 路径：{trainConfig.pretrained.checkpointPath}</div>
                  ) : null}
                  {trainConfig.pretrained.sourceTrainJobId ? (
                    <div>来源任务：{trainConfig.pretrained.sourceTrainJobId}</div>
                  ) : null}
                </div>
              ) : null}
              {row.datasetManifestPath ? <div>manifest：{row.datasetManifestPath}</div> : null}
              {artifacts.length > 0 ? (
                <div style={{ marginTop: 8 }}>
                  artifacts：
                  <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                    {artifacts.map((artifact) => (
                      <li key={artifact.id}>
                        {artifact.artifactType} · {artifact.name}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {loadingDetail ? <div style={{ marginTop: 8, color: '#9ca3af' }}>加载中…</div> : null}
            </div>
          ) : null}

          <button
            type="button"
            onClick={() => setShowLog((v) => !v)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              marginTop: 16,
              padding: 0,
              border: 'none',
              background: 'none',
              fontSize: 12,
              fontWeight: 600,
              color: '#6b7280',
              cursor: 'pointer',
            }}
          >
            <span style={{ fontSize: 10, color: '#9ca3af' }}>{showLog ? '▼' : '▶'}</span>
            日志
          </button>

          {showLog ? (
            <div
              style={{
                marginTop: 10,
                padding: '12px 14px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                backgroundColor: '#0f172a',
                maxHeight: 280,
                overflow: 'auto',
              }}
            >
              {loadingLog && !log ? (
                <div style={{ fontSize: 12, color: '#94a3b8' }}>加载日志…</div>
              ) : log ? (
                <pre
                  style={{
                    margin: 0,
                    fontSize: 11,
                    lineHeight: 1.5,
                    color: '#e2e8f0',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {log}
                </pre>
              ) : (
                <div style={{ fontSize: 12, color: '#94a3b8' }}>暂无日志</div>
              )}
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
          {onRefresh ? <SecondaryButton onClick={() => void onRefresh()}>刷新状态</SecondaryButton> : null}
          {onDelete ? (
            <SecondaryButton onClick={() => onDelete(row)}>删除</SecondaryButton>
          ) : null}
        </div>
      </aside>
    </>
  );
}
