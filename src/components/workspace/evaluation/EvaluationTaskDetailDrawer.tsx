'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import {
  evaluationTaskStatusBadge,
  formatEvalConfig,
  formatEvalConfigList,
  formatEvaluationSampleCountList,
  formatEvaluationSuccessRateList,
} from '@/lib/mock/workspaceEvaluationRecordsMock';
import { findPhysicsProxyModel } from '@/lib/mock/physicsProxiesMock';
import { SecondaryButton, StatusBadge } from '@/components/workspace/workspaceUi';
import { EVALUATION_ROUTES } from '@/components/workspace/evaluation/EvaluationRecordsTable';
import { buildCableThreadingReplayHref } from '@/lib/workspace/cableThreading';
import {
  buildDualArmEvalReplayHref,
  buildDualArmEvalReportHref,
  isDualArmEvalRow,
} from '@/lib/workspace/dualArmEvaluation';
import {
  getWorkspaceJob,
  getWorkspaceJobArtifacts,
  type WorkspaceArtifactItem,
  type WorkspaceJobDetail,
} from '@/lib/api/workspaceJobClient';
import { getEvaluationJobResult } from '@/lib/api/evaluationClient';
import { EvalResolvedMetricsPanel } from '@/components/workspace/evaluation/EvalResolvedMetricsPanel';
import { isIsaacEvalRow, buildIsaacEvalReplayHref, buildIsaacEvalReportHref } from '@/lib/workspace/isaacBlockStacking';
import {
  ISAAC_STACK_DEFAULT_METRIC_IDS,
  normalizeEvaluationJobResultPayload,
  normalizeEvaluationMode,
  normalizeEvaluationTaskType,
} from '@/lib/workspace/evaluationMetricRegistry';
import { resolveEvaluationTypeLabel } from '@/lib/workspace/evaluationDisplay';

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
  width: 520,
  maxWidth: '100vw',
  backgroundColor: '#fff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word', lineHeight: 1.5 }}>
        {children}
      </div>
    </div>
  );
}

function JsonSummary({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).slice(0, 16);
  if (entries.length === 0) return <>—</>;
  return (
    <pre
      style={{
        margin: 0,
        padding: 10,
        borderRadius: 8,
        backgroundColor: '#f8fafc',
        fontSize: 11,
        lineHeight: 1.5,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}
    >
      {JSON.stringify(Object.fromEntries(entries), null, 2)}
    </pre>
  );
}

function artifactLabel(artifact: WorkspaceArtifactItem): string {
  return `${artifact.artifactType} · ${artifact.name}${artifact.urlPath ? ` · ${artifact.urlPath}` : ''}`;
}

export function EvaluationTaskDetailDrawer({
  row,
  onClose,
  onExportReport,
}: {
  row: EvaluationTaskRow | null;
  onClose: () => void;
  onExportReport: (row: EvaluationTaskRow) => void;
}) {
  const [jobDetail, setJobDetail] = useState<WorkspaceJobDetail | null>(null);
  const [artifacts, setArtifacts] = useState<WorkspaceArtifactItem[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [evalAggregate, setEvalAggregate] = useState<Record<string, unknown> | null>(null);
  const [evalPerEpisode, setEvalPerEpisode] = useState<Record<string, unknown> | null>(null);
  const [loadingEvalMetrics, setLoadingEvalMetrics] = useState(false);

  useEffect(() => {
    if (!row) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [row, onClose]);

  useEffect(() => {
    if (!row) {
      setJobDetail(null);
      setArtifacts([]);
      setEvalAggregate(null);
      setEvalPerEpisode(null);
      return;
    }
    let cancelled = false;
    setLoadingDetail(true);
    void Promise.all([
      getWorkspaceJob(row.id).catch(() => null),
      getWorkspaceJobArtifacts(row.id).catch(() => null),
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
  }, [row?.id]);

  useEffect(() => {
    if (!row || !isIsaacEvalRow(row)) {
      setEvalAggregate(null);
      setEvalPerEpisode(null);
      return;
    }
    let cancelled = false;
    setLoadingEvalMetrics(true);
    void getEvaluationJobResult(row.id)
      .then((result) => {
        if (cancelled) return;
        const normalized = normalizeEvaluationJobResultPayload(result);
        setEvalAggregate(normalized.aggregate);
        setEvalPerEpisode(normalized.perEpisode);
      })
      .catch(() => {
        if (!cancelled) {
          setEvalAggregate(null);
          setEvalPerEpisode(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingEvalMetrics(false);
      });
    return () => {
      cancelled = true;
    };
  }, [row?.id]);

  const logArtifacts = useMemo(
    () => artifacts.filter((a) => a.artifactType === 'log'),
    [artifacts]
  );
  const videoArtifacts = useMemo(
    () => artifacts.filter((a) => a.artifactType === 'video'),
    [artifacts]
  );
  const resultArtifacts = useMemo(
    () =>
      artifacts.filter((a) =>
        ['metrics', 'aggregate_result', 'per_episode_result', 'result'].includes(a.artifactType)
      ),
    [artifacts]
  );

  if (!row) return null;

  const isProcessMode = row.evaluationMode === '数据过程评测';
  const isIsaacEval = isIsaacEvalRow(row);
  const metadata = jobDetail?.metadata ?? {};
  const metrics = jobDetail?.metrics ?? row.aggregate ?? {};
  const resolvedTaskType = normalizeEvaluationTaskType(jobDetail?.taskType ?? row.taskType);
  const resolvedEvaluationMode = normalizeEvaluationMode(
    (metadata.evaluationMode as string | undefined) ??
      row.evaluationModeApi ??
      (row.evaluationMode === 'episode 稳定性评测'
        ? 'episode_stability'
        : row.evaluationMode === '数据过程评测'
          ? 'dataset_offline'
          : row.evaluationMode)
  );

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal>
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
          }}
        >
          <div>
            <h2 style={{ margin: 0, fontSize: 17, fontWeight: 600 }}>评测详情</h2>
            <div style={{ marginTop: 8 }}>
              <StatusBadge status={evaluationTaskStatusBadge(row.status)} label={row.status} />
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>
        <div style={{ padding: '20px 24px', overflow: 'auto', flex: 1 }}>
          <Row label="jobId">
            <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{row.id}</span>
          </Row>
          <Row label="jobType">evaluation</Row>
          <Row label="taskType">{jobDetail?.taskType ?? row.taskType ?? '—'}</Row>
          <Row label="evaluationMode">{row.evaluationMode}</Row>
          <Row label="评测类型">{resolveEvaluationTypeLabel(row.evaluationModeApi ?? resolvedEvaluationMode)}</Row>
          <Row label="metricType">
            {row.evaluationMode === 'episode 稳定性评测' ? 'episode_stability' : '—'}
          </Row>
          <Row label="status">{jobDetail?.status ?? row.backendJobStatus ?? row.status}</Row>
          <Row label="runner">{jobDetail?.runner ?? row.runner ?? '—'}</Row>
          <Row label="runtimePath">{jobDetail?.runtimePath ?? row.runtimePath ?? '—'}</Row>
          <Row label="createdAt">{jobDetail?.createdAt ?? row.createdAt}</Row>
          <Row label="updatedAt">{jobDetail?.updatedAt ?? row.updatedAt ?? '—'}</Row>
          <Row label="startedAt">{jobDetail?.startedAt ?? row.startedAt ?? '—'}</Row>
          <Row label="finishedAt">{jobDetail?.finishedAt ?? row.finishedAt ?? '—'}</Row>

          <Row label="评测任务名称">{row.name}</Row>
          {row.rawName && row.rawName !== row.name ? (
            <Row label="原始名称">
              <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{row.rawName}</span>
            </Row>
          ) : null}
          <Row label="关联任务">{row.relatedTask}</Row>

          {!isProcessMode ? (
            <>
              <Row label="模型版本">
                <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>
                  {row.checkpoint && row.checkpoint !== '—' ? row.checkpoint : '—'}
                </span>
              </Row>
              <Row label="模型类型">
                {row.modelType && row.modelType !== '—' ? row.modelType : '—'}
              </Row>
              <Row label="样本数量">{formatEvaluationSampleCountList(row)}</Row>
              <Row label="评测配置">{formatEvalConfigList(row)}</Row>
              <Row label="评测配置（完整）">{formatEvalConfig(row)}</Row>
              {row.physicsProxyMode && row.physicsProxyMode !== 'off' ? (
                <>
                  <Row label="代理模型">{row.physicsProxyModel ?? '—'}</Row>
                  <Row label="误差估计">
                    {row.physicsProxyError ??
                      findPhysicsProxyModel(row.physicsProxyModel)?.errorMetric ??
                      '—'}
                  </Row>
                  <Row label="加速倍率">
                    {row.physicsProxySpeedup ??
                      findPhysicsProxyModel(row.physicsProxyModel)?.speedup ??
                      '—'}
                  </Row>
                </>
              ) : null}
              <Row label="成功率">
                {isIsaacEval
                  ? '见下方评测指标'
                  : formatEvaluationSuccessRateList(row.status, row.successRate)}
              </Row>
              {isIsaacEval ? (
                <div style={{ marginBottom: 16 }}>
                  <EvalResolvedMetricsPanel
                    taskType={resolvedTaskType}
                    evaluationMode={resolvedEvaluationMode}
                    aggregate={evalAggregate ?? (metrics as Record<string, unknown>)}
                    perEpisode={evalPerEpisode}
                    metricIds={[...ISAAC_STACK_DEFAULT_METRIC_IDS]}
                    loading={loadingEvalMetrics}
                  />
                </div>
              ) : null}
            </>
          ) : (
            <>
              <Row label="数据名称">{row.dataName ?? '—'}</Row>
              <Row label="过程评分">{row.processScore ?? '—'}</Row>
            </>
          )}

          <Row label="指标配置">{(row.metrics ?? []).join(' · ') || '—'}</Row>
          <Row label="结果摘要">{row.resultSummary ?? '—'}</Row>

          <Row label="eval log path">
            {logArtifacts.length > 0
              ? logArtifacts.map((a) => a.filePath).join('\n')
              : row.evalCsvPath ?? '—'}
          </Row>
          <Row label="result path">
            {resultArtifacts.length > 0
              ? resultArtifacts.map((a) => a.filePath).join('\n')
              : row.resultPath ?? '—'}
          </Row>
          <Row label="video artifact">
            {videoArtifacts.length > 0
              ? videoArtifacts.map((a) => a.filePath).join('\n')
              : row.videoPath ?? (row.videoExists ? '已生成' : '—')}
          </Row>

          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', margin: '16px 0 8px' }}>
            metadata_json 摘要 {loadingDetail ? '（加载中）' : ''}
          </div>
          <JsonSummary data={metadata} />

          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', margin: '16px 0 8px' }}>
            metrics_json 摘要
          </div>
          <JsonSummary data={metrics as Record<string, unknown>} />

          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', margin: '16px 0 8px' }}>
            artifacts 列表
          </div>
          {artifacts.length === 0 ? (
            <div style={{ fontSize: 13, color: '#6b7280' }}>—</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#334155', lineHeight: 1.6 }}>
              {artifacts.map((artifact) => (
                <li key={artifact.id}>{artifactLabel(artifact)}</li>
              ))}
            </ul>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 16 }}>
            <Link
              href={
                row.taskType === 'cable_threading'
                  ? buildCableThreadingReplayHref({ evalId: row.id })
                  : isDualArmEvalRow(row)
                    ? buildDualArmEvalReplayHref({ evalJobId: row.id })
                    : isIsaacEvalRow(row)
                      ? buildIsaacEvalReplayHref({ evalJobId: row.id })
                      : EVALUATION_ROUTES.replay
              }
              style={{ textDecoration: 'none' }}
            >
              <SecondaryButton>回放</SecondaryButton>
            </Link>
            {isDualArmEvalRow(row) ? (
              <Link
                href={buildDualArmEvalReportHref({ evalJobId: row.id })}
                style={{ textDecoration: 'none' }}
              >
                <SecondaryButton>报告</SecondaryButton>
              </Link>
            ) : null}
            {isIsaacEval ? (
              <Link
                href={buildIsaacEvalReportHref({ evalJobId: row.id })}
                style={{ textDecoration: 'none' }}
              >
                <SecondaryButton>查看报告</SecondaryButton>
              </Link>
            ) : null}
            <SecondaryButton onClick={() => onExportReport(row)}>导出报告</SecondaryButton>
          </div>
        </div>
      </aside>
    </>
  );
}
