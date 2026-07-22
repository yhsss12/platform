'use client';

import { useEffect, useMemo, useState } from 'react';
import type { WorkspaceArtifactItem, WorkspaceJobDetail } from '@/lib/api/workspaceJobClient';
import type { ReplaySession } from '@/lib/mock/workspacePagesMock';
import { getCableThreadingJobLog } from '@/lib/api/cableThreadingClient';
import { getDualArmCableJobLog } from '@/lib/api/dualArmCableClient';
import {
  CollapsiblePanel,
  InfoRow,
  RunLogDrawer,
  SidePanelSection,
  simConsoleCardStyle,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import {
  CABLE_THREADING_TASK_DISPLAY_NAME,
  CABLE_THREADING_TASK_NAME,
} from '@/lib/workspace/cableThreading';
import {
  cableObjectModelLabel,
  cableReplayMetrics,
  cableReplayStatusLabel,
  resolveCableReplayJobId,
  type CableReplayRecord,
} from '@/lib/workspace/replayCableThreadingAdapter';
import {
  DUAL_ARM_CABLE_DEFAULTS,
  DUAL_ARM_CABLE_TASK_NAME,
} from '@/lib/workspace/dualArmCable';
import {
  dualArmReplayMetrics,
  dualArmReplayStatusLabel,
  type DualArmReplayRecord,
} from '@/lib/workspace/replayDualArmCableAdapter';
import { EvalReplaySidePanel } from '@/components/workspace/replay/EvalReplaySidePanel';
import { ReplayActionsPanel } from '@/components/workspace/replay/ReplayActionsPanel';
import { EvaluationWorkbenchBasicInfoRows } from '@/components/workspace/replay/EvaluationWorkbenchBasicInfoRows';
import { resolveEvaluationWorkbenchBasicInfo } from '@/lib/workspace/evaluationWorkbenchBasicInfo';
import { EvaluationReplayMetricsBlock } from '@/components/workspace/replay/EvaluationReplayMetricsBlock';
import {
  ReplayPanelSectionTitle,
  ReplaySidePanelLayout,
} from '@/components/workspace/replay/ReplaySidePanelLayout';
import { buildReplayMetricsInput } from '@/lib/workspace/replayMetricsInput';
import type { ReplayPageKind } from '@/lib/workspace/replayPageKind';

export type DualArmEvalReplayContext = {
  evalJobId: string;
  episode: number;
  aggregate: Record<string, unknown> | null;
  jobStatus: string;
  statusMessage: string;
  loading: boolean;
};

function InfoRowBlock({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '96px 1fr',
        gap: 12,
        fontSize: 13,
        lineHeight: 1.5,
        alignItems: 'start',
      }}
    >
      <span style={{ color: '#6b7280' }}>{label}</span>
      <span style={{ color: '#111827', fontWeight: 400, wordBreak: 'break-word' }}>{value}</span>
    </div>
  );
}
function PathMono({ value }: { value: string }) {
  return (
    <span
      style={{
        display: 'block',
        fontFamily: 'ui-monospace, monospace',
        fontSize: 11,
        lineHeight: 1.5,
        wordBreak: 'break-all',
        overflowWrap: 'anywhere',
      }}
    >
      {value}
    </span>
  );
}

function videoStatusLabel(hasVideo: boolean, status: string): string {
  if (hasVideo) return '已生成';
  if (status === 'running' || status === 'generating' || status === 'pending') return '等待生成';
  return '不可用';
}

function metricsFromWorkspaceJob(job: WorkspaceJobDetail | null | undefined): { label: string; value: string }[] {
  if (!job) return [];
  const raw = { ...(job.metricsSummary ?? {}), ...(job.metrics ?? {}) };
  const labels: Record<string, string> = {
    successRate: '成功率',
    finalSuccessRate: '成功率',
    successfulEpisodes: '成功次数',
    episodes: '轨迹数量',
    numEpisodes: 'Episode 数量',
    num_cables_succeeded: '成功线缆数',
    max_cables: '最大线缆数',
    episodeSuccess: 'episodeSuccess',
  };
  const rows: { label: string; value: string }[] = [];
  for (const [key, label] of Object.entries(labels)) {
    const value = raw[key];
    if (value == null || value === '') continue;
    let display = String(value);
    if ((key === 'successRate' || key === 'finalSuccessRate') && typeof value === 'number' && value <= 1) {
      display = `${Math.round(value * 1000) / 10}%`;
    }
    rows.push({ label, value: display });
  }
  return rows;
}

function logArtifactSummary(artifacts: WorkspaceArtifactItem[]): string {
  const logArtifacts = artifacts.filter((a) => a.artifactType === 'log');
  if (logArtifacts.length === 0) return '日志文件：未生成';
  return `日志文件：已生成（${logArtifacts.length} 个）`;
}

export function ReplayRunInfoPanel({
  emptyLabel,
  replayKind,
  cableRecord,
  dualArmRecord,
  mockSession,
  workspaceJob,
  artifacts = [],
  dualArmEval,
}: {
  emptyLabel: string;
  replayKind: ReplayPageKind;
  cableRecord?: CableReplayRecord | null;
  dualArmRecord?: DualArmReplayRecord | null;
  mockSession?: ReplaySession | null;
  workspaceJob?: WorkspaceJobDetail | null;
  artifacts?: WorkspaceArtifactItem[];
  dualArmEval?: DualArmEvalReplayContext | null;
}) {
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const [logTail, setLogTail] = useState('');
  const [logLoading, setLogLoading] = useState(false);
  const [logSummary, setLogSummary] = useState<string | null>(null);
  const [logAvailable, setLogAvailable] = useState(false);

  const isEval = replayKind === 'evaluation' || cableRecord?.recordType === 'policy_eval';

  const logJobId = useMemo(() => {
    if (dualArmRecord?.backendJobId) return dualArmRecord.backendJobId;
    if (cableRecord) return resolveCableReplayJobId(cableRecord);
    return workspaceJob?.jobId ?? null;
  }, [cableRecord, dualArmRecord, workspaceJob?.jobId]);

  useEffect(() => {
    if (!logJobId) {
      setLogSummary(null);
      setLogAvailable(false);
      return;
    }

    const logFromArtifacts = artifacts.some((a) => a.artifactType === 'log');
    setLogAvailable(logFromArtifacts);

    let cancelled = false;
    setLogLoading(true);

    const fetchLog =
      dualArmRecord || logJobId.startsWith('dac_gen_')
        ? getDualArmCableJobLog(logJobId)
        : getCableThreadingJobLog(logJobId);

    void fetchLog
      .then((response) => {
        if (cancelled) return;
        const tail = response.tail?.trim() ?? '';
        setLogTail(tail);
        setLogAvailable(Boolean(tail) || logFromArtifacts);
        if (tail) {
          const lines = tail.split('\n').filter((l) => l.trim());
          setLogSummary(lines.slice(-3).join('\n'));
        } else if (logFromArtifacts) {
          setLogSummary('日志文件已生成，可点击查看完整日志。');
        } else {
          setLogSummary(null);
        }
      })
      .catch(() => {
        if (cancelled) return;
        if (logFromArtifacts) {
          setLogSummary('日志文件已生成，可点击查看完整日志。');
          setLogAvailable(true);
        } else {
          setLogSummary(null);
          setLogAvailable(false);
        }
      })
      .finally(() => {
        if (!cancelled) setLogLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [logJobId, dualArmRecord, artifacts]);

  const openLogDrawer = () => {
    setLogDrawerOpen(true);
    if (!logJobId || logTail) return;
    setLogLoading(true);
    const fetchLog =
      dualArmRecord || logJobId.startsWith('dac_gen_')
        ? getDualArmCableJobLog(logJobId)
        : getCableThreadingJobLog(logJobId);
    void fetchLog
      .then((response) => setLogTail(response.tail?.trim() ?? ''))
      .finally(() => setLogLoading(false));
  };

  if (dualArmEval) {
    const { aggregate, jobStatus, loading, evalJobId } = dualArmEval;

    const workbenchBasicInfo = resolveEvaluationWorkbenchBasicInfo({
      evalJobId,
      status: {
        status: jobStatus,
        taskType: 'dual_arm_cable_manipulation',
        evaluationMode: aggregate?.evaluationMode,
      },
      aggregate: aggregate ?? undefined,
      fallbackTaskName: DUAL_ARM_CABLE_TASK_NAME,
    });
    if (loading) {
      workbenchBasicInfo.statusLabel = '加载中…';
    }

    return (
      <EvalReplaySidePanel
        evalJobId={evalJobId}
        taskType="dual_arm_cable_manipulation"
        jobStatus={jobStatus}
        loading={loading}
        aggregate={aggregate}
        basicInfo={<EvaluationWorkbenchBasicInfoRows info={workbenchBasicInfo} />}
      />
    );
  }

  if (!cableRecord && !dualArmRecord && !mockSession) {
    return (
      <aside className="replay-run-info-panel" style={simConsoleCardStyle}>
        <div style={{ fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 8 }}>运行信息</div>
        <p style={{ margin: 0, fontSize: 13, color: '#9ca3af', lineHeight: 1.5 }}>{emptyLabel}</p>
      </aside>
    );
  }

  if (mockSession) {
    const metrics = [
      { label: '力矩峰值', value: mockSession.sensors.torquePeak },
      { label: '视觉置信度', value: mockSession.sensors.visualConfidence },
      { label: '末端误差', value: mockSession.sensors.endEffectorError },
      { label: '碰撞次数', value: mockSession.sensors.collisionCount },
    ];
    const hasMetrics = metrics.some((m) => m.value && m.value !== '—');
    const hasLogs = mockSession.logs.length > 0;

    return (
      <aside className="replay-run-info-panel" style={simConsoleCardStyle}>
        <div style={{ fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 4 }}>运行信息</div>
        <div className="replay-run-info-scroll">
          <SidePanelSection title="基本信息">
            <InfoRow label="任务名称" value={mockSession.taskName} />
            <InfoRow label="运行编号" value={mockSession.runNumber} />
            <InfoRow label="场景" value={mockSession.scene} />
            <InfoRow label="机器人" value={mockSession.robot} />
            <InfoRow label="类型" value={isEval ? '策略评测' : '数据生成'} />
          </SidePanelSection>
          <SidePanelSection title="运行状态">
            <InfoRow label="状态" value={mockSession.status === 'completed' ? '已完成' : '失败'} />
            <InfoRow label="视频状态" value="已生成" />
            <InfoRow label="耗时" value={mockSession.duration} />
            {mockSession.successRate != null ? (
              <InfoRow label="成功率" value={`${mockSession.successRate}%`} />
            ) : null}
          </SidePanelSection>
          <CollapsiblePanel title="指标信息">
            {hasMetrics ? (
              metrics.map((m) => <InfoRow key={m.label} label={m.label} value={m.value} />)
            ) : (
              <p style={{ margin: 0, fontSize: 13, color: '#9ca3af' }}>暂无指标信息</p>
            )}
          </CollapsiblePanel>
          <CollapsiblePanel title="日志信息">
            {hasLogs ? (
              <>
                <InfoRow label="日志摘要" value={`${mockSession.logs.length} 条记录`} />
                <button
                  type="button"
                  onClick={openLogDrawer}
                  style={{
                    padding: '4px 0',
                    fontSize: 12,
                    color: '#2563eb',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  查看日志
                </button>
              </>
            ) : (
              <p style={{ margin: 0, fontSize: 13, color: '#9ca3af' }}>暂无日志信息</p>
            )}
          </CollapsiblePanel>
          <CollapsiblePanel title="内部调试信息">
            <InfoRow label="sessionId" value={mockSession.id} />
          </CollapsiblePanel>
        </div>
        <RunLogDrawer
          open={logDrawerOpen}
          logTail={hasLogs ? mockSession.logs.join('\n') : logTail}
          loading={false}
          onClose={() => setLogDrawerOpen(false)}
        />
      </aside>
    );
  }

  if (dualArmRecord) {
    const item = dualArmRecord.dataItem;
    const maxCables = item.dualArmMaxCables ?? 1;
    const metrics = [
      ...dualArmReplayMetrics(item),
      ...metricsFromWorkspaceJob(workspaceJob ?? null),
    ].filter((m) => m.value !== '—');
    const status = dualArmReplayStatusLabel(dualArmRecord.status);
    const jobId = dualArmRecord.backendJobId;

    return (
      <ReplaySidePanelLayout
        footerActions={
          <ReplayActionsPanel variant="footer" />
        }
      >
        <div style={{ marginBottom: 12 }}>
          <ReplayPanelSectionTitle>基础信息</ReplayPanelSectionTitle>
          <InfoRow label="任务名称" value={DUAL_ARM_CABLE_TASK_NAME} />
          <InfoRow label="仿真后端" value="MuJoCo" />
          <InfoRow label="机器人" value={item.robot ?? DUAL_ARM_CABLE_DEFAULTS.robot} />
          <InfoRow label="对象模型" value="杂乱柔性线缆" />
          <InfoRow label="类型" value="数据生成" />
          <InfoRow label="状态" value={status} />
          <InfoRow
            label="视频状态"
            value={videoStatusLabel(dualArmRecord.hasVideo, dualArmRecord.status)}
          />
        </div>

        <div style={{ marginBottom: 12 }}>
          <ReplayPanelSectionTitle>评测指标</ReplayPanelSectionTitle>
          <EvaluationReplayMetricsBlock
            {...buildReplayMetricsInput({ workspaceJob, dualArmRecord })}
          />
        </div>

        <CollapsiblePanel title="内部调试信息">
          <InfoRow label="jobId" value={jobId} />
          <InfoRow label="runtimePath" value={workspaceJob?.runtimePath ?? '—'} />
            {workspaceJob?.metadata ? (
              <InfoRowBlock
                label="metadata_json"
                value={<PathMono value={JSON.stringify(workspaceJob.metadata, null, 2)} />}
              />
            ) : null}
            {artifacts.map((a) => (
              <InfoRowBlock
                key={a.id}
                label={`${a.artifactType}:${a.name}`}
                value={<PathMono value={a.filePath} />}
              />
            ))}
          </CollapsiblePanel>
        <RunLogDrawer
          open={logDrawerOpen}
          logTail={logTail}
          loading={logLoading}
          onClose={() => setLogDrawerOpen(false)}
        />
      </ReplaySidePanelLayout>
    );
  }

  if (cableRecord) {
    const item = cableRecord.dataItem;
    const row = cableRecord.evalRow;
    const isDataGeneration = cableRecord.recordType === 'data_generation';
    const isPolicyEval = !isDataGeneration;

    if (isPolicyEval) {
      const evalStatusLabel = cableReplayStatusLabel(cableRecord.status);
      const evalTypeLabel =
        row?.modelType === '已训练模型' || row?.checkpoint?.startsWith('model_')
          ? '训练模型评测'
          : '专家策略评测';
      const evalJobId = resolveCableReplayJobId(cableRecord) ?? cableRecord.id;

      return (
        <EvalReplaySidePanel
          evalJobId={evalJobId}
          taskType="cable_threading"
          jobStatus={cableRecord.status}
          aggregate={
            (row?.aggregate as Record<string, unknown> | undefined) ??
            (workspaceJob?.metrics as Record<string, unknown> | undefined) ??
            null
          }
          metrics={workspaceJob?.metrics ?? null}
          basicInfo={
            <>
              <InfoRow label="任务名称" value={CABLE_THREADING_TASK_DISPLAY_NAME} />
              <InfoRow label="仿真后端" value="MuJoCo" />
              <InfoRow label="类型" value={evalTypeLabel} />
              <InfoRow label="状态" value={evalStatusLabel} />
            </>
          }
        />
      );
    }

    const robot = row?.robot ?? item?.robot ?? 'Panda';
    const objectModel = cableObjectModelLabel(row?.cableModel ?? item?.cableModel ?? 'composite_cable');
    const difficulty = row?.difficulty ?? item?.difficulty ?? 'easy';
    const rounds = row?.evalRounds ?? (item?.dataVolume ? item.dataVolume.replace(/[^\d]/g, '') : '—');
    const horizon = item?.horizon ?? 600;
    const metrics = [
      ...cableReplayMetrics(cableRecord),
      ...metricsFromWorkspaceJob(workspaceJob ?? null),
    ].filter((m) => m.value !== '—' && m.value !== '');
    const jobId = resolveCableReplayJobId(cableRecord) ?? workspaceJob?.jobId ?? cableRecord.id;

    return (
      <ReplaySidePanelLayout
        footerActions={
          <ReplayActionsPanel variant="footer" />
        }
      >
        <div style={{ marginBottom: 12 }}>
          <ReplayPanelSectionTitle>基础信息</ReplayPanelSectionTitle>
          <InfoRow label="任务名称" value={CABLE_THREADING_TASK_DISPLAY_NAME} />
          <InfoRow label="仿真后端" value="MuJoCo" />
          <InfoRow label="机器人" value={robot} />
          <InfoRow label="对象模型" value={objectModel} />
          <InfoRow label="难度" value={difficulty} />
          <InfoRow label="类型" value="数据生成" />
          <InfoRow label="状态" value={cableReplayStatusLabel(cableRecord.status)} />
          <InfoRow
            label="视频状态"
            value={videoStatusLabel(cableRecord.hasVideo, cableRecord.status)}
          />
          <InfoRow label="采集轮次" value={rounds !== '—' ? String(rounds) : '—'} />
          <InfoRow label="最大步数" value={String(horizon)} />
          {item?.successfulEpisodes != null ? (
            <InfoRow label="成功轨迹" value={String(item.successfulEpisodes)} />
          ) : null}
        </div>

        <div style={{ marginBottom: 12 }}>
          <ReplayPanelSectionTitle>评测指标</ReplayPanelSectionTitle>
          <EvaluationReplayMetricsBlock
            {...buildReplayMetricsInput({ workspaceJob, cableRecord })}
          />
        </div>

        <CollapsiblePanel title="内部调试信息">
          <InfoRow label="jobId" value={jobId} />
          <InfoRow label="jobType" value="generate" />
          <InfoRow label="taskType" value="cable_threading" />
          {workspaceJob?.runtimePath ? (
            <InfoRowBlock label="runtimePath" value={<PathMono value={workspaceJob.runtimePath} />} />
          ) : null}
          {artifacts.map((a) => (
            <InfoRowBlock
              key={`all-${a.id}`}
              label={`${a.artifactType}:${a.name}`}
              value={<PathMono value={a.filePath} />}
            />
          ))}
        </CollapsiblePanel>
        <RunLogDrawer
          open={logDrawerOpen}
          logTail={logTail}
          loading={logLoading}
          onClose={() => setLogDrawerOpen(false)}
        />
      </ReplaySidePanelLayout>
    );
  }

  return null;
}
