'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { WorkspaceCenteredModal } from '@/components/workspace/WorkspaceCenteredModal';
import { PrimaryButton, SecondaryButton, WS } from '@/components/workspace/workspaceUi';
import {
  getNutAssemblyJobLog,
  getNutAssemblyJobResult,
  getNutAssemblyJobStatus,
  type NutAssemblyJobStatusResponse,
} from '@/lib/api/nutAssemblyClient';
import {
  NUT_ASSEMBLY_TASK_DISPLAY_NAME,
  buildNutAssemblyReplayHref,
  formatNutAssemblyElapsedSeconds,
  mergeNutAssemblyJobWithResult,
  nutAssemblyStageStatusLabel,
} from '@/lib/workspace/nutAssembly';
import {
  buildNutAssemblyProgressViewModel,
  type NutAssemblyTimelineStage,
} from '@/lib/workspace/nutAssemblyGenerationProgress';
import {
  isTerminalSimJobStatus,
  usePageVisibleForPolling,
} from '@/lib/workspace/simulationPolling';

const STATUS_POLL_MS = 2000;
const LOG_POLL_MS = 5000;

const logBoxStyle: React.CSSProperties = {
  marginTop: 8,
  padding: 12,
  borderRadius: 8,
  background: '#111827',
  color: '#e5e7eb',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
  fontSize: 11,
  lineHeight: 1.5,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  maxHeight: 220,
  overflow: 'auto',
};

function ProgressBar({ percent, tone }: { percent: number; tone: 'running' | 'completed' | 'failed' }) {
  const fillColor = tone === 'completed' ? '#10b981' : tone === 'failed' ? '#ef4444' : '#2563eb';
  return (
    <div style={{ height: 8, borderRadius: 999, background: '#e5e7eb', overflow: 'hidden' }}>
      <div
        style={{
          width: `${Math.max(0, Math.min(100, percent))}%`,
          height: '100%',
          borderRadius: 999,
          background: fillColor,
          transition: 'width 0.4s ease',
        }}
      />
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        ...WS.card,
        padding: '14px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 12, color: '#6b7280' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600, color: '#111827', lineHeight: 1.2 }}>{value}</div>
    </div>
  );
}

function TimelineIcon({ state }: { state: NutAssemblyTimelineStage['state'] }) {
  if (state === 'done') {
    return (
      <span
        style={{
          width: 22,
          height: 22,
          borderRadius: '50%',
          background: '#d1fae5',
          color: '#047857',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          flexShrink: 0,
        }}
      >
        ✓
      </span>
    );
  }
  if (state === 'active') {
    return (
      <span
        style={{
          width: 22,
          height: 22,
          borderRadius: '50%',
          border: '2px solid #2563eb',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: '#2563eb',
            animation: 'nut-assembly-pulse 1.2s ease-in-out infinite',
          }}
        />
      </span>
    );
  }
  if (state === 'failed') {
    return (
      <span
        style={{
          width: 22,
          height: 22,
          borderRadius: '50%',
          background: '#fee2e2',
          color: '#b91c1c',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          flexShrink: 0,
        }}
      >
        ✕
      </span>
    );
  }
  return (
    <span
      style={{
        width: 22,
        height: 22,
        borderRadius: '50%',
        background: '#f3f4f6',
        border: '1px solid #e5e7eb',
        flexShrink: 0,
      }}
    />
  );
}

function CollapsibleBlock({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ ...WS.card, overflow: 'hidden' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          border: 'none',
          background: 'transparent',
          cursor: 'pointer',
          fontSize: 13,
          fontWeight: 600,
          color: '#374151',
        }}
      >
        {title}
        <span style={{ fontSize: 12, color: '#6b7280' }}>{open ? '收起' : '展开'}</span>
      </button>
      {open ? <div style={{ padding: '0 16px 16px', borderTop: '1px solid #f3f4f6' }}>{children}</div> : null}
    </div>
  );
}

function formatLastUpdated(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('zh-CN', { hour12: false });
  } catch {
    return iso;
  }
}

export function NutAssemblyGenerationProgressModal({
  open,
  jobId,
  dataId,
  onClose,
  onCompleted,
  onRetryConfig,
}: {
  open: boolean;
  jobId: string;
  dataId?: string;
  onClose: () => void;
  onCompleted?: () => void;
  onRetryConfig?: () => void;
}) {
  const router = useRouter();
  const pageVisible = usePageVisibleForPolling();
  const [job, setJob] = useState<NutAssemblyJobStatusResponse | null>(null);
  const [logTail, setLogTail] = useState('');
  const [logExpanded, setLogExpanded] = useState(false);
  const [artifactDetailOpen, setArtifactDetailOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [completedNotified, setCompletedNotified] = useState(false);
  const [pollingStopped, setPollingStopped] = useState(false);
  const logTailRef = useRef('');

  useEffect(() => {
    logTailRef.current = logTail;
  }, [logTail]);

  const vm = buildNutAssemblyProgressViewModel(job, jobId);
  const live = (job?.live ?? {}) as Record<string, unknown>;
  const stageLabel = nutAssemblyStageStatusLabel(job?.stage ?? String(live.stage ?? ''));
  const errorMessage = String(live.error ?? live.mimicgenFallbackError ?? '');
  const traceback =
    (typeof job?.traceback === 'string' && job.traceback) ||
    (typeof live.traceback === 'string' && live.traceback) ||
    (vm.status === 'failed' && logTail.includes('Traceback') ? logTail : '');

  const refreshLog = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await getNutAssemblyJobLog(jobId, 20);
      setLogTail(res.tail?.trim() ?? '');
    } catch {
      /* 日志暂不可读时不中断 status 轮询 */
    }
  }, [jobId]);

  const refreshStatus = useCallback(async () => {
    if (!jobId) return;
    try {
      const nextStatus = await getNutAssemblyJobStatus(jobId);
      let merged = nextStatus;

      if (isTerminalSimJobStatus(nextStatus.status)) {
        const result = await getNutAssemblyJobResult(jobId).catch(() => null);
        merged = mergeNutAssemblyJobWithResult(nextStatus, result, logTailRef.current);
        if (nextStatus.status === 'failed' && !logTailRef.current.trim()) {
          await refreshLog();
          merged = mergeNutAssemblyJobWithResult(nextStatus, result, logTailRef.current);
        }
      }

      setJob(merged);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '无法读取任务状态');
    }
  }, [jobId, refreshLog]);

  useEffect(() => {
    if (!open || !jobId) return;
    setJob(null);
    setLogTail('');
    setCompletedNotified(false);
    setPollingStopped(false);
    setLogExpanded(false);
    setArtifactDetailOpen(false);
    setLoadError(null);
    void refreshStatus();
    void refreshLog();
  }, [open, jobId, refreshStatus, refreshLog]);

  useEffect(() => {
    if (vm.status === 'failed') setLogExpanded(true);
  }, [vm.status]);

  const shouldPollStatus = open && !pollingStopped && pageVisible && !vm.terminal;

  useEffect(() => {
    if (!shouldPollStatus) return;
    const timer = setInterval(() => void refreshStatus(), STATUS_POLL_MS);
    return () => clearInterval(timer);
  }, [shouldPollStatus, refreshStatus]);

  const shouldPollLog = open && !pollingStopped && pageVisible;

  useEffect(() => {
    if (!shouldPollLog) return;
    const timer = setInterval(() => void refreshLog(), LOG_POLL_MS);
    return () => clearInterval(timer);
  }, [shouldPollLog, refreshLog]);

  useEffect(() => {
    if (!open || !vm.terminal || completedNotified) return;
    if (vm.status === 'completed' || vm.status === 'partial_success') {
      void refreshLog();
      onCompleted?.();
    }
    setCompletedNotified(true);
  }, [open, vm.terminal, vm.status, completedNotified, onCompleted, refreshLog]);

  const handleClose = () => {
    if (!vm.terminal && !window.confirm('任务仍在后台运行，确定关闭进度窗口吗？')) {
      return;
    }
    setPollingStopped(true);
    onClose();
  };

  const handleBackgroundRun = () => {
    setPollingStopped(true);
    onClose();
  };

  const handleGoToDataCenter = () => {
    onCompleted?.();
    setPollingStopped(true);
    onClose();
  };

  const progressTone =
    vm.status === 'completed' || vm.status === 'partial_success'
      ? 'completed'
      : vm.status === 'failed'
        ? 'failed'
        : 'running';

  const logHint = logTail
    ? (logTail.split('\n').filter(Boolean).slice(-1)[0] ?? stageLabel)
    : stageLabel;

  const logDisplay = logTail
    ? logTail
    : vm.terminal
      ? '（暂无日志输出）'
      : '暂无详细日志，任务仍在执行。系统将持续检测运行状态。';

  const generationModeRaw =
    job?.generationMode ?? (typeof live.generationMode === 'string' ? live.generationMode : null);
  const policyModeRaw = job?.policyMode ?? (typeof live.policyMode === 'string' ? live.policyMode : null);
  const sourceDemoOriginRaw =
    job?.sourceDemoOrigin ?? (typeof live.sourceDemoOrigin === 'string' ? live.sourceDemoOrigin : null);

  return (
    <WorkspaceCenteredModal
      open={open}
      title="数据生成进度"
      titleId="nut-assembly-generation-progress-title"
      width={860}
      zIndex={1600}
      onClose={handleClose}
      footer={
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, flexWrap: 'wrap' }}>
          {vm.status === 'completed' || vm.status === 'partial_success' ? (
            <>
              {vm.canViewReplay ? (
                <SecondaryButton
                  onClick={() => {
                    router.push(buildNutAssemblyReplayHref({ jobId, datasetId: dataId }));
                  }}
                >
                  查看回放
                </SecondaryButton>
              ) : null}
              <SecondaryButton onClick={handleGoToDataCenter}>前往数据中心</SecondaryButton>
              <PrimaryButton onClick={() => setLogExpanded((v) => !v)}>
                {logExpanded ? '收起日志' : '查看日志'}
              </PrimaryButton>
            </>
          ) : vm.status === 'failed' ? (
            <>
              <SecondaryButton onClick={() => setLogExpanded((v) => !v)}>
                {logExpanded ? '收起日志' : '查看日志'}
              </SecondaryButton>
              <PrimaryButton
                onClick={() => {
                  setPollingStopped(true);
                  onClose();
                  onRetryConfig?.();
                }}
              >
                返回修改配置
              </PrimaryButton>
            </>
          ) : (
            <>
              <SecondaryButton onClick={() => setLogExpanded((v) => !v)}>
                {logExpanded ? '收起日志' : '查看日志'}
              </SecondaryButton>
              <SecondaryButton onClick={handleBackgroundRun}>后台运行</SecondaryButton>
            </>
          )}
        </div>
      }
    >
      <style>{`
        @keyframes nut-assembly-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.85); }
        }
      `}</style>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div
          style={{
            padding: '18px 20px',
            borderRadius: 12,
            background: 'linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%)',
            border: '1px solid #e2e8f0',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
                生成任务数据
              </div>
              <div style={{ fontSize: 13, color: '#64748b', marginBottom: 12 }}>{vm.subtitle}</div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: '6px 24px',
                  fontSize: 13,
                }}
              >
                <div>
                  <span style={{ color: '#94a3b8' }}>任务名称 </span>
                  <span style={{ color: '#334155' }}>{NUT_ASSEMBLY_TASK_DISPLAY_NAME}</span>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>生成方式 </span>
                  <span style={{ color: '#334155' }}>{vm.generationModeLabel}</span>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>示教数据 </span>
                  <span style={{ color: '#334155' }}>{vm.sourceDemoLabel}</span>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>运行时间 </span>
                  <span style={{ color: '#334155' }}>
                    {formatNutAssemblyElapsedSeconds(job?.elapsedSeconds ?? null)}
                  </span>
                </div>
              </div>
            </div>
            <span
              style={{
                display: 'inline-block',
                padding: '4px 12px',
                borderRadius: 999,
                fontSize: 12,
                fontWeight: 600,
                backgroundColor: vm.statusBadge.bg,
                color: vm.statusBadge.color,
                whiteSpace: 'nowrap',
                flexShrink: 0,
              }}
            >
              {vm.statusBadge.label}
            </span>
          </div>
        </div>

        {loadError ? <div style={{ fontSize: 13, color: '#b91c1c' }}>{loadError}</div> : null}

        {vm.status === 'failed' && errorMessage ? (
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 8,
              background: '#fef2f2',
              border: '1px solid #fecaca',
              fontSize: 13,
              color: '#b91c1c',
            }}
          >
            {errorMessage}
          </div>
        ) : null}

        {vm.pinnStatusMessage ? (
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 8,
              background: '#eff6ff',
              border: '1px solid #bfdbfe',
              fontSize: 13,
              color: '#1e40af',
              lineHeight: 1.55,
            }}
          >
            {vm.pinnStatusMessage}
          </div>
        ) : null}

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, gap: 12 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#111827' }}>
              {vm.status === 'completed'
                ? '生成完成'
                : vm.status === 'partial_success'
                  ? '部分完成'
                  : stageLabel}
            </span>
            <span style={{ fontSize: 12, color: '#6b7280' }}>{vm.progressPercent}%</span>
          </div>
          <ProgressBar percent={vm.progressPercent} tone={progressTone} />
          <div style={{ marginTop: 8, fontSize: 12, color: '#6b7280' }}>{vm.progressCaption}</div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#9ca3af' }}>
            最近更新：{formatLastUpdated(job?.lastHeartbeatAt)}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 12 }}>
          <MetricCard label="请求生成" value={vm.metricCards.requested} />
          <MetricCard label="MimicGen 已写入" value={vm.metricCards.mimicgenWritten} />
          <MetricCard label="PINN 修复通过" value={vm.metricCards.pinnRepaired} />
          <MetricCard label="最终数据量" value={vm.metricCards.finalCount} />
        </div>
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 16,
            fontSize: 12,
            color: '#6b7280',
            padding: '0 4px',
          }}
        >
          <span>生成写入率：{vm.summaryRow.datagenWriteRate}</span>
          <span>PINN 增强量：{vm.summaryRow.pinnEnhancementGain}</span>
          <span>任务评测成功率：{vm.summaryRow.taskEvalLabel}</span>
        </div>

        <div style={{ ...WS.card, padding: '16px 18px' }}>
          <div style={{ ...WS.sectionTitle, marginBottom: 14 }}>阶段进度</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {vm.timeline.map((item, index) => (
              <div key={item.id} style={{ display: 'flex', gap: 12, minHeight: 36 }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 22 }}>
                  <TimelineIcon state={item.state} />
                  {index < vm.timeline.length - 1 ? (
                    <div
                      style={{
                        width: 2,
                        flex: 1,
                        minHeight: 12,
                        background: item.state === 'done' ? '#bbf7d0' : '#e5e7eb',
                        marginTop: 4,
                      }}
                    />
                  ) : null}
                </div>
                <div style={{ paddingBottom: index < vm.timeline.length - 1 ? 12 : 0, flex: 1 }}>
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: item.state === 'active' ? 600 : 500,
                      color:
                        item.state === 'failed'
                          ? '#b91c1c'
                          : item.state === 'active'
                            ? '#1d4ed8'
                            : item.state === 'done'
                              ? '#047857'
                              : '#9ca3af',
                    }}
                  >
                    {item.label}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ ...WS.card, padding: '16px 18px' }}>
          <div style={{ ...WS.sectionTitle, marginBottom: 12 }}>生成产物</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 13 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <span style={{ color: '#6b7280' }}>HDF5 数据集</span>
              <span style={{ color: '#111827', fontWeight: 500 }}>{vm.artifacts.hdf5}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <span style={{ color: '#6b7280' }}>回放视频</span>
              <span style={{ color: '#111827', fontWeight: 500 }}>{vm.artifacts.video}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <span style={{ color: '#6b7280' }}>数据中心登记</span>
              <span style={{ color: '#111827', fontWeight: 500 }}>{vm.artifacts.registry}</span>
            </div>
            {vm.terminal ? (
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                <span style={{ color: '#6b7280' }}>任务评测成功率</span>
                <span style={{ color: '#111827', fontWeight: 500 }}>{vm.taskEvalLabel}</span>
              </div>
            ) : null}
          </div>
          {!vm.terminal ? (
            <div style={{ marginTop: 10, fontSize: 12, color: '#9ca3af' }}>
              MimicGen 为离线数据生成，无实时仿真画面。
            </div>
          ) : null}
          <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={() => setArtifactDetailOpen((v) => !v)}
              style={{
                border: 'none',
                background: 'transparent',
                color: '#2563eb',
                fontSize: 12,
                cursor: 'pointer',
                padding: 0,
              }}
            >
              {artifactDetailOpen ? '收起详情' : '查看详情'}
            </button>
          </div>
          {artifactDetailOpen ? (
            <div
              style={{
                marginTop: 10,
                padding: 12,
                borderRadius: 8,
                background: '#f9fafb',
                fontSize: 12,
                lineHeight: 1.6,
                wordBreak: 'break-all',
              }}
            >
              <div>
                <span style={{ color: '#6b7280' }}>数据集 </span>
                {vm.advancedPaths.hdf5Path}
              </div>
              <div>
                <span style={{ color: '#6b7280' }}>回放视频 </span>
                {vm.advancedPaths.videoPath}
              </div>
              <div>
                <span style={{ color: '#6b7280' }}>manifest </span>
                {vm.advancedPaths.manifestPath}
              </div>
              <div>
                <span style={{ color: '#6b7280' }}>summary </span>
                {vm.advancedPaths.summaryPath}
              </div>
            </div>
          ) : null}
        </div>

        <div style={{ ...WS.card, padding: '14px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>运行日志</div>
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                最近状态：{logHint.length > 80 ? `${logHint.slice(0, 80)}…` : logHint}
              </div>
            </div>
            <SecondaryButton onClick={() => setLogExpanded((v) => !v)}>
              {logExpanded ? '收起日志' : '查看日志'}
            </SecondaryButton>
          </div>
          {logExpanded ? (
            <>
              <div style={logBoxStyle}>{logDisplay}</div>
              {traceback ? (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 4 }}>
                    错误 traceback
                  </div>
                  <div style={logBoxStyle}>{traceback}</div>
                </div>
              ) : null}
            </>
          ) : null}
        </div>

        <CollapsibleBlock title="高级信息">
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '140px 1fr',
              gap: '8px 12px',
              fontSize: 12,
              lineHeight: 1.5,
              paddingTop: 12,
            }}
          >
            <div style={{ color: '#6b7280' }}>Job ID</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{jobId}</div>
            <div style={{ color: '#6b7280' }}>generationMode</div>
            <div style={{ color: '#111827' }}>{generationModeRaw ?? '—'}</div>
            <div style={{ color: '#6b7280' }}>policyMode</div>
            <div style={{ color: '#111827' }}>{policyModeRaw ?? '—'}</div>
            <div style={{ color: '#6b7280' }}>示教数据来源</div>
            <div style={{ color: '#111827' }}>{sourceDemoOriginRaw ?? '—'}</div>
            <div style={{ color: '#6b7280' }}>sourceDemoPath</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{vm.advancedPaths.sourceDemoPath}</div>
            <div style={{ color: '#6b7280' }}>sourceDemoHash</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{vm.advancedPaths.sourceDemoHash}</div>
            <div style={{ color: '#6b7280' }}>envName</div>
            <div style={{ color: '#111827' }}>{vm.advancedPaths.envName}</div>
            <div style={{ color: '#6b7280' }}>objectPoseKeys</div>
            <div style={{ color: '#111827' }}>{vm.advancedPaths.objectPoseKeys}</div>
            <div style={{ color: '#6b7280' }}>HDF5 完整路径</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{vm.advancedPaths.hdf5Path}</div>
            <div style={{ color: '#6b7280' }}>视频完整路径</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{vm.advancedPaths.videoPath}</div>
            <div style={{ color: '#6b7280' }}>manifestPath</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{vm.advancedPaths.manifestPath}</div>
            <div style={{ color: '#6b7280' }}>summaryPath</div>
            <div style={{ color: '#111827', wordBreak: 'break-all' }}>{vm.advancedPaths.summaryPath}</div>
          </div>
        </CollapsibleBlock>
      </div>
    </WorkspaceCenteredModal>
  );
}
