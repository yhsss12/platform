'use client';

import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useI18n } from '@/components/common/I18nProvider';
import { EvaluationReplayStatusHeader } from '@/components/workspace/replay/EvaluationReplayStatusHeader';
import { ReplayRunInfoPanel } from '@/components/workspace/replay/ReplayRunInfoPanel';
import type { WorkspaceArtifactItem, WorkspaceJobDetail } from '@/lib/api/workspaceJobClient';
import type { ReplayPageKind } from '@/lib/workspace/replayPageKind';
import {
  replaySessions,
  resolveReplaySessionIdByEvalId,
  type ReplaySession,
} from '@/lib/mock/workspacePagesMock';
import { isCableThreadingReplayMode } from '@/lib/workspace/cableThreading';
import { getWorkspaceJob, getWorkspaceJobArtifacts } from '@/lib/api/workspaceJobClient';
import {
  buildCableReplayRecords,
  cableReplayStatusLabel,
  resolveCableReplayRecordId,
  type CableReplayRecord,
} from '@/lib/workspace/replayCableThreadingAdapter';
import { CableThreadingVideoPlayer } from '@/components/workspace/replay/CableThreadingVideoPlayer';
import { DualArmCableVideoPlayer } from '@/components/workspace/replay/DualArmCableVideoPlayer';
import { CableThreadingLiveFrame } from '@/components/workspace/simulation/CableThreadingLiveFrame';
import { DualArmCableLiveFrame } from '@/components/workspace/simulation/DualArmCableLiveFrame';
import { getDualArmCableJobStatus } from '@/lib/api/dualArmCableClient';
import {
  cableReplayRecordFromWorkspaceJob,
  dualArmReplayRecordFromWorkspaceJob,
  isPersistedWorkspaceJobId,
  resolveWorkspaceReplayJobId,
} from '@/lib/workspace/workspaceJobReplay';
import { isDualArmCableReplayMode } from '@/lib/workspace/dualArmCable';
import { isIsaacBlockStackingReplayMode } from '@/lib/workspace/isaacBlockStacking';
import { UnifiedReplayWorkbench } from '@/components/workspace/replay/UnifiedReplayWorkbench';
import { isUnifiedReplayWorkbenchMode } from '@/lib/workspace/datasetReplayHref';
import type { ReplaySourceKind } from '@/lib/workspace/replayViewModel';
import { mapEvaluationJobStatusLabel } from '@/lib/workspace/evaluationWorkbenchCopy';
import { CABLE_THREADING_DISPLAY_NAME, getTaskDisplayName } from '@/lib/workspace/taskDisplayNames';
import {
  buildDualArmReplayRecords,
  dualArmReplayRecordFromStatus,
  resolveDualArmReplayRecordId,
  type DualArmReplayRecord,
} from '@/lib/workspace/replayDualArmCableAdapter';

export const REPLAY_PAGE_STYLES = `
  .replay-page-stack { display: flex; flex-direction: column; gap: 16px; }
  .replay-workspace-card {
    background: #fff; border-radius: 16px; border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04); padding: 20px;
  }
  .replay-main-area { min-width: 0; }
  .replay-header {
    display: flex; justify-content: space-between; align-items: center;
    gap: 12px; margin-bottom: 14px; flex-wrap: wrap;
  }
  .replay-header-title { font-size: 14px; font-weight: 600; color: #111827; }
  .replay-header-meta { font-size: 12px; color: #6b7280; }
  .replay-content-row { display: flex; gap: 20px; align-items: flex-start; }
  .replay-player-column {
    flex: 1; min-width: 0; display: flex; flex-direction: column;
  }
  .replay-player-shell { width: 100%; margin: 0 auto; }
  .replay-player {
    position: relative;
    aspect-ratio: 16 / 9; width: 100%; background: #0f172a; border-radius: 14px; overflow: hidden;
  }
  .replay-player-media {
    width: 100%; height: 100%;
  }
  .replay-player-media > * { width: 100%; height: 100%; object-fit: contain; }
  .replay-control-bar {
    width: 100%; margin: 10px auto 0; height: 44px; padding: 0 12px;
    display: flex; align-items: center; gap: 10px; background: #f9fafb;
    border: 1px solid #e5e7eb; border-radius: 10px;
  }
  .replay-control-play {
    width: 28px; height: 28px; border-radius: 999px; border: 1px solid #d1d5db;
    background: #fff; color: #374151; font-size: 10px; cursor: default;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .replay-control-time {
    font-size: 11px; color: #6b7280; font-family: ui-monospace, monospace; flex-shrink: 0;
  }
  .replay-control-sep { font-size: 11px; color: #d1d5db; flex-shrink: 0; }
  .replay-control-progress {
    flex: 1; height: 4px; border-radius: 999px; background: #e5e7eb; overflow: hidden; min-width: 60px;
  }
  .replay-control-progress-fill { height: 100%; background: #2563eb; border-radius: 999px; }
  .replay-control-speed { font-size: 11px; color: #6b7280; flex-shrink: 0; }
  .replay-run-info-panel {
    width: 300px; flex-shrink: 0; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 14px; background: #fff; align-self: stretch;
    max-height: calc(min(100vw - 360px, 1200px) * 9 / 16 + 54px);
    display: flex; flex-direction: column; min-height: 0;
  }
  .replay-run-info-scroll { overflow-y: auto; flex: 1; min-height: 0; }
  .replay-side-panel-layout { padding: 14px; }
  .replay-side-panel-scroll { display: flex; flex-direction: column; gap: 0; }
  .replay-side-panel-footer {
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid #f3f4f6;
    flex-shrink: 0;
  }
  .replay-drawer-metrics { display: grid; grid-template-columns: 1fr; gap: 8px; }
  .replay-sensor-card {
    padding: 10px 12px; border-radius: 10px; border: 1px solid #e5e7eb; background: #f9fafb;
  }
  .replay-log-box {
    font-family: ui-monospace, monospace; font-size: 12px; background: #f9fafb;
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px;
    max-height: 240px; overflow: auto; line-height: 1.7;
  }
  @media (max-width: 1100px) {
    .replay-content-row { flex-direction: column; }
    .replay-run-info-panel { width: 100%; max-height: none; }
  }
`;

function formatDurationLabel(duration: string): string {
  const match = duration.match(/^(\d+)m(\d+)s$/);
  if (match) return `${match[1]}:${match[2].padStart(2, '0')}`;
  return duration;
}

function MockPlayerPlaceholder({ emptyHint }: { emptyHint?: string }) {
  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        padding: 20,
        backgroundImage: `
          linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px)
        `,
        backgroundSize: '28px 28px',
      }}
    >
      <div style={{ fontSize: 14, color: '#cbd5e1', fontWeight: 500 }}>轨迹与视频回放</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 6, maxWidth: 360, lineHeight: 1.55 }}>
        {emptyHint ?? '等待加载回放画面…'}
      </div>
    </div>
  );
}

function CableMediaPlayer({ record }: { record: CableReplayRecord | null }) {
  if (!record) {
    return <MockPlayerPlaceholder />;
  }

  if (record.hasVideo && record.videoJobId) {
    return (
      <CableThreadingVideoPlayer videoJobId={record.videoJobId} />
    );
  }

  if (record.recordType === 'data_generation') {
    if (record.frameJobId) {
      return (
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            padding: 24,
          }}
        >
          <CableThreadingLiveFrame
            jobId={record.frameJobId}
            status={record.status === 'failed' ? 'failed' : 'completed'}
            frameCount={1}
          />
          <p
            style={{
              margin: 0,
              textAlign: 'center',
              color: '#94a3b8',
              fontSize: 13,
              lineHeight: 1.6,
            }}
          >
            当前仅有最终采集帧
          </p>
        </div>
      );
    }
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 24,
          textAlign: 'center',
          color: '#94a3b8',
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        暂无回放画面
      </div>
    );
  }

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        textAlign: 'center',
        color: '#94a3b8',
        fontSize: 13,
        lineHeight: 1.6,
        gap: 8,
      }}
    >
      <div style={{ fontSize: 14, color: '#cbd5e1', fontWeight: 500 }}>
        {record.status === 'failed' ? '评测画面未生成' : '评测画面'}
      </div>
      <p style={{ margin: 0, maxWidth: 400 }}>
        {record.status === 'failed'
          ? '当前评测任务执行失败，未生成可回放视频。请查看右侧失败诊断或评测日志。'
          : record.status === 'completed'
            ? '评测已完成，但未生成回放视频。'
            : '评测过程视频尚未生成。请确认评测任务已完成且后端已输出 eval.mp4。'}
      </p>
    </div>
  );
}

function DualArmMediaPlayer({ record }: { record: DualArmReplayRecord | null }) {
  if (!record) {
    return <MockPlayerPlaceholder />;
  }

  if (record.hasVideo && record.backendJobId) {
    return <DualArmCableVideoPlayer videoJobId={record.backendJobId} />;
  }

  if (record.backendJobId) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
          padding: 24,
        }}
      >
        <DualArmCableLiveFrame jobId={record.backendJobId} status="completed" />
        <p style={{ margin: 0, textAlign: 'center', color: '#94a3b8', fontSize: 13, lineHeight: 1.6 }}>
          暂无过程视频，显示最终帧
        </p>
      </div>
    );
  }

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        textAlign: 'center',
        color: '#94a3b8',
        fontSize: 13,
        lineHeight: 1.6,
      }}
    >
      暂无回放画面
    </div>
  );
}

function PlaybackControlBar({
  session,
  cableMode,
}: {
  session: ReplaySession | null;
  cableMode: boolean;
}) {
  if (cableMode) {
    return null;
  }

  const progress = session?.status === 'failed' ? 72 : session ? 48 : 0;
  const currentTime = session ? (session.status === 'failed' ? '00:06:10' : '00:02:14') : '00:00:00';
  const totalTime = session ? formatDurationLabel(session.duration) : '00:00:00';

  return (
    <div className="replay-control-bar">
      <button type="button" aria-label="播放" className="replay-control-play">
        ▶
      </button>
      <span className="replay-control-time">{currentTime}</span>
      <span className="replay-control-sep">/</span>
      <span className="replay-control-time">{totalTime}</span>
      <div className="replay-control-progress">
        <div className="replay-control-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <span className="replay-control-speed">1.0x</span>
    </div>
  );
}

function MainReplayArea({
  session,
  cableRecord,
  dualArmRecord,
  cableMode,
  dualArmMode,
  headerTitle,
  headerMeta,
  emptyLabel,
  replayKind,
  workspaceJob,
  workspaceArtifacts,
}: {
  session: ReplaySession | null;
  cableRecord: CableReplayRecord | null;
  dualArmRecord: DualArmReplayRecord | null;
  cableMode: boolean;
  dualArmMode: boolean;
  headerTitle: ReactNode;
  headerMeta?: string;
  emptyLabel: string;
  replayKind: ReplayPageKind;
  workspaceJob?: WorkspaceJobDetail | null;
  workspaceArtifacts?: WorkspaceArtifactItem[];
}) {
  const realTaskMode = cableMode || dualArmMode;

  return (
    <div className="replay-main-area">
      <div className="replay-header">
        {typeof headerTitle === 'string' ? (
          <span className="replay-header-title">{headerTitle}</span>
        ) : (
          headerTitle
        )}
        {headerMeta ? <span className="replay-header-meta">{headerMeta}</span> : null}
      </div>

      <div className="replay-content-row">
        <div className="replay-player-column">
          <div className="replay-player-shell">
            <div className="replay-player">
              <div className="replay-player-media">
                {dualArmMode ? (
                  <DualArmMediaPlayer record={dualArmRecord} />
                ) : cableMode ? (
                  <CableMediaPlayer record={cableRecord} />
                ) : (
                  <MockPlayerPlaceholder />
                )}
              </div>
            </div>
          </div>
          <PlaybackControlBar session={session} cableMode={realTaskMode} />
        </div>
        <ReplayRunInfoPanel
          emptyLabel={emptyLabel}
          replayKind={replayKind}
          cableRecord={cableRecord}
          dualArmRecord={dualArmRecord}
          mockSession={session}
          workspaceJob={workspaceJob}
          artifacts={workspaceArtifacts}
        />
      </div>
    </div>
  );
}

function ReplayEmptyMain({ message, videoTitle }: { message: string; videoTitle: ReactNode }) {
  return (
    <div className="replay-main-area">
      <div className="replay-header">
        {typeof videoTitle === 'string' ? (
          <span className="replay-header-title">{videoTitle}</span>
        ) : (
          videoTitle
        )}
      </div>
      <div
        style={{
          padding: '48px 24px',
          textAlign: 'center',
          color: '#6b7280',
          fontSize: 14,
          lineHeight: 1.6,
          border: '1px dashed #e5e7eb',
          borderRadius: 12,
        }}
      >
        {message}
      </div>
    </div>
  );
}

export function ReplayWorkbench({
  initialTaskType,
  initialEvalId,
  initialEvalJobId,
  initialJobId,
  initialDatasetId,
  initialReplayJobId,
  initialReplayType,
  replayKind,
  replaySourceKind,
  hasUrlTarget,
  unifiedReplay: unifiedReplayProp,
  returnPath,
  returnLabel,
  onContentResolved,
}: {
  initialTaskType?: string;
  initialEvalId?: string;
  initialEvalJobId?: string;
  initialJobId?: string;
  initialDatasetId?: string;
  initialReplayJobId?: string;
  initialReplayType?: string;
  replayKind: ReplayPageKind;
  replaySourceKind: ReplaySourceKind;
  hasUrlTarget: boolean;
  unifiedReplay?: boolean;
  returnPath?: string;
  returnLabel?: string;
  onContentResolved?: (adapter: import('@/lib/workspace/replayAdapters').ReplayAdapterResult) => void;
}) {
  const { t } = useI18n();
  const [cableRecords, setCableRecords] = useState<CableReplayRecord[]>(() => buildCableReplayRecords());
  const [dualArmRecords, setDualArmRecords] = useState<DualArmReplayRecord[]>(() =>
    buildDualArmReplayRecords()
  );
  const [dualArmFetchError, setDualArmFetchError] = useState<string | null>(null);
  const [dualArmFetching, setDualArmFetching] = useState(false);
  const [workspaceJobDeleted, setWorkspaceJobDeleted] = useState(false);
  const [workspaceJobLoading, setWorkspaceJobLoading] = useState(false);
  const [workspaceJobDetail, setWorkspaceJobDetail] = useState<WorkspaceJobDetail | null>(null);
  const [workspaceArtifacts, setWorkspaceArtifacts] = useState<WorkspaceArtifactItem[]>([]);
  const dualArmFetchAttemptedRef = useRef<string | null>(null);
  const workspaceReplayAttemptedRef = useRef<string | null>(null);

  const workspaceReplayJobId = resolveWorkspaceReplayJobId({
    jobId: initialJobId,
    evalId: initialEvalId ?? initialEvalJobId,
  });

  const unifiedModeParams = {
    replayType: initialReplayType,
    taskType: initialTaskType,
    jobId: initialJobId,
    datasetId: initialDatasetId,
    replayJobId: initialReplayJobId,
    evalId: initialEvalId,
    evalJobId: initialEvalJobId,
    hasUrlTarget,
  };

  const dualArmMode =
    !isUnifiedReplayWorkbenchMode(unifiedModeParams) &&
    (isDualArmCableReplayMode(initialTaskType) || Boolean(initialJobId?.startsWith('dac_gen_')));

  const isaacMode =
    !dualArmMode &&
    (isIsaacBlockStackingReplayMode(initialTaskType) ||
      Boolean(initialJobId?.startsWith('isaac_gen_')) ||
      Boolean(initialReplayJobId?.startsWith('isaac_replay_')) ||
      Boolean(initialDatasetId?.startsWith('isaac_ds_')));

  const unifiedMode = unifiedReplayProp ?? isUnifiedReplayWorkbenchMode(unifiedModeParams);

  const resolvedCableTargetId = useMemo(
    () =>
      resolveCableReplayRecordId(cableRecords, {
        evalId: initialEvalId,
        jobId: initialJobId,
      }),
    [cableRecords, initialEvalId, initialJobId]
  );

  const resolvedDualArmTargetId = useMemo(
    () => resolveDualArmReplayRecordId(dualArmRecords, initialJobId),
    [dualArmRecords, initialJobId]
  );

  const cableMode =
    !unifiedMode &&
    !dualArmMode &&
    !isaacMode &&
    hasUrlTarget &&
    (isCableThreadingReplayMode(initialTaskType) ||
      Boolean(initialEvalId || initialJobId));

  const initialCableId = useMemo(() => {
    if (!cableMode || !hasUrlTarget) return null;
    return resolvedCableTargetId ?? null;
  }, [cableMode, hasUrlTarget, resolvedCableTargetId]);

  const initialDualArmId = useMemo(() => {
    if (!dualArmMode || !initialJobId) return null;
    return resolvedDualArmTargetId ?? initialJobId;
  }, [dualArmMode, initialJobId, resolvedDualArmTargetId]);

  const initialMockId = useMemo(() => {
    if (cableMode || dualArmMode || !hasUrlTarget || !initialEvalId) return null;
    return resolveReplaySessionIdByEvalId(initialEvalId);
  }, [cableMode, dualArmMode, hasUrlTarget, initialEvalId]);

  const [selectedCableId, setSelectedCableId] = useState<string | null>(initialCableId);
  const [selectedDualArmId, setSelectedDualArmId] = useState<string | null>(initialDualArmId);
  const [selectedMockId, setSelectedMockId] = useState<string | null>(initialMockId);

  useEffect(() => {
    setSelectedCableId(initialCableId);
  }, [initialCableId]);

  useEffect(() => {
    setSelectedDualArmId(initialDualArmId);
  }, [initialDualArmId]);

  useEffect(() => {
    setSelectedMockId(initialMockId);
  }, [initialMockId]);

  useEffect(() => {
    if (!workspaceReplayJobId || !isPersistedWorkspaceJobId(workspaceReplayJobId)) return;
    if (workspaceReplayAttemptedRef.current === workspaceReplayJobId) return;
    workspaceReplayAttemptedRef.current = workspaceReplayJobId;

    let cancelled = false;
    setWorkspaceJobLoading(true);
    void Promise.all([
      getWorkspaceJob(workspaceReplayJobId),
      getWorkspaceJobArtifacts(workspaceReplayJobId),
    ])
      .then(([job, artifactRes]) => {
        if (cancelled) return;
        setWorkspaceJobDetail(job);
        setWorkspaceArtifacts(artifactRes.artifacts);
        if (job.jobId.startsWith('dac_gen_')) {
          void getDualArmCableJobStatus(job.jobId)
            .then((status) => {
              if (cancelled) return;
              const record = dualArmReplayRecordFromStatus(status);
              setDualArmRecords((prev) =>
                prev.some((r) => r.id === record.id) ? prev : [record, ...prev]
              );
              setSelectedDualArmId(record.id);
            })
            .catch(() => {
              if (cancelled) return;
              const record = dualArmReplayRecordFromWorkspaceJob(job, artifactRes.artifacts);
              setDualArmRecords((prev) =>
                prev.some((r) => r.id === record.id) ? prev : [record, ...prev]
              );
              setSelectedDualArmId(record.id);
            });
          return;
        }
        if (job.jobId.startsWith('ct_gen_') || job.jobId.startsWith('ct_eval_')) {
          const record = cableReplayRecordFromWorkspaceJob(job, artifactRes.artifacts);
          setCableRecords((prev) =>
            prev.some((r) => r.id === record.id) ? prev : [record, ...prev]
          );
          setSelectedCableId(record.id);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message.toLowerCase() : '';
        if (
          isPersistedWorkspaceJobId(workspaceReplayJobId) &&
          (msg.includes('not found') || msg.includes('404'))
        ) {
          setWorkspaceJobDeleted(true);
        }
      })
      .finally(() => {
        if (!cancelled) setWorkspaceJobLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [workspaceReplayJobId]);

  useEffect(() => {
    if (!dualArmMode || !initialJobId?.startsWith('dac_gen_')) return;

    if (dualArmRecords.some((r) => r.id === initialJobId)) {
      setSelectedDualArmId(initialJobId);
      return;
    }

    if (dualArmFetchAttemptedRef.current === initialJobId) return;
    dualArmFetchAttemptedRef.current = initialJobId;

    let cancelled = false;
    setDualArmFetching(true);
    setDualArmFetchError(null);

    void getDualArmCableJobStatus(initialJobId)
      .then((status) => {
        if (cancelled) return;
        const record = dualArmReplayRecordFromStatus(status);
        setDualArmRecords((prev) => {
          if (prev.some((r) => r.id === record.id)) return prev;
          return [record, ...prev];
        });
        setSelectedDualArmId(record.id);
      })
      .catch(() => {
        if (!cancelled) {
          setDualArmFetchError('无法从后端加载线缆整理回放记录，请检查 jobId 与登录状态。');
        }
      })
      .finally(() => {
        if (!cancelled) setDualArmFetching(false);
      });

    return () => {
      cancelled = true;
    };
  }, [dualArmMode, initialJobId, dualArmRecords.length]);

  const cableRecord = cableRecords.find((r) => r.id === selectedCableId) ?? null;
  const dualArmRecord = dualArmRecords.find((r) => r.id === selectedDualArmId) ?? null;
  const mockSession = replaySessions.find((s) => s.id === selectedMockId) ?? null;

  const cableTargetMissing =
    cableMode &&
    Boolean(initialEvalId || initialJobId) &&
    !resolvedCableTargetId;

  const dualArmTargetMissing =
    dualArmMode &&
    Boolean(initialJobId) &&
    !dualArmFetching &&
    !dualArmRecord &&
    Boolean(dualArmFetchError);

  const emptyCableList = cableMode && cableRecords.length === 0;
  const emptyDualArmList = dualArmMode && dualArmRecords.length === 0 && !dualArmFetching;

  const evaluationContentHeader = useMemo(() => {
    if (replayKind !== 'evaluation') return null;

    const taskNameFromType = initialTaskType ? getTaskDisplayName(initialTaskType) : null;
    const taskName =
      taskNameFromType && taskNameFromType !== '—'
        ? taskNameFromType
        : cableRecord?.recordType === 'policy_eval'
          ? CABLE_THREADING_DISPLAY_NAME
          : mockSession?.taskName
            ? getTaskDisplayName(mockSession.taskName) !== '—'
              ? getTaskDisplayName(mockSession.taskName)
              : mockSession.taskName
            : CABLE_THREADING_DISPLAY_NAME;

    let statusLabel = '加载中…';
    if (cableRecord) {
      statusLabel = cableReplayStatusLabel(cableRecord.status);
    } else if (workspaceJobDetail) {
      statusLabel = mapEvaluationJobStatusLabel(workspaceJobDetail.status);
    } else if (mockSession) {
      statusLabel = mockSession.status === 'completed' ? '已完成' : '运行中';
    }

    return <EvaluationReplayStatusHeader taskName={taskName} statusLabel={statusLabel} />;
  }, [replayKind, initialTaskType, cableRecord, workspaceJobDetail, mockSession]);

  const contentHeader: ReactNode =
    replayKind === 'evaluation'
      ? evaluationContentHeader ?? (
          <EvaluationReplayStatusHeader taskName={CABLE_THREADING_DISPLAY_NAME} statusLabel="加载中…" />
        )
      : replayKind === 'data_generation'
        ? t('workspacePages.replayVideoDataTitle')
        : t('workspacePages.replayVideoGenericTitle');

  const emptyRunInfoLabel = t('workspacePages.replayRunInfoEmpty');

  const headerMeta = dualArmMode ? '视频来源：本次 episode 录制视频（generate.mp4）' : undefined;

  const activeRecord = dualArmMode ? dualArmRecord : cableMode ? cableRecord : mockSession;

  const isLoadingTarget =
    hasUrlTarget &&
    !workspaceJobDeleted &&
    !activeRecord &&
    (dualArmFetching || workspaceJobLoading);

  const showEmptyMain =
    !hasUrlTarget ||
    workspaceJobDeleted ||
    isLoadingTarget ||
    (!activeRecord &&
      (cableTargetMissing ||
        dualArmTargetMissing ||
        emptyCableList ||
        (emptyDualArmList && Boolean(initialJobId))));

  const emptyMessage = !hasUrlTarget
    ? t('workspacePages.replayEmptyNoJob')
    : workspaceJobDeleted
      ? t('workspacePages.replayEmptyDeleted')
      : replaySourceKind === 'evaluation'
        ? '未找到评测回放目标，请返回评测中心重新进入。'
        : isaacMode
          ? '未找到 Isaac Lab 数据集回放目标，请返回数据中心重新进入。'
          : dualArmMode
            ? dualArmFetchError ?? '未找到线缆整理运行记录，请返回数据中心重新进入。'
            : '未找到线缆穿杆运行记录，请返回数据中心重新进入。';

  if (unifiedMode) {
    return (
      <UnifiedReplayWorkbench
        taskType={initialTaskType}
        jobId={initialJobId}
        datasetId={initialDatasetId}
        replayJobId={initialReplayJobId}
        sourceKind={replaySourceKind}
        returnPath={returnPath}
        returnLabel={returnLabel}
        onContentResolved={onContentResolved}
      />
    );
  }

  return (
    <>
      <style>{REPLAY_PAGE_STYLES}</style>
      <div className="replay-page-stack">
        <section className="replay-workspace-card">
          {showEmptyMain ? (
            <ReplayEmptyMain
              videoTitle={contentHeader}
              message={
                isLoadingTarget
                  ? dualArmFetching
                    ? '正在从后端加载线缆整理回放记录…'
                    : '正在加载回放记录…'
                  : emptyMessage
              }
            />
          ) : (
            <MainReplayArea
              session={mockSession}
              cableRecord={cableRecord}
              dualArmRecord={dualArmRecord}
              cableMode={cableMode}
              dualArmMode={dualArmMode}
              headerTitle={contentHeader}
              headerMeta={headerMeta}
              emptyLabel={emptyRunInfoLabel}
              replayKind={replayKind}
              workspaceJob={workspaceJobDetail}
              workspaceArtifacts={workspaceArtifacts}
            />
          )}
        </section>
      </div>
    </>
  );
}
