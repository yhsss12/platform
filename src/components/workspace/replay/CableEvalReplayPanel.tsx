'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { CableThreadingVideoPlayer } from '@/components/workspace/replay/CableThreadingVideoPlayer';
import { EvalReplaySidePanel } from '@/components/workspace/replay/EvalReplaySidePanel';
import { EvaluationReplayEpisodeStats } from '@/components/workspace/replay/EvaluationReplayEpisodeStats';
import { EvaluationReplayStatusHeader } from '@/components/workspace/replay/EvaluationReplayStatusHeader';
import { EvaluationTrajectorySelector } from '@/components/workspace/replay/EvaluationTrajectorySelector';
import { EvaluationWorkbenchBasicInfoRows } from '@/components/workspace/replay/EvaluationWorkbenchBasicInfoRows';
import { REPLAY_PAGE_STYLES } from '@/components/workspace/replay/ReplayWorkbench';
import {
  getCableThreadingEvalResult,
  getCableThreadingJobLog,
  getCableThreadingJobStatus,
  type CableThreadingJobStatusResponse,
} from '@/lib/api/cableThreadingClient';
import {
  CABLE_THREADING_TASK_DISPLAY_NAME,
  resolveCableThreadingHasValidLiveFrame,
} from '@/lib/workspace/cableThreading';
import { findEvaluationTaskById } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { listWorkspaceEvaluationTasksForUi } from '@/lib/workspace/workspaceDataSources';
import { resolveEvaluationWorkbenchBasicInfo } from '@/lib/workspace/evaluationWorkbenchBasicInfo';
import {
  extractMetricResultsFromAggregate,
  extractSelectedMetricIds,
} from '@/lib/workspace/evaluationMetricResultsDisplay';
import { resolveEvalFailureDiagnosis } from '@/lib/workspace/evaluationFailureDiagnosis';
import {
  buildRepresentativeVideoHint,
  findTrajectoryIndexByRound,
  getEpisodeIndex,
  getTrajectoryLabel,
  mergeEvaluationReplayInfo,
  normalizeReplayTrajectoryItems,
} from '@/lib/workspace/evaluationReplayInfo';
import { resolveEvaluationViewportState } from '@/lib/workspace/evaluationViewportState';
import { replayVideoSourceUserLabel } from '@/lib/workspace/replayAdapters';
import type { ReplayPageKind } from '@/lib/workspace/replayPageKind';
import { CableThreadingLiveFrame } from '@/components/workspace/simulation/CableThreadingLiveFrame';

const POLL_MS = 2000;

const viewportCardStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 24,
  textAlign: 'center',
  gap: 8,
};

function mapStatusLabel(status: string): string {
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'queued') return '排队中';
  if (status === 'running') return '运行中';
  return status;
}

function EvaluationViewport({
  evalJobId,
  status,
  videoApiPath,
}: {
  evalJobId: string;
  status: CableThreadingJobStatusResponse | null;
  videoApiPath: string | null;
}) {
  const jobStatus = status?.status ?? 'loading';
  const evalVideoExists = Boolean(status?.evalVideoExists || status?.evalBrowserVideoExists);
  const hasValidLiveFrame = resolveCableThreadingHasValidLiveFrame(status);
  const viewport = resolveEvaluationViewportState({
    jobStatus,
    evalJobId,
    evalVideoExists,
    hasValidLiveFrame,
  });

  if (viewport.kind === 'running_live') {
    const frameStatus =
      jobStatus === 'completed' ? 'completed' : jobStatus === 'failed' ? 'failed' : 'running';
    return (
      <div style={{ width: '100%', height: '100%' }}>
        <CableThreadingLiveFrame jobId={viewport.evalJobId} status={frameStatus} embedded />
      </div>
    );
  }

  if (viewport.kind === 'video' || viewport.kind === 'failed_partial_video') {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          overflow: 'hidden',
          borderRadius: 14,
          background: '#0f172a',
        }}
      >
        <CableThreadingVideoPlayer videoJobId={evalJobId} videoApiPath={videoApiPath} />
      </div>
    );
  }

  if (viewport.kind === 'running') {
    return (
      <div style={viewportCardStyle}>
        <div style={{ fontSize: 14, color: '#cbd5e1', fontWeight: 500 }}>MuJoCo 评测画面</div>
        <p style={{ margin: 0, fontSize: 13, color: '#94a3b8', lineHeight: 1.6, maxWidth: 360 }}>
          {viewport.message}
        </p>
      </div>
    );
  }
  return (
    <div style={viewportCardStyle}>
      <div style={{ fontSize: 14, color: '#cbd5e1', fontWeight: 500 }}>
        {'title' in viewport ? viewport.title : '评测画面未生成'}
      </div>
      <p style={{ margin: 0, fontSize: 13, color: '#94a3b8', lineHeight: 1.6, maxWidth: 400 }}>
        {viewport.message}
      </p>
    </div>
  );
}

export function CableEvalReplayPanel({
  evalJobId,
  replayKind,
  initialEpisode,
}: {
  evalJobId: string;
  replayKind: ReplayPageKind;
  /** URL ?episode=N，1-based 轮次 */
  initialEpisode?: number;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [status, setStatus] = useState<CableThreadingJobStatusResponse | null>(null);
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [aggregate, setAggregate] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(0);

  const fetchLog = useCallback(async () => {
    const response = await getCableThreadingJobLog(evalJobId);
    return response.tail?.trim() ?? '';
  }, [evalJobId]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | undefined;

    const load = async () => {
      try {
        const nextStatus = await getCableThreadingJobStatus(evalJobId);
        if (cancelled) return;
        setStatus(nextStatus);

        const terminal = nextStatus.status === 'completed' || nextStatus.status === 'failed';
        if (terminal) {
          try {
            const result = await getCableThreadingEvalResult(evalJobId);
            if (!cancelled && result && typeof result === 'object') {
              setResultPayload(result as Record<string, unknown>);
              if (result.aggregate && typeof result.aggregate === 'object') {
                setAggregate(result.aggregate as Record<string, unknown>);
              } else if (nextStatus.metrics?.aggregate) {
                setAggregate(nextStatus.metrics.aggregate);
              }
            } else if (!cancelled && nextStatus.metrics?.aggregate) {
              setResultPayload(null);
              setAggregate(nextStatus.metrics.aggregate);
            }
          } catch {
            if (!cancelled) {
              setResultPayload(null);
              if (nextStatus.metrics?.aggregate) {
                setAggregate(nextStatus.metrics.aggregate);
              }
            }
          }
          if (timer) {
            clearInterval(timer);
            timer = undefined;
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '加载评测工作台失败');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();
    timer = setInterval(() => {
      void load();
    }, POLL_MS);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [evalJobId]);

  const live = (status?.live ?? {}) as Record<string, unknown>;

  const listItem = useMemo(() => {
    const row = findEvaluationTaskById(evalJobId, listWorkspaceEvaluationTasksForUi());
    if (!row) return null;
    return {
      taskName: row.taskName ?? row.name,
      name: row.name,
      evaluationTypeLabel: row.evaluationTypeLabel,
      evaluationObject: row.evaluationObject,
      evaluationMode: row.evaluationMode,
      taskType: row.taskType,
      status: row.status,
    };
  }, [evalJobId]);

  const workbenchBasicInfo = useMemo(() => {
    const info = resolveEvaluationWorkbenchBasicInfo({
      evalJobId,
      status: status as unknown as Record<string, unknown>,
      aggregate: aggregate ?? undefined,
      result: resultPayload ?? undefined,
      live,
      listItem: listItem ?? undefined,
      fallbackTaskName: CABLE_THREADING_TASK_DISPLAY_NAME,
    });
    if (loading && !status) {
      return { ...info, statusLabel: '加载中…' };
    }
    return info;
  }, [status, aggregate, resultPayload, live, evalJobId, loading, listItem]);

  const horizon = Number(live.horizon ?? 600);
  const seed = Number(live.seed ?? 0);

  const replayInfo = useMemo(
    () =>
      mergeEvaluationReplayInfo(
        (status as unknown as Record<string, unknown> | undefined) ?? undefined,
        resultPayload ?? undefined,
        aggregate ?? undefined
      ),
    [status, resultPayload, aggregate]
  );

  const selectedMetricIds = useMemo(
    () =>
      extractSelectedMetricIds([
        aggregate,
        resultPayload,
        status as unknown as Record<string, unknown> | null,
        status?.metrics as Record<string, unknown> | null,
      ]),
    [aggregate, resultPayload, status]
  );

  const metricResults = useMemo(() => {
    const fromAggregate = extractMetricResultsFromAggregate(aggregate);
    if (fromAggregate) return fromAggregate;
    const statusBlock = status as unknown as Record<string, unknown> | null;
    if (statusBlock?.metricResults && typeof statusBlock.metricResults === 'object') {
      return extractMetricResultsFromAggregate(statusBlock);
    }
    if (resultPayload?.metricResults && typeof resultPayload.metricResults === 'object') {
      return extractMetricResultsFromAggregate(resultPayload);
    }
    const metricsBlock = status?.metrics as Record<string, unknown> | undefined;
    if (metricsBlock?.metricResults && typeof metricsBlock.metricResults === 'object') {
      return extractMetricResultsFromAggregate(metricsBlock);
    }
    return null;
  }, [aggregate, resultPayload, status]);

  const replayVideoOptions = useMemo(
    () => normalizeReplayTrajectoryItems(replayInfo),
    [replayInfo]
  );

  useEffect(() => {
    setSelectedVideoIndex(0);
  }, [evalJobId]);

  useEffect(() => {
    if (replayVideoOptions.length <= 1) return;
    const urlRound =
      (initialEpisode && initialEpisode > 0 ? initialEpisode : null) ??
      (Number(searchParams.get('episode') ?? '0') || 0);
    if (urlRound > 0) {
      setSelectedVideoIndex(findTrajectoryIndexByRound(replayVideoOptions, urlRound));
    }
    // 仅在任务切换或 replay 条数变化时从 URL 初始化，避免轮询导致重复重置
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalJobId, replayVideoOptions.length, initialEpisode]);

  useEffect(() => {
    if (selectedVideoIndex >= replayVideoOptions.length) {
      setSelectedVideoIndex(0);
    }
  }, [replayVideoOptions.length, selectedVideoIndex]);

  const handleSelectVideo = useCallback(
    (index: number) => {
      setSelectedVideoIndex(index);
      const item = replayVideoOptions[index];
      if (!item || replayVideoOptions.length <= 1) return;
      const round = getEpisodeIndex(item, index);
      const params = new URLSearchParams(searchParams.toString());
      params.set('episode', String(round));
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [replayVideoOptions, searchParams, pathname, router]
  );

  const selectedVideoOption =
    replayVideoOptions[selectedVideoIndex] ?? replayVideoOptions[0] ?? null;
  const videoApiPath = selectedVideoOption?.uri ?? null;
  const replayVideoHint = useMemo(() => buildRepresentativeVideoHint(replayInfo), [replayInfo]);

  const failureDiagnosis = useMemo(
    () =>
      resolveEvalFailureDiagnosis(
        status
          ? {
              status: status.status,
              failedStage: status.failedStage,
              failureReason: status.failureReason,
              errorMessage: status.errorMessage,
              logPaths: status.logPaths,
              live: status.live,
            }
          : null
      ),
    [status]
  );

  const displayTaskName = workbenchBasicInfo.taskName;

  if (error) {
    return <section style={{ padding: 20, color: '#b45309', fontSize: 13 }}>{error}</section>;
  }

  const jobStatusLabel = mapStatusLabel(status?.status ?? (loading ? 'loading' : 'running'));
  const videoResolution = status?.videoResolution ?? null;
  const evaluationMode =
    replayInfo.evaluationMode ??
    (typeof status?.evaluationMode === 'string' ? status.evaluationMode : null) ??
    listItem?.evaluationMode ??
    null;
  const rolloutFooterLabel = replayVideoSourceUserLabel('evaluation', 'evaluation', evaluationMode);
  const footerLabel = selectedVideoOption
    ? getTrajectoryLabel(selectedVideoOption, selectedVideoIndex)
    : rolloutFooterLabel;
  const showTrajectorySelector = replayVideoOptions.length > 1;
  const evalVideoExists = Boolean(status?.evalVideoExists || status?.evalBrowserVideoExists);
  const hasValidLiveFrame = resolveCableThreadingHasValidLiveFrame(status);
  const viewportKind = resolveEvaluationViewportState({
    jobStatus: status?.status ?? 'loading',
    evalJobId,
    evalVideoExists,
    hasValidLiveFrame,
  }).kind;
  const showVideoHints =
    viewportKind === 'video' || viewportKind === 'failed_partial_video';
  const partialVideoMessage =
    viewportKind === 'failed_partial_video'
      ? '评测任务失败，但已生成部分回放画面。'
      : null;

  return (
    <>
      <style>{REPLAY_PAGE_STYLES}</style>
      <section className="replay-workspace-card">
        <div className="replay-main-area">
          <div className="replay-header">
            <EvaluationReplayStatusHeader
              taskName={displayTaskName}
              statusLabel={jobStatusLabel}
            />
          </div>
          <div className="replay-content-row">
            <div className="replay-player-column">
              {showTrajectorySelector ? (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    marginBottom: 10,
                  }}
                >
                  <span style={{ fontSize: 14, color: '#64748b', flexShrink: 0 }}>选择轨迹：</span>
                  <EvaluationTrajectorySelector
                    replayItems={replayVideoOptions}
                    selectedIndex={selectedVideoIndex}
                    onSelect={handleSelectVideo}
                  />
                </div>
              ) : null}
              <div className="replay-player-shell">
                <div className="replay-player">
                  <div className="replay-player-media">
                    <EvaluationViewport
                      evalJobId={evalJobId}
                      status={status}
                      videoApiPath={videoApiPath}
                    />
                  </div>
                </div>
              </div>
              {showVideoHints && replayVideoHint ? (
                <p style={{ margin: '8px 0 0', fontSize: 12, color: '#64748b', textAlign: 'center' }}>
                  {replayVideoHint}
                </p>
              ) : null}
              {showVideoHints && partialVideoMessage ? (
                <p style={{ margin: '4px 0 0', fontSize: 12, color: '#b45309', textAlign: 'center' }}>
                  {partialVideoMessage}
                </p>
              ) : null}
              <div
                style={{
                  marginTop: 8,
                  fontSize: 12,
                  color: '#64748b',
                  textAlign: 'center',
                }}
              >
                {footerLabel}
                {videoResolution ? (
                  <span
                    style={{
                      marginLeft: 8,
                      color: videoResolution.startsWith('256') ? '#b45309' : '#64748b',
                    }}
                  >
                    {videoResolution}
                  </span>
                ) : null}
              </div>
            </div>
            <EvalReplaySidePanel
              evalJobId={evalJobId}
              taskType="cable_threading"
              jobStatus={status?.status ?? null}
              loading={loading}
              aggregate={aggregate}
              metrics={status?.metrics ?? null}
              live={live}
              metricResults={metricResults}
              selectedMetricIds={selectedMetricIds}
              failureDiagnosis={failureDiagnosis}
              onFetchLog={fetchLog}
              basicInfo={<EvaluationWorkbenchBasicInfoRows info={workbenchBasicInfo} />}
              progressInfo={
                <EvaluationReplayEpisodeStats
                  replay={replayInfo}
                  horizon={horizon}
                  seed={seed}
                  statusLabel={jobStatusLabel}
                />
              }
            />
          </div>
        </div>
      </section>
    </>
  );
}
