'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { REPLAY_PAGE_STYLES } from '@/components/workspace/replay/ReplayWorkbench';
import { ReplayActionsPanel } from '@/components/workspace/replay/ReplayActionsPanel';
import { ReplayInfoPanel } from '@/components/workspace/replay/ReplayInfoPanel';
import { ReplayVideoPlayer } from '@/components/workspace/replay/ReplayVideoPlayer';
import {
  generateReplayForAdapter,
  resolveReplayAdapter,
  type ReplayAdapterResult,
} from '@/lib/workspace/replayAdapters';
import { getIsaacLabRuntimeStatus } from '@/lib/api/isaacLabClient';
import { inferReplayTaskTypeFromJobId } from '@/lib/workspace/datasetReplayHref';
import { isStackCubeJobInProgress } from '@/lib/workspace/isaaclabFrankaStackCubeReplayLogic';
import type { ReplaySourceKind } from '@/lib/workspace/replayViewModel';
import { REPLAY_DATA_CENTER_HREF } from '@/lib/workspace/replayPanelNavigation';
import {
  REPLAY_CONTENT_COPY,
  resolveReplayTabVideoPlayable,
  type ReplayContentKind,
} from '@/lib/workspace/replayContentKind';

const POLL_MS = 3000;

interface UnifiedReplayWorkbenchProps {
  taskType?: string;
  jobId?: string;
  datasetId?: string;
  replayJobId?: string;
  sourceKind: ReplaySourceKind;
  returnPath?: string;
  returnLabel?: string;
  onContentResolved?: (adapter: ReplayAdapterResult) => void;
}

type PlayerState = 'loading' | 'playing' | 'trajectory' | 'no_video' | 'load_failed';

function resolvePlayerState(params: {
  loading: boolean;
  adapter: ReplayAdapterResult | null;
  playbackError: string | null;
  activeTab: ReplayContentKind | null;
  videoPlayable: boolean;
}): PlayerState {
  const { loading, adapter, playbackError, activeTab, videoPlayable } = params;
  if (loading || !adapter) return 'loading';
  if (adapter.replayInProgress) return 'loading';
  if (playbackError) return 'load_failed';
  if (activeTab === 'dataset_trajectory_replay' && (adapter.trajectories?.length ?? 0) > 0) {
    return 'trajectory';
  }
  if (videoPlayable && adapter.videoJobId && adapter.videoBackend !== 'none') {
    return 'playing';
  }
  return 'no_video';
}

function VideoTag({ label }: { label: string }) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        borderRadius: 999,
        fontSize: 11,
        fontWeight: 500,
        color: '#64748b',
        background: '#f1f5f9',
        border: '1px solid #e2e8f0',
      }}
    >
      {label}
    </span>
  );
}

function ReplayTabBar({
  tabs,
  activeTab,
  onChange,
}: {
  tabs: Array<{ id: string; label: string }>;
  activeTab: ReplayContentKind;
  onChange: (tab: ReplayContentKind) => void;
}) {
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
      {tabs.map((tab) => {
        const selected = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id as ReplayContentKind)}
            style={{
              borderRadius: 999,
              border: selected ? '1px solid #2563eb' : '1px solid #e5e7eb',
              background: selected ? '#eff6ff' : '#fff',
              color: selected ? '#1d4ed8' : '#475569',
              fontSize: 12,
              fontWeight: selected ? 600 : 500,
              padding: '6px 12px',
              cursor: 'pointer',
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

function TrajectorySelector({
  trajectories,
  selectedIndex,
  onSelect,
}: {
  trajectories: string[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
      {trajectories.map((name, index) => {
        const selected = index === selectedIndex;
        return (
          <button
            key={name}
            type="button"
            onClick={() => onSelect(index)}
            style={{
              borderRadius: 8,
              border: selected ? '1px solid #2563eb' : '1px solid #334155',
              background: selected ? '#1d4ed8' : 'rgba(15, 23, 42, 0.72)',
              color: '#e2e8f0',
              fontSize: 12,
              padding: '6px 10px',
              cursor: 'pointer',
            }}
          >
            {name}
          </button>
        );
      })}
    </div>
  );
}

export function UnifiedReplayWorkbench({
  taskType,
  jobId,
  datasetId,
  replayJobId,
  sourceKind,
  returnPath = REPLAY_DATA_CENTER_HREF,
  returnLabel = '返回数据中心',
  onContentResolved,
}: UnifiedReplayWorkbenchProps) {
  const router = useRouter();
  const [adapter, setAdapter] = useState<ReplayAdapterResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ReplayContentKind | null>(null);
  const [selectedTrajectoryIndex, setSelectedTrajectoryIndex] = useState(0);

  const resolvedTaskType =
    inferReplayTaskTypeFromJobId(jobId) ?? inferReplayTaskTypeFromJobId(replayJobId) ?? taskType;

  const refresh = useCallback(async () => {
    const result = await resolveReplayAdapter({
      taskType: resolvedTaskType,
      jobId,
      datasetId,
      replayJobId,
    });
    setAdapter(result);
    setLoading(false);
    onContentResolved?.(result);
    return result;
  }, [resolvedTaskType, jobId, datasetId, replayJobId, onContentResolved]);

  useEffect(() => {
    setLoading(true);
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!adapter?.defaultReplayTab) return;
    setActiveTab(adapter.defaultReplayTab);
  }, [adapter?.defaultReplayTab, adapter?.sourceJobId]);

  useEffect(() => {
    if (!adapter?.replayInProgress) return undefined;
    const timer = window.setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [adapter?.replayInProgress, refresh]);

  useEffect(() => {
    if (adapter?.taskType !== 'isaaclab_franka_stack_cube') return undefined;
    if (!isStackCubeJobInProgress(adapter.sourceJobStatus)) return undefined;
    const timer = window.setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [adapter?.sourceJobStatus, adapter?.taskType, refresh]);

  const handleGenerateReplay = async () => {
    if (!adapter || generating) return;
    setGenerating(true);
    setActionError(null);
    setPlaybackError(null);
    try {
      if (adapter.taskType === 'isaac_block_stacking') {
        const runtime = await getIsaacLabRuntimeStatus();
        if (!runtime.configured || !runtime.enabled) {
          setActionError('需配置 Isaac Lab 运行节点');
          return;
        }
      }
      const started = await generateReplayForAdapter(adapter);
      setToast(started.reused ? '已复用已有回放任务' : '回放任务已创建');
      setTimeout(() => setToast(null), 2800);
      router.replace(started.refreshHref);
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : '回放生成启动失败');
    } finally {
      setGenerating(false);
    }
  };

  const effectiveTab = activeTab ?? adapter?.defaultReplayTab ?? adapter?.replayContentKind ?? null;
  const replayContent = adapter?.replayContent;
  const videoPlayable = useMemo(() => {
    if (!adapter || !effectiveTab) return false;
    if (effectiveTab === 'generation_process_preview') {
      return resolveReplayTabVideoPlayable(effectiveTab, {
        hasGenerationPreview: adapter.hasGenerationPreview ?? false,
      });
    }
    if (effectiveTab === 'evaluation_replay') {
      return adapter.videoPlayable;
    }
    return false;
  }, [adapter, effectiveTab]);

  const playerState = resolvePlayerState({
    loading,
    adapter,
    playbackError,
    activeTab: effectiveTab,
    videoPlayable,
  });

  const headerCopy =
    effectiveTab && REPLAY_CONTENT_COPY[effectiveTab]
      ? REPLAY_CONTENT_COPY[effectiveTab]
      : null;
  const cardTitle = headerCopy?.title ?? adapter?.pageTitle ?? adapter?.taskName ?? '回放';
  const cardSubtitle = headerCopy?.subtitle ?? adapter?.pageSubtitle ?? '';
  const videoTag =
    effectiveTab === 'generation_process_preview'
      ? headerCopy?.tag ?? adapter?.videoTag
      : effectiveTab === 'evaluation_replay'
        ? adapter?.videoTag
        : null;

  const emptyMessage =
    playerState === 'load_failed'
      ? '回放视频加载失败，可重新生成回放视频后查看。'
      : playerState === 'loading' && adapter?.replayInProgress
        ? '正在生成回放视频…'
        : playerState === 'loading'
          ? '正在加载回放视频…'
          : adapter?.videoPlaceholderMessage ??
            adapter?.error ??
            '暂无可播放回放视频，可生成回放视频后查看。';

  const trajectories = adapter?.trajectories ?? [];
  const selectedTrajectory = trajectories[selectedTrajectoryIndex] ?? trajectories[0] ?? null;
  const showTabs = (adapter?.replayTabs?.length ?? 0) > 1;

  return (
    <>
      <style>{REPLAY_PAGE_STYLES}</style>
      <div className="replay-page-stack">
        <section className="replay-workspace-card">
          <div className="replay-main-area">
            <div className="replay-header">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 }}>
                <span className="replay-header-title">{cardTitle}</span>
                {cardSubtitle ? (
                  <span className="replay-header-meta">{cardSubtitle}</span>
                ) : null}
              </div>
              {videoTag && playerState === 'playing' ? <VideoTag label={videoTag} /> : null}
            </div>

            {showTabs && adapter?.replayTabs && effectiveTab ? (
              <ReplayTabBar
                tabs={adapter.replayTabs}
                activeTab={effectiveTab}
                onChange={(tab) => {
                  setActiveTab(tab);
                  setPlaybackError(null);
                }}
              />
            ) : null}

            <div className="replay-content-row">
              <div className="replay-player-column">
                <div className="replay-player-shell">
                  <div className="replay-player">
                    <div className="replay-player-media">
                      {playerState === 'playing' && adapter?.videoJobId ? (
                        <ReplayVideoPlayer
                          key={`${adapter.videoBackend}:${adapter.videoJobId}:${effectiveTab}`}
                          videoBackend={adapter.videoBackend}
                          videoJobId={adapter.videoJobId}
                          onPlaybackError={setPlaybackError}
                          onReady={() => setPlaybackError(null)}
                        />
                      ) : playerState === 'trajectory' ? (
                        <div
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 18,
                            padding: 24,
                            textAlign: 'center',
                            color: '#e2e8f0',
                          }}
                        >
                          <div style={{ fontSize: 15, fontWeight: 600 }}>
                            {selectedTrajectory ?? '有效轨迹'}
                          </div>
                          <div style={{ fontSize: 13, color: '#94a3b8', maxWidth: 420, lineHeight: 1.7 }}>
                            当前展示 HDF5 数据集中的有效轨迹。可在下方切换 demo 条目查看。
                          </div>
                          <TrajectorySelector
                            trajectories={trajectories}
                            selectedIndex={selectedTrajectoryIndex}
                            onSelect={setSelectedTrajectoryIndex}
                          />
                        </div>
                      ) : (
                        <div
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 16,
                            padding: 24,
                            textAlign: 'center',
                            color: '#94a3b8',
                            fontSize: 13,
                            lineHeight: 1.6,
                          }}
                        >
                          <p style={{ margin: 0, maxWidth: 360 }}>{emptyMessage}</p>
                          {adapter &&
                          (playerState === 'no_video' || playerState === 'load_failed') ? (
                            <ReplayActionsPanel
                              adapter={adapter}
                              generating={generating}
                              variant="inline"
                              onGenerateReplay={() => void handleGenerateReplay()}
                            />
                          ) : null}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {adapter ? (
                <>
                  <ReplayInfoPanel
                    adapter={adapter}
                    sourceKind={sourceKind}
                    activeReplayTab={effectiveTab}
                    selectedTrajectory={selectedTrajectory}
                    generating={generating}
                    returnPath={returnPath}
                    returnLabel={returnLabel}
                    onGenerateReplay={() => void handleGenerateReplay()}
                  />
                  {adapter.replayFailed && adapter.replayFailureMessage ? (
                    <div
                      style={{
                        margin: '0 14px 14px',
                        padding: '10px 12px',
                        borderRadius: 8,
                        background: '#fef2f2',
                        border: '1px solid #fecaca',
                        color: '#991b1b',
                        fontSize: 12,
                        lineHeight: 1.55,
                      }}
                    >
                      {adapter.replayFailureMessage}
                    </div>
                  ) : null}
                  {actionError ? (
                    <div
                      style={{
                        margin: '0 14px 14px',
                        padding: '10px 12px',
                        borderRadius: 8,
                        background: '#fef2f2',
                        border: '1px solid #fecaca',
                        color: '#991b1b',
                        fontSize: 12,
                      }}
                    >
                      {actionError}
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>
          </div>

          {toast ? (
            <div
              style={{
                marginTop: 12,
                padding: '8px 12px',
                borderRadius: 8,
                background: '#ecfdf5',
                border: '1px solid #a7f3d0',
                color: '#065f46',
                fontSize: 12,
              }}
            >
              {toast}
            </div>
          ) : null}
        </section>
      </div>
    </>
  );
}
