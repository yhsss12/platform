'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { IsaacLabVideoPlayer } from '@/components/workspace/replay/IsaacLabVideoPlayer';
import { ReplayActionsPanel } from '@/components/workspace/replay/ReplayActionsPanel';
import { EvaluationReplayMetricsBlock } from '@/components/workspace/replay/EvaluationReplayMetricsBlock';
import {
  ReplayPanelSectionTitle,
  ReplaySidePanelLayout,
} from '@/components/workspace/replay/ReplaySidePanelLayout';
import {
  buildIsaacBlockStackingReplayHref,
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
} from '@/lib/workspace/isaacBlockStacking';
import {
  getIsaacLabDatasetReplayContext,
  getIsaacLabRunJobLog,
  getIsaacLabRuntimeStatus,
  startIsaacLabReplayFromDataset,
  type IsaacLabDatasetReplayContext,
} from '@/lib/api/isaacLabClient';

interface IsaacBlockStackingReplayPanelProps {
  sourceJobId?: string;
  datasetId?: string;
  replayJobId?: string;
}

const POLL_MS = 3000;

function InfoRow({ label, value }: { label: string; value?: string | number | null }) {
  if (value == null || value === '') return null;
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 12, lineHeight: 1.6 }}>
      <span style={{ color: '#9ca3af', minWidth: 88, flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#374151', wordBreak: 'break-all' }}>{value}</span>
    </div>
  );
}

export function IsaacBlockStackingReplayPanel({
  sourceJobId,
  datasetId,
  replayJobId: initialReplayJobId,
}: IsaacBlockStackingReplayPanelProps) {
  const router = useRouter();
  const [context, setContext] = useState<IsaacLabDatasetReplayContext | null>(null);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [replayFailureLog, setReplayFailureLog] = useState<string | null>(null);

  const resolvedSourceJobId =
    context?.sourceJobId?.startsWith('isaac_gen_')
      ? context.sourceJobId
      : sourceJobId?.startsWith('isaac_gen_')
        ? sourceJobId
        : undefined;

  const activeReplayJobId = context?.replayJobId ?? initialReplayJobId ?? '';
  const playback = context?.playback;
  const activeVideoJobId = playback?.playable ? playback.videoJobId ?? null : null;

  const refreshContext = useCallback(async () => {
    if (!datasetId) {
      setLoading(false);
      setContext(null);
      return;
    }
    try {
      const next = await getIsaacLabDatasetReplayContext(datasetId);
      setContext(next);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载回放上下文失败');
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    setLoading(true);
    void refreshContext();
  }, [refreshContext]);

  useEffect(() => {
    if (!context?.replayInProgress) return undefined;
    const timer = window.setInterval(() => {
      void refreshContext();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [context?.replayInProgress, refreshContext]);

  useEffect(() => {
    if (!context?.replayFailed || !activeReplayJobId) {
      setReplayFailureLog(null);
      return;
    }
    void getIsaacLabRunJobLog(activeReplayJobId, 'stderr', 30)
      .then((tail) => setReplayFailureLog(tail || null))
      .catch(() => setReplayFailureLog(null));
  }, [context?.replayFailed, activeReplayJobId]);

  const headerMeta = context?.videoSourceLabel;

  const previewNotice = useMemo(() => {
    if (!context?.usingPreviewFallback) return null;
    return '当前播放生成预览视频；正式数据集回放视频可通过「生成回放视频」创建。';
  }, [context?.usingPreviewFallback]);

  const handleGenerateReplay = async () => {
    if (!datasetId) {
      setError('缺少 datasetId，无法发起回放生成。');
      return;
    }
    if (generating) return;

    setGenerating(true);
    setError(null);
    setPlaybackError(null);
    try {
      const runtime = await getIsaacLabRuntimeStatus();
      if (!runtime.configured || !runtime.enabled) {
        setError('需配置 Isaac Lab 运行节点');
        return;
      }
      const started = await startIsaacLabReplayFromDataset(datasetId);
      setToast(
        started.reused
          ? `复用已有回放任务：${started.jobId}`
          : `回放任务已创建：${started.jobId}`
      );
      setTimeout(() => setToast(null), 2800);
      router.replace(
        buildIsaacBlockStackingReplayHref({
          jobId: resolvedSourceJobId,
          datasetId,
          replayJobId: started.jobId,
        })
      );
      await refreshContext();
    } catch (err) {
      setError(err instanceof Error ? err.message : '回放生成启动失败');
    } finally {
      setGenerating(false);
    }
  };

  const showPlayer = Boolean(activeVideoJobId && !playbackError);
  const showEmpty = !loading && !showPlayer;

  return (
    <div className="replay-main-area">
      <div className="replay-header">
        <span className="replay-header-title">{ISAAC_BLOCK_STACKING_DISPLAY_NAME} / Isaac Lab 回放</span>
        {headerMeta ? <span className="replay-header-meta">{headerMeta}</span> : null}
      </div>

      {previewNotice ? (
        <div
          style={{
            marginBottom: 12,
            padding: '8px 12px',
            borderRadius: 8,
            background: '#fffbeb',
            border: '1px solid #fde68a',
            color: '#92400e',
            fontSize: 12,
            lineHeight: 1.55,
          }}
        >
          {previewNotice}
        </div>
      ) : null}

      {context?.replayInProgress ? (
        <div
          style={{
            marginBottom: 12,
            padding: '8px 12px',
            borderRadius: 8,
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
            color: '#1e40af',
            fontSize: 12,
            lineHeight: 1.55,
          }}
        >
          回放任务进行中（{activeReplayJobId || '—'}），完成后将自动切换至 replay.mp4。
        </div>
      ) : null}

      <div className="replay-content-row">
        <div className="replay-player-column">
          <div className="replay-player-shell">
            <div className="replay-player">
              <div className="replay-player-media">
                {loading ? (
                  <div
                    style={{
                      width: '100%',
                      height: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#94a3b8',
                      fontSize: 14,
                    }}
                  >
                    正在加载回放上下文…
                  </div>
                ) : showPlayer && activeVideoJobId ? (
                  <IsaacLabVideoPlayer
                    key={`${activeVideoJobId}:${playback?.videoSource ?? 'none'}`}
                    videoJobId={activeVideoJobId}
                    onPlaybackError={(message) => setPlaybackError(message)}
                    onReady={() => setPlaybackError(null)}
                  />
                ) : showEmpty ? (
                  <div
                    style={{
                      width: '100%',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 14,
                      padding: 24,
                      textAlign: 'center',
                      color: '#94a3b8',
                      fontSize: 13,
                      lineHeight: 1.6,
                    }}
                  >
                    <p style={{ margin: 0 }}>
                      {playbackError ??
                        (context?.replayInProgress
                          ? '回放视频生成中，请稍候…'
                          : '当前数据集暂无可播放回放视频。')}
                    </p>
                    {datasetId && context?.hasDatasetFile && !context.replayInProgress ? (
                      <button
                        type="button"
                        onClick={() => void handleGenerateReplay()}
                        disabled={generating}
                        style={{
                          padding: '8px 14px',
                          borderRadius: 8,
                          border: '1px solid #d1d5db',
                          background: '#fff',
                          color: '#374151',
                          fontSize: 13,
                          cursor: generating ? 'not-allowed' : 'pointer',
                        }}
                      >
                        {generating ? '正在创建回放任务…' : '生成回放视频'}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>

        <ReplaySidePanelLayout
          footerActions={
            <ReplayActionsPanel variant="footer" />
          }
        >
          <div style={{ marginBottom: 12 }}>
            <ReplayPanelSectionTitle>基础信息</ReplayPanelSectionTitle>
            <InfoRow label="数据集名称" value={context?.dataset.name} />
            <InfoRow label="来源任务" value={ISAAC_BLOCK_STACKING_DISPLAY_NAME} />
            <InfoRow label="仿真后端" value="Isaac Lab" />
            <InfoRow label="数据格式" value="HDF5" />
            <InfoRow label="Episode 数" value={context?.dataset.episodeCount} />
            <InfoRow
              label="状态"
              value={context?.dataset.status === 'available' ? '可用' : context?.dataset.status}
            />
          </div>

          <div style={{ marginBottom: 12 }}>
            <ReplayPanelSectionTitle>评测指标</ReplayPanelSectionTitle>
            <EvaluationReplayMetricsBlock />
          </div>

          {context?.replayFailed ? (
            <div
              style={{
                padding: '10px 12px',
                borderRadius: 8,
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: 12,
                lineHeight: 1.55,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4 }}>回放任务失败</div>
              <div>{context.replayJobs.find((j) => j.jobId === activeReplayJobId)?.message ?? '请查看 stderr 日志'}</div>
              {replayFailureLog ? (
                <pre
                  style={{
                    marginTop: 8,
                    whiteSpace: 'pre-wrap',
                    fontSize: 11,
                    color: '#7f1d1d',
                    maxHeight: 120,
                    overflow: 'auto',
                  }}
                >
                  {replayFailureLog}
                </pre>
              ) : null}
            </div>
          ) : null}

          {error ? (
            <div
              style={{
                padding: '10px 12px',
                borderRadius: 8,
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: 12,
                lineHeight: 1.55,
              }}
            >
              {error}
            </div>
          ) : null}
        </ReplaySidePanelLayout>
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
    </div>
  );
}
