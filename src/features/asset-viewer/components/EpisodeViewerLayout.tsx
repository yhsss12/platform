'use client';

/**
 * 数据查看页与标注执行页共用布局：左列表 + 中视口 + 底时间轴，可选右侧标注面板与顶部标题栏。
 * 通过 source 区分数据来源：assetId（数据资产查看）或 taskId（标注任务）。
 */
import { useEffect, useState, useRef, useCallback } from 'react';
import LeftPanel from './LeftPanel';
import CenterPanel from './CenterPanel';
import BottomTimeline from './BottomTimeline';
import type { Episode, EpisodeInfo } from '../api/labelApi';
import { getEpisodes, getEpisode, getAssetFrame } from '../api/labelApi';
import { apiGet } from '@/features/data-platform/api/client';
import { useFrameCache } from '../context/FrameCacheContext';

export type ViewerSource =
  | { type: 'asset'; assetId: string }
  | { type: 'task'; taskId: string };

export interface EpisodeViewerLayoutProps {
  source: ViewerSource;
  /** 顶部标题栏（如数据查看页的资产名称与路径） */
  header?: { title: string; subtitle?: string };
  /** 右侧面板（标注页传入 RightPanel），不传则两列布局 */
  rightPanel?: React.ReactNode;
  /** 缺失 source 时的错误提示 */
  missingContextMessage?: string;
  /** 用于 LeftPanel 的标注相关回调（仅 task 模式需要；asset 模式可传空函数） */
  onNewAnnotation?: () => void;
  onSave?: () => void;
  /** 标注页专用：任务名、描述更新 ref、instructions 刷新、episodes 刷新等 */
  taskName?: string;
  onDescriptionUpdateRef?: React.MutableRefObject<((desc: string) => void) | null>;
  instructionsRefreshRef?: React.MutableRefObject<{ refresh: () => void } | null>;
  /** 重新拉取 episodes 列表（更新左侧 instruction_text / 已标注状态）；用于自动标注保存后由页面向外触发 */
  episodesRefreshRef?: React.MutableRefObject<{ refresh: () => Promise<void> } | null>;
  /** 标注页专用：预加载提示是否显示（由外部 FrameCache 等控制） */
  isPreloading?: boolean;
  preloadProgress?: number;
  /** 标注页专用：把当前选中 episode/视窗/帧数/taskId/相机列表 传给父组件，用于预加载与 RightPanel */
  onViewerStateChange?: (state: {
    selectedEpisode: string | null;
    viewportTopics: (string | null)[];
    maxFrame: number;
    taskId: string;
    cameras: string[];
  }) => void;
  /** 子组件需要用的 i18n */
  t?: (key: string, params?: Record<string, string | number>) => string;
}

async function getEpisodesByAsset(assetId: string) {
  return apiGet<Episode[]>(`/api/data-assets/episodes?assetId=${encodeURIComponent(assetId)}`);
}

async function getEpisodeByAsset(assetId: string, episodeId: string) {
  return apiGet<EpisodeInfo>(`/api/data-assets/episodes/${encodeURIComponent(episodeId)}?assetId=${encodeURIComponent(assetId)}`);
}

export default function EpisodeViewerLayout({
  source,
  header,
  rightPanel,
  missingContextMessage = '缺少资产或任务上下文',
  onNewAnnotation = () => {},
  onSave = () => {},
  taskName = '',
  onDescriptionUpdateRef,
  instructionsRefreshRef,
  episodesRefreshRef,
  isPreloading = false,
  preloadProgress = 0,
  onViewerStateChange,
  t: tProp,
}: EpisodeViewerLayoutProps) {
  const assetId = source.type === 'asset' ? source.assetId : undefined;
  const taskId = source.type === 'task' ? source.taskId : '';
  const { setCachedFrame } = useFrameCache();

  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [episodesError, setEpisodesError] = useState<string>('');

  const [selectedEpisode, setSelectedEpisode] = useState<string | null>(null);
  const [selectedEpisodeInfo, setSelectedEpisodeInfo] = useState<EpisodeInfo | null>(null);

  const [currentFrame, setCurrentFrame] = useState(0);
  /** HDF5 播放时由定时器推进，用于请求下一帧；进度条用 currentFrame（由视窗上报「已显示」同步） */
  const [requestedFrame, setRequestedFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  // 播放速度：1 = 正常速度（按 MCAP 真实时长推算 fps；无时间信息时约 24fps），支持 0.25~2 倍
  const [speed, setSpeed] = useState(1);
  const [maxFrame, setMaxFrame] = useState(0);
  const [loopPlayback, setLoopPlayback] = useState(true);
  const [viewportTopics, setViewportTopics] = useState<(string | null)[]>(() =>
    Array(3).fill(null) as (string | null)[]
  );

  const isUserDraggingSliderRef = useRef(false);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const requestedFrameRef = useRef(requestedFrame);
  requestedFrameRef.current = requestedFrame;

  const isMcap = selectedEpisodeInfo?.startTimeNs != null;
  /** MCAP 与 HDF5 均走 WebSocket 时由服务端推帧，进度条由 onFrameIndexChange 驱动，不跑本地定时器 */
  const useWebSocketPlayback = !!(taskId || assetId);

  // 依据 MCAP 的 startTimeNs/endTimeNs 与帧数，推算接近真实时间的基础 fps，使「30 秒数据 ≈ 播放 30 秒」
  const basePlaybackFps = (() => {
    if (isMcap && selectedEpisodeInfo?.startTimeNs != null && selectedEpisodeInfo?.endTimeNs != null && maxFrame > 1) {
      const durationSec =
        (selectedEpisodeInfo.endTimeNs - selectedEpisodeInfo.startTimeNs) / 1e9;
      if (durationSec > 0.1) {
        const approxFps = maxFrame / durationSec;
        return Math.max(5, Math.min(60, approxFps));
      }
    }
    // 没有时间信息时退化为一个合理的默认值
    return 24;
  })();

  // WebSocket 推流时只传 currentFrame；否则 HDF5 HTTP 用 requestedFrame 推进
  const frameToLoad = useWebSocketPlayback ? currentFrame : (isPlaying ? requestedFrame : currentFrame);
  const syncDisplayedFrame = (idx: number) => {
    if (isUserDraggingSliderRef.current) return;
    if (!useWebSocketPlayback && isPlaying && maxFrame > 10) {
      const req = requestedFrameRef.current;
      if (req <= 5 && idx >= maxFrame - 5) return;
    }
    setCurrentFrame((prev) => (isPlaying ? Math.max(prev, idx) : idx));
  };

  const refreshEpisodes = useCallback(async () => {
    if (source.type === 'asset' && assetId) {
      const res = await getEpisodesByAsset(assetId);
      if (res.ok && res.data) setEpisodes(res.data);
    } else if (source.type === 'task' && taskId) {
      const res = await getEpisodes(taskId);
      if (res.ok && res.data) setEpisodes(res.data);
    }
  }, [source.type, assetId, taskId]);

  useEffect(() => {
    if (!episodesRefreshRef) return;
    episodesRefreshRef.current = { refresh: refreshEpisodes };
    return () => {
      episodesRefreshRef.current = null;
    };
  }, [episodesRefreshRef, refreshEpisodes]);

  // 加载 episodes 列表
  useEffect(() => {
    if (source.type === 'asset') {
      if (!assetId) {
        setEpisodesError(tProp?.('labelExecutePage.alertNoDatasets') || '请选择数据资产');
        setEpisodesLoading(false);
        return;
      }
      setEpisodesLoading(true);
      setEpisodesError('');
      getEpisodesByAsset(assetId)
        .then((res) => {
          if (res.ok && res.data) setEpisodes(res.data);
          else {
            setEpisodes([]);
            setEpisodesError(res.error || (tProp?.('labelExecutePage.loadEpisodesFailed') ?? '加载失败'));
          }
        })
        .catch((e: unknown) => {
          setEpisodes([]);
          setEpisodesError((e as Error)?.message || (tProp?.('labelExecutePage.loadEpisodesFailedRetry') ?? '请重试'));
        })
        .finally(() => setEpisodesLoading(false));
    } else {
      if (!taskId) {
        setEpisodesError(tProp?.('labelExecutePage.missingTaskId') ?? '缺少任务 ID');
        setEpisodesLoading(false);
        return;
      }
      setEpisodesLoading(true);
      setEpisodesError('');
      getEpisodes(taskId)
        .then((res) => {
          if (res.ok && res.data) setEpisodes(res.data);
          else {
            setEpisodes([]);
            setEpisodesError(res.error || (tProp?.('labelExecutePage.loadEpisodesFailed') ?? '加载失败'));
          }
        })
        .catch((e: unknown) => {
          setEpisodes([]);
          setEpisodesError((e as Error)?.message || (tProp?.('labelExecutePage.loadEpisodesFailedRetry') ?? '请重试'));
        })
        .finally(() => setEpisodesLoading(false));
    }
  }, [source.type, assetId, taskId, tProp]);

  /** 任务切换时清空选中，避免沿用上一任务的 episode id */
  const prevTaskIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (source.type !== 'task' || !taskId) {
      prevTaskIdRef.current = null;
      return;
    }
    if (prevTaskIdRef.current === taskId) return;
    prevTaskIdRef.current = taskId;
    setSelectedEpisode(null);
    setSelectedEpisodeInfo(null);
    setCurrentFrame(0);
    setRequestedFrame(0);
    setIsPlaying(false);
    setMaxFrame(0);
  }, [source.type, taskId]);

  const prevPlayingRef = useRef(false);
  useEffect(() => {
    if (useWebSocketPlayback) return;
    if (isPlaying && !prevPlayingRef.current) {
      setRequestedFrame(currentFrame);
    }
    prevPlayingRef.current = !!isPlaying;
  }, [isPlaying, useWebSocketPlayback, currentFrame]);

  // 播放定时器：仅在不使用 WebSocket 时（无 taskId/assetId）推进 requestedFrame
  useEffect(() => {
    if (useWebSocketPlayback) return;
    if (isPlaying && maxFrame > 0) {
      const viewportCount = viewportTopics.filter(Boolean).length;
      const baseFps = viewportCount > 3 ? 12 : 24;
      const intervalMs = Math.max(33, Math.round(1000 / (baseFps * speed)));
      playIntervalRef.current = setInterval(() => {
        setRequestedFrame((prev) => {
          if (isUserDraggingSliderRef.current) return prev;
          if (maxFrame > 0 && prev >= maxFrame - 1) {
            if (loopPlayback) {
              setCurrentFrame(0);
              return 0;
            }
            setIsPlaying(false);
            return Math.max(0, maxFrame - 1);
          }
          return prev + 1;
        });
      }, intervalMs);
    } else {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, speed, maxFrame, useWebSocketPlayback, loopPlayback, viewportTopics]);

  useEffect(() => {
    if (maxFrame > 0 && currentFrame >= maxFrame) {
      setCurrentFrame(Math.max(0, maxFrame - 1));
    }
  }, [maxFrame, currentFrame]);
  useEffect(() => {
    if (maxFrame > 0 && requestedFrame >= maxFrame) {
      setRequestedFrame(Math.max(0, maxFrame - 1));
    }
  }, [maxFrame, requestedFrame]);

  // 数据资产查看：无 task 的 batch 接口，用逐帧 getAssetFrame 预加载前 N 帧到缓存，减轻 HDF5/MCAP 播放卡顿
  const PRELOAD_FRAMES = 80;
  const PRELOAD_CONCURRENCY = 3;
  useEffect(() => {
    if (source.type !== 'asset' || !assetId || !selectedEpisode || !selectedEpisodeInfo?.cameras?.length) return;
    const camera = selectedEpisodeInfo.cameras[0];
    const total = Math.min(selectedEpisodeInfo.frameCount ?? 0, PRELOAD_FRAMES);
    if (total <= 0) return;
    let cancelled = false;
    const run = async () => {
      for (let start = 0; start < total && !cancelled; start += PRELOAD_CONCURRENCY) {
        const batch = Array.from(
          { length: Math.min(PRELOAD_CONCURRENCY, total - start) },
          (_, i) => start + i
        );
        await Promise.all(
          batch.map(async (frame) => {
            if (cancelled) return;
            try {
              const blob = await getAssetFrame(assetId, selectedEpisode!, camera, frame, 72);
              if (cancelled) return;
              const url = URL.createObjectURL(blob);
              setCachedFrame(selectedEpisode!, camera, frame, url);
            } catch {
              // ignore single frame failure
            }
          })
        );
      }
    };
    run();
    return () => { cancelled = true; };
  }, [source.type, assetId, selectedEpisode, selectedEpisodeInfo?.cameras, selectedEpisodeInfo?.frameCount, setCachedFrame]);

  // 通知父组件当前视窗状态（标注页用于帧预加载与 RightPanel）
  useEffect(() => {
    onViewerStateChange?.({
      selectedEpisode,
      viewportTopics,
      maxFrame,
      taskId,
      cameras: selectedEpisodeInfo?.cameras ?? [],
    });
  }, [selectedEpisode, viewportTopics, maxFrame, taskId, selectedEpisodeInfo?.cameras, onViewerStateChange]);

  const loadEpisodeAndCameras = useCallback(
    async (episodeId: string) => {
      setSelectedEpisode(episodeId);
      setCurrentFrame(0);
      setRequestedFrame(0);
      setIsPlaying(false);
      try {
        if (source.type === 'asset' && assetId) {
          const res = await getEpisodeByAsset(assetId, episodeId);
          if (res.ok && res.data) {
            const ep = res.data;
            setSelectedEpisodeInfo(ep);
            setMaxFrame(ep.frameCount || 0);
            if (ep.cameras?.length) {
              setViewportTopics((prev) => {
                const len = prev.length;
                const cams = ep.cameras!.slice(0, len) as string[];
                return cams.concat(Array(len - cams.length).fill(null)) as (string | null)[];
              });
            }
          }
        } else if (source.type === 'task' && taskId) {
          const res = await getEpisode(episodeId, taskId);
          if (res.ok && res.data) {
            const ep = res.data;
            setSelectedEpisodeInfo(ep);
            setMaxFrame(ep.frameCount || 0);
            if (ep.cameras?.length) {
              setViewportTopics((prev) => {
                const len = prev.length;
                const cams = ep.cameras!.slice(0, len) as string[];
                return cams.concat(Array(len - cams.length).fill(null)) as (string | null)[];
              });
            }
          }
        }
      } catch (e) {
        console.error('Failed to load episode detail:', e);
      }
    },
    [source.type, assetId, taskId]
  );

  /** 标注任务进入执行页：列表加载完成后自动选中并拉取第一条 episode */
  useEffect(() => {
    if (source.type !== 'task' || !taskId) return;
    if (episodesLoading || episodesError) return;
    if (episodes.length === 0) return;
    const validSelected =
      selectedEpisode != null && episodes.some((e) => e.id === selectedEpisode);
    if (validSelected) return;
    void loadEpisodeAndCameras(episodes[0].id);
  }, [
    source.type,
    taskId,
    episodesLoading,
    episodesError,
    episodes,
    selectedEpisode,
    loadEpisodeAndCameras,
  ]);

  const handleSelectEpisode = (episodeId: string) => loadEpisodeAndCameras(episodeId);
  const handleDoubleClickEpisode = (episodeId: string) => loadEpisodeAndCameras(episodeId);
  const handleViewportTopicChange = (index: number, topic: string | null) => {
    const next = [...viewportTopics];
    next[index] = topic;
    setViewportTopics(next);
  };
  const handleAddSlot = () => {
    if (viewportTopics.length >= 9) return;
    setViewportTopics([...viewportTopics, null]);
  };
  const handleRemoveSlot = (index: number) => {
    if (viewportTopics.length > 3) {
      setViewportTopics(viewportTopics.filter((_, i) => i !== index));
    } else {
      const next = [...viewportTopics];
      next[index] = null;
      setViewportTopics(next);
    }
  };
  const handleRefresh = () => {
    setCurrentFrame(0);
    setRequestedFrame(0);
    setIsPlaying(false);
  };

  const hasContext = (source.type === 'asset' && !!assetId) || (source.type === 'task' && !!taskId);
  if (!hasContext) {
    return (
      <div style={{ padding: '24px' }}>
        <div
          style={{
            padding: '24px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            color: '#dc2626',
          }}
        >
          {missingContextMessage}
        </div>
      </div>
    );
  }

  const gridColumns = rightPanel ? '320px 1fr 360px' : '320px 1fr';
  const contentHeight = header ? 'calc(100% - 52px)' : '100%';

  return (
    <div
      style={{
        height: 'calc(100vh - 60px)',
        width: '100%',
        backgroundColor: '#f6f7f9',
      }}
    >
      {header && (
        <div
          style={{
            padding: '12px 24px',
            borderBottom: '1px solid #e5e7eb',
            backgroundColor: '#ffffff',
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 4 }}>
            {header.title}
          </div>
          {header.subtitle && (
            <div style={{ fontSize: 12, color: '#6b7280' }}>{header.subtitle}</div>
          )}
        </div>
      )}
      <div
        style={{
          height: contentHeight,
          display: 'grid',
          gridTemplateRows: '1fr 84px',
          position: 'relative',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: gridColumns,
            gap: 0,
            height: '100%',
            overflow: 'hidden',
          }}
        >
          <LeftPanel
            selectedEpisode={selectedEpisode}
            onSelectEpisode={handleSelectEpisode}
            onDoubleClickEpisode={handleDoubleClickEpisode}
            onNewAnnotation={onNewAnnotation}
            onSave={onSave}
            taskId={taskId}
            assetId={assetId}
            episodes={episodes}
            episodesLoading={episodesLoading}
            episodesError={episodesError}
            onDescriptionUpdateRef={onDescriptionUpdateRef ? (ref) => { onDescriptionUpdateRef.current = ref; } : undefined}
            instructionsRefreshRef={instructionsRefreshRef}
            onEpisodesRefresh={refreshEpisodes}
            showLabelActions={source.type === 'task'}
          />
          <CenterPanel
            viewportTopics={viewportTopics}
            onViewportTopicChange={handleViewportTopicChange}
            onAddSlot={handleAddSlot}
            onRemoveSlot={handleRemoveSlot}
            currentFrame={frameToLoad}
            selectedEpisode={selectedEpisode}
            selectedEpisodeInfo={selectedEpisodeInfo}
            taskId={taskId || undefined}
            assetId={assetId}
            isPlaying={isPlaying}
            currentTimeSec={
              selectedEpisodeInfo?.startTimeNs != null &&
              selectedEpisodeInfo?.endTimeNs != null &&
              maxFrame > 0
                ? ((selectedEpisodeInfo.endTimeNs - selectedEpisodeInfo.startTimeNs) * currentFrame) /
                  Math.max(1, maxFrame) /
                  1e9
                : undefined
            }
            loopPlayback={loopPlayback}
            onPlaybackLooped={() => { setCurrentFrame(0); setRequestedFrame(0); }}
            onFrameIndexChange={syncDisplayedFrame}
            // WebSocket 推流帧率：按数据真实时长推算基础 fps，随速度成比例调整，限制在 8~60fps 之间
            playbackFps={Math.max(8, Math.min(60, Math.round(basePlaybackFps * speed)))}
          />
          {rightPanel}
        </div>
        {isPreloading && (
          <div
            style={{
              position: 'absolute',
              bottom: '96px',
              left: '50%',
              transform: 'translateX(-50%)',
              padding: '6px 12px',
              backgroundColor: 'rgba(0,0,0,0.7)',
              color: '#fff',
              fontSize: '12px',
              borderRadius: '6px',
              zIndex: 10,
            }}
          >
            {tProp?.('labelExecutePage.preloadingFrames', { n: preloadProgress }) ?? `预加载帧: ${preloadProgress}`}
          </div>
        )}
        <BottomTimeline
          currentFrame={currentFrame}
          maxFrame={maxFrame}
          isPlaying={isPlaying}
          speed={speed}
          onFrameChange={(v) => { setCurrentFrame(v); setRequestedFrame(v); }}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onSliderDragStart={() => { isUserDraggingSliderRef.current = true; }}
          onSliderDragEnd={() => { isUserDraggingSliderRef.current = false; }}
          onPreviousFrame={() => {
            const next = Math.max(0, currentFrame - 1);
            setCurrentFrame(next);
            setRequestedFrame(next);
            setIsPlaying(false);
          }}
          onNextFrame={() => {
            const next = Math.min(maxFrame - 1, currentFrame + 1);
            setCurrentFrame(next);
            setRequestedFrame(next);
            setIsPlaying(false);
          }}
          onSpeedChange={setSpeed}
          onRefresh={handleRefresh}
          startTimeNs={selectedEpisodeInfo?.startTimeNs}
          endTimeNs={selectedEpisodeInfo?.endTimeNs}
          loopPlayback={loopPlayback}
          onLoopPlaybackChange={setLoopPlayback}
        />
      </div>
    </div>
  );
}
