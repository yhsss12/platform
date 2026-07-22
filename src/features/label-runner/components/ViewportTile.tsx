'use client';

import { useState, useEffect, useRef, memo } from 'react';
import { getFrameQueued, getAssetFrame } from '../api/labelApi';
import { useFrameCache } from '../context/FrameCacheContext';
import { useMcapPlaybackWs } from '../hooks/useMcapPlaybackWs';

interface ViewportTileProps {
  topic: string | null;
  onTopicChange: (topic: string | null) => void;
  currentFrame: number;
  onClose?: () => void;
  showClose?: boolean;
  selectedEpisode: string | null;
  cameras: string[];
  taskId?: string;
  /** 按数据资产查看时传入，用于走 /api/data-assets/frames */
  assetId?: string;
  /** 是否为 MCAP 格式（用于 UI，如时间角标）；MCAP 与 HDF5 均使用 WebSocket 推流 */
  isMcap?: boolean;
  /** 是否正在播放 */
  isPlaying?: boolean;
  /** 当前时间（秒），MCAP 时用于角标显示 */
  currentTimeSec?: number;
  /** MCAP WebSocket 收到帧时回调，用于更新时间轴 */
  onFrameIndexChange?: (index: number) => void;
  /** 播放帧率（MCAP） */
  playbackFps?: number;
  /** 是否循环播放 */
  loopPlayback?: boolean;
  /** MCAP 播放到底后循环时，用于将 currentFrame 置 0 */
  onPlaybackLooped?: () => void;
}

function ViewportTile({
  topic,
  onTopicChange,
  currentFrame,
  onClose,
  showClose = false,
  selectedEpisode,
  cameras,
  taskId,
  assetId,
  isMcap = false,
  isPlaying = false,
  currentTimeSec,
  onFrameIndexChange,
  playbackFps = 15,
  loopPlayback = true,
  onPlaybackLooped,
}: ViewportTileProps) {
  const { getCached, setCachedFrame } = useFrameCache();
  const getCachedRef = useRef(getCached);
  getCachedRef.current = getCached;

  const [frameImage, setFrameImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  /** 帧加载失败时的友好提示（如该相机无可用帧） */
  const [frameErrorHint, setFrameErrorHint] = useState<string | null>(null);
  /** 请求完成后 +1，用于在「仅允许 1 个请求在飞」时再次触发拉取当前帧，避免大量并发导致 Failed to fetch */
  const [loadTick, setLoadTick] = useState(0);
  const currentFrameRef = useRef(currentFrame);
  const displayedUrlRef = useRef<string | null>(null);
  const sessionRef = useRef(0);
  const lastDisplayedFrameRef = useRef(-1);
  const controllerRef = useRef<AbortController | null>(null);
  const inFlightRef = useRef(false);
  const lastFrameToLoadRef = useRef(-1);

  currentFrameRef.current = currentFrame;

  // MCAP 与 HDF5 均使用 WebSocket 推流（标注页 taskId，数据查看页 assetId），协议一致
  const usePlaybackWs = !!topic && !!selectedEpisode && (!!taskId || !!assetId) && cameras.includes(topic || '');
  const wsResult = useMcapPlaybackWs(
    usePlaybackWs ? selectedEpisode : null,
    usePlaybackWs ? topic : null,
    usePlaybackWs ? taskId ?? null : null,
    isPlaying,
    currentFrame,
    playbackFps,
    loopPlayback,
    usePlaybackWs ? assetId ?? null : null
  );

  useEffect(() => {
    if (!usePlaybackWs) return;
    wsResult.setOnFrameIndex(onFrameIndexChange ?? null);
    wsResult.setOnLoopToStart(onPlaybackLooped ?? null);
  }, [usePlaybackWs, onFrameIndexChange, onPlaybackLooped]);

  // 未使用 WebSocket 时（无 taskId/assetId）：HTTP 拉取 + 缓存
  const useHttp = !usePlaybackWs;
  useEffect(() => {
    sessionRef.current += 1;
    inFlightRef.current = false;
    lastFrameToLoadRef.current = -1;
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, [topic, selectedEpisode, taskId, assetId]);

  useEffect(() => {
    if (!useHttp) return;
    const hasTaskOrAsset = !!taskId || !!assetId;
    if (!topic || !selectedEpisode || !cameras.includes(topic) || !hasTaskOrAsset) {
      setFrameImage(null);
      setFrameErrorHint(null);
      setLoading(false);
      lastDisplayedFrameRef.current = -1;
      if (displayedUrlRef.current) {
        URL.revokeObjectURL(displayedUrlRef.current);
        displayedUrlRef.current = null;
      }
      return;
    }

    const episodeId = selectedEpisode;
    const camera = topic;
    const frameToLoad = currentFrame;
    const mySession = sessionRef.current;
    if (frameToLoad <= 5 && lastFrameToLoadRef.current > 100) {
      controllerRef.current?.abort();
      controllerRef.current = null;
      inFlightRef.current = false;
    }
    lastFrameToLoadRef.current = frameToLoad;

    const cached = getCachedRef.current(episodeId, camera, frameToLoad);
    if (cached) {
      setFrameErrorHint(null);
      lastDisplayedFrameRef.current = frameToLoad;
      if (displayedUrlRef.current) {
        URL.revokeObjectURL(displayedUrlRef.current);
        displayedUrlRef.current = null;
      }
      setFrameImage(cached);
      setLoading(false);
      return;
    }

    // 每视窗只允许 1 个请求在飞，避免 MCAP/多视窗播放时大量并发导致 Failed to fetch
    if (inFlightRef.current) return;

    inFlightRef.current = true;
    if (!isPlaying) setLoading(true);
    const quality = isPlaying ? 72 : 85;
    const controller = new AbortController();
    controllerRef.current = controller;
    getFramePromise()
      .then((blob) => {
        if (sessionRef.current !== mySession) return;
        if (isPlaying) {
          if (frameToLoad < lastDisplayedFrameRef.current) return;
        } else {
          if (currentFrameRef.current !== frameToLoad) return;
        }
        lastDisplayedFrameRef.current = frameToLoad;
        const imageUrl = URL.createObjectURL(blob);
        if (displayedUrlRef.current) {
          URL.revokeObjectURL(displayedUrlRef.current);
        }
        displayedUrlRef.current = imageUrl;
        setFrameErrorHint(null);
        setFrameImage(imageUrl);
        setCachedFrame(episodeId, camera, frameToLoad, imageUrl);
        onFrameIndexChange?.(frameToLoad);
      })
      .catch((err) => {
        if (sessionRef.current !== mySession) return;
        if (err?.name === 'AbortError') return;
        const msg = (err?.message || String(err)) as string;
        const isNoFrames = /has no frames|无帧|无可用帧/i.test(msg);
        if (isNoFrames) {
          setFrameErrorHint('该相机无可用帧');
        } else {
          setFrameErrorHint(null);
          console.error('Failed to load frame:', err);
        }
        if (currentFrameRef.current === frameToLoad) setFrameImage(null);
      })
      .finally(() => {
        if (sessionRef.current !== mySession) return;
        inFlightRef.current = false;
        controllerRef.current = null;
        if (currentFrameRef.current === frameToLoad || isPlaying) setLoading(false);
        // 仅当已落后于当前帧时再触发拉取，避免重复请求同一帧
        if (currentFrameRef.current !== frameToLoad) setLoadTick((t) => t + 1);
      });

    function getFramePromise(): Promise<Blob> {
      if (taskId) {
        return getFrameQueued(episodeId, camera, frameToLoad, quality, taskId, controller.signal as any);
      }
      if (assetId) {
        return getAssetFrame(assetId, episodeId, camera, frameToLoad, quality, controller.signal);
      }
      return Promise.reject(new Error('缺少任务或资产上下文，无法加载帧'));
    }

    return () => {};
  }, [useHttp, topic, selectedEpisode, currentFrame, cameras, taskId, assetId, isPlaying, loadTick]);

  const wsErr = usePlaybackWs ? wsResult.connectError : null;
  const showImage = usePlaybackWs ? !!wsResult.frameImage : !!frameImage;
  const showLoadingPlaceholder = usePlaybackWs
    ? !wsErr && !wsResult.connected && !wsResult.frameImage
    : loading && !frameImage && !isPlaying;
  const imgSrc = usePlaybackWs ? wsResult.frameImage : frameImage;

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#ffffff',
        border: '1px solid #e5e7eb',
        position: 'relative',
      }}
    >
      {/* Topic 下拉 */}
      <div style={{ padding: '8px', borderBottom: '1px solid #e5e7eb' }}>
        <select
          value={topic || ''}
          onChange={(e) => onTopicChange(e.target.value || null)}
          style={{
            width: '100%',
            height: '32px',
            padding: '0 28px 0 8px',
            backgroundColor: '#ffffff',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            color: '#111827',
            fontSize: '12px',
            outline: 'none',
            cursor: 'pointer',
            appearance: 'none',
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M6 9L1 4h10z'/%3E%3C/svg%3E")`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'right 8px center',
            textOverflow: 'ellipsis',
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            transition: 'all 0.2s',
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = '#2563eb';
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = '#d1d5db';
          }}
        >
          <option value="">选择相机</option>
          {(() => {
            const groups: Record<string, string[]> = {};
            for (const c of cameras) {
              const parts = c.split('/').filter(Boolean);
              const first = (parts[0] || c).trim() || c;
              if (!groups[first]) groups[first] = [];
              groups[first].push(c);
            }
            const entries = Object.entries(groups);
            if (entries.length <= 1 && entries[0]?.[1].length <= 8) {
              return cameras.map((camera) => (
                <option key={camera} value={camera}>
                  {camera}
                </option>
              ));
            }
            return entries.map(([label, list]) => (
              <optgroup key={label} label={label}>
                {list.map((camera) => (
                  <option key={camera} value={camera}>
                    {camera}
                  </option>
                ))}
              </optgroup>
            ));
          })()}
        </select>
      </div>

      {/* 视窗主体 */}
      <div
        style={{
          flex: 1,
          backgroundColor: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {topic ? (
          wsErr ? (
            <div
              style={{
                width: '100%',
                height: '100%',
                padding: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#b45309',
                fontSize: '12px',
                textAlign: 'center',
                lineHeight: 1.5,
              }}
            >
              {wsErr}
            </div>
          ) : showLoadingPlaceholder ? (
            <div
              style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#6b7280',
                fontSize: '12px',
              }}
            >
              加载中...
            </div>
          ) : showImage && imgSrc ? (
            <img
              key={`viewport-${topic}`}
              src={imgSrc}
              alt={isMcap ? 'Camera' : `Frame ${currentFrame}`}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                backgroundColor: '#000000',
                display: 'block',
              }}
            />
          ) : (
            <div
              style={{
                width: '100%',
                height: '100%',
                backgroundColor: '#f9fafb',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundImage: 'linear-gradient(45deg, #f3f4f6 25%, transparent 25%), linear-gradient(-45deg, #f3f4f6 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f3f4f6 75%), linear-gradient(-45deg, transparent 75%, #f3f4f6 75%)',
                backgroundSize: '20px 20px',
                backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px',
              }}
            >
              <div
                style={{
                  textAlign: 'center',
                  color: '#6b7280',
                  fontSize: '12px',
                }}
              >
                <div style={{ marginBottom: '8px' }}>
                  {frameErrorHint || '无图像'}
                </div>
                <div style={{ fontSize: '11px', color: '#9ca3af' }}>{topic}</div>
              </div>
            </div>
          )
        ) : (
          <div
            style={{
              textAlign: 'center',
              color: '#9ca3af',
              fontSize: '12px',
            }}
          >
            请选择相机
          </div>
        )}

        {/* 角标：HDF5 显示 Frame，MCAP 显示时间（秒）或隐藏 */}
        {topic && (
          <div
            style={{
              position: 'absolute',
              top: '8px',
              right: '8px',
              padding: '4px 8px',
              backgroundColor: 'rgba(0, 0, 0, 0.6)',
              color: '#ffffff',
              borderRadius: '4px',
              fontSize: '11px',
            }}
          >
            {isMcap && typeof currentTimeSec === 'number'
              ? `${currentTimeSec.toFixed(1)}s`
              : !isMcap
              ? `Frame: ${currentFrame}`
              : null}
          </div>
        )}

        {/* 关闭按钮 */}
        {showClose && onClose && (
          <button
            onClick={onClose}
            style={{
              position: 'absolute',
              top: '8px',
              left: '8px',
              width: '24px',
              height: '24px',
              backgroundColor: 'rgba(0, 0, 0, 0.6)',
              color: '#ffffff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '14px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';
            }}
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}

export default memo(ViewportTile);
