'use client';

import { useState, useEffect, useRef } from 'react';
import { useFrameCache } from '../context/FrameCacheContext';
import { getAccessToken } from '@/lib/auth/session';

function isLoopbackHost(hostname: string): boolean {
  const h = (hostname || '').toLowerCase();
  return h === 'localhost' || h === '127.0.0.1' || h === '::1' || h === '[::1]';
}

/**
 * 播放 WS 使用的 HTTP(S) origin。
 * 典型坑：NEXT_PUBLIC_API_URL=http://127.0.0.1:8000，但从其它电脑用 http://192.168.x.x:3000 打开前端，
 * 浏览器会去连「自己电脑」的 127.0.0.1:8000 → 1006。此处若在局域网访问且 env 为 loopback，则改用当前页面的 hostname，保留端口。
 */
function resolvePlaybackHttpOrigin(apiBaseRaw: string): string {
  if (typeof window === 'undefined') return (apiBaseRaw || '').trim();
  const pageHost = window.location.hostname;
  const pageLoopback = isLoopbackHost(pageHost);
  const portFallback = (process.env.NEXT_PUBLIC_API_PORT || '').trim() || '8000';

  const trimmed = apiBaseRaw.trim();
  if (!trimmed) {
    const proto = window.location.protocol === 'https:' ? 'https:' : 'http:';
    return `${proto}//${pageHost}:${portFallback}`;
  }
  const u = new URL(trimmed);
  if (isLoopbackHost(u.hostname) && !pageLoopback) {
    u.hostname = pageHost;
  }
  return u.origin;
}

/** 方案1：WebSocket 推流播放，服务端控制节奏。标注页用 taskId，数据查看页用 assetId，同一套协议。 */
export function useMcapPlaybackWs(
  episodeId: string | null,
  camera: string | null,
  taskId: string | null,
  isPlaying: boolean,
  seekFrame: number,
  fps: number,
  loop: boolean = true,
  assetId?: string | number | null
) {
  const { getCached, setCachedFrame } = useFrameCache();
  const [frameImage, setFrameImage] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  /** 握手失败 / 被服务端关闭 / 无 token 等，避免界面永远停在「加载中」 */
  const [connectError, setConnectError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const openedRef = useRef(false);
  const displayedUrlRef = useRef<string | null>(null);
  const lastSeekRef = useRef(-1);
  const onFrameIndexRef = useRef<((index: number) => void) | null>(null);
  const seekFrameRef = useRef(seekFrame);
  const isPlayingRef = useRef(isPlaying);
  const fpsRef = useRef(fps);
  seekFrameRef.current = seekFrame;
  isPlayingRef.current = isPlaying;
  fpsRef.current = fps;
  const loopRef = useRef(loop);
  loopRef.current = loop;
  const loopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setOnFrameIndex = (cb: ((index: number) => void) | null) => {
    onFrameIndexRef.current = cb;
  };
  const onLoopToStartRef = useRef<(() => void) | null>(null);
  const setOnLoopToStart = (cb: (() => void) | null) => {
    onLoopToStartRef.current = cb;
  };

  const hasContext = !!taskId || (assetId != null && assetId !== '');
  useEffect(() => {
    if (!episodeId || !camera || !hasContext) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setConnected(false);
      }
      setConnectError(null);
      openedRef.current = false;
      setFrameImage(null);
      if (displayedUrlRef.current) {
        URL.revokeObjectURL(displayedUrlRef.current);
        displayedUrlRef.current = null;
      }
      return;
    }

    const params = taskId
      ? new URLSearchParams({ camera, taskId })
      : new URLSearchParams({ camera, assetId: String(assetId) });
    // WebSocket 无法带 Header，后端要求 token 放在 query（routes_label.get_current_user_ws）
    const token = typeof window !== 'undefined' ? getAccessToken() : null;
    if (!token) {
      setConnectError('未登录或访问令牌缺失，无法连接播放服务（请重新登录）');
      setConnected(false);
      openedRef.current = false;
      return;
    }
    params.set('token', token);

    const basePath = taskId ? '/api/label/ws/playback' : '/api/data-assets/ws/playback';
    const path = `${basePath}/${episodeId}?${params}`;
    const apiBaseRaw = typeof window !== 'undefined' ? (process.env.NEXT_PUBLIC_API_URL || '').trim() : '';

    let httpOrigin: string;
    try {
      if (
        apiBaseRaw &&
        !apiBaseRaw.startsWith('http://') &&
        !apiBaseRaw.startsWith('https://')
      ) {
        throw new Error('bad_scheme');
      }
      httpOrigin = resolvePlaybackHttpOrigin(apiBaseRaw);
      new URL(httpOrigin);
    } catch {
      setConnectError(
        apiBaseRaw
          ? `NEXT_PUBLIC_API_URL 无效：${apiBaseRaw.slice(0, 96)}… 请使用完整 URL（如 http://127.0.0.1:8000）`
          : '无法解析后端地址，请设置 NEXT_PUBLIC_API_URL',
      );
      setConnected(false);
      openedRef.current = false;
      return;
    }

    const wsUrl = `${httpOrigin.replace(/^http/, 'ws').replace(/\/$/, '')}${path}`;

    setConnectError(null);
    openedRef.current = false;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      openedRef.current = true;
      setConnectError(null);
      setConnected(true);
      const frame = seekFrameRef.current;
      lastSeekRef.current = frame;
      ws.send(JSON.stringify({ action: 'seek', frame }));
      if (isPlayingRef.current) {
        ws.send(JSON.stringify({ action: 'play', fps: fpsRef.current }));
      }
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data === 'string') {
        try {
          const j = JSON.parse(ev.data);
          if (j.type === 'ended') {
            if (loopRef.current) {
              lastSeekRef.current = 0;
              onFrameIndexRef.current?.(j.frame ?? 0);
              if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current);
              const delayMs = 280;
              loopTimeoutRef.current = setTimeout(() => {
                loopTimeoutRef.current = null;
                ws.send(JSON.stringify({ action: 'seek', frame: 0 }));
                ws.send(JSON.stringify({ action: 'play', fps: fpsRef.current }));
                onLoopToStartRef.current?.();
                onFrameIndexRef.current?.(0);
              }, delayMs);
            } else {
              onFrameIndexRef.current?.(j.frame ?? 0);
            }
          }
        } catch {
          // ignore
        }
        return;
      }
      const buf = ev.data as ArrayBuffer;
      if (buf.byteLength < 4) return;
      const view = new DataView(buf);
      const index = view.getUint32(0, false);
      const jpeg = new Uint8Array(buf, 4);
      const blob = new Blob([jpeg], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      if (displayedUrlRef.current) {
        URL.revokeObjectURL(displayedUrlRef.current);
      }
      displayedUrlRef.current = url;
      setFrameImage(url);
      onFrameIndexRef.current?.(index);
      if (episodeId && camera) {
        setCachedFrame(episodeId, camera, index, url);
      }
    };

    ws.onclose = (ev: CloseEvent) => {
      setConnected(false);
      if (!openedRef.current) {
        const code = ev.code;
        let hint = '播放 WebSocket 未连接成功。';
        if (code === 1008) hint += ' 可能被服务端拒绝（权限或任务不可见）。';
        else if (code === 4404) hint += ' 服务端返回 4404：文件不存在或该相机无帧。';
        else if (code === 1006 || code === 0)
          hint +=
            ' 常见原因：NEXT_PUBLIC_API_URL 指向本机 127.0.0.1，但从其它电脑访问前端；或后端未启动、端口错误。';
        setConnectError(`${hint}（关闭码 ${code}）`);
      }
      setFrameImage(null);
      if (displayedUrlRef.current) {
        URL.revokeObjectURL(displayedUrlRef.current);
        displayedUrlRef.current = null;
      }
      openedRef.current = false;
    };

    ws.onerror = () => {
      setConnected(false);
      // 具体原因多在 onclose；此处避免重复覆盖文案
      if (!openedRef.current) {
        setConnectError((prev) => prev ?? '播放 WebSocket 连接出错，请检查网络与后端地址');
      }
    };

    return () => {
      if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = null;
      ws.close();
      wsRef.current = null;
    };
  }, [episodeId, camera, taskId, assetId, hasContext]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (isPlaying) {
      ws.send(JSON.stringify({ action: 'play', fps }));
    } else {
      ws.send(JSON.stringify({ action: 'pause' }));
    }
  }, [isPlaying, fps]);

  useEffect(() => {
    if (!episodeId || !camera || !hasContext) return;
    if (isPlaying) return;
    if (seekFrame === lastSeekRef.current) return;
    const cached = getCached(episodeId, camera, seekFrame);
    if (cached) {
      lastSeekRef.current = seekFrame;
      if (displayedUrlRef.current) {
        URL.revokeObjectURL(displayedUrlRef.current);
      }
      displayedUrlRef.current = cached;
      setFrameImage(cached);
      onFrameIndexRef.current?.(seekFrame);
    }
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    lastSeekRef.current = seekFrame;
    ws.send(JSON.stringify({ action: 'seek', frame: seekFrame }));
  }, [episodeId, camera, taskId, assetId, hasContext, seekFrame, isPlaying, getCached]);

  return { frameImage, connected, connectError, setOnFrameIndex, setOnLoopToStart };
}
