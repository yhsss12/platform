'use client';

import { useEffect, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { SimulationViewportMessage } from '@/components/workspace/simulation/SimulationViewport';

function describeMediaError(error: MediaError | null): string {
  if (!error) return '视频编码可能不受浏览器支持，请下载后本地播放。';
  switch (error.code) {
    case MediaError.MEDIA_ERR_ABORTED:
      return '视频加载被中断。';
    case MediaError.MEDIA_ERR_NETWORK:
      return '视频网络加载失败。';
    case MediaError.MEDIA_ERR_DECODE:
      return '视频编码不支持，请下载后本地播放。';
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return '当前浏览器无法播放该视频。';
    default:
      return '视频播放失败。';
  }
}

export function RunConsoleVideoPlayer({
  apiPath,
  title = 'generate.mp4',
}: {
  apiPath: string;
  title?: string;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;

    const fail = (message: string) => {
      if (cancelled) return;
      setError(message);
      setLoading(false);
    };

    const load = async () => {
      setLoading(true);
      setError(null);
      setSrc(null);

      const token = getAccessToken();
      if (!token) {
        fail('未登录，无法加载视频。请重新登录后重试。');
        return;
      }

      try {
        const response = await fetch(apiPath, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
          fail(
            response.status === 404
              ? '视频文件不存在。'
              : `视频文件存在但无法访问（HTTP ${response.status}）。`
          );
          return;
        }
        const blob = await response.blob();
        if (cancelled) return;
        if (!blob.size) {
          fail('视频文件为空。');
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
        setLoading(false);
      } catch {
        fail('视频加载失败，请检查网络或后端 artifact 路由。');
      }
    };

    void load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [apiPath]);

  if (loading) {
    return <SimulationViewportMessage>正在加载 {title}…</SimulationViewportMessage>;
  }

  if (error) {
    return <SimulationViewportMessage>{error}</SimulationViewportMessage>;
  }

  if (!src) {
    return <SimulationViewportMessage>后端未返回可播放的视频 URL。</SimulationViewportMessage>;
  }

  return (
    <video
      src={src}
      controls
      autoPlay
      muted
      playsInline
      style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#111827' }}
      onError={(event) => {
        const mediaError = (event.currentTarget as HTMLVideoElement).error;
        setError(describeMediaError(mediaError));
        setSrc(null);
      }}
    />
  );
}
