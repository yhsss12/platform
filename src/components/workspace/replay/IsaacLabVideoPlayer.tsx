'use client';

import { useEffect, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { getIsaacLabJobVideoUrl } from '@/lib/api/isaacLabClient';

interface IsaacLabVideoPlayerProps {
  videoJobId: string;
  onPlaybackError?: (message: string) => void;
  onReady?: () => void;
}

function describeMediaError(error: MediaError | null): string {
  if (!error) return '视频解码失败，请尝试重新生成回放视频。';
  switch (error.code) {
    case MediaError.MEDIA_ERR_ABORTED:
      return '视频加载被中断。';
    case MediaError.MEDIA_ERR_NETWORK:
      return '视频网络加载失败。';
    case MediaError.MEDIA_ERR_DECODE:
      return '回放视频编码不兼容或文件损坏，请尝试重新生成回放视频。';
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return '浏览器不支持该回放视频格式，请尝试重新生成回放视频。';
    default:
      return '视频播放失败，请尝试重新生成回放视频。';
  }
}

export function IsaacLabVideoPlayer({
  videoJobId,
  onPlaybackError,
  onReady,
}: IsaacLabVideoPlayerProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [transcoded, setTranscoded] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;

    const fail = (message: string) => {
      if (cancelled) return;
      setError(message);
      setLoading(false);
      onPlaybackError?.(message);
    };

    const load = async () => {
      setLoading(true);
      setError(null);
      setSrc(null);
      setTranscoded(false);

      const token = getAccessToken();
      if (!token) {
        fail('未登录，无法加载视频。请重新登录后重试。');
        return;
      }

      try {
        const response = await fetch(getIsaacLabJobVideoUrl(videoJobId), {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
          let detail = `视频加载失败（HTTP ${response.status}）`;
          try {
            const payload = (await response.json()) as { detail?: string };
            if (payload.detail) detail = payload.detail;
          } catch {
            /* not json */
          }
          fail(response.status === 404 ? '视频文件不存在，请重新生成视频。' : detail);
          return;
        }

        if (response.headers.get('X-Isaac-Video-Transcoded') === '1') {
          setTranscoded(true);
        }

        const contentType = response.headers.get('Content-Type') ?? '';
        const blob = await response.blob();
        if (blob.size <= 0) {
          fail('视频文件为空，请重新生成回放视频。');
          return;
        }
        if (contentType && !contentType.startsWith('video/') && !blob.type.startsWith('video/')) {
          fail('视频接口返回了非视频内容，请检查登录状态或稍后重试。');
          return;
        }

        const videoBlob =
          blob.type.startsWith('video/') || !contentType.startsWith('video/')
            ? blob
            : new Blob([blob], { type: contentType || 'video/mp4' });

        objectUrl = URL.createObjectURL(videoBlob);
        if (!cancelled) {
          setSrc(objectUrl);
          setLoading(false);
        }
      } catch {
        fail('视频加载失败，请检查网络或稍后重试。');
      }
    };

    void load();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [videoJobId, onPlaybackError]);

  if (loading) {
    return (
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
        正在加载视频…
      </div>
    );
  }

  if (error) {
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
        {error}
      </div>
    );
  }

  if (!src) return null;

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {transcoded ? (
        <div
          style={{
            padding: '6px 10px',
            fontSize: 11,
            color: '#94a3b8',
            textAlign: 'center',
          }}
        >
          回放视频编码不兼容，已自动转码为浏览器可播放格式。
        </div>
      ) : null}
      <video
        controls
        src={src}
        onLoadedMetadata={(event) => {
          const duration = event.currentTarget.duration;
          if (!Number.isFinite(duration) || duration <= 0) {
            const message = '回放视频时长异常，可能无法播放。';
            setError(message);
            onPlaybackError?.(message);
            return;
          }
          onReady?.();
        }}
        onCanPlay={() => onReady?.()}
        onError={(event) => {
          const message = describeMediaError(event.currentTarget.error);
          setError(message);
          onPlaybackError?.(message);
        }}
        style={{
          width: '100%',
          flex: 1,
          minHeight: 0,
          objectFit: 'contain',
          backgroundColor: '#0f172a',
        }}
      />
    </div>
  );
}
