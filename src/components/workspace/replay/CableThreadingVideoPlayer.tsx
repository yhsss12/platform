'use client';

import { useEffect, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { resolveReplayVideoApiPath } from '@/lib/workspace/evaluationReplayInfo';

interface CableThreadingVideoPlayerProps {
  videoJobId: string;
  videoApiPath?: string | null;
  onCurrentTimeChange?: (timeSec: number) => void;
}

export function CableThreadingVideoPlayer({
  videoJobId,
  videoApiPath,
  onCurrentTimeChange,
}: CableThreadingVideoPlayerProps) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const resolvedApiPath =
    videoApiPath ??
    `/api/workspace/cable-threading/jobs/${encodeURIComponent(videoJobId)}/video`;

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);
      setSrc(null);

      const token = getAccessToken();
      if (!token) {
        setError('未登录，无法加载视频。请重新登录后重试。');
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(resolveReplayVideoApiPath(resolvedApiPath), {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
          const detail =
            response.status === 404
              ? '视频文件不存在，请重新生成视频。'
              : `视频加载失败（HTTP ${response.status}）`;
          if (!cancelled) {
            setError(detail);
            setLoading(false);
          }
          return;
        }
        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        if (!cancelled) {
          setSrc(objectUrl);
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setError('视频加载失败，请检查网络或稍后重试。');
          setLoading(false);
        }
      }
    };

    void load();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [resolvedApiPath]);

  const containerStyle: React.CSSProperties = {
    width: '100%',
    height: '100%',
    minHeight: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  if (loading) {
    return (
      <div style={containerStyle}>
        <div style={{ color: '#94a3b8', fontSize: 14 }}>正在加载视频…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ ...containerStyle, padding: 16 }}>
        <div
          style={{
            padding: '14px 16px',
            borderRadius: 10,
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#991b1b',
            fontSize: 13,
            lineHeight: 1.55,
            width: '100%',
          }}
        >
          {error}
        </div>
      </div>
    );
  }

  if (!src) return null;

  return (
    <video
      key={resolvedApiPath}
      controls
      src={src}
      onTimeUpdate={(event) => {
        onCurrentTimeChange?.(event.currentTarget.currentTime);
      }}
      onSeeked={(event) => {
        onCurrentTimeChange?.(event.currentTarget.currentTime);
      }}
      onEnded={(event) => {
        onCurrentTimeChange?.(event.currentTarget.duration);
      }}
      onLoadedMetadata={(event) => {
        onCurrentTimeChange?.(event.currentTarget.currentTime);
      }}
      style={{
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        backgroundColor: '#0f172a',
      }}
    />
  );
}
