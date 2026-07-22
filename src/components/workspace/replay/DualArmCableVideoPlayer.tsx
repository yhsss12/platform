'use client';

import { useEffect, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { buildDualArmCableVideoApiPath } from '@/lib/api/dualArmCableClient';

export function DualArmCableVideoPlayer({ videoJobId }: { videoJobId: string }) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);
      setSrc(null);

      const token = getAccessToken();
      if (!token) {
        setError('未登录，无法加载视频。');
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(buildDualArmCableVideoApiPath(videoJobId), {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
          setError(response.status === 404 ? '视频文件不存在。' : `视频加载失败（HTTP ${response.status}）`);
          setLoading(false);
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
          setError('视频加载失败，请稍后重试。');
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [videoJobId]);

  if (loading) {
    return (
      <div style={{ aspectRatio: '16 / 9', background: '#0f172a', borderRadius: 14, color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        正在加载过程视频…
      </div>
    );
  }

  if (error || !src) {
    return (
      <div style={{ aspectRatio: '16 / 9', background: '#0f172a', borderRadius: 14, color: '#fca5a5', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, textAlign: 'center' }}>
        {error ?? '无法播放视频'}
      </div>
    );
  }

  return (
    <video
      src={src}
      controls
      style={{
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        borderRadius: 14,
        backgroundColor: '#0f172a',
      }}
    />
  );
}
