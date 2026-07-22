'use client';

import { useEffect, useRef, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { buildDualArmCableFrameApiPath } from '@/lib/api/dualArmCableClient';
import {
  SimulationViewportImage,
  SimulationViewportMessage,
  SimulationViewportPlaceholder,
} from '@/components/workspace/simulation/SimulationViewport';

export type DualArmCableFrameJobStatus = 'queued' | 'running' | 'completed' | 'failed';

export function DualArmCableLiveFrame({
  jobId,
  status,
  liveFrameSeq,
  liveFrameUpdatedAt,
  onFrameReadyChange,
}: {
  jobId?: string;
  status: DualArmCableFrameJobStatus;
  phase?: string | null;
  liveFrameSeq?: number | null;
  liveFrameUpdatedAt?: string | null;
  onFrameReadyChange?: (ready: boolean) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const lastFrameSignatureRef = useRef<string | null>(null);

  const revokeObjectUrl = () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  };

  const applyBlob = (blob: Blob) => {
    revokeObjectUrl();
    const url = URL.createObjectURL(blob);
    objectUrlRef.current = url;
    setSrc(url);
  };

  useEffect(() => {
    revokeObjectUrl();
    setSrc(null);
    lastFrameSignatureRef.current = null;
    onFrameReadyChange?.(false);
  }, [jobId, onFrameReadyChange]);

  useEffect(() => () => revokeObjectUrl(), []);

  useEffect(() => {
    onFrameReadyChange?.(Boolean(src));
  }, [src, onFrameReadyChange]);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const cacheKey =
      liveFrameSeq != null
        ? String(liveFrameSeq)
        : liveFrameUpdatedAt ?? '0';

    const fetchFrame = async (): Promise<boolean> => {
      const token = getAccessToken();
      if (!token) return false;

      try {
        const response = await fetch(
          `${buildDualArmCableFrameApiPath(jobId)}?v=${encodeURIComponent(cacheKey)}`,
          {
            headers: { Authorization: `Bearer ${token}` },
            cache: 'no-store',
          }
        );
        if (cancelled) return false;
        if (response.status === 204 || response.status === 404) return false;
        if (!response.ok) return false;
        const blob = await response.blob();
        if (cancelled || blob.size < 64) return false;
        const signature = `${cacheKey}:${blob.size}`;
        if (signature === lastFrameSignatureRef.current && objectUrlRef.current) {
          return true;
        }
        lastFrameSignatureRef.current = signature;
        applyBlob(blob);
        return true;
      } catch {
        return false;
      }
    };

    void fetchFrame();

    const pollMs = status === 'running' || status === 'queued' ? 800 : 2500;
    timer = setInterval(() => {
      void fetchFrame();
    }, pollMs);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [jobId, status, liveFrameSeq, liveFrameUpdatedAt]);

  if (!jobId) {
    return <SimulationViewportPlaceholder message="等待后端任务启动…" />;
  }

  if (status === 'failed') {
    return <SimulationViewportMessage>任务失败，请查看日志。</SimulationViewportMessage>;
  }

  if (src) {
    return <SimulationViewportImage src={src} alt="MuJoCo 运行画面" />;
  }

  if (status === 'completed') {
    return <SimulationViewportMessage>任务已完成，可查看过程回放。</SimulationViewportMessage>;
  }

  return <SimulationViewportPlaceholder backend="mujoco" />;
}
