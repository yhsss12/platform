'use client';

import { useEffect, useRef, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { buildCableThreadingFrameApiPath } from '@/lib/api/cableThreadingClient';
import {
  SimulationViewportImage,
  SimulationViewportMessage,
  SimulationViewportPlaceholder,
} from '@/components/workspace/simulation/SimulationViewport';
import {
  isTerminalSimJobStatus,
  usePageVisibleForPolling,
} from '@/lib/workspace/simulationPolling';

export type CableThreadingFrameJobStatus = 'running' | 'completed' | 'failed';

const FRAME_POLL_MS = 1200;

function buildFrameSignature(blob: Blob, response: Response): string {
  const lastModified = response.headers.get('last-modified') ?? '';
  const etag = response.headers.get('etag') ?? '';
  if (lastModified || etag) {
    return `${blob.size}:${lastModified}:${etag}`;
  }
  return `${blob.size}:${response.status}`;
}

export function CableThreadingLiveFrame({
  jobId,
  status,
  embedded = false,
  onFrameReadyChange,
}: {
  /** backendJobId — 帧 API 必须使用后端 jobId，而非 localRunId */
  jobId?: string;
  status: CableThreadingFrameJobStatus;
  frameCount?: number;
  /** 外层已有 16:9 容器时设为 true，避免嵌套 SimulationViewportShell */
  embedded?: boolean;
  onFrameReadyChange?: (ready: boolean) => void;
}) {
  const pageVisible = usePageVisibleForPolling();
  const [src, setSrc] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const lastFrameSignatureRef = useRef<string | null>(null);
  const preloadGenerationRef = useRef(0);
  const pollingActive =
    Boolean(jobId) &&
    pageVisible &&
    status === 'running' &&
    !isTerminalSimJobStatus(status);

  const revokeObjectUrl = () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  };

  const preloadAndApplyBlob = (blob: Blob, signature: string): Promise<boolean> => {
    if (signature === lastFrameSignatureRef.current && objectUrlRef.current) {
      return Promise.resolve(true);
    }

    const nextUrl = URL.createObjectURL(blob);
    const preloadId = ++preloadGenerationRef.current;

    return new Promise((resolve) => {
      const img = new Image();

      img.onload = () => {
        if (preloadId !== preloadGenerationRef.current) {
          URL.revokeObjectURL(nextUrl);
          resolve(false);
          return;
        }

        const previousUrl = objectUrlRef.current;
        objectUrlRef.current = nextUrl;
        lastFrameSignatureRef.current = signature;
        setSrc(nextUrl);
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl);
        }
        resolve(true);
      };

      img.onerror = () => {
        URL.revokeObjectURL(nextUrl);
        resolve(false);
      };

      img.src = nextUrl;
    });
  };

  useEffect(() => {
    preloadGenerationRef.current += 1;
    revokeObjectUrl();
    setSrc(null);
    lastFrameSignatureRef.current = null;
    onFrameReadyChange?.(false);
  }, [jobId, onFrameReadyChange]);

  useEffect(() => {
    return () => {
      preloadGenerationRef.current += 1;
      revokeObjectUrl();
    };
  }, []);

  useEffect(() => {
    onFrameReadyChange?.(Boolean(src));
  }, [src, onFrameReadyChange]);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const fetchFrame = async (): Promise<boolean> => {
      const token = getAccessToken();
      if (!token) return false;

      try {
        const response = await fetch(
          `${buildCableThreadingFrameApiPath(jobId)}?t=${Date.now()}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (cancelled) return false;
        if (response.status === 204 || response.status === 404) return false;
        if (!response.ok) return false;
        const blob = await response.blob();
        if (cancelled || blob.size < 64) return false;

        const signature = buildFrameSignature(blob, response);
        if (signature === lastFrameSignatureRef.current && objectUrlRef.current) {
          return true;
        }

        return preloadAndApplyBlob(blob, signature);
      } catch {
        return false;
      }
    };

    if (pollingActive) {
      void fetchFrame();
      timer = setInterval(() => {
        void fetchFrame();
      }, FRAME_POLL_MS);
    } else if (status === 'completed') {
      void fetchFrame();
    }

    return () => {
      cancelled = true;
      preloadGenerationRef.current += 1;
      if (timer) clearInterval(timer);
    };
  }, [jobId, pollingActive, status]);

  if (!jobId) {
    return <SimulationViewportPlaceholder message="等待后端任务启动…" embedded={embedded} />;
  }

  if (status === 'failed') {
    return (
      <SimulationViewportMessage embedded={embedded}>采集失败，请查看日志。</SimulationViewportMessage>
    );
  }

  if (src) {
    return (
      <SimulationViewportImage src={src} alt="MuJoCo 实时采集画面" embedded={embedded} />
    );
  }

  if (status === 'completed') {
    return (
      <SimulationViewportMessage embedded={embedded}>
        采集已完成，但未获取到最终帧。
      </SimulationViewportMessage>
    );
  }

  return (
    <SimulationViewportPlaceholder
      backend="mujoco"
      message="等待仿真首帧…"
      embedded={embedded}
    />
  );
}
