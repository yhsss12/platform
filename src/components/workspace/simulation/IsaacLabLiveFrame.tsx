'use client';

import { useEffect, useRef, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import { buildIsaacLabLiveFrameApiPath } from '@/lib/api/isaacLabClient';
import { isImageBlobMostlyBlack } from '@/lib/workspace/isaacLiveFrameUtils';
import {
  SimulationViewportImage,
  SimulationViewportMessage,
  SimulationViewportPlaceholder,
  SimulationViewportShell,
} from '@/components/workspace/simulation/SimulationViewport';

export function IsaacLabLiveFrame({
  jobId,
  enabled,
  pollEnabled,
  status,
  liveFrameApiPath,
  onFrameReadyChange,
  onFrameUsableChange,
  onLoadError,
  silentMode = false,
}: {
  jobId: string;
  enabled: boolean;
  /** 仅当 status API 报告有效 live 帧时为 true，避免 annotate 阶段 404 */
  pollEnabled: boolean;
  status: 'queued' | 'running' | 'completed' | 'failed';
  liveFrameApiPath?: string;
  onFrameReadyChange?: (ready: boolean) => void;
  onFrameUsableChange?: (usable: boolean) => void;
  onLoadError?: (message: string | null) => void;
  /** 统一运行控制台：不展示工程错误文案，加载中仅显示暗色画面区 */
  silentMode?: boolean;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const revokeObjectUrl = () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  };

  const reportLoadError = (message: string | null) => {
    setLoadError(message);
    onLoadError?.(message);
  };

  useEffect(() => {
    revokeObjectUrl();
    setSrc(null);
    setLoadError(null);
    onFrameReadyChange?.(false);
    onFrameUsableChange?.(false);
    onLoadError?.(null);
  }, [jobId, onFrameReadyChange, onFrameUsableChange, onLoadError]);

  useEffect(() => () => revokeObjectUrl(), []);

  useEffect(() => {
    onFrameReadyChange?.(Boolean(src));
  }, [src, onFrameReadyChange]);

  useEffect(() => {
    if (!pollEnabled) {
      revokeObjectUrl();
      setSrc(null);
      setLoadError(null);
      onFrameUsableChange?.(false);
      onLoadError?.(null);
    }
  }, [pollEnabled, onFrameUsableChange, onLoadError]);

  useEffect(() => {
    if (!jobId || !enabled || !pollEnabled) return;

    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const fetchFrame = async () => {
      const token = getAccessToken();
      if (!token) {
        if (!cancelled) reportLoadError('未登录，无法加载实时画面。');
        return;
      }
      try {
        const response = await fetch(
          `${liveFrameApiPath ?? buildIsaacLabLiveFrameApiPath(jobId)}?t=${Date.now()}`,
          {
          headers: { Authorization: `Bearer ${token}` },
          }
        );
        if (cancelled) return;
        if (!response.ok) {
          reportLoadError('实时画面加载失败，请稍后重试。');
          onFrameUsableChange?.(false);
          return;
        }
        const contentType = response.headers.get('content-type') ?? '';
        if (!contentType.includes('image/')) {
          reportLoadError('实时画面响应不是图片格式（可能为登录页或错误页）。');
          onFrameUsableChange?.(false);
          return;
        }
        const blob = await response.blob();
        if (cancelled || blob.size < 64) {
          onFrameUsableChange?.(false);
          return;
        }
        if (await isImageBlobMostlyBlack(blob)) {
          revokeObjectUrl();
          setSrc(null);
          onFrameUsableChange?.(false);
          reportLoadError('实时帧为全黑图，无法显示有效画面。');
          return;
        }
        revokeObjectUrl();
        const url = URL.createObjectURL(blob);
        objectUrlRef.current = url;
        setSrc(url);
        reportLoadError(null);
        onFrameUsableChange?.(true);
      } catch {
        if (!cancelled) {
          reportLoadError('实时画面网络请求失败。');
          onFrameUsableChange?.(false);
        }
      }
    };

    void fetchFrame();
    timer = setInterval(() => void fetchFrame(), status === 'running' ? 1500 : 3000);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [jobId, enabled, pollEnabled, status, liveFrameApiPath, onFrameUsableChange]);

  if (!enabled) {
    return (
      <SimulationViewportMessage>
        当前任务未启用相机输出，无法显示实时画面。
      </SimulationViewportMessage>
    );
  }

  if (!pollEnabled) {
    return silentMode ? null : null;
  }

  if (src) {
    return <SimulationViewportImage src={src} alt="Isaac Lab live preview" />;
  }

  if (loadError && !silentMode) {
    return <SimulationViewportMessage>{loadError}</SimulationViewportMessage>;
  }

  if (status === 'failed' && !silentMode) {
    return <SimulationViewportMessage>生成失败，暂无实时画面。</SimulationViewportMessage>;
  }

  if (silentMode) {
    return <SimulationViewportShell>{null}</SimulationViewportShell>;
  }

  return (
    <SimulationViewportPlaceholder
      message={
        status === 'running' ? '正在加载 Isaac 实时画面…' : '当前阶段尚未输出实时画面。'
      }
    />
  );
}
