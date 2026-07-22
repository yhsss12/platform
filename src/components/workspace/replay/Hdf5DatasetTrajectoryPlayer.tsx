'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getAccessToken } from '@/lib/auth/session';
import {
  getCableThreadingHdf5TrajectoryMeta,
  getCableThreadingHdf5TrajectoryStep,
  type CableThreadingHdf5TrajectoryMeta,
  type CableThreadingHdf5TrajectoryStep,
} from '@/lib/api/cableThreadingClient';
import {
  buildCableThreadingHdf5TrajectoryFrameApiPath,
} from '@/lib/workspace/cableThreading';

import type { ReplayTrajectoryRecord } from '@/lib/workspace/replayContentKind';

interface Hdf5DatasetTrajectoryPlayerProps {
  jobId: string;
  demoName: string;
  trajectories: string[];
  trajectoryRecords?: ReplayTrajectoryRecord[];
  selectedIndex: number;
  onSelectDemo: (index: number) => void;
  trajectoryDisplayMode?: 'rgb_frame_replay' | 'state_trajectory';
}

function formatDemoButtonLabel(name: string, record?: ReplayTrajectoryRecord): string {
  if (!record) return name;
  const ordinal = (record.successfulTrajectoryIndex ?? 0) + 1;
  const source =
    record.sourceEpisodeIndex != null ? ` · 原轮次 ${record.sourceEpisodeIndex + 1}` : '';
  return `${name}（第 ${ordinal} 条成功轨迹${source}）`;
}

function formatVector(values: number[] | undefined): string {
  if (!values?.length) return '—';
  return values.map((value) => value.toFixed(3)).join(', ');
}

export function Hdf5DatasetTrajectoryPlayer({
  jobId,
  demoName,
  trajectories,
  trajectoryRecords,
  selectedIndex,
  onSelectDemo,
  trajectoryDisplayMode,
}: Hdf5DatasetTrajectoryPlayerProps) {
  const [meta, setMeta] = useState<CableThreadingHdf5TrajectoryMeta | null>(null);
  const [stepDetail, setStepDetail] = useState<CableThreadingHdf5TrajectoryStep | null>(null);
  const [frameSrc, setFrameSrc] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [camera, setCamera] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const displayMode =
    meta?.trajectoryDisplayMode ?? trajectoryDisplayMode ?? 'state_trajectory';
  const isRgbMode = displayMode === 'rgb_frame_replay';
  const modeLabel = isRgbMode ? 'HDF5 RGB 帧序列回放' : 'HDF5 状态轨迹详情';

  const revokeObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPlaying(false);
    setStepIndex(0);
    revokeObjectUrl();
    setFrameSrc(null);

    void getCableThreadingHdf5TrajectoryMeta(jobId, demoName)
      .then((nextMeta) => {
        if (cancelled) return;
        setMeta(nextMeta);
        setCamera(nextMeta.defaultCamera ?? nextMeta.rgbCameras[0] ?? null);
      })
      .catch((err) => {
        if (cancelled) return;
        setMeta(null);
        setError(err instanceof Error ? err.message : '轨迹元数据加载失败');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [jobId, demoName, revokeObjectUrl]);

  const loadFrame = useCallback(
    async (index: number, activeCamera: string) => {
      const token = getAccessToken();
      if (!token) {
        setError('未登录，无法加载 HDF5 轨迹帧。');
        return;
      }
      const apiPath = buildCableThreadingHdf5TrajectoryFrameApiPath(jobId, demoName, {
        camera: activeCamera,
        index,
      });
      const response = await fetch(apiPath, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        throw new Error(`帧加载失败（HTTP ${response.status}）`);
      }
      const blob = await response.blob();
      revokeObjectUrl();
      const objectUrl = URL.createObjectURL(blob);
      objectUrlRef.current = objectUrl;
      setFrameSrc(objectUrl);
    },
    [jobId, demoName, revokeObjectUrl]
  );

  const loadStepDetail = useCallback(
    async (index: number) => {
      const detail = await getCableThreadingHdf5TrajectoryStep(jobId, demoName, index);
      setStepDetail(detail);
    },
    [jobId, demoName]
  );

  useEffect(() => {
    if (!meta || loading) return;
    let cancelled = false;

    const run = async () => {
      try {
        setError(null);
        if (isRgbMode && camera) {
          await loadFrame(stepIndex, camera);
        } else {
          await loadStepDetail(stepIndex);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '轨迹内容加载失败');
        }
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [meta, loading, isRgbMode, camera, stepIndex, loadFrame, loadStepDetail]);

  useEffect(() => {
    return () => revokeObjectUrl();
  }, [revokeObjectUrl]);

  useEffect(() => {
    if (!playing || !meta?.stepCount) return undefined;
    const timer = window.setInterval(() => {
      setStepIndex((current) => {
        if (current + 1 >= meta.stepCount) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, meta?.stepCount]);

  const maxStep = Math.max((meta?.stepCount ?? 1) - 1, 0);
  const stepLabel = `${stepIndex + 1} / ${meta?.stepCount ?? 0}`;

  const obsPreview = useMemo(() => {
    if (!stepDetail?.obs) return [];
    return Object.entries(stepDetail.obs).slice(0, 4);
  }, [stepDetail]);

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
          fontSize: 13,
        }}
      >
        正在加载 HDF5 轨迹…
      </div>
    );
  }

  if (error && !meta) {
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
          color: '#fca5a5',
          fontSize: 13,
        }}
      >
        {error}
      </div>
    );
  }

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: '#0f172a',
        color: '#e2e8f0',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 12,
          padding: '10px 12px',
          borderBottom: '1px solid rgba(148, 163, 184, 0.18)',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontSize: 12, fontWeight: 600 }}>{modeLabel}</span>
          <span style={{ fontSize: 11, color: '#94a3b8' }}>
            {demoName} · {meta?.stepCount ?? 0} steps
            {meta?.actionDim ? ` · action_dim=${meta.actionDim}` : ''}
          </span>
        </div>
        {isRgbMode && (meta?.rgbCameras.length ?? 0) > 1 ? (
          <select
            value={camera ?? undefined}
            onChange={(event) => {
              setCamera(event.target.value);
              setPlaying(false);
            }}
            style={{
              fontSize: 11,
              borderRadius: 8,
              border: '1px solid #334155',
              background: '#111827',
              color: '#e2e8f0',
              padding: '4px 8px',
            }}
          >
            {meta?.rgbCameras.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        ) : null}
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {isRgbMode ? (
          frameSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={frameSrc}
              alt={`${demoName} step ${stepIndex}`}
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
          ) : (
            <div style={{ color: '#94a3b8', fontSize: 13 }}>正在加载帧…</div>
          )
        ) : (
          <div
            style={{
              width: '100%',
              height: '100%',
              padding: 20,
              overflow: 'auto',
              textAlign: 'left',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>状态轨迹详情</div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 12, lineHeight: 1.6 }}>
              当前数据集未提供 RGB observation，展示 action 与低维 state/observation。
            </div>
            <div style={{ fontSize: 12, marginBottom: 8 }}>
              <span style={{ color: '#94a3b8' }}>action: </span>
              {formatVector(stepDetail?.action)}
            </div>
            {obsPreview.map(([key, values]) => (
              <div key={key} style={{ fontSize: 12, marginBottom: 8 }}>
                <span style={{ color: '#94a3b8' }}>{key}: </span>
                {formatVector(values)}
              </div>
            ))}
            {stepDetail?.reward != null ? (
              <div style={{ fontSize: 12, marginBottom: 8 }}>
                <span style={{ color: '#94a3b8' }}>reward: </span>
                {stepDetail.reward.toFixed(4)}
              </div>
            ) : null}
          </div>
        )}
      </div>

      <div
        style={{
          padding: '10px 12px 12px',
          borderTop: '1px solid rgba(148, 163, 184, 0.18)',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {trajectories.map((name, index) => {
            const selected = index === selectedIndex;
            const record = trajectoryRecords?.[index];
            return (
              <button
                key={name}
                type="button"
                onClick={() => onSelectDemo(index)}
                title={formatDemoButtonLabel(name, record)}
                style={{
                  borderRadius: 8,
                  border: selected ? '1px solid #2563eb' : '1px solid #334155',
                  background: selected ? '#1d4ed8' : 'rgba(15, 23, 42, 0.72)',
                  color: '#e2e8f0',
                  fontSize: 12,
                  padding: '6px 10px',
                  cursor: 'pointer',
                }}
              >
                {formatDemoButtonLabel(name, record)}
              </button>
            );
          })}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {isRgbMode ? (
            <button
              type="button"
              onClick={() => setPlaying((value) => !value)}
              style={{
                borderRadius: 999,
                border: '1px solid #334155',
                background: '#111827',
                color: '#e2e8f0',
                fontSize: 12,
                padding: '4px 10px',
                cursor: 'pointer',
              }}
            >
              {playing ? '暂停' : '播放'}
            </button>
          ) : null}
          <input
            type="range"
            min={0}
            max={maxStep}
            value={Math.min(stepIndex, maxStep)}
            onChange={(event) => {
              setStepIndex(Number(event.target.value));
              setPlaying(false);
            }}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11, color: '#94a3b8', minWidth: 72, textAlign: 'right' }}>
            step {stepLabel}
          </span>
        </div>
        {error ? <div style={{ fontSize: 11, color: '#fca5a5' }}>{error}</div> : null}
      </div>
    </div>
  );
}
