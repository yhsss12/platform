'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  getIsaacLabReplayVideoUrl,
  getIsaacLabRunJobLog,
  getIsaacLabRunJobStatus,
  startIsaacLabReplayDemo,
  type IsaacLabRunJobStatus,
  type IsaacLabRuntimeStatus,
} from '@/lib/api/isaacLabClient';
import { ISAAC_BLOCK_STACKING_DEFAULT_ENV } from '@/lib/workspace/isaacBlockStacking';
import {
  buildTaskTemplatesPathWithParams,
  clearIsaacReplayQueryParams,
  readIsaacReplayJobId,
} from '@/lib/workspace/isaacReplayNavigation';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';

const TERMINAL_STATUSES = new Set(['completed', 'failed']);
const POLL_INTERVAL_MS = 2000;

function isJobNotFoundError(err: unknown): boolean {
  const message = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return message.includes('not found') || message.includes('404');
}

export function IsaacLabReplayDemoPanel({
  runtime,
  autoReplayJobId,
  onReplayJobCleared,
}: {
  runtime: IsaacLabRuntimeStatus | null;
  autoReplayJobId?: string | null;
  onReplayJobCleared?: () => void;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [datasetFile, setDatasetFile] = useState('./datasets/dataset.hdf5');
  const [taskId, setTaskId] = useState(ISAAC_BLOCK_STACKING_DEFAULT_ENV);
  const [headless, setHeadless] = useState(true);
  const [enableCameras, setEnableCameras] = useState(true);
  const [video, setVideo] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobNotFound, setJobNotFound] = useState<string | null>(null);
  const [job, setJob] = useState<IsaacLabRunJobStatus | null>(null);
  const [stdoutTail, setStdoutTail] = useState('');
  const [stderrTail, setStderrTail] = useState('');
  const [trackingJobId, setTrackingJobId] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const trackedJobRef = useRef<string | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  const urlReplayJobId =
    autoReplayJobId ?? readIsaacReplayJobId(searchParams) ?? null;
  const autoTracking = Boolean(urlReplayJobId && trackingJobId === urlReplayJobId);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const refreshJob = useCallback(
    async (jobId: string) => {
      try {
        const status = await getIsaacLabRunJobStatus(jobId);
        setJob(status);
        setJobNotFound(null);

        try {
          const stdout = await getIsaacLabRunJobLog(jobId, 'stdout', 40);
          setStdoutTail(stdout);
        } catch {
          setStdoutTail('');
        }

        if (status.status === 'failed') {
          try {
            const stderr = await getIsaacLabRunJobLog(jobId, 'stderr', 40);
            setStderrTail(stderr);
          } catch {
            setStderrTail('');
          }
        } else {
          setStderrTail('');
        }

        if (TERMINAL_STATUSES.has(status.status)) {
          stopPolling();
        }
      } catch (err) {
        if (isJobNotFoundError(err)) {
          setJob(null);
          setJobNotFound('未找到该 Isaac Replay Job');
          stopPolling();
          return;
        }
        throw err;
      }
    },
    [stopPolling]
  );

  const beginTrackingJob = useCallback(
    async (jobId: string) => {
      setTrackingJobId(jobId);
      setError(null);
      setJobNotFound(null);
      stopPolling();
      try {
        await refreshJob(jobId);
        pollRef.current = setInterval(() => {
          void refreshJob(jobId).catch(() => undefined);
        }, POLL_INTERVAL_MS);
      } catch (err) {
        if (isJobNotFoundError(err)) {
          setJobNotFound('未找到该 Isaac Replay Job');
        } else {
          setError(err instanceof Error ? err.message : '加载 Replay Job 失败');
        }
      }
    },
    [refreshJob, stopPolling]
  );

  const clearTrackedJob = useCallback(() => {
    stopPolling();
    trackedJobRef.current = null;
    setTrackingJobId(null);
    setJob(null);
    setStdoutTail('');
    setStderrTail('');
    setJobNotFound(null);
    setError(null);

    const nextParams = clearIsaacReplayQueryParams(new URLSearchParams(searchParams.toString()));
    router.replace(buildTaskTemplatesPathWithParams(nextParams));
    onReplayJobCleared?.();
  }, [onReplayJobCleared, router, searchParams, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  useEffect(() => {
    if (!urlReplayJobId) return;
    if (trackedJobRef.current === urlReplayJobId) return;
    trackedJobRef.current = urlReplayJobId;
    void beginTrackingJob(urlReplayJobId);
  }, [urlReplayJobId, beginTrackingJob]);

  useEffect(() => {
    if (!autoTracking || !panelRef.current) return;
    panelRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [autoTracking]);

  const canStart = Boolean(runtime?.configured && runtime?.enabled);
  const startDisabled = submitting || !canStart || !datasetFile.trim();

  const handleStart = async () => {
    setSubmitting(true);
    setError(null);
    setJobNotFound(null);
    setStdoutTail('');
    setStderrTail('');
    stopPolling();
    trackedJobRef.current = null;
    try {
      const started = await startIsaacLabReplayDemo({
        taskId: taskId.trim() || ISAAC_BLOCK_STACKING_DEFAULT_ENV,
        datasetFile: datasetFile.trim(),
        headless,
        enableCameras,
        video,
      });
      trackedJobRef.current = started.jobId;
      await beginTrackingJob(started.jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Replay 启动失败');
    } finally {
      setSubmitting(false);
    }
  };

  const videoUrl =
    job?.videoAvailable && job.jobId ? getIsaacLabReplayVideoUrl(job.jobId) : null;

  return (
    <div
      ref={panelRef}
      style={{
        marginTop: 12,
        padding: 14,
        borderRadius: 10,
        border: autoTracking ? '1px solid #93c5fd' : '1px solid #e5e7eb',
        backgroundColor: autoTracking ? '#f8fbff' : '#fff',
      }}
    >
      <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 8 }}>
        Replay Demo 验证
      </div>

      {autoTracking && urlReplayJobId ? (
        <div
          style={{
            marginBottom: 12,
            padding: '8px 10px',
            borderRadius: 8,
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            fontSize: 12,
            color: '#1e40af',
            lineHeight: 1.55,
          }}
        >
          正在跟踪数据中心发起的回放任务：
          <span style={{ fontFamily: 'monospace' }}> {urlReplayJobId}</span>
        </div>
      ) : null}

      <p style={{ margin: '0 0 12px', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
        调用 <code style={{ fontFamily: 'monospace' }}>replay_demos.py</code> 回放物块堆叠 HDF5 demo。
        需提供 Isaac Lab 官方或 teleop 录制的 HDF5 文件路径；若无 demo 文件将无法启动。
      </p>

      {!canStart && !trackingJobId ? (
        <p style={{ margin: '0 0 12px', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          需配置 Isaac Lab 运行节点后启用（ISAACLAB_ROOT + ISAACLAB_RUNTIME_ENABLED=true）。
        </p>
      ) : null}

      {!canStart && trackingJobId ? (
        <p style={{ margin: '0 0 12px', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
          运行节点未配置，仍可查看当前 Replay Job 的状态与日志；新建回放需先配置运行节点。
        </p>
      ) : null}

      <div style={{ display: 'grid', gap: 10 }}>
        <label style={{ display: 'grid', gap: 4, fontSize: 12, color: '#374151' }}>
          dataset_file
          <input
            type="text"
            value={datasetFile}
            onChange={(e) => setDatasetFile(e.target.value)}
            placeholder="./datasets/dataset.hdf5"
            style={{
              padding: '8px 10px',
              borderRadius: 8,
              border: '1px solid #d1d5db',
              fontSize: 13,
            }}
            disabled={submitting || Boolean(trackingJobId)}
          />
        </label>
        <label style={{ display: 'grid', gap: 4, fontSize: 12, color: '#374151' }}>
          taskId
          <input
            type="text"
            value={taskId}
            onChange={(e) => setTaskId(e.target.value)}
            style={{
              padding: '8px 10px',
              borderRadius: 8,
              border: '1px solid #d1d5db',
              fontSize: 13,
              fontFamily: 'monospace',
            }}
            disabled={submitting || Boolean(trackingJobId)}
          />
        </label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 13 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              checked={headless}
              onChange={(e) => setHeadless(e.target.checked)}
              disabled={submitting || Boolean(trackingJobId)}
            />
            headless
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              checked={enableCameras}
              onChange={(e) => setEnableCameras(e.target.checked)}
              disabled={submitting || Boolean(trackingJobId)}
            />
            enable cameras
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              checked={video}
              onChange={(e) => setVideo(e.target.checked)}
              disabled={submitting || Boolean(trackingJobId)}
            />
            尝试从 HDF5 生成视频
          </label>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
        <PrimaryButton disabled={startDisabled} onClick={() => void handleStart()}>
          {submitting ? '正在启动…' : '启动回放'}
        </PrimaryButton>
        {job?.jobId ? (
          <SecondaryButton onClick={() => void refreshJob(job.jobId).catch(() => undefined)}>
            刷新状态
          </SecondaryButton>
        ) : null}
        {trackingJobId ? (
          <SecondaryButton onClick={clearTrackedJob}>清除当前 job</SecondaryButton>
        ) : null}
      </div>

      {error ? (
        <p style={{ margin: '12px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>{error}</p>
      ) : null}

      {jobNotFound ? (
        <p style={{ margin: '12px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>{jobNotFound}</p>
      ) : null}

      {job ? (
        <div style={{ marginTop: 12, fontSize: 12, color: '#374151', lineHeight: 1.6 }}>
          <div>
            Job: <span style={{ fontFamily: 'monospace' }}>{job.jobId}</span>
          </div>
          <div>
            Status:{' '}
            <strong style={{ color: job.status === 'failed' ? '#b45309' : '#111827' }}>{job.status}</strong>
            {job.phase ? ` · ${job.phase}` : ''}
          </div>
          {job.message ? <div>{job.message}</div> : null}
          {job.status === 'failed' ? (
            <div style={{ color: '#b45309' }}>Replay 任务失败，请查看下方日志。</div>
          ) : null}
          {job.videoAvailable === false && job.status === 'completed' ? (
            <div style={{ color: '#6b7280' }}>videoAvailable=false（HDF5 可能不含相机帧或未生成 mp4）</div>
          ) : null}
          {job.videoNote ? <div style={{ color: '#6b7280' }}>{job.videoNote}</div> : null}
          {videoUrl ? (
            <div style={{ marginTop: 8 }}>
              <a href={videoUrl} target="_blank" rel="noreferrer" style={{ color: '#2563eb' }}>
                查看 replay 视频
              </a>
            </div>
          ) : null}
          {stdoutTail ? (
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>stdout</div>
              <pre
                style={{
                  margin: 0,
                  padding: 10,
                  borderRadius: 8,
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  fontSize: 11,
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {stdoutTail}
              </pre>
            </div>
          ) : null}
          {stderrTail ? (
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>stderr</div>
              <pre
                style={{
                  margin: 0,
                  padding: 10,
                  borderRadius: 8,
                  backgroundColor: '#fef2f2',
                  border: '1px solid #fecaca',
                  fontSize: 11,
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {stderrTail}
              </pre>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
