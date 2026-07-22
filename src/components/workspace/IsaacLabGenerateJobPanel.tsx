'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  getIsaacLabJobVideoUrl,
  getIsaacLabRunJobLog,
  getIsaacLabRunJobStatus,
  type IsaacLabRunJobStatus,
  type IsaacLabRuntimeStatus,
} from '@/lib/api/isaacLabClient';
import {
  buildTaskTemplatesPathWithParams,
  clearIsaacReplayQueryParams,
  readIsaacGenerateJobId,
} from '@/lib/workspace/isaacReplayNavigation';
import { SecondaryButton } from '@/components/workspace/workspaceUi';

const TERMINAL_STATUSES = new Set(['completed', 'failed']);
const POLL_INTERVAL_MS = 2000;

function isJobNotFoundError(err: unknown): boolean {
  const message = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return message.includes('not found') || message.includes('404');
}

export function IsaacLabGenerateJobPanel({
  runtime,
  autoGenerateJobId,
  onGenerateJobCleared,
}: {
  runtime: IsaacLabRuntimeStatus | null;
  autoGenerateJobId?: string | null;
  onGenerateJobCleared?: () => void;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [error, setError] = useState<string | null>(null);
  const [jobNotFound, setJobNotFound] = useState<string | null>(null);
  const [job, setJob] = useState<IsaacLabRunJobStatus | null>(null);
  const [stdoutTail, setStdoutTail] = useState('');
  const [stderrTail, setStderrTail] = useState('');
  const [trackingJobId, setTrackingJobId] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const trackedJobRef = useRef<string | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  const urlGenerateJobId =
    autoGenerateJobId ?? readIsaacGenerateJobId(searchParams) ?? null;
  const autoTracking = Boolean(urlGenerateJobId && trackingJobId === urlGenerateJobId);

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
          setJobNotFound('未找到该 Isaac 数据生成 Job');
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
          setJobNotFound('未找到该 Isaac 数据生成 Job');
        } else {
          setError(err instanceof Error ? err.message : '加载数据生成 Job 失败');
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
    onGenerateJobCleared?.();
  }, [onGenerateJobCleared, router, searchParams, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  useEffect(() => {
    if (!urlGenerateJobId) return;
    if (trackedJobRef.current === urlGenerateJobId) return;
    trackedJobRef.current = urlGenerateJobId;
    void beginTrackingJob(urlGenerateJobId);
  }, [urlGenerateJobId, beginTrackingJob]);

  useEffect(() => {
    if (!autoTracking || !panelRef.current) return;
    panelRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [autoTracking]);

  const videoUrl =
    job?.videoAvailable && job.jobId ? getIsaacLabJobVideoUrl(job.jobId) : null;

  return (
    <div
      ref={panelRef}
      style={{
        marginTop: 12,
        padding: 14,
        borderRadius: 10,
        border: autoTracking ? '1px solid #86efac' : '1px solid #e5e7eb',
        backgroundColor: autoTracking ? '#f0fdf4' : '#fff',
      }}
    >
      <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 8 }}>
        数据生成 Job 控制台
      </div>

      {autoTracking && urlGenerateJobId ? (
        <div
          style={{
            marginBottom: 12,
            padding: '8px 10px',
            borderRadius: 8,
            backgroundColor: '#ecfdf5',
            border: '1px solid #bbf7d0',
            fontSize: 12,
            color: '#166534',
            lineHeight: 1.55,
          }}
        >
          正在跟踪数据中心发起的 Isaac Lab 数据生成任务：
          <span style={{ fontFamily: 'monospace' }}> {urlGenerateJobId}</span>
        </div>
      ) : null}

      <p style={{ margin: '0 0 12px', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
        通过 Mimic 或 teleop 生成物块堆叠 HDF5 数据集。完成后自动登记到 Dataset Registry，可在数据中心查看与回放。
      </p>

      {!runtime?.configured && !trackingJobId ? (
        <p style={{ margin: '0 0 12px', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          需配置 Isaac Lab 运行节点后启用（ISAACLAB_ROOT + ISAACLAB_RUNTIME_ENABLED=true）。
        </p>
      ) : null}

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
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
          {job.generationMode ? <div>generationMode: {job.generationMode}</div> : null}
          {job.datasetName ? <div>datasetName: {job.datasetName}</div> : null}
          {job.numDemos != null ? <div>numDemos: {job.numDemos}</div> : null}
          {job.message ? <div>{job.message}</div> : null}
          <div>
            dataset.hdf5:{' '}
            <strong style={{ color: job.datasetAvailable ? '#15803d' : '#6b7280' }}>
              {job.datasetAvailable ? '已生成' : '未就绪'}
            </strong>
          </div>
          {job.status === 'completed' && job.datasetAvailable && job.datasetId ? (
            <div style={{ color: '#15803d', marginTop: 4 }}>
              已登记为数据集：<span style={{ fontFamily: 'monospace' }}>{job.datasetId}</span>
            </div>
          ) : null}
          {job.status === 'failed' ? (
            <div style={{ color: '#b45309' }}>数据生成失败，请查看下方 stderr 日志。</div>
          ) : null}
          {job.videoAvailable === false && job.status === 'completed' ? (
            <div style={{ color: '#6b7280' }}>preview.mp4 不可用（可能未启用相机或后处理失败）</div>
          ) : null}
          {job.videoNote ? <div style={{ color: '#6b7280' }}>{job.videoNote}</div> : null}
          {videoUrl ? (
            <div style={{ marginTop: 8 }}>
              <a href={videoUrl} target="_blank" rel="noreferrer" style={{ color: '#2563eb' }}>
                查看 preview 视频
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
      ) : trackingJobId ? (
        <p style={{ margin: '12px 0 0', fontSize: 12, color: '#6b7280' }}>正在加载 Job 状态…</p>
      ) : null}
    </div>
  );
}
