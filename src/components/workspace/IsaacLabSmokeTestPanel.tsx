'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  getIsaacLabRunJobLog,
  getIsaacLabRunJobStatus,
  startIsaacLabSmokeTest,
  type IsaacLabRunJobStatus,
  type IsaacLabRuntimeStatus,
} from '@/lib/api/isaacLabClient';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';

const TERMINAL_STATUSES = new Set(['completed', 'failed']);

export function IsaacLabSmokeTestPanel({
  runtime,
}: {
  runtime: IsaacLabRuntimeStatus | null;
}) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<IsaacLabRunJobStatus | null>(null);
  const [stdoutTail, setStdoutTail] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const refreshJob = useCallback(async (jobId: string) => {
    const status = await getIsaacLabRunJobStatus(jobId);
    setJob(status);
    if (TERMINAL_STATUSES.has(status.status)) {
      stopPolling();
      try {
        const tail = await getIsaacLabRunJobLog(jobId, 'stdout', 30);
        setStdoutTail(tail);
      } catch {
        setStdoutTail('');
      }
    }
  }, [stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const canStart = Boolean(runtime?.configured && runtime?.enabled);
  const startDisabled = submitting || !canStart;

  const handleStart = async () => {
    setSubmitting(true);
    setError(null);
    setStdoutTail('');
    stopPolling();
    try {
      const started = await startIsaacLabSmokeTest('Stack');
      setJob({ jobId: started.jobId, status: started.status, kind: started.kind });
      pollRef.current = setInterval(() => {
        void refreshJob(started.jobId).catch(() => undefined);
      }, 2000);
      await refreshJob(started.jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Smoke test 启动失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      style={{
        marginTop: 12,
        padding: 14,
        borderRadius: 10,
        border: '1px solid #e5e7eb',
        backgroundColor: '#fff',
      }}
    >
      <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 8 }}>
        CLI Smoke Test
      </div>
      <p style={{ margin: '0 0 12px', fontSize: 12, color: '#6b7280', lineHeight: 1.55 }}>
        通过 <code style={{ fontFamily: 'monospace' }}>isaaclab.sh -p scripts/environments/list_envs.py --keyword Stack</code>{' '}
        验证外部 Isaac Lab 节点是否可执行。
      </p>

      {!canStart ? (
        <p style={{ margin: '0 0 12px', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          需配置 ISAACLAB_ROOT 并设置 ISAACLAB_RUNTIME_ENABLED=true 后启用。
        </p>
      ) : null}

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <PrimaryButton disabled={startDisabled} onClick={() => void handleStart()}>
          {submitting ? '正在启动…' : '运行 Smoke Test'}
        </PrimaryButton>
        {job?.jobId ? (
          <SecondaryButton
            onClick={() => {
              void refreshJob(job.jobId).catch(() => undefined);
            }}
          >
            刷新状态
          </SecondaryButton>
        ) : null}
      </div>

      {error ? (
        <p style={{ margin: '12px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>{error}</p>
      ) : null}

      {job ? (
        <div style={{ marginTop: 12, fontSize: 12, color: '#374151', lineHeight: 1.6 }}>
          <div>
            Job: <span style={{ fontFamily: 'monospace' }}>{job.jobId}</span>
          </div>
          <div>
            Status: <strong>{job.status}</strong>
            {job.phase ? ` · ${job.phase}` : ''}
          </div>
          {job.message ? <div>{job.message}</div> : null}
          {typeof job.stackEnvMatches === 'number' ? (
            <div>Stack 环境匹配行数: {job.stackEnvMatches}</div>
          ) : null}
          {job.paths?.jobRoot ? (
            <div style={{ color: '#6b7280', marginTop: 4, wordBreak: 'break-all' }}>{job.paths.jobRoot}</div>
          ) : null}
          {stdoutTail ? (
            <pre
              style={{
                marginTop: 10,
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
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
