'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  getIsaacLabRunJobLog,
  getIsaacLabRunJobStatus,
  type IsaacLabRunJobStatus,
} from '@/lib/api/isaacLabClient';
import {
  buildIsaacBlockStackingReplayHref,
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
} from '@/lib/workspace/isaacBlockStacking';
import { resolvePreviewStatusDisplay } from '@/lib/workspace/isaacPreviewStatus';
import { RunConsoleView } from '@/components/workspace/simulation/RunConsoleView';
import { buildIsaacBlockStackingRunConsoleViewModel } from '@/lib/workspace/runConsoleAdapters';
import {
  simConsoleCardStyle,
  type SimConsoleHeaderState,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';

const POLL_INTERVAL_MS = 2000;

export function IsaacLabGenerateConsole({
  jobId,
  onHeaderStateChange,
}: {
  jobId: string;
  onHeaderStateChange?: (state: SimConsoleHeaderState) => void;
}) {
  const router = useRouter();
  const [jobStatus, setJobStatus] = useState<IsaacLabRunJobStatus | null>(null);
  const [frameLoaded, setFrameLoaded] = useState(false);
  const [stdoutTail, setStdoutTail] = useState('');
  const [stderrTail, setStderrTail] = useState('');
  const [logLoading, setLogLoading] = useState(false);
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);

  const displayStatusRaw = jobStatus?.status ?? 'queued';
  const previewDisplay = resolvePreviewStatusDisplay({
    previewStatus: jobStatus?.previewStatus,
    videoAvailable: jobStatus?.previewVideoAvailable || jobStatus?.videoAvailable,
    videoNote: jobStatus?.videoNote,
    jobStatus: displayStatusRaw,
    phase: jobStatus?.phase,
  });
  const canOpenReplay = previewDisplay.canOpenReplay;
  const replayDisabledTitle = previewDisplay.hint ?? '当前暂无回放视频';

  const openReplay = useCallback(() => {
    router.push(
      buildIsaacBlockStackingReplayHref({
        jobId,
        datasetId: jobStatus?.datasetId ?? undefined,
      })
    );
  }, [router, jobId, jobStatus?.datasetId]);

  useEffect(() => {
    onHeaderStateChange?.({
      canViewReplay: canOpenReplay,
      openReplay,
      replayDisabledTitle,
    });
  }, [canOpenReplay, openReplay, replayDisabledTitle, onHeaderStateChange]);

  useEffect(() => {
    setFrameLoaded(false);
  }, [jobId]);

  const refreshLogs = useCallback(async () => {
    setLogLoading(true);
    try {
      const [stdout, stderr] = await Promise.all([
        getIsaacLabRunJobLog(jobId, 'stdout', 60).catch(() => ''),
        getIsaacLabRunJobLog(jobId, 'stderr', 60).catch(() => ''),
      ]);
      setStdoutTail(stdout.trim());
      setStderrTail(stderr.trim());
    } finally {
      setLogLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      try {
        const status = await getIsaacLabRunJobStatus(jobId);
        if (cancelled) return;
        setJobStatus(status);
        if (status.status === 'completed' || status.status === 'failed') {
          if (timer) {
            clearInterval(timer);
            timer = null;
          }
        }
      } catch {
        /* polling continues */
      }
    };

    void poll();
    timer = setInterval(() => void poll(), POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [jobId]);

  useEffect(() => {
    void refreshLogs();
    const timer = setInterval(() => void refreshLogs(), logDrawerOpen ? 1500 : 3000);
    return () => clearInterval(timer);
  }, [jobId, logDrawerOpen, refreshLogs]);

  const viewModel = useMemo(
    () =>
      buildIsaacBlockStackingRunConsoleViewModel({
        jobId,
        jobStatus,
        frameLoaded,
        canViewReplay: canOpenReplay,
      }),
    [jobId, jobStatus, frameLoaded, canOpenReplay]
  );

  const combinedLogTail = [stdoutTail, stderrTail ? `--- stderr ---\n${stderrTail}` : '']
    .filter(Boolean)
    .join('\n\n');

  if (!jobStatus && !viewModel) {
    return (
      <div style={simConsoleCardStyle}>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          正在加载 {ISAAC_BLOCK_STACKING_DISPLAY_NAME} 运行状态…
        </p>
      </div>
    );
  }

  return (
    <RunConsoleView
      vm={viewModel}
      logTail={combinedLogTail}
      logLoading={logLoading}
      logDrawerOpen={logDrawerOpen}
      onOpenLog={() => setLogDrawerOpen(true)}
      onCloseLog={() => setLogDrawerOpen(false)}
      onFrameLoadedChange={setFrameLoaded}
    />
  );
}
