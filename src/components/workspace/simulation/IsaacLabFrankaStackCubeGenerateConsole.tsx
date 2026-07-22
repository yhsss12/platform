'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  getIsaacLabFrankaStackCubeJobLog,
  getIsaacLabFrankaStackCubeJobStatus,
  type IsaacLabFrankaStackCubeJobStatusResponse,
} from '@/lib/api/isaaclabFrankaStackCubeClient';
import {
  buildIsaacLabFrankaStackCubeReplayHref,
  ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME,
} from '@/lib/workspace/isaaclabFrankaStackCube';
import { isValidDataGenJobId } from '@/lib/workspace/backendJobIds';
import { buildIsaacLabFrankaStackCubeRunConsoleViewModel } from '@/lib/workspace/runConsoleAdapters';
import { RunConsoleView } from '@/components/workspace/simulation/RunConsoleView';
import {
  simConsoleCardStyle,
  type SimConsoleHeaderState,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import { SecondaryButton } from '@/components/workspace/workspaceUi';

const POLL_INTERVAL_MS = 2000;

export function IsaacLabFrankaStackCubeGenerateConsole({
  jobId,
  onHeaderStateChange,
}: {
  jobId: string;
  onHeaderStateChange?: (state: SimConsoleHeaderState) => void;
}) {
  const router = useRouter();
  const [jobStatus, setJobStatus] = useState<IsaacLabFrankaStackCubeJobStatusResponse | null>(null);
  const [frameLoaded, setFrameLoaded] = useState(false);
  const [logTail, setLogTail] = useState('');
  const [logLoading, setLogLoading] = useState(false);
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);

  const isBackendJobId = isValidDataGenJobId(jobId);
  const displayStatusRaw = jobStatus?.status ?? 'queued';
  const canOpenReplay =
    (displayStatusRaw === 'completed' && jobStatus?.videoExists === true) ||
    ((jobStatus?.video_status ?? jobStatus?.videoStatus) === 'available' && jobStatus?.videoExists === true);

  const openReplay = useCallback(() => {
    router.push(
      buildIsaacLabFrankaStackCubeReplayHref({
        jobId,
      })
    );
  }, [router, jobId]);

  useEffect(() => {
    onHeaderStateChange?.({
      canViewReplay: canOpenReplay,
      openReplay,
      replayDisabledTitle: canOpenReplay ? undefined : '回放视频尚未就绪',
    });
  }, [canOpenReplay, openReplay, onHeaderStateChange]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      try {
        const status = await getIsaacLabFrankaStackCubeJobStatus(jobId);
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

  const refreshLog = useCallback(async () => {
    setLogLoading(true);
    try {
      const res = await getIsaacLabFrankaStackCubeJobLog(jobId);
      setLogTail(res.tail?.trim() || '');
    } catch {
      /* ignore */
    } finally {
      setLogLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    void refreshLog();
    const timer = setInterval(() => void refreshLog(), logDrawerOpen ? 1500 : 3000);
    return () => clearInterval(timer);
  }, [refreshLog, logDrawerOpen]);

  const viewModel = useMemo(
    () =>
      buildIsaacLabFrankaStackCubeRunConsoleViewModel({
        jobId,
        jobStatus,
        canViewReplay: canOpenReplay,
        frameLoaded,
      }),
    [jobId, jobStatus, canOpenReplay, frameLoaded]
  );

  if (!isBackendJobId) {
    return (
      <div style={simConsoleCardStyle}>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          无效的物块堆叠后端 jobId：{jobId}。请从数据中心重新启动任务。
        </p>
        <div style={{ marginTop: 12 }}>
          <SecondaryButton onClick={() => router.push('/workspace/data')}>返回数据中心</SecondaryButton>
        </div>
      </div>
    );
  }

  if (!jobStatus) {
    return (
      <div style={simConsoleCardStyle}>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          正在加载 {ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME} 运行状态…
        </p>
      </div>
    );
  }

  return (
    <RunConsoleView
      vm={viewModel}
      logTail={logTail}
      logLoading={logLoading}
      logDrawerOpen={logDrawerOpen}
      onOpenLog={() => setLogDrawerOpen(true)}
      onCloseLog={() => setLogDrawerOpen(false)}
      onFrameLoadedChange={setFrameLoaded}
    />
  );
}
