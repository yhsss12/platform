'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  getDualArmCableJobLog,
  getDualArmCableJobStatus,
  type DualArmCableJobStatusResponse,
} from '@/lib/api/dualArmCableClient';
import {
  getDualArmCableGenerateRun,
  replaceMockDataItem,
  updateDualArmCableGenerateRun,
} from '@/lib/mock/workspaceMockFlowStore';
import {
  buildDualArmCableReplayHref,
  dualArmCableDataItemFromJobStatus,
} from '@/lib/workspace/dualArmCable';
import { RunConsoleView } from '@/components/workspace/simulation/RunConsoleView';
import {
  simConsoleCardStyle,
  type SimConsoleHeaderState,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { isValidDualArmGenerateJobId } from '@/lib/workspace/backendJobIds';
import { buildDualArmCableRunConsoleViewModel } from '@/lib/workspace/runConsoleAdapters';

export function DualArmCableGenerateConsole({
  jobId,
  dataId,
  onHeaderStateChange,
}: {
  jobId: string;
  dataId?: string;
  onHeaderStateChange?: (state: SimConsoleHeaderState) => void;
}) {
  const router = useRouter();
  const [jobStatus, setJobStatus] = useState<DualArmCableJobStatusResponse | null>(null);
  const [logTail, setLogTail] = useState('');
  const [logLoading, setLogLoading] = useState(false);
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const [frameLoaded, setFrameLoaded] = useState(false);

  const run = getDualArmCableGenerateRun(jobId);
  const isBackendJobId = isValidDualArmGenerateJobId(jobId);

  useEffect(() => {
    setFrameLoaded(false);
  }, [jobId]);

  const refreshStatus = useCallback(async () => {
    try {
      const status = await getDualArmCableJobStatus(jobId);
      setJobStatus(status);
      if (status.status === 'completed' || status.status === 'failed') {
        const runRecord = getDualArmCableGenerateRun(jobId);
        if (runRecord && runRecord.status === 'running') {
          replaceMockDataItem(runRecord.dataItemId, dualArmCableDataItemFromJobStatus(status, runRecord.payload));
          updateDualArmCableGenerateRun(jobId, {
            status: status.status === 'completed' ? 'completed' : 'failed',
          });
        }
      }
    } catch {
      /* polling continues */
    }
  }, [jobId]);

  useEffect(() => {
    void refreshStatus();
    const timer = setInterval(() => void refreshStatus(), 1200);
    return () => clearInterval(timer);
  }, [refreshStatus]);

  const refreshLog = useCallback(async () => {
    setLogLoading(true);
    try {
      const res = await getDualArmCableJobLog(jobId);
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

  const displayStatusRaw = jobStatus?.status ?? run?.status ?? 'running';
  const canOpenReplay = displayStatusRaw === 'completed' && Boolean(jobStatus?.videoExists);

  const openReplay = useCallback(() => {
    router.push(buildDualArmCableReplayHref({ jobId }));
  }, [router, jobId]);

  useEffect(() => {
    onHeaderStateChange?.({ canViewReplay: canOpenReplay, openReplay });
  }, [canOpenReplay, openReplay, onHeaderStateChange]);

  const viewModel = useMemo(
    () =>
      buildDualArmCableRunConsoleViewModel({
        jobId,
        dataId,
        jobStatus,
        payload: run?.payload,
        frameLoaded,
        canViewReplay: canOpenReplay,
      }),
    [jobId, dataId, jobStatus, run?.payload, frameLoaded, canOpenReplay]
  );

  if (!isBackendJobId) {
    return (
      <div style={simConsoleCardStyle}>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          无效的线缆整理后端 jobId：{jobId}。请从数据中心重新启动任务。
        </p>
        <div style={{ marginTop: 12 }}>
          <SecondaryButton onClick={() => router.push('/workspace/data')}>返回数据中心</SecondaryButton>
        </div>
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
