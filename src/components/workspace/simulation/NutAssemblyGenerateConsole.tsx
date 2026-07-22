'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  getNutAssemblyJobLog,
  getNutAssemblyJobResult,
  getNutAssemblyJobStatus,
  type NutAssemblyJobStatusResponse,
} from '@/lib/api/nutAssemblyClient';
import { RunConsoleView } from '@/components/workspace/simulation/RunConsoleView';
import { buildNutAssemblyRunConsoleViewModel } from '@/lib/workspace/runConsoleAdapters';
import { isValidNutAssemblyGenerateJobId } from '@/lib/workspace/backendJobIds';
import { buildNutAssemblyReplayHref, mergeNutAssemblyJobWithResult } from '@/lib/workspace/nutAssembly';
import {
  isTerminalSimJobStatus,
  usePageVisibleForPolling,
} from '@/lib/workspace/simulationPolling';
import type { SimConsoleHeaderState } from '@/components/workspace/simulation/SimulationRunConsoleLayout';

const STATUS_POLL_MS = 2000;
const LOG_POLL_MS = 5000;

export function NutAssemblyGenerateConsole({
  jobId,
  onHeaderStateChange,
}: {
  jobId: string;
  dataId?: string;
  onHeaderStateChange?: (state: SimConsoleHeaderState) => void;
}) {
  const router = useRouter();
  const pageVisible = usePageVisibleForPolling();
  const [jobStatus, setJobStatus] = useState<NutAssemblyJobStatusResponse | null>(null);
  const [jobResult, setJobResult] = useState<Record<string, unknown> | null>(null);
  const [logTail, setLogTail] = useState('');
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const [logLoading, setLogLoading] = useState(false);
  const [frameLoaded, setFrameLoaded] = useState(false);
  const logTailRef = useRef('');

  useEffect(() => {
    logTailRef.current = logTail;
  }, [logTail]);

  const backendJobId = isValidNutAssemblyGenerateJobId(jobId) ? jobId : undefined;
  const displayStatusRaw = jobStatus?.status ?? 'running';
  const pollingActive = Boolean(backendJobId) && pageVisible && !isTerminalSimJobStatus(displayStatusRaw);

  const refreshLog = useCallback(async () => {
    if (!backendJobId) return;
    setLogLoading(true);
    try {
      const res = await getNutAssemblyJobLog(backendJobId, 20);
      setLogTail(res.tail?.trim() || '');
    } catch {
      /* keep polling */
    } finally {
      setLogLoading(false);
    }
  }, [backendJobId]);

  const refreshStatus = useCallback(async () => {
    if (!backendJobId) return;
    try {
      const status = await getNutAssemblyJobStatus(backendJobId);
      if (isTerminalSimJobStatus(status.status)) {
        const result = await getNutAssemblyJobResult(backendJobId).catch(() => null);
        setJobResult(result);
        setJobStatus(mergeNutAssemblyJobWithResult(status, result, logTailRef.current));
      } else {
        setJobStatus(status);
      }
    } catch {
      /* keep polling */
    }
  }, [backendJobId]);

  useEffect(() => {
    if (!backendJobId) return;
    void refreshStatus();
    void refreshLog();
  }, [backendJobId, refreshStatus, refreshLog]);

  useEffect(() => {
    if (!backendJobId || !pollingActive) return;
    const timer = setInterval(() => void refreshStatus(), STATUS_POLL_MS);
    return () => clearInterval(timer);
  }, [backendJobId, pollingActive, refreshStatus]);

  useEffect(() => {
    if (!backendJobId || !pageVisible) return;
    void refreshLog();
    const timer = setInterval(() => void refreshLog(), LOG_POLL_MS);
    return () => clearInterval(timer);
  }, [backendJobId, pageVisible, refreshLog]);

  const viewModel = buildNutAssemblyRunConsoleViewModel({
    jobId: backendJobId ?? jobId,
    jobStatus,
    jobResult,
    logTail,
  });

  const canOpenReplay = viewModel.actions.canViewReplay;

  const openReplay = useCallback(() => {
    if (backendJobId) {
      router.push(buildNutAssemblyReplayHref({ jobId: backendJobId }));
    }
  }, [backendJobId, router]);

  useEffect(() => {
    onHeaderStateChange?.({
      canViewReplay: canOpenReplay,
      openReplay,
    });
  }, [onHeaderStateChange, canOpenReplay, openReplay]);

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
