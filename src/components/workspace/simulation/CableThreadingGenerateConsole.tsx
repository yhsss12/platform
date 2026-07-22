'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  generateCableThreadingDataAsync,
  getCableThreadingJobLog,
  getCableThreadingJobStatus,
  type CableThreadingJobStatusResponse,
} from '@/lib/api/cableThreadingClient';
import {
  getCableThreadingGenerateRun,
  replaceMockDataItem,
  updateCableThreadingGenerateRun,
  updateMockDataItem,
} from '@/lib/mock/workspaceMockFlowStore';
import {
  buildCableThreadingReplayHref,
  cableThreadingDataItemFromJobStatus,
  cableThreadingGenerateRunResultFromStatus,
} from '@/lib/workspace/cableThreading';
import { RunConsoleView } from '@/components/workspace/simulation/RunConsoleView';
import {
  simConsoleCardStyle,
  type SimConsoleHeaderState,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { isValidCableThreadingGenerateJobId } from '@/lib/workspace/backendJobIds';
import { buildCableThreadingRunConsoleViewModel } from '@/lib/workspace/runConsoleAdapters';
import {
  isTerminalSimJobStatus,
  usePageVisibleForPolling,
} from '@/lib/workspace/simulationPolling';

const STATUS_POLL_MS = 1000;
const LOG_POLL_MS = 3000;
const LOG_POLL_OPEN_MS = 3000;

export function CableThreadingGenerateConsole({
  jobId,
  dataId,
  onHeaderStateChange,
}: {
  jobId: string;
  dataId?: string;
  onHeaderStateChange?: (state: SimConsoleHeaderState) => void;
}) {
  const router = useRouter();
  const pageVisible = usePageVisibleForPolling();
  const [run, setRun] = useState(() => getCableThreadingGenerateRun(jobId));
  const [jobStatus, setJobStatus] = useState<CableThreadingJobStatusResponse | null>(null);
  const [logTail, setLogTail] = useState('');
  const [logLoading, setLogLoading] = useState(false);
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const [frameLoaded, setFrameLoaded] = useState(false);

  const refreshRun = useCallback(() => {
    setRun(getCableThreadingGenerateRun(jobId));
  }, [jobId]);

  const isBackendJobId = isValidCableThreadingGenerateJobId(jobId);
  const backendJobId = run?.backendJobId ?? (isBackendJobId ? jobId : undefined);
  const displayStatusRaw = jobStatus?.status ?? run?.status ?? 'running';
  const pollingActive = Boolean(backendJobId) && pageVisible && !isTerminalSimJobStatus(displayStatusRaw);

  useEffect(() => {
    setFrameLoaded(false);
  }, [backendJobId]);

  useEffect(() => {
    if (isBackendJobId) return;

    const current = getCableThreadingGenerateRun(jobId);
    if (!current) return;
    setRun(current);
    if (current.status !== 'running' || current.apiStarted) return;

    updateCableThreadingGenerateRun(jobId, { apiStarted: true });
    refreshRun();

    const payload = current.payload;
    void generateCableThreadingDataAsync({
      episodes: payload.episodes,
      robot: payload.cableThreadingRobot,
      cableModel: payload.cableThreadingCableModel,
      difficulty: payload.cableThreadingDifficulty,
      horizon: payload.cableThreadingHorizon,
      seed: payload.seed,
      outputFormat: payload.dataFormat === 'hdf5' ? 'hdf5' : 'npz',
      saveHdf5: payload.dataFormat === 'hdf5',
      saveProcessVideo: payload.cableThreadingSaveProcessVideo ?? true,
    })
      .then((response) => {
        updateCableThreadingGenerateRun(jobId, { backendJobId: response.jobId });
        refreshRun();
      })
      .catch((err) => {
        updateMockDataItem(current.dataItemId, { status: 'failed', backendJobStatus: 'failed' });
        updateCableThreadingGenerateRun(jobId, {
          status: 'failed',
          errorMessage: err instanceof Error ? err.message : '线缆穿杆异步任务启动失败',
        });
        refreshRun();
      });
  }, [isBackendJobId, jobId, refreshRun]);

  useEffect(() => {
    if (!backendJobId || !pollingActive) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getCableThreadingJobStatus(backendJobId);
        if (cancelled) return;
        setJobStatus(status);

        const current = getCableThreadingGenerateRun(jobId);
        const dataItemId = current?.dataItemId ?? dataId;
        if (!dataItemId) return;

        if (status.status === 'completed' || status.status === 'failed') {
          const runStatus = current?.status ?? 'running';
          if (runStatus !== status.status) {
            const liveSnapshot = (status.live ?? {}) as Record<string, unknown>;
            if (status.status === 'completed' && current?.payload) {
              replaceMockDataItem(dataItemId, cableThreadingDataItemFromJobStatus(status, current.payload));
            } else if (status.status === 'failed') {
              updateMockDataItem(dataItemId, { status: 'failed', backendJobStatus: 'failed' });
            }
            if (current) {
              updateCableThreadingGenerateRun(jobId, {
                status: status.status === 'completed' ? 'completed' : 'failed',
                backendJobId,
                result: cableThreadingGenerateRunResultFromStatus(status),
                errorMessage:
                  status.status === 'failed'
                    ? String(liveSnapshot.error ?? '线缆穿杆数据生成失败')
                    : undefined,
              });
              refreshRun();
            }
          }
        }
      } catch {
        /* polling continues */
      }
    };

    void poll();
    const timer = setInterval(() => void poll(), STATUS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [backendJobId, dataId, jobId, pollingActive, refreshRun]);

  const refreshLog = useCallback(async () => {
    if (!backendJobId) return;
    setLogLoading(true);
    try {
      const res = await getCableThreadingJobLog(backendJobId);
      setLogTail(res.tail?.trim() || '');
    } catch {
      /* ignore */
    } finally {
      setLogLoading(false);
    }
  }, [backendJobId]);

  useEffect(() => {
    if (!backendJobId || !pageVisible || isTerminalSimJobStatus(displayStatusRaw)) return;
    void refreshLog();
    const intervalMs = logDrawerOpen ? LOG_POLL_OPEN_MS : LOG_POLL_MS;
    const timer = setInterval(() => void refreshLog(), intervalMs);
    return () => clearInterval(timer);
  }, [backendJobId, displayStatusRaw, logDrawerOpen, pageVisible, refreshLog]);

  const payload = run?.payload;
  const result = run?.result;
  const live = (jobStatus?.live ?? {}) as Record<string, unknown>;
  const generateVideoExists =
    result?.generateVideoExists === true ||
    live.generateVideoExists === true ||
    jobStatus?.generateVideoExists === true ||
    jobStatus?.paths.generateVideo?.exists === true;
  const processVideoEnabled = payload?.cableThreadingSaveProcessVideo !== false;

  const canOpenReplay =
    displayStatusRaw === 'completed' &&
    Boolean(backendJobId) &&
    (generateVideoExists || !processVideoEnabled);

  const openReplay = useCallback(() => {
    if (backendJobId) router.push(buildCableThreadingReplayHref({ jobId: backendJobId }));
  }, [router, backendJobId]);

  useEffect(() => {
    onHeaderStateChange?.({ canViewReplay: canOpenReplay, openReplay });
  }, [canOpenReplay, openReplay, onHeaderStateChange]);

  const viewModel = useMemo(() => {
    if (!backendJobId) return null;
    return buildCableThreadingRunConsoleViewModel({
      jobId,
      backendJobId,
      dataId,
      localRunId: isBackendJobId ? undefined : jobId,
      jobStatus,
      run,
      frameLoaded,
      canViewReplay: canOpenReplay,
    });
  }, [backendJobId, jobId, dataId, isBackendJobId, jobStatus, run, frameLoaded, canOpenReplay]);

  if (!backendJobId || !viewModel) {
    return (
      <div style={simConsoleCardStyle}>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          无效的线缆穿杆后端 jobId：{jobId}。请从数据中心重新启动任务。
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
