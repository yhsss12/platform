'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  getCableThreadingJobLog,
  getCableThreadingJobStatus,
  type CableThreadingJobStatusResponse,
} from '@/lib/api/cableThreadingClient';
import {
  appendEvaluationTask,
  getCableThreadingEvaluateRun,
  updateCableThreadingEvaluateRun,
} from '@/lib/mock/workspaceMockFlowStore';
import {
  cableThreadingEvalRowFromJobStatus,
  cableThreadingEvalRunResultFromStatus,
} from '@/lib/workspace/cableThreading';
import { adaptCableThreadingEvalJobToConsoleView } from '@/lib/workspace/cableThreadingEvaluationRunAdapter';
import {
  SimulationEventLogDrawer,
  SimulationTaskSummaryBar,
} from '@/components/workspace/simulation/SimulationConsoleSections';
import {
  CableThreadingEvaluationStatusPanel,
  CableThreadingEvaluationViewport,
} from '@/components/workspace/simulation/CableThreadingEvaluationConsoleSections';

export function CableThreadingEvaluateConsole({ evalJobId }: { evalJobId: string }) {
  const router = useRouter();
  const [run, setRun] = useState(() => getCableThreadingEvaluateRun(evalJobId));
  const [jobStatus, setJobStatus] = useState<CableThreadingJobStatusResponse | null>(null);
  const [logTail, setLogTail] = useState('');
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const recordWrittenRef = useRef(false);

  const payload = run?.payload;

  const refreshRun = useCallback(() => {
    setRun(getCableThreadingEvaluateRun(evalJobId));
  }, [evalJobId]);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await getCableThreadingJobStatus(evalJobId);
        if (cancelled) return;
        setJobStatus(status);

        const current = getCableThreadingEvaluateRun(evalJobId);
        const payloadForRecord = current?.payload ?? payload;

        if (status.status === 'completed' || status.status === 'failed') {
          if (current && current.status !== status.status) {
            const liveSnapshot = (status.live ?? {}) as Record<string, unknown>;
            updateCableThreadingEvaluateRun(evalJobId, {
              status: status.status === 'completed' ? 'completed' : 'failed',
              result: cableThreadingEvalRunResultFromStatus(status),
              errorMessage:
                status.status === 'failed'
                  ? String(liveSnapshot.error ?? '线缆穿杆策略评测失败')
                  : undefined,
            });
            refreshRun();
          }

          const written = current?.recordWritten ?? recordWrittenRef.current;
          if (status.status === 'completed' && !written && payloadForRecord) {
            const row = cableThreadingEvalRowFromJobStatus(status, payloadForRecord);
            appendEvaluationTask(row);
            recordWrittenRef.current = true;
            if (current) {
              updateCableThreadingEvaluateRun(evalJobId, { recordWritten: true });
              refreshRun();
            }
          }
        }
      } catch {
        // 轮询失败时保持上一次状态
      }
    };

    void poll();
    const timer = setInterval(() => {
      void poll();
    }, 800);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [evalJobId, payload, refreshRun]);

  useEffect(() => {
    let cancelled = false;
    const pollLog = async () => {
      try {
        const res = await getCableThreadingJobLog(evalJobId);
        if (!cancelled) setLogTail(res.tail);
      } catch {
        // ignore
      }
    };
    void pollLog();
    const timer = setInterval(() => {
      void pollLog();
    }, 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [evalJobId]);

  const view = useMemo(
    () =>
      adaptCableThreadingEvalJobToConsoleView({
        evalJobId,
        status: jobStatus,
        payload,
        logTail,
      }),
    [evalJobId, jobStatus, payload, logTail]
  );

  const openReport = useCallback(() => {
    router.push(view.reportHref);
  }, [router, view.reportHref]);

  const openReplay = useCallback(() => {
    router.push(view.replayHref);
  }, [router, view.replayHref]);

  const openRecords = useCallback(() => {
    router.push(view.recordsHref);
  }, [router, view.recordsHref]);

  return (
    <>
      <div
        style={{
          display: 'flex',
          gap: 16,
          alignItems: 'flex-start',
        }}
      >
        <section style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <SimulationTaskSummaryBar
            sim={view.sim}
            runStatus={view.runStatus}
            context={view.context}
            onControl={() => {}}
            onViewData={openRecords}
            onViewEvaluation={openReport}
            onViewLogs={() => setLogDrawerOpen(true)}
            onViewRecords={openRecords}
            compact
            summaryOverride={{
              taskLabel: view.summary.taskName,
              statusLabel: view.summary.statusLabel,
              progressText: view.summary.progressText,
              showProgressBar: false,
              progressPercent: view.summary.progressPercent,
            }}
          />
          <CableThreadingEvaluationViewport viewport={view.viewport} />
        </section>
        <aside style={{ width: 360, flexShrink: 0 }}>
          <CableThreadingEvaluationStatusPanel view={view} />
        </aside>
      </div>

      <SimulationEventLogDrawer
        open={logDrawerOpen}
        logs={view.logEvents}
        onClose={() => setLogDrawerOpen(false)}
      />
    </>
  );
}
