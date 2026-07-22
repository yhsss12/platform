'use client';

import { type ReactNode, useMemo, useState } from 'react';
import { EvaluationReplayMetricsBlock } from '@/components/workspace/replay/EvaluationReplayMetricsBlock';
import {
  ReplayPanelSectionTitle,
  ReplaySidePanelLayout,
} from '@/components/workspace/replay/ReplaySidePanelLayout';
import { RunLogDrawer } from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import {
  evalFailedStageLabel,
  evalFailureReasonLabel,
  type EvalFailureDiagnosis,
} from '@/lib/workspace/evaluationFailureDiagnosis';
import {
  evaluationMetricsInputFromCableStatus,
  shouldShowMetricsNotGenerated,
  type EvaluationMetricsInput,
} from '@/lib/workspace/evaluationLiveMetrics';
import { InfoRow } from '@/components/workspace/simulation/SimulationRunConsoleLayout';

function LogLinkButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        background: 'none',
        border: 'none',
        padding: 0,
        color: '#2563eb',
        fontSize: 13,
        cursor: 'pointer',
        textDecoration: 'underline',
      }}
    >
      {label}
    </button>
  );
}

export function EvalReplaySidePanel({
  evalJobId,
  taskType,
  basicInfo,
  progressInfo,
  aggregate,
  metrics,
  live,
  jobStatus,
  loading = false,
  failureDiagnosis,
  metricResults,
  selectedMetricIds,
  onFetchLog,
}: {
  evalJobId?: string;
  taskType?: string;
  basicInfo: ReactNode;
  progressInfo?: ReactNode;
  aggregate?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  live?: Record<string, unknown> | null;
  jobStatus?: string | null;
  loading?: boolean;
  failureDiagnosis?: EvalFailureDiagnosis | null;
  metricResults?: Record<string, import('@/lib/workspace/evaluationMetricResultsDisplay').EvaluationMetricResultEntry> | null;
  selectedMetricIds?: string[] | null;
  onFetchLog?: () => Promise<string>;
}) {
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const [logTail, setLogTail] = useState('');
  const [logLoading, setLogLoading] = useState(false);

  const metricsInput: EvaluationMetricsInput = useMemo(
    () => ({
      aggregate,
      metrics,
      live,
      jobStatus,
    }),
    [aggregate, metrics, live, jobStatus]
  );

  const metricsNotGenerated = shouldShowMetricsNotGenerated(metricsInput);
  const showFailure = jobStatus === 'failed' && failureDiagnosis;

  const openLogDrawer = () => {
    setLogDrawerOpen(true);
    if (!onFetchLog || logTail) return;
    setLogLoading(true);
    void onFetchLog()
      .then((tail) => setLogTail(tail))
      .finally(() => setLogLoading(false));
  };

  return (
    <>
      <ReplaySidePanelLayout title="运行状态">
        <div style={{ marginBottom: 12 }}>
          <ReplayPanelSectionTitle>基础信息</ReplayPanelSectionTitle>
          {basicInfo}
        </div>

        {progressInfo ? (
          <div style={{ marginBottom: 12 }}>
            <ReplayPanelSectionTitle>运行进度</ReplayPanelSectionTitle>
            {progressInfo}
          </div>
        ) : null}

        <div style={{ marginBottom: 12 }}>
          <ReplayPanelSectionTitle>评测指标</ReplayPanelSectionTitle>
          <EvaluationReplayMetricsBlock
            aggregate={aggregate}
            metrics={metrics}
            live={live}
            jobStatus={jobStatus}
            loading={loading}
            metricsNotGenerated={metricsNotGenerated}
            metricResults={metricResults}
            selectedMetricIds={selectedMetricIds}
          />
        </div>

        {showFailure ? (
          <div style={{ marginBottom: 12 }}>
            <ReplayPanelSectionTitle>失败诊断</ReplayPanelSectionTitle>
            <InfoRow
              label="失败阶段"
              value={evalFailedStageLabel(failureDiagnosis?.failedStage)}
            />
            <InfoRow
              label="失败原因"
              value={evalFailureReasonLabel(failureDiagnosis?.failureReason)}
            />
            {failureDiagnosis?.errorMessage ? (
              <InfoRow label="错误说明" value={failureDiagnosis.errorMessage} />
            ) : null}
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{ fontSize: 12, color: '#6b7280' }}>日志</span>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <LogLinkButton label="查看 run.log" onClick={openLogDrawer} />
                {failureDiagnosis?.logPaths?.stdout ? (
                  <LogLinkButton label="查看 stdout" onClick={openLogDrawer} />
                ) : null}
                {failureDiagnosis?.logPaths?.stderr ? (
                  <LogLinkButton label="查看 stderr" onClick={openLogDrawer} />
                ) : null}
              </div>
            </div>
          </div>
        ) : null}

        {evalJobId && onFetchLog ? (
          <div style={{ marginBottom: 4 }}>
            <ReplayPanelSectionTitle>日志文件</ReplayPanelSectionTitle>
            <LogLinkButton label="查看完整日志" onClick={openLogDrawer} />
          </div>
        ) : null}
      </ReplaySidePanelLayout>

      <RunLogDrawer
        open={logDrawerOpen}
        logTail={logTail}
        loading={logLoading}
        onClose={() => setLogDrawerOpen(false)}
      />
    </>
  );
}

export { evaluationMetricsInputFromCableStatus };
