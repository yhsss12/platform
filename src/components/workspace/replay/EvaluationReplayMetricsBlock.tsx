'use client';

import { InfoRow } from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import {
  buildLegacySuccessRateOnlyRow,
  buildMetricResultDisplayRows,
  type EvaluationMetricResultEntry,
} from '@/lib/workspace/evaluationMetricResultsDisplay';
import {
  buildEvaluationMetricDisplayRows,
  resolveEvaluationMetricsSource,
  type EvaluationMetricsInput,
} from '@/lib/workspace/evaluationLiveMetrics';

export function EvaluationReplayMetricsBlock({
  aggregate,
  metrics,
  live,
  jobStatus,
  loading = false,
  metricsNotGenerated = false,
  metricResults,
  selectedMetricIds,
}: {
  aggregate?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  live?: Record<string, unknown> | null;
  jobStatus?: string | null;
  loading?: boolean;
  metricsNotGenerated?: boolean;
  metricResults?: Record<string, EvaluationMetricResultEntry> | null;
  selectedMetricIds?: string[] | null;
}) {
  if (loading) {
    return (
      <>
        <InfoRow label="评测指标" value="加载中…" />
      </>
    );
  }

  if (metricsNotGenerated) {
    return <InfoRow label="评测指标" value="未生成" />;
  }

  const resolvedMetricResults =
    metricResults ??
    (aggregate?.metricResults as Record<string, EvaluationMetricResultEntry> | undefined) ??
    null;

  const metricRows = buildMetricResultDisplayRows(resolvedMetricResults, selectedMetricIds);
  if (metricRows.length > 0) {
    return (
      <>
        {metricRows.map((row) => (
          <div key={row.metricId} title={row.reason}>
            <InfoRow label={row.label} value={row.value} />
          </div>
        ))}
      </>
    );
  }

  const legacyRows = buildLegacySuccessRateOnlyRow(
    aggregate ?? resolveEvaluationMetricsSource({ aggregate, metrics, live, jobStatus })
  );
  if (legacyRows.length > 0) {
    return (
      <>
        {legacyRows.map((row) => (
          <InfoRow key={row.metricId} label={row.label} value={row.value} />
        ))}
      </>
    );
  }

  const input: EvaluationMetricsInput = { aggregate, metrics, live, jobStatus };
  const rows = buildEvaluationMetricDisplayRows(input, { loading, metricsNotGenerated });
  const successOnly = rows.filter((row) => row.label === '成功率' && row.value !== '-');
  if (successOnly.length > 0) {
    return (
      <>
        {successOnly.map((row) => (
          <InfoRow key={row.label} label={row.label} value={row.value} />
        ))}
      </>
    );
  }

  return null;
}

export { resolveEvaluationMetricsSource, buildEvaluationMetricDisplayRows };
