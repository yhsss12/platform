'use client';

import {
  buildLegacySuccessRateOnlyRow,
  buildMetricResultDisplayRows,
  extractMetricResultsFromAggregate,
  extractSelectedMetricIds,
  type EvaluationMetricResultDisplayRow,
} from '@/lib/workspace/evaluationMetricResultsDisplay';
import {
  buildEvaluationReportCoreMetrics,
  EVALUATION_REPORT_CORE_METRIC_LABELS,
  type EvaluationReportCoreMetricRow,
} from '@/lib/workspace/evaluationReportCoreMetrics';

/** 评测报告页核心指标：不展示的项（仅影响本面板渲染） */
const HIDDEN_REPORT_CORE_METRIC_LABELS = new Set<string>([
  EVALUATION_REPORT_CORE_METRIC_LABELS.everSuccessRate,
  EVALUATION_REPORT_CORE_METRIC_LABELS.evalStability,
  EVALUATION_REPORT_CORE_METRIC_LABELS.taskCompletion,
  EVALUATION_REPORT_CORE_METRIC_LABELS.operationSpread,
]);

function filterReportPageCoreMetrics(
  items: EvaluationReportCoreMetricRow[]
): EvaluationReportCoreMetricRow[] {
  return items.filter((row) => !HIDDEN_REPORT_CORE_METRIC_LABELS.has(row.label));
}

function metricResultRowsToReportRows(
  rows: EvaluationMetricResultDisplayRow[]
): EvaluationReportCoreMetricRow[] {
  return rows.map((row) => ({
    label: row.label,
    value: row.value,
    hint: row.reason,
  }));
}

function InfoGrid({ items }: { items: EvaluationReportCoreMetricRow[] }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
        gap: '12px 24px',
      }}
    >
      {items.map((item) => (
        <div key={item.label} title={item.hint}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.label}</div>
          <div style={{ fontSize: 14, color: '#111827', wordBreak: 'break-word' }}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}

export function EvaluationReportCoreMetricsPanel({
  aggregate,
  perEpisode,
  loading,
  error,
}: {
  aggregate: Record<string, unknown> | null | undefined;
  perEpisode?: unknown[];
  loading?: boolean;
  error?: string | null;
}) {
  if (loading) {
    return <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>正在加载评测结果…</p>;
  }

  if (error) {
    return (
      <p style={{ margin: 0, fontSize: 13, color: '#b45309', lineHeight: 1.6 }}>
        {error}
      </p>
    );
  }

  const metricResults = extractMetricResultsFromAggregate(aggregate);
  const selectedMetricIds = extractSelectedMetricIds([aggregate ?? null]);
  const selectedRows = buildMetricResultDisplayRows(metricResults, selectedMetricIds);
  if (selectedRows.length > 0) {
    return <InfoGrid items={metricResultRowsToReportRows(selectedRows)} />;
  }

  const legacyRows = buildLegacySuccessRateOnlyRow(aggregate);
  if (legacyRows.length > 0) {
    return <InfoGrid items={metricResultRowsToReportRows(legacyRows)} />;
  }

  const items = filterReportPageCoreMetrics(
    buildEvaluationReportCoreMetrics(aggregate, { perEpisode })
  );
  const successOnly = items.filter((row) => row.label === '成功率' && row.value !== '-');
  if (successOnly.length > 0) {
    return <InfoGrid items={successOnly} />;
  }

  return items.length > 0 ? <InfoGrid items={items} /> : null;
}
