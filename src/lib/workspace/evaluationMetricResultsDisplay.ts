import {
  filterMetricIdsForDisplay,
  isReportBodyMetricId,
} from '@/lib/workspace/evaluationMetricPolicy';

export interface EvaluationMetricResultEntry {
  metricId: string;
  displayName: string;
  value: number | null;
  formattedValue: string;
  unit?: string;
  available: boolean;
  reason?: string;
  source?: string;
  description?: string;
}

const METRIC_DISPLAY_NAME_ALIASES: Record<string, string> = {
  metric_runtime_mean_runtime_sec_v1: '平均仿真时长',
};

function resolveMetricDisplayName(entry: EvaluationMetricResultEntry): string {
  if (entry.displayName === '平均耗时') {
    return '平均仿真时长';
  }
  return METRIC_DISPLAY_NAME_ALIASES[entry.metricId] ?? entry.displayName;
}

export interface EvaluationMetricResultDisplayRow {
  metricId: string;
  label: string;
  value: string;
  reason?: string;
  available: boolean;
}

export function extractMetricResultsFromAggregate(
  aggregate: Record<string, unknown> | null | undefined
): Record<string, EvaluationMetricResultEntry> | null {
  if (!aggregate) return null;
  const block = aggregate.metricResults;
  if (!block || typeof block !== 'object' || Array.isArray(block)) return null;
  return block as Record<string, EvaluationMetricResultEntry>;
}

export function extractSelectedMetricIds(
  sources: Array<Record<string, unknown> | null | undefined>
): string[] | null {
  for (const source of sources) {
    if (!source) continue;
    const ids = source.selectedMetricIds;
    if (Array.isArray(ids) && ids.length > 0) {
      return ids.map((item) => String(item)).filter(Boolean);
    }
  }
  return null;
}

export function buildMetricResultDisplayRows(
  metricResults: Record<string, EvaluationMetricResultEntry> | null | undefined,
  selectedMetricIds?: string[] | null
): EvaluationMetricResultDisplayRow[] {
  if (!metricResults || Object.keys(metricResults).length === 0) return [];

  const orderedIds = filterMetricIdsForDisplay(
    selectedMetricIds && selectedMetricIds.length > 0
      ? selectedMetricIds
      : Object.keys(metricResults)
  );

  return orderedIds
    .filter((metricId) => isReportBodyMetricId(metricId))
    .map((metricId) => {
      const entry = metricResults[metricId];
      if (entry) return entry;
      return {
        metricId,
        displayName: metricId,
        value: null,
        formattedValue: '-',
        available: false,
        reason: '指标结果尚未生成',
      } satisfies EvaluationMetricResultEntry;
    })
    .map((entry) => ({
      metricId: entry.metricId,
      label: resolveMetricDisplayName(entry),
      value: entry.available ? entry.formattedValue || String(entry.value ?? '-') : '-',
      reason: entry.available ? entry.description ?? undefined : entry.reason,
      available: entry.available,
    }));
}

export function buildLegacySuccessRateOnlyRow(
  aggregate: Record<string, unknown> | null | undefined
): EvaluationMetricResultDisplayRow[] {
  if (!aggregate) return [];
  const rate =
    aggregate.success_rate ??
    aggregate.final_success_rate ??
    aggregate.successRate ??
    aggregate.finalSuccessRate;
  if (rate == null || rate === '') return [];
  const numeric = Number(rate);
  if (!Number.isFinite(numeric)) return [];
  const percent = numeric <= 1 ? numeric * 100 : numeric;
  const rounded = Math.round(percent * 10) / 10;
  const formatted = `${Number.isInteger(rounded) ? rounded : rounded}%`;
  return [
    {
      metricId: 'metric_cable_success_rate_v1',
      label: '成功率',
      value: formatted,
      available: true,
    },
  ];
}
