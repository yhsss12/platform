function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function pickFirst(...values: unknown[]): unknown {
  for (const value of values) {
    if (value == null || value === '') continue;
    return value;
  }
  return null;
}

function formatRatio(value: unknown): string {
  if (value == null || value === '') return '-';
  if (typeof value === 'number' && Number.isFinite(value)) {
    const percent = value <= 1 ? value * 100 : value;
    return `${Math.round(percent * 10) / 10}%`;
  }
  if (typeof value === 'string' && value.trim() !== '') return value.trim();
  return '-';
}

function formatNumber(value: unknown, unit?: string): string {
  if (value == null || value === '') return '-';
  if (typeof value === 'number' && Number.isFinite(value)) {
    const rounded = Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000);
    return unit ? `${rounded}${unit}` : rounded;
  }
  if (typeof value === 'string' && value.trim() !== '') return value.trim();
  return '-';
}

function formatDurationSec(value: unknown): string {
  const formatted = formatNumber(value);
  if (formatted === '-') return '-';
  return formatted.endsWith('s') ? formatted : `${formatted}s`;
}

export interface EvaluationReplayCoreMetricRow {
  label: string;
  value: string;
}

export function buildEvaluationReplayCoreMetricRows(
  aggregate: Record<string, unknown> | null | undefined
): EvaluationReplayCoreMetricRow[] {
  const block = aggregate ?? {};
  const metrics = asRecord(block.metrics);
  const summary = asRecord(block.summary);

  return [
    {
      label: '成功率',
      value: formatRatio(
        pickFirst(
          block.successRate,
          block.success_rate,
          block.final_success_rate,
          summary.successRate,
          summary.success_rate,
          metrics.successRate,
          metrics.success_rate,
          metrics.final_success_rate
        )
      ),
    },
    {
      label: '操作效率',
      value: (() => {
        const explicit = pickFirst(
          block.operationEfficiency,
          block.operation_efficiency,
          metrics.operationEfficiency,
          metrics.operation_efficiency
        );
        if (explicit != null && explicit !== '') {
          const asNum = Number(explicit);
          if (Number.isFinite(asNum) && asNum > 0 && asNum <= 1) return formatRatio(explicit);
          return formatNumber(explicit);
        }
        const meanSteps = pickFirst(
          block.meanSteps,
          block.mean_steps,
          block.meanEpisodeLength,
          metrics.meanSteps,
          metrics.mean_steps
        );
        if (meanSteps != null && Number(meanSteps) > 0) {
          return formatNumber(1 / Number(meanSteps));
        }
        return '-';
      })(),
    },
    {
      label: '平均步数',
      value: formatNumber(
        pickFirst(
          block.meanSteps,
          block.mean_steps,
          block.meanEpisodeLength,
          block.mean_episode_length,
          metrics.meanSteps,
          metrics.mean_steps,
          metrics.meanEpisodeLength
        )
      ),
    },
    {
      label: '平均耗时',
      value: formatDurationSec(
        pickFirst(
          block.meanDurationSec,
          block.mean_duration_sec,
          block.avg_duration_sec,
          summary.meanDurationSec,
          metrics.meanDurationSec,
          metrics.mean_duration_sec
        )
      ),
    },
    {
      label: '失败次数',
      value: formatNumber(
        pickFirst(block.failureCount, block.failure_count, metrics.failureCount, metrics.failure_count)
      ),
    },
    {
      label: '超时率',
      value: formatRatio(
        pickFirst(block.timeoutRate, block.timeout_rate, metrics.timeoutRate, metrics.timeout_rate)
      ),
    },
  ];
}

export function hasEvaluationReplayCoreMetrics(
  aggregate: Record<string, unknown> | null | undefined
): boolean {
  if (!aggregate || Object.keys(aggregate).length === 0) return false;
  return buildEvaluationReplayCoreMetricRows(aggregate).some((row) => row.value !== '-');
}
