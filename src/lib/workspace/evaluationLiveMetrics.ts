import {
  buildEvaluationReplayCoreMetricRows,
  hasEvaluationReplayCoreMetrics,
  type EvaluationReplayCoreMetricRow,
} from '@/lib/workspace/evaluationReplayCoreMetrics';

export interface EvaluationMetricsInput {
  aggregate?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  live?: Record<string, unknown> | null;
  jobStatus?: string | null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asEpisodeRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object');
}

function isTimeoutEpisode(row: Record<string, unknown>): boolean {
  const reason = String(row.failure_reason ?? row.failureReason ?? row.fail_reason ?? '').toLowerCase();
  return reason.includes('timeout') || reason.includes('超时') || row.timed_out === true || row.timedOut === true;
}

function computeFromEpisodes(episodes: Record<string, unknown>[]): Record<string, unknown> | null {
  if (episodes.length === 0) return null;

  let successCount = 0;
  let timeoutCount = 0;
  let stepSum = 0;
  let stepCount = 0;
  let durationSum = 0;
  let durationCount = 0;
  let failureCount = 0;

  for (const row of episodes) {
    const success = row.success === true || row.final_success === true || row.finalSuccess === true;
    if (success) successCount += 1;
    else failureCount += 1;
    if (isTimeoutEpisode(row)) timeoutCount += 1;

    const steps = row.steps ?? row.step_count ?? row.episode_length ?? row.episodeLength;
    if (typeof steps === 'number' && Number.isFinite(steps)) {
      stepSum += steps;
      stepCount += 1;
    }

    const duration = row.duration_sec ?? row.durationSec ?? row.duration ?? row.elapsed_sec;
    if (typeof duration === 'number' && Number.isFinite(duration)) {
      durationSum += duration;
      durationCount += 1;
    }
  }

  const finished = episodes.length;
  const computed: Record<string, unknown> = {};
  if (finished > 0) computed.successRate = successCount / finished;
  if (failureCount > 0) computed.failureCount = failureCount;
  if (finished > 0 && timeoutCount > 0) computed.timeoutRate = timeoutCount / finished;
  if (stepCount > 0) computed.meanSteps = stepSum / stepCount;
  if (durationCount > 0) computed.meanDurationSec = durationSum / durationCount;
  return Object.keys(computed).length > 0 ? computed : null;
}

function computeFromFailureSummary(
  items: Array<Record<string, unknown>>,
  finishedEpisodes: number
): Record<string, unknown> | null {
  if (items.length === 0) return null;

  let successCount = 0;
  let timeoutCount = 0;
  let failureCount = 0;

  for (const item of items) {
    if (item.success === true) successCount += 1;
    else failureCount += 1;
    const reason = String(item.failureReason ?? item.failure_reason ?? '').toLowerCase();
    if (reason.includes('timeout') || reason.includes('超时')) timeoutCount += 1;
  }

  const inferredFinished = Math.max(finishedEpisodes, items.length, 1);
  const computed: Record<string, unknown> = {
    successRate: successCount / inferredFinished,
  };
  if (failureCount > 0) computed.failureCount = failureCount;
  else if (inferredFinished > successCount) computed.failureCount = inferredFinished - successCount;
  if (timeoutCount > 0) computed.timeoutRate = timeoutCount / inferredFinished;
  return computed;
}

function applyCountBasedMetrics(
  merged: Record<string, unknown>,
  metrics: Record<string, unknown>,
  live: Record<string, unknown>
): void {
  const completedEpisodes = Number(
    live.completedEpisodes ?? live.episode ?? metrics.completedEpisodes ?? metrics.numEpisodes ?? 0
  );
  const successfulEpisodes = Number(
    metrics.successfulEpisodes ?? live.successfulEpisodes ?? 0
  );
  const failedEpisodes = Number(metrics.failedEpisodes ?? live.failedEpisodes ?? 0);

  if (merged.successRate == null && completedEpisodes > 0 && successfulEpisodes >= 0) {
    merged.successRate = successfulEpisodes / completedEpisodes;
  }
  if (merged.failureCount == null) {
    if (failedEpisodes > 0) merged.failureCount = failedEpisodes;
    else if (completedEpisodes > 0 && successfulEpisodes >= 0) {
      const failed = completedEpisodes - successfulEpisodes;
      if (failed > 0) merged.failureCount = failed;
    }
  }
}

function applyOperationEfficiencyFallback(merged: Record<string, unknown>): void {
  if (merged.operationEfficiency != null || merged.operation_efficiency != null) return;
  const meanSteps = Number(merged.meanSteps ?? merged.mean_steps ?? merged.meanEpisodeLength);
  if (Number.isFinite(meanSteps) && meanSteps > 0) {
    merged.operationEfficiency = 1 / meanSteps;
  }
}

export function resolveEvaluationMetricsSource(input: EvaluationMetricsInput): Record<string, unknown> {
  const metrics = asRecord(input.metrics);
  const aggregateBlock = asRecord(input.aggregate);
  const nestedAggregate = asRecord(metrics.aggregate);
  const live = asRecord(input.live);

  const merged: Record<string, unknown> = {
    ...nestedAggregate,
    ...aggregateBlock,
    ...metrics,
  };

  const episodeSources = [
    aggregateBlock.episodes,
    aggregateBlock.perEpisode,
    nestedAggregate.episodes,
    nestedAggregate.perEpisode,
    metrics.episodes,
    metrics.perEpisode,
    live.episodes,
    live.perEpisode,
  ];
  const episodes = asEpisodeRows(
    episodeSources.find((source) => Array.isArray(source) && source.length > 0)
  );
  const computedFromEpisodes = computeFromEpisodes(episodes);

  const failureSummary = Array.isArray(metrics.failureSummary)
    ? (metrics.failureSummary as Array<Record<string, unknown>>)
    : [];
  const finishedEpisodes = Number(
    live.completedEpisodes ?? live.episode ?? metrics.completedEpisodes ?? episodes.length ?? 0
  );
  const computedFromFailures = computeFromFailureSummary(failureSummary, finishedEpisodes);

  for (const partial of [computedFromFailures, computedFromEpisodes]) {
    if (!partial) continue;
    for (const [key, value] of Object.entries(partial)) {
      if (merged[key] == null || merged[key] === '') merged[key] = value;
    }
  }

  applyCountBasedMetrics(merged, metrics, live);

  if (merged.successRate == null && metrics.successRate != null) {
    merged.successRate = metrics.successRate;
  }
  if (merged.successRate == null && metrics.finalSuccessRate != null) {
    merged.successRate = metrics.finalSuccessRate;
  }
  if (merged.success_rate == null && merged.successRate != null) {
    merged.success_rate = merged.successRate;
  }

  applyOperationEfficiencyFallback(merged);

  return merged;
}

export function buildEvaluationMetricDisplayRows(
  input: EvaluationMetricsInput,
  options?: { loading?: boolean; metricsNotGenerated?: boolean }
): EvaluationReplayCoreMetricRow[] {
  if (options?.loading) {
    return buildEvaluationReplayCoreMetricRows({});
  }
  if (options?.metricsNotGenerated) {
    return [{ label: '评测指标', value: '未生成' }];
  }
  const source = resolveEvaluationMetricsSource(input);
  return buildEvaluationReplayCoreMetricRows(source);
}

export function shouldShowMetricsNotGenerated(input: EvaluationMetricsInput): boolean {
  const status = String(input.jobStatus ?? '').toLowerCase();
  if (status !== 'failed' && status !== 'completed') return false;

  const aggregate = input.aggregate;
  const hasAggregateFile =
    aggregate != null &&
    typeof aggregate === 'object' &&
    Object.keys(aggregate).length > 0 &&
    Number(
      (aggregate as Record<string, unknown>).total_episodes ??
        (aggregate as Record<string, unknown>).numEpisodes ??
        (aggregate as Record<string, unknown>).num_episodes ??
        0
    ) > 0;

  if (hasAggregateFile) return false;

  const source = resolveEvaluationMetricsSource(input);
  return !hasEvaluationReplayCoreMetrics(source);
}

export function evaluationMetricsInputFromCableStatus(
  status: {
    status?: string;
    metrics?: Record<string, unknown>;
    live?: Record<string, unknown>;
  } | null | undefined
): EvaluationMetricsInput {
  return {
    aggregate: (status?.metrics?.aggregate as Record<string, unknown> | undefined) ?? null,
    metrics: status?.metrics ?? null,
    live: status?.live ?? null,
    jobStatus: status?.status ?? null,
  };
}

export { hasEvaluationReplayCoreMetrics };
