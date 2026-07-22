/**
 * 评测报告 — 统一核心指标（固定布局、中文标签）
 */

export interface EvaluationReportCoreMetricRow {
  label: string;
  value: string;
  hint?: string;
}

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

function pickNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

function pickRate(...values: unknown[]): number | null {
  const raw = pickNumber(...values);
  if (raw == null) return null;
  if (raw > 1) return raw / 100;
  return raw;
}

function formatPercent(rate: number | null | undefined): string {
  if (rate == null || !Number.isFinite(rate)) return '-';
  return `${(rate * 100).toFixed(1)}%`;
}

function formatMetricNumber(value: unknown): string {
  if (value == null || value === '') return '-';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '-';
    if (Number.isInteger(value)) return String(value);
    if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
    return value.toFixed(4);
  }
  const trimmed = String(value).trim();
  return trimmed || '-';
}

function computeMeanStepsFromEpisodes(perEpisode: unknown[]): number | null {
  if (!Array.isArray(perEpisode) || perEpisode.length === 0) return null;
  const values: number[] = [];
  for (const item of perEpisode) {
    const ep = asRecord(item);
    const steps = pickNumber(
      ep.stepsExecuted,
      ep.steps_executed,
      ep.meanSteps,
      ep.mean_steps,
      ep.horizon,
      ep.episodeLength,
      ep.episode_length
    );
    if (steps != null) values.push(steps);
  }
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function normalizeReportAggregate(
  payload: Record<string, unknown> | null | undefined
): Record<string, unknown> {
  if (!payload) return {};
  const nested = asRecord(payload.aggregate);
  const taskMetrics = asRecord(payload.taskMetrics);
  const metrics = asRecord(payload.metrics);
  if (Object.keys(nested).length > 0) return nested;
  if (Object.keys(taskMetrics).length > 0) {
    return asRecord(taskMetrics.aggregate).final_success_rate != null ||
      taskMetrics.final_success_rate != null
      ? { ...taskMetrics, ...asRecord(taskMetrics.aggregate) }
      : taskMetrics;
  }
  if (payload.final_success_rate != null || payload.success_rate != null) return payload;
  if (Object.keys(metrics).length > 0) {
    return asRecord(metrics.aggregate).final_success_rate != null
      ? asRecord(metrics.aggregate)
      : metrics;
  }
  return payload;
}

export function resolveReportPerEpisode(payload: Record<string, unknown> | null | undefined): unknown[] {
  if (!payload) return [];
  if (Array.isArray(payload.perEpisode)) return payload.perEpisode;
  if (Array.isArray(payload.per_episode_results)) return payload.per_episode_results;
  if (Array.isArray(payload.episodes)) return payload.episodes;
  const taskMetrics = asRecord(payload.taskMetrics);
  if (Array.isArray(taskMetrics.episodes)) return taskMetrics.episodes;
  const aggregate = normalizeReportAggregate(payload);
  if (Array.isArray(aggregate.perEpisode)) return aggregate.perEpisode as unknown[];
  return [];
}

/** 固定顺序的核心指标中文标签 */
export const EVALUATION_REPORT_CORE_METRIC_LABELS = {
  finalSuccessRate: '最终成功率',
  everSuccessRate: '历史成功率',
  evalStability: '评测稳定性',
  meanSteps: '平均执行步数',
  taskCompletion: '平均任务完成度',
  targetError: '平均目标误差',
  trajectoryError: '平均轨迹误差',
  endpointError: '平均端点误差',
  operationSpread: '平均操作扩散度',
} as const;

export function buildEvaluationReportCoreMetrics(
  aggregate: Record<string, unknown> | null | undefined,
  options?: { perEpisode?: unknown[] }
): EvaluationReportCoreMetricRow[] {
  const block = aggregate ?? {};
  const metrics = asRecord(block.metrics);
  const summary = asRecord(block.summary);
  const taskMetrics = asRecord(block.taskMetrics);
  const perEpisode = options?.perEpisode ?? [];

  const finalSuccessRate = pickRate(
    block.final_success_rate,
    block.finalSuccessRate,
    metrics.final_success_rate,
    metrics.finalSuccessRate,
    summary.final_success_rate,
    summary.finalSuccessRate
  );

  const everSuccessRate = pickRate(
    block.ever_success_rate,
    block.everSuccessRate,
    metrics.ever_success_rate,
    metrics.everSuccessRate,
    summary.ever_success_rate,
    summary.everSuccessRate
  );

  const evalStability = (() => {
    const direct = pickRate(
      block.successRate,
      block.success_rate,
      summary.successRate,
      summary.success_rate,
      metrics.successRate,
      metrics.success_rate
    );
    if (direct != null) return direct;
    const successEpisodes = pickNumber(
      block.success_episodes,
      block.successEpisodes,
      block.successfulEpisodes,
      summary.successEpisodes,
      summary.successfulEpisodes
    );
    const completedEpisodes = pickNumber(
      block.completedEpisodes,
      block.total_episodes,
      block.totalEpisodes,
      block.episodeCount,
      block.episodes,
      summary.totalEpisodes,
      summary.completedEpisodes,
      perEpisode.length > 0 ? perEpisode.length : null
    );
    if (successEpisodes != null && completedEpisodes != null && completedEpisodes > 0) {
      return successEpisodes / completedEpisodes;
    }
    return null;
  })();

  const meanSteps =
    pickNumber(
      block.meanSteps,
      block.mean_steps,
      metrics.meanSteps,
      metrics.mean_steps,
      metrics.averageSteps,
      summary.meanSteps,
      summary.mean_steps,
      block.meanEpisodeLength,
      block.mean_episode_length,
      metrics.meanEpisodeLength
    ) ?? computeMeanStepsFromEpisodes(perEpisode);

  const taskCompletion = pickNumber(
    block.mean_thread_completion_max,
    metrics.mean_thread_completion_max,
    block.mean_task_completion,
    metrics.mean_task_completion,
    block.task_completion,
    metrics.task_completion,
    taskMetrics.meanTaskCompletion,
    taskMetrics.mean_task_completion,
    taskMetrics.stretchReachedRate,
    taskMetrics.contactSuccessRate
  );

  const targetError = pickNumber(
    block.mean_endpoint_goal_error_final,
    metrics.mean_endpoint_goal_error_final,
    block.mean_target_error,
    metrics.mean_target_error
  );

  const trajectoryError = pickNumber(
    block.mean_straightness_error_final,
    metrics.mean_straightness_error_final,
    block.mean_trajectory_error,
    metrics.mean_trajectory_error
  );

  const endpointError = pickNumber(
    block.mean_anchor_error_final,
    metrics.mean_anchor_error_final,
    block.mean_endpoint_error,
    metrics.mean_endpoint_error
  );

  const operationSpread = pickNumber(
    block.mean_tabletop_spread_final,
    metrics.mean_tabletop_spread_final,
    block.mean_operation_spread,
    metrics.mean_operation_spread
  );

  const finalSuccessDisplay = finalSuccessRate ?? evalStability;

  return [
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.finalSuccessRate, value: formatPercent(finalSuccessDisplay) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.everSuccessRate, value: formatPercent(everSuccessRate) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.evalStability, value: formatPercent(evalStability) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.meanSteps, value: formatMetricNumber(meanSteps) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.taskCompletion, value: formatMetricNumber(taskCompletion) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.targetError, value: formatMetricNumber(targetError) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.trajectoryError, value: formatMetricNumber(trajectoryError) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.endpointError, value: formatMetricNumber(endpointError) },
    { label: EVALUATION_REPORT_CORE_METRIC_LABELS.operationSpread, value: formatMetricNumber(operationSpread) },
  ];
}

export function hasAnyEvaluationReportCoreMetricValue(
  aggregate: Record<string, unknown> | null | undefined,
  options?: { perEpisode?: unknown[] }
): boolean {
  return buildEvaluationReportCoreMetrics(aggregate, options).some((row) => row.value !== '-');
}
