import {
  buildEvaluationReportCoreMetrics,
  hasAnyEvaluationReportCoreMetricValue,
  normalizeReportAggregate,
  resolveReportPerEpisode,
} from '@/lib/workspace/evaluationReportCoreMetrics';
import { findEvaluationTaskById } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { listWorkspaceEvaluationTasksForUi } from '@/lib/workspace/workspaceDataSources';

export interface EvaluationReportEpisodeRow {
  episode: number;
  seed: number | string;
  success: boolean | null;
  finalSuccess: boolean | null;
  threadCompletion: string;
  failureReason: string;
  videoPath: string;
}

export interface EvaluationReportArtifacts {
  aggregateResultPath?: string | null;
  perEpisodeResultsPath?: string | null;
  resultsJsonPath?: string | null;
  evalCsvPath?: string | null;
  logPath?: string | null;
  videoPath?: string | null;
  failuresPath?: string | null;
}

export interface EvaluationReportFileChecks {
  aggregateResult: boolean;
  perEpisodeResults: boolean;
  statusCompleted: boolean;
  resultsDirectory: boolean;
}

export interface ParsedEvaluationReport {
  evalJobId: string;
  status: string;
  hasCoreMetrics: boolean;
  conclusion: '成功' | '失败' | '未完成';
  totalEpisodes: number | null;
  successEpisodes: number | null;
  finalSuccessRate: number | null;
  everSuccessRate: number | null;
  meanDurationSec: number | null;
  primaryFailureReason: string | null;
  failureReasonsText: string | null;
  coreMetrics: { label: string; value: string }[];
  episodes: EvaluationReportEpisodeRow[];
  artifacts: EvaluationReportArtifacts;
  fileChecks: EvaluationReportFileChecks;
  rawAggregate: Record<string, unknown>;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function pickNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const n = Number(value);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

function pickRate(...values: unknown[]): number | null {
  const n = pickNumber(...values);
  if (n == null) return null;
  if (n > 1) return n / 100;
  return n;
}

function formatPercent(rate: number | null): string {
  if (rate == null) return '—';
  return `${(rate * 100).toFixed(1)}%`;
}

function formatMetricNumber(value: unknown): string {
  if (value == null) return '—';
  if (typeof value === 'number') {
    if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
    return value.toFixed(4);
  }
  return String(value);
}

function normalizeEpisodeRows(source: unknown): EvaluationReportEpisodeRow[] {
  if (!Array.isArray(source)) return [];
  return source.map((item, index) => {
    const ep = asRecord(item);
    const success =
      ep.success === true || ep.final_success === true || ep.episode_success === true
        ? true
        : ep.success === false || ep.final_success === false || ep.episode_success === false
          ? false
          : null;
    const finalSuccess =
      ep.final_success === true || ep.finalSuccess === true
        ? true
        : ep.final_success === false || ep.finalSuccess === false
          ? false
          : success;
    const threadCompletion = pickNumber(
      ep.thread_completion_max,
      ep.threadCompletionMax,
      ep.max_thread_completion,
      ep.thread_completion_final,
      ep.thread_completion
    );
    const failureReason = String(
      ep.failure_reason ?? ep.fail_reason ?? ep.error ?? ep.failureReason ?? ''
    ).trim();
    const videoPath = String(
      ep.video_path ?? ep.videoPath ?? ep.video ?? ''
    ).trim();
    return {
      episode: pickNumber(ep.episode, ep.episodeIndex, ep.episode_index) ?? index,
      seed: (ep.seed as number | string | undefined) ?? '—',
      success,
      finalSuccess,
      threadCompletion: threadCompletion != null ? formatMetricNumber(threadCompletion) : '—',
      failureReason: failureReason || '—',
      videoPath: videoPath || '—',
    };
  });
}

function extractFailureReasons(aggregate: Record<string, unknown>, perEpisode: unknown[]): string | null {
  const reasons = aggregate.failure_reasons ?? aggregate.failures ?? aggregate.error_summary;
  if (reasons && typeof reasons === 'object' && !Array.isArray(reasons)) {
    const entries = Object.entries(reasons as Record<string, unknown>).filter(
      ([, v]) => v != null && v !== '' && v !== 0
    );
    if (entries.length > 0) {
      return entries.map(([k, v]) => `${k}: ${String(v)}`).join('；');
    }
  }
  if (Array.isArray(reasons) && reasons.length > 0) {
    return reasons.map(String).join('；');
  }
  const fromEpisodes = normalizeEpisodeRows(perEpisode)
    .map((row) => row.failureReason)
    .filter((r) => r !== '—');
  if (fromEpisodes.length > 0) {
    const counts = new Map<string, number>();
    for (const reason of fromEpisodes) {
      counts.set(reason, (counts.get(reason) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([reason, count]) => `${reason}（${count} 次）`)
      .join('；');
  }
  return null;
}

function countSuccessfulEpisodes(perEpisode: unknown[], aggregate: Record<string, unknown>): number | null {
  const direct = pickNumber(
    aggregate.success_episodes,
    aggregate.num_success,
    aggregate.passed,
    asRecord(aggregate.summary).success_episodes
  );
  if (direct != null) return Math.round(direct);
  const rows = normalizeEpisodeRows(perEpisode);
  if (rows.length === 0) return null;
  return rows.filter((row) => row.success === true || row.finalSuccess === true).length;
}

function resolveAggregateBlock(payload: Record<string, unknown>): Record<string, unknown> {
  return normalizeReportAggregate(payload);
}

function resolvePerEpisode(payload: Record<string, unknown>): unknown[] {
  return resolveReportPerEpisode(payload);
}

export function parseEvaluationReportPayload(
  evalJobId: string,
  payload: Record<string, unknown>
): ParsedEvaluationReport {
  const status = String(payload.status ?? 'unknown');
  const aggregateBlock = resolveAggregateBlock(payload);
  const metricsNested = asRecord(aggregateBlock.metrics);
  const summaryNested = asRecord(aggregateBlock.summary);
  const perEpisode = resolvePerEpisode(payload);

  const totalEpisodes = pickNumber(
    aggregateBlock.episodes,
    aggregateBlock.total_episodes,
    aggregateBlock.num_episodes,
    summaryNested.episodes,
    summaryNested.totalEpisodes,
    payload.numEpisodes,
    asRecord(payload.summary).totalEpisodes,
    perEpisode.length > 0 ? perEpisode.length : null
  );

  const finalSuccessRate = pickRate(
    aggregateBlock.final_success_rate,
    aggregateBlock.success_rate,
    aggregateBlock.task_success_rate,
    aggregateBlock.episode_success_rate,
    metricsNested.final_success_rate,
    metricsNested.success_rate,
    summaryNested.success_rate,
    summaryNested.final_success_rate,
    payload.successRate,
    payload.success_rate
  );

  const everSuccessRate = pickRate(
    aggregateBlock.ever_success_rate,
    metricsNested.ever_success_rate,
    summaryNested.ever_success_rate,
    payload.everSuccessRate,
    payload.ever_success_rate
  );

  const successEpisodes = countSuccessfulEpisodes(perEpisode, aggregateBlock);
  const failureReasonsText = extractFailureReasons(aggregateBlock, perEpisode);
  const primaryFailureReason = failureReasonsText?.split('；')[0] ?? null;

  const meanDurationSec = pickNumber(
    aggregateBlock.mean_duration_sec,
    aggregateBlock.meanDurationSec,
    aggregateBlock.avg_duration_sec,
    summaryNested.meanDurationSec
  );

  const hasCoreMetrics = hasAnyEvaluationReportCoreMetricValue(aggregateBlock, { perEpisode });

  let conclusion: ParsedEvaluationReport['conclusion'] = '未完成';
  if (status === 'completed' || status === 'failed') {
    if (finalSuccessRate != null && finalSuccessRate >= 0.5) conclusion = '成功';
    else if (status === 'failed') conclusion = '失败';
    else if (finalSuccessRate != null) conclusion = '失败';
    else conclusion = '未完成';
  }

  const artifactsRaw = asRecord(payload.artifacts);
  const pathsRaw = asRecord(payload.paths);
  const resultsJsonInfo = asRecord(artifactsRaw.resultsJson);
  const evalCsvInfo = asRecord(artifactsRaw.evalCsv);
  const evalVideoInfo = asRecord(artifactsRaw.evalVideo);
  const pathsResultsJson = asRecord(pathsRaw.resultsJson);
  const pathsEvalCsv = asRecord(pathsRaw.evalCsv);
  const pathsFailuresJson = asRecord(pathsRaw.failuresJson);
  const pathsLog = asRecord(pathsRaw.log);
  const fileChecksRaw = asRecord(payload.fileChecks);

  const artifacts: EvaluationReportArtifacts = {
    aggregateResultPath:
      (artifactsRaw.aggregateResult as string | undefined) ??
      (pathsRaw.aggregateResult as string | undefined) ??
      (pathsResultsJson.path as string | undefined) ??
      null,
    perEpisodeResultsPath:
      (artifactsRaw.perEpisodeResults as string | undefined) ??
      (pathsRaw.perEpisodeResults as string | undefined) ??
      null,
    resultsJsonPath:
      (resultsJsonInfo.path as string | undefined) ??
      (pathsResultsJson.path as string | undefined) ??
      null,
    evalCsvPath:
      (evalCsvInfo.path as string | undefined) ?? (pathsEvalCsv.path as string | undefined) ?? null,
    logPath:
      (pathsLog.path as string | undefined) ??
      (artifactsRaw.log as string | undefined) ??
      null,
    videoPath:
      (evalVideoInfo.path as string | undefined) ??
      (payload.evalVideoPath as string | undefined) ??
      null,
    failuresPath:
      (pathsFailuresJson.path as string | undefined) ??
      (artifactsRaw.failuresJson as string | undefined) ??
      null,
  };

  const fileChecks: EvaluationReportFileChecks = {
    aggregateResult: Boolean(fileChecksRaw.aggregateResult ?? artifacts.resultsJsonPath),
    perEpisodeResults: Boolean(fileChecksRaw.perEpisodeResults ?? perEpisode.length > 0),
    statusCompleted: status === 'completed',
    resultsDirectory: Boolean(fileChecksRaw.resultsDirectory ?? hasCoreMetrics),
  };

  const coreMetrics = buildEvaluationReportCoreMetrics(aggregateBlock, { perEpisode });

  return {
    evalJobId,
    status,
    hasCoreMetrics,
    conclusion,
    totalEpisodes,
    successEpisodes,
    finalSuccessRate,
    everSuccessRate,
    meanDurationSec,
    primaryFailureReason,
    failureReasonsText,
    coreMetrics,
    episodes: normalizeEpisodeRows(perEpisode),
    artifacts,
    fileChecks,
    rawAggregate: aggregateBlock,
  };
}

/** @deprecated 使用 resolveEvaluationReportTitle */
export function buildCableThreadingReportTitle(taskName: string): string {
  return resolveEvaluationReportTitle({ jobName: taskName });
}

function pickFirstNonEmptyString(...values: unknown[]): string | null {
  for (const value of values) {
    if (value == null) continue;
    const trimmed = String(value).trim();
    if (trimmed) return trimmed;
  }
  return null;
}

function nestedRecord(value: unknown, ...keys: string[]): Record<string, unknown> {
  let current: unknown = value;
  for (const key of keys) {
    if (!current || typeof current !== 'object' || Array.isArray(current)) {
      return {};
    }
    current = (current as Record<string, unknown>)[key];
  }
  return asRecord(current);
}

export interface EvaluationTaskDisplayNameSources {
  jobName?: string | null;
  reportJobName?: string | null;
  sourceJobName?: string | null;
  replayRecordName?: string | null;
  recordName?: string | null;
  taskName?: string | null;
  evaluationName?: string | null;
  metadata?: Record<string, unknown> | null;
  fallback?: string;
}

function resolveEvaluationTaskDisplayNameInternal(
  sources: EvaluationTaskDisplayNameSources,
  defaultFallback: string
): string {
  const meta = sources.metadata ?? {};
  const evaluationRequest = nestedRecord(meta, 'evaluationRequest');
  const config = asRecord(meta.config ?? evaluationRequest.config);

  const resolved = pickFirstNonEmptyString(
    sources.jobName,
    sources.reportJobName,
    sources.taskName,
    sources.evaluationName,
    evaluationRequest.taskName,
    meta.taskName,
    nestedRecord(meta, 'job').name,
    sources.sourceJobName,
    sources.replayRecordName,
    sources.recordName,
    meta.name,
    meta.templateDisplayName,
    meta.displayName,
    evaluationRequest.modelName,
    meta.modelName,
    config.name,
    config.modelName
  );

  if (!resolved) return sources.fallback ?? defaultFallback;
  const stripped = resolved.replace(/\s*[·•]\s*评测回放\s*$/u, '').trim();
  return stripped || resolved;
}

/** 评测回放页视频区标题：优先评测任务名称 */
export function resolveEvaluationTaskDisplayName(
  sources: EvaluationTaskDisplayNameSources
): string {
  return resolveEvaluationTaskDisplayNameInternal(sources, '评测回放');
}

/** 评测回放页：优先列表/DB 中的用户任务名称 */
export function resolveEvaluationReplayTaskName(
  evalJobId: string,
  fallback: string
): string {
  const row = findEvaluationTaskById(evalJobId, listWorkspaceEvaluationTasksForUi());
  return resolveEvaluationTaskDisplayName({
    taskName: row?.taskName ?? row?.name,
    recordName: row?.rawName ?? row?.name,
    fallback,
  });
}

/** 评测报告页卡片内标题：优先评测任务名称 */
export function resolveEvaluationReportCardTitle(
  sources: EvaluationTaskDisplayNameSources
): string {
  return resolveEvaluationTaskDisplayNameInternal(sources, '评测任务');
}

export interface EvaluationReportTitleSources {
  jobName?: string | null;
  jobTaskName?: string | null;
  metadata?: Record<string, unknown> | null;
  reportTaskName?: string | null;
  evaluationName?: string | null;
  listRowName?: string | null;
  mockReportTitle?: string | null;
}

/** 报告页主标题：优先评测任务名称，不拼接「· 评测报告」后缀 */
export function resolveEvaluationReportTitle(sources: EvaluationReportTitleSources): string {
  const meta = sources.metadata ?? {};
  const evaluationRequest = nestedRecord(meta, 'evaluationRequest');
  const config = asRecord(meta.config ?? evaluationRequest.config);
  const cableThreading = asRecord(meta.cableThreading ?? evaluationRequest.cableThreading);
  const dualArmCable = asRecord(meta.dualArmCable ?? evaluationRequest.dualArmCable);
  const jobRecord = asRecord(meta.job);

  const resolved = pickFirstNonEmptyString(
    sources.jobName,
    nestedRecord(meta, 'job').name,
    jobRecord.name,
    sources.reportTaskName,
    sources.evaluationName,
    meta.name,
    meta.modelName,
    evaluationRequest.modelName,
    cableThreading.modelName,
    dualArmCable.modelName,
    config.name,
    config.modelName,
    sources.jobTaskName,
    sources.listRowName,
    meta.taskName,
    evaluationRequest.taskName,
    sources.mockReportTitle
  );

  if (!resolved) return '评测报告';
  return resolved.replace(/\s*[·•]\s*评测报告\s*$/u, '').trim() || resolved;
}

export interface EvaluationReportRobotSources {
  metadata?: Record<string, unknown> | null;
  taskType?: string | null;
  reportPayload?: Record<string, unknown> | null;
}

function formatSingleRobotLabel(raw: string): string {
  const value = raw.trim();
  if (!value) return '—';
  if (/isaac-stack-cube-franka/i.test(value) || /^franka$/i.test(value)) return 'Franka';
  if (/^panda$/i.test(value)) return 'Panda';
  if (/^ur5e?$/i.test(value)) return 'UR5e';
  if (/双臂|dual/i.test(value)) return value;
  return value;
}

function formatDualArmRobotLabel(robots: string[]): string {
  const labels = [...new Set(robots.map(formatSingleRobotLabel).filter((item) => item !== '—'))];
  if (labels.length === 0) return '—';
  if (labels.length === 1) {
    const base = labels[0];
    if (/panda/i.test(base)) return '双臂 Panda';
    return `双臂 ${base}`;
  }
  return `双臂 ${labels.join('/')}`;
}

/** 任务配置详情 — 机器人字段：单值展示，按任务类型归一化 */
export function resolveEvaluationReportRobotDisplay(sources: EvaluationReportRobotSources): string {
  const meta = sources.metadata ?? {};
  const report = sources.reportPayload ?? {};
  const evaluationRequest = nestedRecord(meta, 'evaluationRequest');
  const config = asRecord(meta.config ?? evaluationRequest.config ?? meta.trainConfig);
  const cableThreading = asRecord(meta.cableThreading ?? evaluationRequest.cableThreading);
  const simConfig = asRecord(meta.simConfig ?? config.simConfig);
  const dualArmCable = asRecord(meta.dualArmCable ?? evaluationRequest.dualArmCable);
  const taskType = String(sources.taskType ?? meta.taskType ?? report.taskType ?? '').trim();

  const directRobot = pickFirstNonEmptyString(
    config.robot,
    cableThreading.robot,
    meta.cableThreadingRobot,
    simConfig.robot,
    meta.robot,
    evaluationRequest.robot,
    dualArmCable.robot,
    nestedRecord(meta, 'job').robot,
    report.robot,
    report.envName,
    config.envName
  );

  if (directRobot) {
    if (taskType === 'dual_arm_cable_manipulation' && !/双臂|dual/i.test(directRobot)) {
      return formatDualArmRobotLabel([directRobot]);
    }
    return formatSingleRobotLabel(directRobot);
  }

  const resolvedResources = asRecord(meta.resolvedResources);
  const resolvedRobots = resolvedResources.robots;
  if (Array.isArray(resolvedRobots) && resolvedRobots.length > 0) {
    const names = resolvedRobots
      .map((item) => {
        if (typeof item === 'object' && item && 'name' in item) {
          return String((item as { name: string }).name);
        }
        return String(item);
      })
      .filter(Boolean);
    if (taskType === 'dual_arm_cable_manipulation' || names.length > 1) {
      return formatDualArmRobotLabel(names);
    }
    if (names.length === 1) return formatSingleRobotLabel(names[0]);
  }

  if (taskType === 'block_stacking' || taskType === 'isaac_block_stacking') {
    const env = pickFirstNonEmptyString(simConfig.envName, config.envName, report.envName);
    if (env && /franka/i.test(env)) return 'Franka';
  }

  return '—';
}
