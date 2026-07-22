import type { WorkspaceJobDetail } from '@/lib/api/workspaceJobClient';
import type { EvaluationMetricsInput } from '@/lib/workspace/evaluationLiveMetrics';
import type { ReplayAdapterResult } from '@/lib/workspace/replayAdapters';
import type { CableReplayRecord } from '@/lib/workspace/replayCableThreadingAdapter';
import type { DualArmReplayRecord } from '@/lib/workspace/replayDualArmCableAdapter';

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function mergeMetricsInput(...parts: EvaluationMetricsInput[]): EvaluationMetricsInput {
  const merged: EvaluationMetricsInput = {};
  for (const part of parts) {
    if (!part) continue;
    if (part.aggregate) {
      merged.aggregate = { ...asRecord(merged.aggregate), ...asRecord(part.aggregate) };
    }
    if (part.metrics) {
      merged.metrics = { ...asRecord(merged.metrics), ...asRecord(part.metrics) };
    }
    if (part.live) {
      merged.live = { ...asRecord(merged.live), ...asRecord(part.live) };
    }
    if (part.jobStatus) merged.jobStatus = part.jobStatus;
  }
  return merged;
}

export function metricsInputFromWorkspaceJob(
  job: WorkspaceJobDetail | null | undefined
): EvaluationMetricsInput {
  if (!job) return {};
  const metrics = asRecord(job.metrics);
  const metadata = asRecord(job.metadata);
  const aggregate = asRecord(metrics.aggregate ?? metadata.aggregate ?? metrics);
  return {
    aggregate: Object.keys(aggregate).length > 0 ? aggregate : null,
    metrics: Object.keys(metrics).length > 0 ? metrics : null,
    jobStatus: job.status,
  };
}

export function metricsInputFromCableReplayRecord(
  record: CableReplayRecord | null | undefined
): EvaluationMetricsInput {
  if (!record) return {};

  if (record.recordType === 'policy_eval' && record.evalRow) {
    const aggregate = asRecord(record.evalRow.aggregate);
    const metrics: Record<string, unknown> = {};
    if (record.successRate != null) {
      metrics.successRate = record.successRate / 100;
    }
    if (record.evalRow.everSuccessRate != null) {
      metrics.everSuccessRate = record.evalRow.everSuccessRate / 100;
    }
    return {
      aggregate: Object.keys(aggregate).length > 0 ? aggregate : null,
      metrics,
      jobStatus: record.status,
    };
  }

  if (record.dataItem) {
    const item = record.dataItem;
    const episodes = item.successfulEpisodes != null ? Number(item.successfulEpisodes) : undefined;
    const metrics: Record<string, unknown> = {};
    if (record.successRate != null) {
      metrics.successRate = record.successRate / 100;
      metrics.finalSuccessRate = record.successRate / 100;
    }
    if (item.successfulEpisodes != null) {
      metrics.successfulEpisodes = item.successfulEpisodes;
    }
    if (item.horizon != null) {
      metrics.meanSteps = item.horizon;
    }
    return {
      metrics,
      live: episodes != null ? { completedEpisodes: episodes } : undefined,
      jobStatus: record.status,
    };
  }

  return { jobStatus: record.status };
}

export function metricsInputFromDualArmReplayRecord(
  record: DualArmReplayRecord | null | undefined
): EvaluationMetricsInput {
  if (!record) return {};
  const item = record.dataItem;
  const metrics: Record<string, unknown> = {};
  if (record.episodeSuccess != null) {
    metrics.successRate = record.episodeSuccess ? 1 : 0;
    metrics.successfulEpisodes = record.episodeSuccess ? 1 : 0;
    metrics.failureCount = record.episodeSuccess ? 0 : 1;
  }
  if (item?.dualArmMaxCables != null) {
    metrics.numEpisodes = item.dualArmMaxCables;
  }
  return { metrics, jobStatus: record.status };
}

export function metricsInputFromReplayAdapter(
  adapter: ReplayAdapterResult | null | undefined
): EvaluationMetricsInput {
  if (!adapter) return {};
  const metrics = asRecord(adapter.metricsSummary);
  const aggregate = asRecord(adapter.metricsAggregate);
  const metaSuccess = adapter.metadata?.successRate;
  if (metaSuccess != null && metrics.successRate == null) {
    metrics.successRate =
      typeof metaSuccess === 'number' && metaSuccess > 1 ? metaSuccess / 100 : metaSuccess;
  }
  return {
    aggregate: Object.keys(aggregate).length > 0 ? aggregate : null,
    metrics: Object.keys(metrics).length > 0 ? metrics : null,
    jobStatus: adapter.status,
  };
}

export function buildReplayMetricsInput(params: {
  workspaceJob?: WorkspaceJobDetail | null;
  cableRecord?: CableReplayRecord | null;
  dualArmRecord?: DualArmReplayRecord | null;
  adapter?: ReplayAdapterResult | null;
}): EvaluationMetricsInput {
  return mergeMetricsInput(
    metricsInputFromWorkspaceJob(params.workspaceJob),
    metricsInputFromCableReplayRecord(params.cableRecord),
    metricsInputFromDualArmReplayRecord(params.dualArmRecord),
    metricsInputFromReplayAdapter(params.adapter)
  );
}
