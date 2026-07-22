'use client';

import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import type { EvaluationJobStatusResponse } from '@/lib/api/evaluationClient';
import { DUAL_ARM_CABLE_TASK_NAME, DUAL_ARM_CABLE_TASK_TYPE } from '@/lib/workspace/dualArmCable';

export const DUAL_ARM_EVAL_DEFAULTS = {
  numEpisodes: 1,
  seeds: [42] as number[],
  maxCables: 1,
  record: true,
  headless: true,
  stretchMode: 'fixed_distance' as const,
  releaseMode: 'three_phase' as const,
};

export function generateDualArmEvalTaskName(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const seq = String(Math.floor(Math.random() * 999) + 1).padStart(3, '0');
  return `${DUAL_ARM_CABLE_TASK_NAME} · episode 稳定性评测_${date}_${seq}`;
}

export function buildDualArmEvalSeeds(numEpisodes: number, baseSeed: number): number[] {
  return Array.from({ length: numEpisodes }, (_, i) => baseSeed + i);
}

export function buildDualArmEvalReplayHref(params: { evalJobId: string; episode?: number }): string {
  const search = new URLSearchParams({
    replayType: 'evaluation',
    taskType: DUAL_ARM_CABLE_TASK_TYPE,
    evalJobId: params.evalJobId,
  });
  if (params.episode != null) search.set('episode', String(params.episode));
  return `/workspace/replay?${search.toString()}`;
}

export function buildDualArmEvalReportHref(params: { evalJobId: string }): string {
  return `/workspace/evaluation/report?evalId=${encodeURIComponent(params.evalJobId)}&taskType=${DUAL_ARM_CABLE_TASK_TYPE}`;
}

export function isDualArmEvalRow(row: EvaluationTaskRow): boolean {
  return row.taskType === DUAL_ARM_CABLE_TASK_TYPE && row.id.startsWith('eval_');
}

export function buildDualArmEvalResultSummary(input: {
  jobStatus: string;
  evaluationMode?: string | null;
  message?: string | null;
  metrics?: Record<string, unknown> | null;
}): string {
  const metrics = input.metrics ?? {};
  const successEpisodes = metrics.successEpisodes;
  const totalEpisodes = metrics.totalEpisodes;
  const evaluationMode = input.evaluationMode ?? undefined;
  const isPolicyEval = evaluationMode === 'trained_model_evaluation';

  if (input.jobStatus === 'completed') {
    if (typeof successEpisodes === 'number' && typeof totalEpisodes === 'number') {
      const taskOutcome =
        successEpisodes > 0
          ? `${successEpisodes}/${totalEpisodes} 任务成功`
          : `运行完成，0/${totalEpisodes} 任务成功`;
      return isPolicyEval
        ? `训练模型 rollout 评测完成：${taskOutcome}`
        : `episode 稳定性评测完成：${taskOutcome}`;
    }
    return isPolicyEval ? '训练模型 rollout 评测完成' : 'episode 稳定性评测完成';
  }
  if (input.jobStatus === 'failed') {
    return input.message?.trim() || '评测失败';
  }
  return (
    input.message?.trim() ||
    (isPolicyEval ? '训练模型 rollout 评测运行中…' : 'episode 稳定性评测运行中…')
  );
}

export function dualArmEvalRowFromStatus(
  evalJobId: string,
  status: EvaluationJobStatusResponse,
  _name: string,
  seeds: number[]
): Partial<EvaluationTaskRow> {
  const metrics = status.metrics ?? {};
  const successRateRaw = metrics.successRate;
  const successRate =
    typeof successRateRaw === 'number' ? Math.round(successRateRaw * 1000) / 10 : null;

  let rowStatus: EvaluationTaskRow['status'] = '评测中';
  if (status.status === 'completed') rowStatus = '已完成';
  else if (status.status === 'failed') rowStatus = '失败';

  return {
    status: rowStatus,
    successRate,
    resultSummary: buildDualArmEvalResultSummary({
      jobStatus: status.status,
      evaluationMode: status.evaluationMode,
      message: status.message,
      metrics,
    }),
    backendJobStatus: status.status,
    aggregate: metrics,
    dualArmEvalCurrentEpisode: status.currentEpisode ?? undefined,
    dualArmEvalTotalEpisodes: status.totalEpisodes ?? undefined,
    dualArmEvalSeeds: seeds,
    dualArmMeanFinalSag: metrics.meanFinalSag as number | undefined,
    dualArmMeanFinalSpan: metrics.meanFinalSpan as number | undefined,
  };
}

export function formatDualArmEvalProgress(row: EvaluationTaskRow): string {
  if (row.dualArmEvalCurrentEpisode != null && row.dualArmEvalTotalEpisodes != null) {
    return `${row.dualArmEvalCurrentEpisode}/${row.dualArmEvalTotalEpisodes}`;
  }
  return '—';
}

export function dualArmEvalReportSections(aggregate: Record<string, unknown>): {
  basic: { label: string; value: string }[];
  metrics: { label: string; value: string }[];
  perEpisode: { label: string; value: string }[];
} {
  const summary = (aggregate.summary as Record<string, unknown>) ?? {};
  const taskMetrics = (aggregate.taskMetrics as Record<string, unknown>) ?? {};
  const perEpisode = (aggregate.perEpisode as Record<string, unknown>[]) ?? [];
  const evaluationMode = String(aggregate.evaluationMode ?? 'episode_stability');
  const modeLabel =
    evaluationMode === 'trained_model_evaluation' ? '训练模型 rollout 评测' : 'episode 稳定性评测';

  const fmtRate = (v: unknown) =>
    typeof v === 'number' ? `${Math.round(v * 1000) / 10}%` : '—';
  const fmtNum = (v: unknown) => (typeof v === 'number' ? String(v) : '—');
  const fmtBool = (v: unknown) => (v === true ? '是' : v === false ? '否' : '—');

  return {
    basic: [
      { label: '评测模式', value: modeLabel },
      { label: '总评测轮次', value: String(summary.totalEpisodes ?? '—') },
      { label: '任务成功轮次', value: String(summary.successEpisodes ?? '—') },
      { label: '任务成功率', value: fmtRate(summary.successRate) },
    ],
    metrics: [
      { label: 'contactSuccessRate', value: fmtRate(taskMetrics.contactSuccessRate) },
      { label: 'stretchReachedRate', value: fmtRate(taskMetrics.stretchReachedRate) },
      { label: 'meanFinalSag', value: fmtNum(taskMetrics.meanFinalSag) },
      { label: 'meanFinalSpan', value: fmtNum(taskMetrics.meanFinalSpan) },
      { label: 'meanSag', value: fmtNum(taskMetrics.meanSag) },
      { label: 'meanSpan', value: fmtNum(taskMetrics.meanSpan) },
      {
        label: 'failureSeeds',
        value: Array.isArray(taskMetrics.failureSeeds)
          ? (taskMetrics.failureSeeds as number[]).join(', ') || '—'
          : '—',
      },
      {
        label: 'failureReasons',
        value: taskMetrics.failureReasons ? JSON.stringify(taskMetrics.failureReasons) : '—',
      },
    ],
    perEpisode: perEpisode.flatMap((ep, idx) => [
      { label: `Episode ${idx} seed`, value: String(ep.seed ?? '—') },
      { label: `Episode ${idx} 运行状态`, value: String(ep.episodeStatus ?? ep.status ?? '—') },
      { label: `Episode ${idx} 任务成功`, value: fmtBool(ep.episodeSuccess) },
      { label: `Episode ${idx} taskSuccess`, value: fmtBool(ep.taskSuccess) },
      { label: `Episode ${idx} graspSuccess`, value: fmtBool(ep.graspSuccess) },
      { label: `Episode ${idx} stretchSuccess`, value: fmtBool(ep.stretchSuccess) },
      { label: `Episode ${idx} failureReason`, value: String(ep.failureReason ?? '—') },
      { label: `Episode ${idx} finalSag`, value: fmtNum(ep.finalSagM) },
      { label: `Episode ${idx} finalSpan`, value: fmtNum(ep.finalSpanM) },
    ]),
  };
}

export function dualArmEvalReplayMetrics(
  aggregate: Record<string, unknown> | null,
  episodeIndex = 0
): { label: string; value: string }[] {
  if (!aggregate) return [];
  const perEpisode = (aggregate.perEpisode as Record<string, unknown>[]) ?? [];
  const ep = perEpisode[episodeIndex];
  const taskMetrics = (aggregate.taskMetrics as Record<string, unknown>) ?? {};
  const summary = (aggregate.summary as Record<string, unknown>) ?? {};
  const fmt = (v: unknown) =>
    v === true ? '是' : v === false ? '否' : v != null ? String(v) : '—';

  return [
    { label: 'evaluationMode', value: 'episode_stability' },
    { label: 'seed', value: fmt(ep?.seed) },
    { label: 'episodeIndex', value: String(ep?.episodeIndex ?? episodeIndex) },
    { label: 'episodeSuccess', value: fmt(ep?.episodeSuccess) },
    { label: 'finalSag', value: fmt(ep?.finalSagM) },
    { label: 'finalSpan', value: fmt(ep?.finalSpanM) },
    {
      label: 'successRate',
      value: typeof summary.successRate === 'number' ? String(summary.successRate) : '—',
    },
    { label: 'meanFinalSag', value: fmt(taskMetrics.meanFinalSag) },
    { label: 'meanFinalSpan', value: fmt(taskMetrics.meanFinalSpan) },
  ];
}
