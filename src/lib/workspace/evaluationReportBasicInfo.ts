import { formatMetricTaskType } from '@/lib/workspace/evaluationMetricRegistry';
import {
  formatTaskTemplateDisplayName,
  getTaskDisplayName,
  resolveTemplateIdFromBackendTaskType,
} from '@/lib/workspace/taskTemplateMapping';
import {
  formatSimulatorBackendLabel,
  resolveEvalBackendLabel,
  resolveTaskTemplateCapabilities,
  resolveTaskTemplateCapabilitiesById,
  type SimulatorBackendId,
} from '@/lib/workspace/taskTemplateCapabilities';
import { normalizeReportAggregate } from '@/lib/workspace/evaluationReportCoreMetrics';
import { pickFirstNonEmptyString } from '@/lib/workspace/evaluationReportBasicInfoUtils';

export interface ReportBasicInfoField {
  label: string;
  value: string;
}

export interface BuildReportBasicInfoInput {
  taskName?: string | null;
  relatedTask?: string | null;
  taskDisplayName?: string | null;
  templateDisplayName?: string | null;
  taskTemplateName?: string | null;
  taskTemplateId?: string | null;
  taskType?: string | null;
  metadata?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  aggregate?: Record<string, unknown> | null;
  evaluationMode?: string | null;
  modelType?: string | null;
  episodeCount?: number | string | null;
}

const REPORT_EVALUATION_MODE_LABELS: Record<string, string> = {
  trained_model_evaluation: '已训练模型',
  expert_policy_evaluation: '专家策略',
  policy_evaluation: '已训练模型',
  robomimic: '已训练模型',
  episode_stability: '专家策略',
  dataset_evaluation: '数据集评测',
  dataset_offline_eval: '数据集评测',
  offline_dataset_evaluation: '数据集评测',
  数据过程评测: '数据集评测',
  'episode 稳定性评测': '专家策略',
  已训练模型: '已训练模型',
  专家策略: '专家策略',
  数据集评测: '数据集评测',
};

function nestedRecord(value: unknown, ...keys: string[]): Record<string, unknown> {
  let current: unknown = value;
  for (const key of keys) {
    if (!current || typeof current !== 'object' || Array.isArray(current)) {
      return {};
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current && typeof current === 'object' && !Array.isArray(current)
    ? (current as Record<string, unknown>)
    : {};
}

function displayOrDash(value: string | null | undefined): string {
  const trimmed = value?.trim();
  return trimmed ? trimmed : '-';
}

function resolveTemplateId(input: BuildReportBasicInfoInput): string | null {
  const meta = input.metadata ?? {};
  const metrics = input.metrics ?? {};
  return (
    pickFirstNonEmptyString(
      input.taskTemplateId,
      meta.taskTemplateId,
      metrics.taskTemplateId,
      nestedRecord(meta, 'evaluationRequest').taskTemplateId,
      nestedRecord(meta, 'config').taskTemplateId,
      resolveTemplateIdFromBackendTaskType(String(input.taskType ?? ''))
    ) ?? null
  );
}

export function resolveReportRelatedTask(input: BuildReportBasicInfoInput): string {
  const meta = input.metadata ?? {};
  const metrics = input.metrics ?? {};
  const templateId = resolveTemplateId(input);

  const resolved = pickFirstNonEmptyString(
    input.relatedTask,
    input.taskDisplayName,
    meta.taskDisplayName,
    metrics.taskDisplayName,
    input.templateDisplayName,
    meta.templateDisplayName,
    metrics.templateDisplayName,
    input.taskTemplateName,
    meta.taskTemplateName,
    metrics.taskTemplateName,
    formatTaskTemplateDisplayName(templateId),
    input.taskType ? formatMetricTaskType(input.taskType) : null,
    input.taskType ? getTaskDisplayName(input.taskType) : null
  );

  return displayOrDash(resolved);
}

function normalizeSimulatorBackendId(raw: string | null | undefined): SimulatorBackendId | null {
  if (!raw?.trim()) return null;
  const value = raw.trim().toLowerCase();
  if (value === 'isaac_lab' || value === 'isaac' || value.includes('isaac')) {
    return 'isaac_lab';
  }
  if (value === 'mujoco' || value.includes('mujoco')) {
    return 'mujoco';
  }
  return null;
}

export function resolveReportSimulatorPlatform(input: BuildReportBasicInfoInput): string {
  const meta = input.metadata ?? {};
  const metrics = input.metrics ?? {};
  const templateId = resolveTemplateId(input);

  const explicitBackend = pickFirstNonEmptyString(
    meta.simulatorBackend,
    metrics.simulatorBackend,
    nestedRecord(meta, 'config').simulatorBackend,
    nestedRecord(meta, 'resolvedResources').simulatorBackend
  );
  const normalizedExplicit = normalizeSimulatorBackendId(explicitBackend);
  if (normalizedExplicit) {
    return formatSimulatorBackendLabel(normalizedExplicit);
  }

  const bindingLabel = resolveEvalBackendLabel(templateId);
  if (bindingLabel && bindingLabel !== 'MuJoCo') {
    return bindingLabel;
  }

  const profile =
    resolveTaskTemplateCapabilitiesById(templateId) ??
    (input.taskType ? resolveTaskTemplateCapabilities(input.taskType) : null);
  if (profile) {
    return formatSimulatorBackendLabel(profile.simulatorBackend);
  }

  const taskType = String(input.taskType ?? '').trim();
  if (
    taskType === 'cable_threading' ||
    taskType === 'cable_threading_single_arm' ||
    taskType === 'dual_arm_cable_manipulation'
  ) {
    return 'MuJoCo';
  }
  if (
    taskType === 'isaac_block_stacking' ||
    taskType === 'block_stacking'
  ) {
    return 'Isaac Lab';
  }

  return bindingLabel ? bindingLabel : '-';
}

export function resolveReportEvaluationModeLabel(input: BuildReportBasicInfoInput): string {
  const meta = input.metadata ?? {};
  const metrics = input.metrics ?? {};
  const aggregate = normalizeReportAggregate(input.aggregate ?? undefined);
  const summary = nestedRecord(aggregate, 'summary');

  const rawMode = pickFirstNonEmptyString(
    input.evaluationMode,
    aggregate.evaluationMode,
    summary.evaluationMode,
    metrics.evaluationMode,
    meta.evaluationMode,
    nestedRecord(meta, 'evaluationRequest').evaluationMode,
    metrics.policy,
    meta.policy
  );

  if (rawMode && REPORT_EVALUATION_MODE_LABELS[rawMode]) {
    return REPORT_EVALUATION_MODE_LABELS[rawMode];
  }

  const modelType = pickFirstNonEmptyString(input.modelType, meta.modelType, metrics.modelType);
  if (modelType === '已训练模型') return '已训练模型';
  if (modelType === '专家策略') return '专家策略';

  const runner = pickFirstNonEmptyString(meta.runner, metrics.runner);
  if (runner === 'dataset_offline_eval' || /离线数据集评测|dataset/i.test(String(rawMode ?? ''))) {
    return '数据集评测';
  }

  if (rawMode === '策略评测') {
    return modelType === '专家策略' ? '专家策略' : '已训练模型';
  }

  if (rawMode && !/episode_stability|episode/i.test(rawMode)) {
    return rawMode;
  }

  return '-';
}

function resolveReportEpisodeCount(input: BuildReportBasicInfoInput): string {
  const aggregate = normalizeReportAggregate(input.aggregate ?? undefined);
  const metrics = input.metrics ?? {};
  const meta = input.metadata ?? {};
  const summary = nestedRecord(aggregate, 'summary');

  if (input.episodeCount != null && input.episodeCount !== '') {
    const direct = String(input.episodeCount).trim();
    if (direct) return direct;
  }

  const count = pickFirstNonEmptyString(
    aggregate.episodeCount,
    aggregate.totalEpisodes,
    aggregate.total_episodes,
    aggregate.episodes,
    summary.totalEpisodes,
    summary.episodeCount,
    metrics.numEpisodes,
    metrics.episodes,
    metrics.totalEpisodes,
    meta.numEpisodes,
    meta.episodes
  );

  if (count == null) return '-';
  const numeric = Number(count);
  if (Number.isFinite(numeric)) return String(numeric);
  return String(count).trim() || '-';
}

export function buildReportBasicInfo(input: BuildReportBasicInfoInput): ReportBasicInfoField[] {
  return [
    { label: '任务名称', value: displayOrDash(input.taskName) },
    { label: '关联任务', value: resolveReportRelatedTask(input) },
    { label: '仿真平台', value: resolveReportSimulatorPlatform(input) },
    { label: '评测模式', value: resolveReportEvaluationModeLabel(input) },
    { label: 'Episode 数', value: resolveReportEpisodeCount(input) },
  ];
}
