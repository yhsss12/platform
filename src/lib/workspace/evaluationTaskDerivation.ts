import type { TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import type { RegistryResource } from '@/lib/api/resourceRegistryClient';
import { CABLE_THREADING_DEFAULTS } from '@/lib/workspace/cableThreading';
import { DUAL_ARM_EVAL_DEFAULTS } from '@/lib/workspace/dualArmEvaluation';
import {
  formatSimulatorBackendLabel,
  normalizeSimulatorBackendId,
  resolveEvalBackendLabel,
  resolveEvaluationUiBinding,
} from '@/lib/workspace/taskTemplateCapabilities';
import { resolveFrankStackCubeEvaluationTemplateId } from '@/lib/workspace/isaacStackCubeProduct';
import { isFrankStackCubeEvalTask } from '@/lib/workspace/isaacStackCubeProduct';
import {
  CABLE_THREADING_COMPUTABLE_METRIC_IDS,
  DUAL_ARM_COMPUTABLE_METRIC_IDS,
  filterMetricIdsForTaskSelection,
  ISAAC_STACK_COMPUTABLE_METRIC_IDS,
} from '@/lib/workspace/evaluationMetricPolicy';

export interface EvaluationMetricOption {
  key: string;
  label: string;
  description?: string;
  defaultSelected?: boolean;
  requiresStepMetrics?: boolean;
}

export interface DerivedEvaluationConfig {
  taskType: string;
  taskTemplateId: string;
  simulationPlatform: string;
  robotType: string;
  cableModel: string | null;
  cableModelLabel: string | null;
  difficulty: string | null;
  episodes: number;
  episodesMin: number;
  episodesMax: number;
  horizon: number;
  horizonMin: number;
  horizonMax: number;
  seed: number;
  recordVideo: boolean;
  taskConfigId: string | null;
  defaultTaskEnv: string | null;
  stretchMode: string | null;
  releaseMode: string | null;
  maxCables: number | null;
  config: Record<string, unknown>;
}

export interface DerivedMetricDefinitions {
  availableMetrics: EvaluationMetricOption[];
  defaultSelectedMetricKeys: string[];
}

export const DEFAULT_MIN_EVALUATION_EPISODES = 1;
export const DEFAULT_MAX_EVALUATION_EPISODES = 100;

export function clampEpisodes(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function readEpisodesBounds(...sources: unknown[]): { min: number; max: number; defaultValue?: number } {
  let min: number | null = null;
  let max: number | null = null;
  let defaultValue: number | null = null;

  for (const raw of sources) {
    const source = asRecord(raw);
    const episodesBlock = asRecord(source.episodes);
    const constraints = asRecord(source.parameterConstraints);
    const episodeConstraints = asRecord(constraints.episodes);

    min =
      pickNumber(
        min,
        episodesBlock.min,
        source.episodesMin,
        episodeConstraints.min,
        constraints.episodesMin
      ) ?? min;
    max =
      pickNumber(
        max,
        episodesBlock.max,
        source.episodesMax,
        source.episodes_max,
        episodeConstraints.max,
        constraints.episodesMax
      ) ?? max;
    defaultValue =
      pickNumber(
        defaultValue,
        episodesBlock.default,
        source.episodesDefault,
        episodeConstraints.default
      ) ?? defaultValue;
  }

  return {
    min: min ?? DEFAULT_MIN_EVALUATION_EPISODES,
    max: max ?? DEFAULT_MAX_EVALUATION_EPISODES,
    defaultValue: defaultValue ?? undefined,
  };
}
export const FALLBACK_MODEL_EVAL_METRICS: EvaluationMetricOption[] = [
  { key: 'metric_cable_success_rate_v1', label: '成功率', defaultSelected: true },
];

const METRIC_ID_CATALOG: Record<string, Omit<EvaluationMetricOption, 'key'>> = {
  metric_success_rate_v1: {
    label: '成功率',
    description: '成功 episode 数占实际完成 episode 数的比例',
    defaultSelected: true,
  },
  metric_cable_success_rate_v1: { label: '成功率', defaultSelected: true },
  metric_episode_stability_v1: { label: 'Episode 稳定性', defaultSelected: false },
  isaac_stack_success_rate_v1: { label: '成功率', defaultSelected: true },
  isaac_stack_mean_reward_v1: { label: '平均奖励', defaultSelected: true },
  isaac_stack_mean_episode_length_v1: { label: '平均 Episode 长度', defaultSelected: false },
  isaac_stack_failure_count_v1: { label: '失败次数', defaultSelected: true },
  isaac_stack_timeout_rate_v1: { label: '超时率', defaultSelected: false },
  metric_runtime_mean_steps_v1: { label: '平均步数', description: '每 episode 实际 env.step 次数的平均值', defaultSelected: false },
  metric_runtime_max_steps_v1: { label: '最大步数', description: '所有 episode 中 env.step 次数的最大值', defaultSelected: false },
  metric_runtime_video_fps_v1: { label: '视频帧率', description: '回放 MP4 的编码/播放帧率（优先读取视频 metadata）', defaultSelected: false },
  metric_runtime_control_frequency_v1: { label: '控制频率', description: '策略控制循环频率，通常为 1/control_dt，不等同于视频帧率', defaultSelected: false },
  metric_runtime_mean_sim_time_sec_v1: {
    label: '平均仿真时长',
    description: '每轮 episode 的仿真时间，按 stepCount / controlFrequencyHz 计算，不等同于 wall time 或视频播放时长',
    defaultSelected: false,
  },
  metric_runtime_max_action_norm_v1: { label: '最大动作范数', description: '单步 action 向量 L2 范数最大值，不等同于末端位移', defaultSelected: false },
  metric_runtime_smoothness_v1: {
    label: '动作平稳性',
    description:
      '基于相邻 step 的 action 向量变化量（L2 范数）计算：smoothnessScore = 1/(1+meanActionDelta)。数值越接近 1 表示 action 输出越平稳；不是末端轨迹或关节动力学平稳性。',
    defaultSelected: false,
  },
  metric_runtime_ee_path_length_v1: {
    label: '末端轨迹长度',
    description: '需记录 step 数据（eePosition）',
    defaultSelected: false,
    requiresStepMetrics: true,
  },
  metric_runtime_path_efficiency_v1: {
    label: '轨迹效率',
    description: '需记录 step 数据（eePosition）',
    defaultSelected: false,
    requiresStepMetrics: true,
  },
  metric_runtime_mean_joint_speed_v1: {
    label: '平均关节速度',
    description: '需记录 step 数据（qvel）',
    defaultSelected: false,
    requiresStepMetrics: true,
  },
  metric_runtime_max_joint_speed_v1: {
    label: '最大关节速度',
    description: '需记录 step 数据（qvel）',
    defaultSelected: false,
    requiresStepMetrics: true,
  },
  metric_runtime_mean_joint_acceleration_v1: {
    label: '平均关节加速度',
    description: '需记录 step 数据（qvel）',
    defaultSelected: false,
    requiresStepMetrics: true,
  },
  metric_runtime_max_joint_acceleration_v1: {
    label: '最大关节加速度',
    description: '需记录 step 数据（qvel）',
    defaultSelected: false,
    requiresStepMetrics: true,
  },
};

export const METRIC_ID_ALIASES: Record<string, string> = {
  metric_runtime_mean_runtime_sec_v1: 'metric_runtime_mean_sim_time_sec_v1',
};

export const COMMON_RUNTIME_METRIC_IDS = [
  'metric_runtime_mean_steps_v1',
  'metric_runtime_max_steps_v1',
  'metric_runtime_video_fps_v1',
  'metric_runtime_control_frequency_v1',
  'metric_runtime_mean_sim_time_sec_v1',
] as const;

/** @deprecated use COMMON_RUNTIME_METRIC_IDS */
export const CABLE_THREADING_RUNTIME_METRIC_IDS = COMMON_RUNTIME_METRIC_IDS;

export const COMMON_EVALUATION_AVAILABLE_METRIC_IDS = {
  cable_threading: [...CABLE_THREADING_COMPUTABLE_METRIC_IDS],
  dual_arm_cable_manipulation: [...DUAL_ARM_COMPUTABLE_METRIC_IDS],
  isaac_stack: [...ISAAC_STACK_COMPUTABLE_METRIC_IDS],
} as const;

export const CABLE_THREADING_AVAILABLE_METRIC_IDS = COMMON_EVALUATION_AVAILABLE_METRIC_IDS.cable_threading;

export const DUAL_ARM_AVAILABLE_METRIC_IDS = COMMON_EVALUATION_AVAILABLE_METRIC_IDS.dual_arm_cable_manipulation;

export const ISAAC_STACK_AVAILABLE_METRIC_IDS = COMMON_EVALUATION_AVAILABLE_METRIC_IDS.isaac_stack;

const SUCCESS_RATE_METRIC_IDS = new Set([
  'metric_success_rate_v1',
  'metric_cable_success_rate_v1',
  'isaac_stack_success_rate_v1',
]);

export function isSuccessRateMetric(metric: Pick<EvaluationMetricOption, 'key' | 'label'>): boolean {
  const key = String(metric.key ?? '').toLowerCase();
  const label = String(metric.label ?? '').trim();
  return (
    key === 'success_rate' ||
    key.includes('success_rate') ||
    key.includes('successrate') ||
    label === '成功率' ||
    label.includes('成功率')
  );
}

function normalizeSuccessRateMetricLabels(metrics: EvaluationMetricOption[]): EvaluationMetricOption[] {
  return metrics.map((metric) =>
    isSuccessRateMetric(metric)
      ? {
          ...metric,
          label: '成功率',
          defaultSelected: metric.defaultSelected !== false,
        }
      : metric
  );
}

function dedupeSuccessRateMetrics(metrics: EvaluationMetricOption[]): EvaluationMetricOption[] {
  let hasSuccessRate = false;
  return metrics.filter((metric) => {
    if (!isSuccessRateMetric(metric)) return true;
    if (hasSuccessRate) return false;
    hasSuccessRate = true;
    return true;
  });
}

function ensureSuccessRateMetric(metrics: EvaluationMetricOption[]): EvaluationMetricOption[] {
  const normalized = dedupeSuccessRateMetrics(normalizeSuccessRateMetricLabels(metrics));
  if (normalized.some(isSuccessRateMetric)) {
    return normalized;
  }
  return [
    {
      key: 'metric_cable_success_rate_v1',
      label: '成功率',
      description: '成功 episode 数占实际完成 episode 数的比例',
      defaultSelected: true,
    },
    ...normalized,
  ];
}

const CABLE_MODEL_LABELS: Record<string, string> = {
  composite_cable: '复合线缆模型',
  composite_soft: '复合软线缆',
  rmb: 'RMB 线缆',
  flex: 'Flex（实验性）',
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
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

function pickString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return null;
}

function pickStringList(...values: unknown[]): string[] {
  for (const value of values) {
    if (!Array.isArray(value)) continue;
    const items = value.map((item) => String(item).trim()).filter(Boolean);
    if (items.length > 0) return items;
  }
  return [];
}

function formatRobotTypeLabel(template: TaskTemplateDto | null, robotRaw: string | null): string {
  if (robotRaw?.trim()) return robotRaw.trim();
  const robots = template?.supportedRobotTypes ?? [];
  if (template?.id === 'dual_arm_cable_manipulation') return 'Dual FR3';
  if (robots.length === 0) return '—';
  return robots
    .map((item) => {
      if (item === 'dual_fr3') return 'Dual FR3';
      if (item === 'franka_panda') return 'Franka Panda';
      return item;
    })
    .join(' / ');
}

function formatCableModelLabel(model: string | null): string | null {
  if (!model?.trim()) return null;
  return CABLE_MODEL_LABELS[model] ?? model;
}

function resolveTemplateDefaults(templateId: string): Partial<DerivedEvaluationConfig> {
  if (templateId === 'cable_threading_single_arm') {
    return {
      robotType: CABLE_THREADING_DEFAULTS.robot,
      cableModel: CABLE_THREADING_DEFAULTS.cableModel,
      difficulty: CABLE_THREADING_DEFAULTS.difficulty,
      episodes: CABLE_THREADING_DEFAULTS.evalEpisodes,
      episodesMax: 100,
      horizon: CABLE_THREADING_DEFAULTS.horizon,
      horizonMin: 100,
      horizonMax: 1000,
      seed: CABLE_THREADING_DEFAULTS.seed,
      recordVideo: true,
    };
  }
  if (templateId === 'dual_arm_cable_manipulation') {
    return {
      robotType: 'Dual FR3',
      cableModel: null,
      difficulty: null,
      episodes: DUAL_ARM_EVAL_DEFAULTS.numEpisodes,
      episodesMax: DEFAULT_MAX_EVALUATION_EPISODES,
      horizon: 2000,
      horizonMin: 100,
      horizonMax: 2000,
      seed: DUAL_ARM_EVAL_DEFAULTS.seeds[0] ?? 42,
      recordVideo: DUAL_ARM_EVAL_DEFAULTS.record,
      stretchMode: DUAL_ARM_EVAL_DEFAULTS.stretchMode,
      releaseMode: DUAL_ARM_EVAL_DEFAULTS.releaseMode,
      maxCables: DUAL_ARM_EVAL_DEFAULTS.maxCables,
    };
  }
  if (isFrankStackCubeEvalTask(templateId)) {
    return {
      robotType: 'Franka Panda',
      cableModel: null,
      difficulty: null,
      episodes: 1,
      episodesMax: 20,
      horizon: 400,
      horizonMin: 1,
      horizonMax: 2000,
      seed: 0,
      recordVideo: true,
      defaultTaskEnv: 'Isaac-Stack-Cube-Franka-IK-Rel-v0',
    };
  }
  return {
    episodes: 10,
    episodesMax: 100,
    horizon: 600,
    horizonMin: 1,
    horizonMax: 1000,
    seed: 0,
    recordVideo: true,
  };
}

function metricOptionFromId(metricId: string): EvaluationMetricOption | null {
  const catalog = METRIC_ID_CATALOG[metricId];
  if (!catalog) {
    return {
      key: metricId,
      label: metricId,
      defaultSelected: true,
    };
  }
  return { key: metricId, ...catalog, requiresStepMetrics: catalog.requiresStepMetrics };
}

function metricOptionFromRaw(raw: unknown): EvaluationMetricOption | null {
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    if (METRIC_ID_CATALOG[trimmed]) return metricOptionFromId(trimmed);
    return { key: trimmed, label: trimmed, defaultSelected: true };
  }
  const record = asRecord(raw);
  const key = pickString(record.key, record.id, record.metricId);
  if (!key) return null;
  return {
    key,
    label: pickString(record.label, record.displayName, record.name) ?? key,
    description: pickString(record.description) ?? undefined,
    defaultSelected: record.defaultSelected === true,
  };
}

function asMetricIdList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function normalizeMetricId(metricId: string): string {
  return METRIC_ID_ALIASES[metricId] ?? metricId;
}

function dedupeMetricIds(ids: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const rawId of ids) {
    const metricId = normalizeMetricId(rawId);
    if (seen.has(metricId)) continue;
    seen.add(metricId);
    result.push(metricId);
  }
  return result;
}

function isKnownMetricId(metricId: string): boolean {
  const normalized = normalizeMetricId(metricId);
  return (
    normalized in METRIC_ID_CATALOG ||
    metricId in METRIC_ID_ALIASES ||
    normalized.startsWith('metric_') ||
    normalized.startsWith('isaac_stack_')
  );
}

function isIsaacStackTemplate(templateId: string, taskType: string): boolean {
  return (
    isFrankStackCubeEvalTask(templateId) ||
    templateId === 'isaac_block_stacking' ||
    templateId === 'isaaclab_franka_stack_cube' ||
    taskType === 'block_stacking' ||
    taskType === 'isaaclab_franka_stack_cube' ||
    taskType === 'stacking'
  );
}

function isCableThreadingTemplate(templateId: string, taskType: string): boolean {
  return templateId === 'cable_threading_single_arm' || taskType === 'cable_threading';
}

function isDualArmCableTemplate(templateId: string, taskType: string): boolean {
  return templateId === 'dual_arm_cable_manipulation' || taskType === 'dual_arm_cable_manipulation';
}

function normalizeMetricIdForTask(metricId: string, templateId: string, taskType: string): string {
  const trimmed = metricId.trim();
  if (trimmed === 'success_rate' && isCableThreadingTemplate(templateId, taskType)) {
    return 'metric_cable_success_rate_v1';
  }
  return trimmed;
}

function enrichMetricOptionFromRegistry(
  option: EvaluationMetricOption,
  metricId: string,
  registryMetrics?: RegistryResource[]
): EvaluationMetricOption {
  if (!registryMetrics?.length) return option;
  const registryMetric = registryMetrics.find((m) => m.assetId === metricId);
  if (!registryMetric) return option;
  const meta = asRecord(registryMetric.metadata);
  const registryLabel = pickString(meta.displayName, registryMetric.name);
  return {
    key: registryMetric.assetId,
    label:
      SUCCESS_RATE_METRIC_IDS.has(metricId) ||
      isSuccessRateMetric({ key: registryMetric.assetId, label: registryLabel ?? option.label })
        ? '成功率'
        : registryLabel ?? option.label,
    description: pickString(registryMetric.description, meta.description) ?? option.description,
    defaultSelected: option.defaultSelected,
    requiresStepMetrics:
      option.requiresStepMetrics || meta.availability === 'requires_step_metrics',
  };
}

function buildMetricOptionFromId(
  metricId: string,
  registryMetrics: RegistryResource[] | undefined,
  defaultSelectedIds: Set<string>
): EvaluationMetricOption | null {
  const normalizedId = metricId.trim();
  if (!normalizedId) return null;

  let option: EvaluationMetricOption | null = null;
  if (METRIC_ID_CATALOG[normalizedId]) {
    option = metricOptionFromId(normalizedId);
  } else if (registryMetrics?.length) {
    const registryMetric = registryMetrics.find((m) => m.assetId === normalizedId);
    if (registryMetric) {
      const meta = asRecord(registryMetric.metadata);
      option = {
        key: normalizedId,
        label: pickString(meta.displayName, registryMetric.name) ?? normalizedId,
        description: pickString(registryMetric.description, meta.description) ?? undefined,
        defaultSelected: meta.defaultSelected === true,
        requiresStepMetrics: meta.availability === 'requires_step_metrics',
      };
    }
  }

  if (!option) return null;
  option = enrichMetricOptionFromRegistry(option, normalizedId, registryMetrics);
  option.defaultSelected = defaultSelectedIds.has(normalizedId);
  return option;
}

function collectAvailableMetricIds(
  template: TaskTemplateDto | null,
  registryTask: RegistryResource | null | undefined,
  templateRecord: Record<string, unknown>,
  templateConfig: Record<string, unknown>,
  registryMeta: Record<string, unknown>,
  evaluationConfig: Record<string, unknown>
): string[] {
  const templateId = template?.id ?? '';
  const taskType = String(template?.taskType ?? registryTask?.taskType ?? '');

  const ids: string[] = [];
  const add = (value: unknown) => {
    for (const metricId of asMetricIdList(value)) {
      const normalized = normalizeMetricId(metricId);
      if (isKnownMetricId(metricId) || isKnownMetricId(normalized)) {
        ids.push(normalized);
      }
    }
  };

  add(registryTask?.metrics);
  add(templateRecord.availableMetricIds);
  add(registryMeta.availableMetricIds);
  add(template?.availableMetricIds);
  add(templateRecord.metrics);
  add(templateRecord.evaluationMetrics);
  add(templateConfig.metrics);
  add(evaluationConfig.metrics);
  add(template?.defaultMetricIds);

  const rawMetricSources = [
    templateRecord.metricDefinitions,
    templateConfig.metricDefinitions,
    asRecord(templateRecord.metadata).metricDefinitions,
    evaluationConfig.metricDefinitions,
    registryMeta.metricDefinitions,
  ];
  for (const source of rawMetricSources) {
    if (!source) continue;
    const items = Array.isArray(source) ? source : [source];
    for (const item of items) {
      if (typeof item === 'string') {
        add([item]);
        continue;
      }
      const record = asRecord(item);
      const key = pickString(record.key, record.id, record.metricId);
      if (key) add([key]);
    }
  }

  if (isCableThreadingTemplate(templateId, taskType)) {
    add(CABLE_THREADING_AVAILABLE_METRIC_IDS);
  } else if (isDualArmCableTemplate(templateId, taskType)) {
    add(DUAL_ARM_AVAILABLE_METRIC_IDS);
  } else if (isIsaacStackTemplate(templateId, taskType)) {
    add(ISAAC_STACK_AVAILABLE_METRIC_IDS);
  }

  return filterMetricIdsForTaskSelection(dedupeMetricIds(ids), templateId, taskType);
}

function collectDefaultSelectedMetricIds(
  template: TaskTemplateDto | null,
  availableMetricIds: string[],
  templateId: string,
  taskType: string
): Set<string> {
  const selected = new Set<string>();
  for (const rawId of asMetricIdList(template?.defaultMetricIds)) {
    const normalized = normalizeMetricIdForTask(rawId, templateId, taskType);
    if (availableMetricIds.includes(normalized)) {
      selected.add(normalized);
    }
  }

  for (const metricId of availableMetricIds) {
    const catalog = METRIC_ID_CATALOG[metricId];
    if (catalog?.defaultSelected === true) {
      selected.add(metricId);
    }
  }

  if (isCableThreadingTemplate(templateId, taskType)) {
    selected.add('metric_cable_success_rate_v1');
  } else if (isDualArmCableTemplate(templateId, taskType)) {
    selected.add('metric_success_rate_v1');
  } else if (isIsaacStackTemplate(templateId, taskType)) {
    selected.add('isaac_stack_success_rate_v1');
  }

  return selected;
}

export function deriveMetricDefinitionsFromTask(
  template: TaskTemplateDto | null,
  registryTask?: RegistryResource | null,
  registryMetrics?: RegistryResource[]
): DerivedMetricDefinitions {
  const templateRecord = asRecord(template);
  const templateConfig = asRecord(templateRecord.config);
  const registryMeta = asRecord(registryTask?.metadata);
  const evaluationConfig = asRecord(
    templateRecord.evaluationConfig ?? templateConfig.evaluation ?? registryMeta.evaluationConfig
  );
  const templateId = template?.id ?? '';
  const taskType = String(template?.taskType ?? registryTask?.taskType ?? '');

  const availableMetricIds = collectAvailableMetricIds(
    template,
    registryTask,
    templateRecord,
    templateConfig,
    registryMeta,
    evaluationConfig
  );

  const defaultSelectedIds = collectDefaultSelectedMetricIds(
    template,
    availableMetricIds,
    templateId,
    taskType
  );

  const options: EvaluationMetricOption[] = [];
  for (const metricId of availableMetricIds) {
    const option = buildMetricOptionFromId(metricId, registryMetrics, defaultSelectedIds);
    if (option) options.push(option);
  }

  const availableMetrics = ensureSuccessRateMetric(
    options.length > 0 ? options : [...FALLBACK_MODEL_EVAL_METRICS]
  );

  const defaultSelectedMetricKeys = availableMetrics
    .filter((item) => defaultSelectedIds.has(item.key))
    .map((item) => item.key);

  const successRateKeys = availableMetrics.filter(isSuccessRateMetric).map((item) => item.key);

  return {
    availableMetrics,
    defaultSelectedMetricKeys:
      defaultSelectedMetricKeys.length > 0
        ? defaultSelectedMetricKeys
        : successRateKeys.length > 0
          ? successRateKeys
          : availableMetrics.map((item) => item.key).slice(0, 1),
  };
}

export function deriveEvaluationConfigFromTask(
  template: TaskTemplateDto | null,
  registryTask?: RegistryResource | null
): DerivedEvaluationConfig {
  const templateId = resolveFrankStackCubeEvaluationTemplateId(template?.id ?? '');
  const binding = resolveEvaluationUiBinding(templateId);
  const templateRecord = asRecord(template);
  const templateConfig = asRecord(templateRecord.config);
  const registryMeta = asRecord(registryTask?.metadata);
  const registryDefault = asRecord(registryTask?.defaultConfig);
  const evaluationConfig = asRecord(
    templateRecord.evaluationConfig ?? templateConfig.evaluation ?? registryMeta.evaluationConfig
  );
  const resourceConfig = asRecord(templateRecord.resourceConfig ?? registryMeta.resourceConfig);
  const mergedConfig = templateConfig;
  const templateDefaults = resolveTemplateDefaults(templateId);
  const episodesBounds = readEpisodesBounds(
    evaluationConfig,
    registryMeta.evaluationConfig,
    registryMeta.parameterConstraints,
    registryDefault,
    templateConfig,
    templateRecord.evaluationConfig
  );
  const episodesMin = episodesBounds.min;
  const episodesMax =
    episodesBounds.max ?? templateDefaults.episodesMax ?? DEFAULT_MAX_EVALUATION_EPISODES;

  const seeds = pickStringList(
    evaluationConfig.seeds,
    mergedConfig.seeds,
    resourceConfig.seeds,
    registryDefault.seeds
  );
  const seed =
    pickNumber(
      evaluationConfig.seed,
      mergedConfig.seed,
      resourceConfig.seed,
      seeds[0],
      registryDefault.seed
    ) ?? templateDefaults.seed ?? 0;

  const episodesRaw =
    pickNumber(
      evaluationConfig.episodes,
      evaluationConfig.episodeCount,
      evaluationConfig.numEpisodes,
      mergedConfig.episodes,
      mergedConfig.episodeCount,
      resourceConfig.episodes,
      registryDefault.episode_count,
      registryDefault.episodes,
      episodesBounds.defaultValue
    ) ?? templateDefaults.episodes ?? DEFAULT_MIN_EVALUATION_EPISODES;
  const episodes = clampEpisodes(episodesRaw, episodesMin, episodesMax);

  const horizon =
    pickNumber(
      evaluationConfig.horizon,
      evaluationConfig.maxSteps,
      mergedConfig.horizon,
      mergedConfig.maxSteps,
      resourceConfig.horizon,
      registryDefault.max_steps,
      registryDefault.horizon
    ) ?? templateDefaults.horizon ?? 600;

  const robotRaw = pickString(
    evaluationConfig.robotType,
    evaluationConfig.robot,
    mergedConfig.robot,
    resourceConfig.robot,
    registryDefault.robot
  );

  const cableModel = pickString(
    evaluationConfig.cableModel,
    evaluationConfig.lineModel,
    mergedConfig.cableModel,
    resourceConfig.cableModel,
    registryDefault.cable_model
  );

  const difficulty = pickString(
    evaluationConfig.difficulty,
    mergedConfig.difficulty,
    resourceConfig.difficulty,
    registryDefault.difficulty
  );

  const simulationPlatform =
    pickString(
      evaluationConfig.simulationPlatform,
      template?.simulatorBackendLabel,
      templateRecord.simulationPlatform,
      registryTask?.simBackend,
      binding?.evalBackendLabel
    ) ??
    resolveEvalBackendLabel(templateId) ??
    formatSimulatorBackendLabel(
      normalizeSimulatorBackendId(binding?.simulatorBackend ?? template?.simulatorBackend ?? 'mujoco')
    );

  const robotType = formatRobotTypeLabel(template, robotRaw ?? templateDefaults.robotType ?? null);
  const cableModelValue = cableModel ?? templateDefaults.cableModel ?? null;

  const taskConfigId =
    pickString(
      template?.registryTaskConfigId,
      templateRecord.taskConfigId,
      evaluationConfig.taskConfigId,
      registryTask?.assetId
    ) ?? null;

  const defaultTaskEnv =
    pickString(
      evaluationConfig.defaultTaskEnv,
      evaluationConfig.taskEnv,
      template?.defaultEnv,
      registryDefault.isaac_lab_task_id,
      binding?.defaultTaskEnv
    ) ?? templateDefaults.defaultTaskEnv ?? null;

  const recordVideo =
    evaluationConfig.recordVideo === false || evaluationConfig.record === false
      ? false
      : evaluationConfig.recordVideo === true ||
          evaluationConfig.record === true ||
          registryDefault.save_video === true ||
          templateDefaults.recordVideo !== false;

  const config: Record<string, unknown> = {
    simulationPlatform,
    robotType: robotRaw ?? robotType,
    cableModel: cableModelValue,
    difficulty: difficulty ?? templateDefaults.difficulty ?? null,
    episodes,
    horizon,
    seed,
    recordVideo,
    taskConfigId,
    defaultTaskEnv,
    stretchMode:
      pickString(evaluationConfig.stretchMode, mergedConfig.stretchMode) ??
      templateDefaults.stretchMode ??
      null,
    releaseMode:
      pickString(evaluationConfig.releaseMode, mergedConfig.releaseMode) ??
      templateDefaults.releaseMode ??
      null,
    maxCables:
      pickNumber(evaluationConfig.maxCables, mergedConfig.maxCables) ??
      templateDefaults.maxCables ??
      null,
    simulatorBackend: template?.simulatorBackend ?? binding?.simulatorBackend ?? null,
    taskTemplateId: templateId,
    taskType: template?.taskType ?? templateId,
  };

  return {
    taskType: template?.taskType ?? templateId,
    taskTemplateId: templateId,
    simulationPlatform,
    robotType,
    cableModel: cableModelValue,
    cableModelLabel: formatCableModelLabel(cableModelValue),
    difficulty: difficulty ?? templateDefaults.difficulty ?? null,
    episodes,
    episodesMin,
    episodesMax,
    horizon,
    horizonMin: templateDefaults.horizonMin ?? 1,
    horizonMax: templateDefaults.horizonMax ?? 1000,
    seed,
    recordVideo,
    taskConfigId,
    defaultTaskEnv,
    stretchMode: (config.stretchMode as string | null) ?? null,
    releaseMode: (config.releaseMode as string | null) ?? null,
    maxCables: (config.maxCables as number | null) ?? null,
    config,
  };
}
