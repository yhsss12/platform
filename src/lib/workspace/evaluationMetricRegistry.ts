import type { RegistryResource } from '@/lib/api/resourceRegistryClient';
import {
  getTaskDisplayName,
  TASK_TEMPLATE_DISPLAY_NAMES,
} from '@/lib/workspace/taskDisplayNames';

export type MetricCalculationMode =
  | 'aggregate_field'
  | 'per_episode_failure_reason'
  | 'aggregate_fields';

export type MetricValueType = 'ratio' | 'number' | 'integer' | 'composite';

export interface MetricDefinition {
  metricId: string;
  displayName: string;
  description: string;
  implemented: boolean;
  selectable?: boolean;
  availability?: string;
  requiredFields?: string[];
  unavailableReason?: string;
  calculationMode: MetricCalculationMode;
  sourceField?: string;
  sourceFields?: string[];
  failureReasonValue?: string;
  valueType: MetricValueType;
  unit?: string;
  applicableTaskTypes: string[];
  applicableEvaluationModes: string[];
}

export interface ResolvedEvalMetric {
  metricId: string;
  displayName: string;
  valueText: string;
  status: 'computed' | 'missing' | 'unavailable';
  sourceHint?: string;
  unavailableReason?: string;
}

const EVALUATION_MODE_LABELS: Record<string, string> = {
  trained_model_evaluation: '已训练模型',
  expert_policy_evaluation: '专家策略',
  episode_stability: 'Episode 稳定性评测',
  policy_evaluation: '策略评测',
};

const TASK_TYPE_LABELS: Record<string, string> = {
  ...TASK_TEMPLATE_DISPLAY_NAMES,
  cable_threading: getTaskDisplayName('cable_threading'),
  dual_arm_cable_manipulation: getTaskDisplayName('dual_arm_cable_manipulation'),
  isaac_block_stacking: getTaskDisplayName('isaac_block_stacking'),
  block_stacking: getTaskDisplayName('isaac_block_stacking'),
};

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item)).filter(Boolean);
}

function readNested(data: Record<string, unknown>, path: string): unknown {
  let current: unknown = data;
  for (const part of path.split('.')) {
    if (!current || typeof current !== 'object') return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

export function parseMetricDefinition(resource: RegistryResource): MetricDefinition | null {
  const parsed = parseRegistryMetricDefinition(resource);
  if (!parsed?.implemented) return null;
  return parsed;
}

export function parseRegistryMetricDefinition(resource: RegistryResource): MetricDefinition | null {
  if (resource.assetType !== 'metric') return null;
  const metadata = resource.metadata ?? {};
  const calculationMode = String(metadata.calculationMode || 'aggregate_field') as MetricCalculationMode;
  return {
    metricId: resource.assetId,
    displayName: String(metadata.displayName || resource.name || resource.assetId),
    description: resource.description || '',
    implemented: metadata.implemented === true,
    selectable: metadata.selectable === true || metadata.implemented === true,
    availability: metadata.availability ? String(metadata.availability) : undefined,
    requiredFields: asStringList(metadata.requiredFields),
    unavailableReason: metadata.unavailableReason ? String(metadata.unavailableReason) : undefined,
    calculationMode,
    sourceField: metadata.sourceField ? String(metadata.sourceField) : undefined,
    sourceFields: asStringList(metadata.sourceFields),
    failureReasonValue: metadata.failureReasonValue ? String(metadata.failureReasonValue) : undefined,
    valueType: (metadata.valueType ? String(metadata.valueType) : 'number') as MetricValueType,
    unit: metadata.unit != null ? String(metadata.unit) : undefined,
    applicableTaskTypes: asStringList(metadata.applicableTaskTypes),
    applicableEvaluationModes: asStringList(metadata.applicableEvaluationModes),
  };
}

export function listSelectableMetricDefinitions(resources: RegistryResource[]): MetricDefinition[] {
  return resources
    .map(parseRegistryMetricDefinition)
    .filter((item): item is MetricDefinition => item != null && (item.implemented || item.selectable === true));
}

export function listImplementedMetricDefinitions(resources: RegistryResource[]): MetricDefinition[] {
  return resources
    .map(parseMetricDefinition)
    .filter((item): item is MetricDefinition => item != null);
}

export function filterMetricDefinitions(
  definitions: MetricDefinition[],
  taskType: string,
  evaluationMode: string
): MetricDefinition[] {
  const normalizedTaskType = taskType === 'block_stacking' ? 'isaac_block_stacking' : taskType;
  return definitions.filter((metric) => {
    const taskMatch =
      metric.applicableTaskTypes.length === 0 ||
      metric.applicableTaskTypes.includes(normalizedTaskType) ||
      metric.applicableTaskTypes.includes(taskType);
    const modeMatch =
      metric.applicableEvaluationModes.length === 0 ||
      metric.applicableEvaluationModes.includes(evaluationMode);
    return taskMatch && modeMatch;
  });
}

function formatMetricValue(
  value: unknown,
  metric: MetricDefinition
): string {
  if (metric.valueType === 'composite' && value && typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, raw]) => `${key}: ${formatScalarValue(raw, { ...metric, valueType: 'number' })}`)
      .join(' · ');
  }
  return formatScalarValue(value, metric);
}

function formatScalarValue(value: unknown, metric: MetricDefinition): string {
  if (value == null || value === '') return '—';
  if (metric.valueType === 'ratio' && typeof value === 'number') {
    const percent = value <= 1 ? value * 100 : value;
    return `${Math.round(percent * 10) / 10}${metric.unit === '%' ? '%' : ''}`;
  }
  if (typeof value === 'number') {
    const rounded = Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000);
    return metric.unit && metric.unit !== '%' ? `${rounded} ${metric.unit}` : rounded;
  }
  return String(value);
}

function resolveMetricRawValue(
  metric: MetricDefinition,
  aggregate: Record<string, unknown>,
  perEpisode: Record<string, unknown> | null | undefined
): unknown {
  const metricsBlock =
    aggregate.metrics && typeof aggregate.metrics === 'object'
      ? (aggregate.metrics as Record<string, unknown>)
      : {};

  if (metric.calculationMode === 'aggregate_field' && metric.sourceField) {
    const nested = readNested(aggregate, metric.sourceField);
    if (nested != null) return nested;
    if (metricsBlock[metric.sourceField] != null) return metricsBlock[metric.sourceField];
    return aggregate[metric.sourceField];
  }

  if (metric.calculationMode === 'per_episode_failure_reason') {
    if (metricsBlock.timeoutRate != null) return metricsBlock.timeoutRate;
    const episodes = Array.isArray(perEpisode?.episodes) ? perEpisode?.episodes : [];
    if (episodes.length === 0) return null;
    const field = metric.sourceField || 'failureReason';
    const target = metric.failureReasonValue || 'horizon_reached';
    const matched = episodes.filter(
      (row) =>
        row &&
        typeof row === 'object' &&
        String((row as Record<string, unknown>)[field] ?? '') === target
    ).length;
    return matched / episodes.length;
  }

  if (metric.calculationMode === 'aggregate_fields' && metric.sourceFields?.length) {
    const resolved: Record<string, unknown> = {};
    for (const path of metric.sourceFields) {
      const value = readNested(aggregate, path);
      if (value != null) resolved[path] = value;
    }
    return Object.keys(resolved).length > 0 ? resolved : null;
  }

  return null;
}

export function resolveEvalMetrics(params: {
  definitions: MetricDefinition[];
  aggregate: Record<string, unknown> | null | undefined;
  perEpisode?: Record<string, unknown> | null;
  metricIds?: string[];
}): ResolvedEvalMetric[] {
  const aggregate = params.aggregate ?? {};
  const selected =
    params.metricIds && params.metricIds.length > 0
      ? params.definitions.filter((metric) => params.metricIds?.includes(metric.metricId))
      : params.definitions;

  return selected.map((metric) => {
    if (!metric.implemented) {
      return {
        metricId: metric.metricId,
        displayName: metric.displayName,
        valueText: '—',
        status: 'unavailable' as const,
        unavailableReason:
          metric.unavailableReason ||
          (metric.requiredFields?.length
            ? `缺少 step_metrics.${metric.requiredFields.join(',')}`
            : '暂不可计算'),
        sourceHint: metric.availability,
      };
    }

    const raw = resolveMetricRawValue(metric, aggregate, params.perEpisode);
    const hasValue = raw != null && raw !== '';
    const sourceHint =
      metric.calculationMode === 'per_episode_failure_reason'
        ? `${metric.sourceField || 'failureReason'}=${metric.failureReasonValue || 'horizon_reached'}`
        : metric.sourceField || metric.sourceFields?.join(', ') || undefined;

    return {
      metricId: metric.metricId,
      displayName: metric.displayName,
      valueText: hasValue ? formatMetricValue(raw, metric) : '—',
      status: hasValue ? 'computed' : 'missing',
      unavailableReason: hasValue
        ? undefined
        : metric.unavailableReason || '缺少 step_metrics summary',
      sourceHint,
    };
  });
}

export function formatMetricTaskTypes(taskTypes: string[]): string {
  return taskTypes.map((taskType) => TASK_TYPE_LABELS[taskType] ?? taskType).join('、') || '—';
}

export function formatMetricEvaluationModes(modes: string[]): string {
  return modes.map((mode) => EVALUATION_MODE_LABELS[mode] ?? mode).join('、') || '—';
}

export const ISAAC_STACK_DEFAULT_METRIC_IDS = [
  'isaac_stack_success_rate_v1',
  'isaac_stack_mean_reward_v1',
  'isaac_stack_mean_episode_length_v1',
  'isaac_stack_failure_count_v1',
  'isaac_stack_timeout_rate_v1',
] as const;

export function normalizeEvaluationTaskType(taskType?: string | null): string {
  const value = (taskType ?? '').trim();
  if (value === 'block_stacking') return 'isaac_block_stacking';
  return value;
}

export function normalizeEvaluationMode(mode?: string | null): string {
  const value = (mode ?? '').trim();
  if (value === '策略评测') return 'trained_model_evaluation';
  if (value === 'episode 稳定性评测') return 'episode_stability';
  if (value === '数据过程评测') return 'dataset_offline';
  return value;
}

export function normalizeEvaluationJobResultPayload(result: Record<string, unknown>): {
  aggregate: Record<string, unknown>;
  perEpisode: Record<string, unknown> | null;
} {
  const aggregate = (
    result.aggregate && typeof result.aggregate === 'object'
      ? result.aggregate
      : result
  ) as Record<string, unknown>;

  if (Array.isArray(result.episodes)) {
    return { aggregate, perEpisode: { episodes: result.episodes } };
  }
  if (result.perEpisode && typeof result.perEpisode === 'object') {
    return { aggregate, perEpisode: result.perEpisode as Record<string, unknown> };
  }
  if (result.per_episode && typeof result.per_episode === 'object') {
    return { aggregate, perEpisode: result.per_episode as Record<string, unknown> };
  }
  return { aggregate, perEpisode: null };
}

export type GenericMetricStatus = 'implemented' | 'planned';

export interface GenericMetricSpec {
  groupKey: string;
  displayName: string;
  description: string;
  valueType: MetricValueType;
  unit?: string;
  plannedOnly?: boolean;
}

export interface MetricTaskMapping {
  metricId: string;
  taskType: string;
  taskLabel: string;
  evaluationMode: string;
  evaluationModeLabel: string;
  sourceField?: string;
  sourceFields?: string[];
  calculationMode: MetricCalculationMode;
  implemented: boolean;
}

export interface GenericMetricGroup {
  groupKey: string;
  displayName: string;
  description: string;
  valueType: MetricValueType;
  unit?: string;
  status: GenericMetricStatus;
  mappings: MetricTaskMapping[];
  applicableTaskLabels: string[];
  applicableEvaluationModeLabels: string[];
  calculationModeSummary: string;
}

export const GENERIC_METRIC_SPECS: GenericMetricSpec[] = [
  {
    groupKey: 'success_rate',
    displayName: '成功率',
    description: '成功 episode 数占总 episode 数的比例。',
    valueType: 'ratio',
    unit: '%',
  },
  {
    groupKey: 'mean_reward',
    displayName: '平均奖励',
    description: '评测 rollout 中 episode 奖励的平均值。',
    valueType: 'number',
    unit: '',
  },
  {
    groupKey: 'mean_episode_length',
    displayName: '平均步长',
    description: '模型完成或终止一个 episode 的平均仿真步数。',
    valueType: 'number',
    unit: 'steps',
  },
  {
    groupKey: 'failure_count',
    displayName: '失败次数',
    description: '评测中未达成任务成功条件的 episode 数量。',
    valueType: 'integer',
    unit: 'episodes',
  },
  {
    groupKey: 'timeout_rate',
    displayName: '超时率',
    description: '因达到 horizon 上限而失败的 episode 比例。',
    valueType: 'ratio',
    unit: '%',
  },
  {
    groupKey: 'episode_stability',
    displayName: 'Episode 稳定性',
    description: 'Episode 稳定性评测中聚合的多维稳定性指标。',
    valueType: 'composite',
    unit: '',
  },
  {
    groupKey: 'trajectory_error',
    displayName: '轨迹误差',
    description: '策略轨迹与参考轨迹之间的偏差统计。',
    valueType: 'number',
    plannedOnly: true,
  },
  {
    groupKey: 'collision_count',
    displayName: '碰撞次数',
    description: '评测过程中发生碰撞事件的次数统计。',
    valueType: 'integer',
    plannedOnly: true,
  },
  {
    groupKey: 'action_smoothness',
    displayName: '动作平滑度',
    description: '策略输出动作序列的平滑程度评估。',
    valueType: 'number',
    plannedOnly: true,
  },
];

export function formatMetricTaskType(taskType: string): string {
  if (!taskType) return '—';
  return TASK_TYPE_LABELS[taskType] ?? taskType;
}

export function formatMetricEvaluationMode(mode: string): string {
  if (!mode) return '—';
  return EVALUATION_MODE_LABELS[mode] ?? mode;
}

export function formatCalculationModeSummary(mode: MetricCalculationMode): string {
  if (mode === 'aggregate_field') return 'aggregate 字段映射';
  if (mode === 'per_episode_failure_reason') return 'per-episode 统计';
  if (mode === 'aggregate_fields') return 'aggregate 复合字段';
  return mode;
}

export function formatSourceFieldsLabel(mapping: MetricTaskMapping): string {
  if (mapping.sourceField) return mapping.sourceField;
  if (mapping.sourceFields?.length) return mapping.sourceFields.join(', ');
  return '—';
}

export function truncateLabels(labels: string[], maxVisible = 2): string {
  const unique = [...new Set(labels.filter(Boolean))];
  if (unique.length === 0) return '—';
  if (unique.length <= maxVisible) return unique.join('、');
  return `${unique.slice(0, maxVisible).join('、')} +${unique.length - maxVisible}`;
}

function inferGenericMetricGroupKey(metric: MetricDefinition): string | null {
  if (metric.calculationMode === 'per_episode_failure_reason') return 'timeout_rate';
  if (metric.calculationMode === 'aggregate_fields') return 'episode_stability';
  if (metric.sourceField === 'successRate') return 'success_rate';
  if (metric.sourceField === 'meanReward') return 'mean_reward';
  if (metric.sourceField === 'meanEpisodeLength') return 'mean_episode_length';
  if (metric.sourceField === 'failureCount') return 'failure_count';
  return null;
}

function expandMetricToMappings(metric: MetricDefinition): MetricTaskMapping[] {
  const taskTypes = metric.applicableTaskTypes.length > 0 ? metric.applicableTaskTypes : [''];
  const modes =
    metric.applicableEvaluationModes.length > 0 ? metric.applicableEvaluationModes : [''];
  const rows: MetricTaskMapping[] = [];
  for (const taskType of taskTypes) {
    for (const mode of modes) {
      rows.push({
        metricId: metric.metricId,
        taskType,
        taskLabel: formatMetricTaskType(taskType),
        evaluationMode: mode,
        evaluationModeLabel: formatMetricEvaluationMode(mode),
        sourceField: metric.sourceField,
        sourceFields: metric.sourceFields,
        calculationMode: metric.calculationMode,
        implemented: metric.implemented,
      });
    }
  }
  return rows;
}

function summarizeCalculationModes(mappings: MetricTaskMapping[]): string {
  const labels = [
    ...new Set(mappings.map((mapping) => formatCalculationModeSummary(mapping.calculationMode))),
  ];
  return labels.join('、') || '—';
}

export function groupMetricsByGenericMetric(resources: RegistryResource[]): GenericMetricGroup[] {
  const registryMetrics = resources
    .map(parseRegistryMetricDefinition)
    .filter((item): item is MetricDefinition => item != null);

  const mappingsByGroup = new Map<string, MetricTaskMapping[]>();
  for (const metric of registryMetrics) {
    const groupKey = inferGenericMetricGroupKey(metric);
    if (!groupKey) continue;
    const existing = mappingsByGroup.get(groupKey) ?? [];
    mappingsByGroup.set(groupKey, [...existing, ...expandMetricToMappings(metric)]);
  }

  return GENERIC_METRIC_SPECS.map((spec) => {
    const mappings = mappingsByGroup.get(spec.groupKey) ?? [];
    const activeMappings = mappings.filter((mapping) => mapping.implemented);
    const status: GenericMetricStatus =
      spec.plannedOnly || activeMappings.length === 0 ? 'planned' : 'implemented';

    const applicableTaskLabels = [
      ...new Set(activeMappings.map((mapping) => mapping.taskLabel).filter((label) => label !== '—')),
    ];
    const applicableEvaluationModeLabels = [
      ...new Set(
        activeMappings
          .map((mapping) => mapping.evaluationModeLabel)
          .filter((label) => label !== '—')
      ),
    ];

    return {
      groupKey: spec.groupKey,
      displayName: spec.displayName,
      description: spec.description,
      valueType: spec.valueType,
      unit: spec.unit,
      status,
      mappings,
      applicableTaskLabels,
      applicableEvaluationModeLabels,
      calculationModeSummary: summarizeCalculationModes(activeMappings.length > 0 ? activeMappings : mappings),
    };
  });
}

export function summarizeGenericMetricLibrary(groups: GenericMetricGroup[]): {
  total: number;
  implemented: number;
  planned: number;
  applicableTaskCount: number;
} {
  const taskLabels = new Set<string>();
  for (const group of groups) {
    if (group.status === 'implemented') {
      group.applicableTaskLabels.forEach((label) => taskLabels.add(label));
    }
  }
  return {
    total: groups.length,
    implemented: groups.filter((group) => group.status === 'implemented').length,
    planned: groups.filter((group) => group.status === 'planned').length,
    applicableTaskCount: taskLabels.size,
  };
}
