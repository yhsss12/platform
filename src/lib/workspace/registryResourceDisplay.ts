import type { RegistryResource } from '@/lib/api/resourceRegistryClient';
import { registryStatusLabel } from '@/lib/api/resourceRegistryClient';

const ASSET_TYPE_LABELS: Record<string, string> = {
  metric: '评测指标',
  task: '任务模板',
  robot: '机器人',
  scene: '场景',
  object: '物体',
  policy: '策略',
  end_effector: '末端执行器',
};

const SCENE_TAG_LABELS: Record<string, string> = {
  cable_threading: '线缆穿杆',
  dual_arm: '线缆整理',
  episode_stability: 'Episode 稳定性',
  success_rate: '成功率',
};

export function registryAssetTypeLabel(assetType?: string | null): string {
  const key = (assetType ?? '').trim();
  return (ASSET_TYPE_LABELS[key] ?? key) || '资源';
}

export function resolveRegistryScenarioLabel(resource: RegistryResource): string {
  if (resource.assetType === 'metric') {
    const taskTypes = extractStringList(resource.metadata?.applicableTaskTypes);
    if (taskTypes.length > 0) {
      return formatMetricTaskTypesShort(taskTypes);
    }
  }
  const tags = resource.tags ?? [];
  const fromTags = tags
    .map((tag) => SCENE_TAG_LABELS[tag] ?? tag)
    .filter(Boolean)
    .slice(0, 3)
    .join(' · ');
  if (fromTags) return fromTags;
  if (resource.taskType) return String(resource.taskType);
  return registryAssetTypeLabel(resource.assetType);
}

export function formatRegistryUpdatedAt(value?: string | null): string {
  if (!value?.trim()) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value.slice(0, 10);
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
}

export function extractStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item)).filter(Boolean);
}

export function formatThresholds(metadata: Record<string, unknown>): string | null {
  const thresholds = metadata.thresholds;
  if (!thresholds || typeof thresholds !== 'object') return null;
  const row = thresholds as Record<string, unknown>;
  const parts: string[] = [];
  if (row.pass != null) parts.push(`通过线 ≥ ${row.pass}`);
  if (row.warn != null) parts.push(`预警线 ≥ ${row.warn}`);
  return parts.length > 0 ? parts.join('；') : null;
}

export function registryResourceSummaryLine(resource: RegistryResource): string {
  if (resource.assetType === 'metric') {
    return formatMetricRegistrySummary(resource);
  }
  return [
    registryAssetTypeLabel(resource.assetType),
    resource.version || '—',
    `更新 ${formatRegistryUpdatedAt(resource.lastModifiedAt)}`,
  ].join(' · ');
}

export function formatMetricRegistrySummary(resource: RegistryResource): string {
  const metadata = resource.metadata ?? {};
  const taskTypes = extractStringList(metadata.applicableTaskTypes);
  const modes = extractStringList(metadata.applicableEvaluationModes);
  const implemented = metadata.implemented === true;
  const parts = [
    implemented ? '已接入' : '待接入',
    taskTypes.length > 0 ? formatMetricTaskTypesShort(taskTypes) : '通用',
    modes.length > 0 ? formatMetricModesShort(modes) : '',
  ].filter(Boolean);
  return parts.join(' · ');
}

function formatMetricTaskTypesShort(taskTypes: string[]): string {
  const labels: Record<string, string> = {
    isaac_block_stacking: '物块堆叠',
    isaaclab_franka_stack_cube: '物块堆叠',
    cable_threading: '线缆穿杆',
    dual_arm_cable_manipulation: '线缆整理',
  };
  return taskTypes.map((taskType) => labels[taskType] ?? taskType).join('、');
}

function formatMetricModesShort(modes: string[]): string {
  const labels: Record<string, string> = {
    trained_model_evaluation: '已训练模型',
    expert_policy_evaluation: '专家策略',
    episode_stability: 'Episode 稳定性',
  };
  return modes.map((mode) => labels[mode] ?? mode).join('、');
}

export function metricImplementationLabel(resource: RegistryResource): string {
  const implemented = resource.metadata?.implemented === true;
  return implemented ? '已接入' : '待接入';
}
