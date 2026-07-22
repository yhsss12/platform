import type { TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import { ISAAC_BLOCK_STACKING_TEMPLATE_ID } from '@/lib/workspace/isaacBlockStacking';
import { ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID } from '@/lib/workspace/isaaclabFrankaStackCube';
import { ISAACSIM_FRANKA_PICK_PLACE_TEMPLATE_ID } from '@/lib/workspace/isaacsimFrankaPickPlace';
import type { TrajectoryQualitySeverity } from '@/lib/workspace/isaacTrajectoryQuality';
import {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
} from '@/lib/workspace/taskDisplayNames';
import {
  FRANKA_STACK_CUBE_EVAL_CAPABILITY_LABEL,
  FRANKA_STACK_CUBE_PRODUCT_DESCRIPTION,
  FRANKA_STACK_CUBE_PRODUCT_NAME,
  isFrankStackCubeInternalTemplateId,
} from '@/lib/workspace/isaacStackCubeProduct';

export type TaskTemplateStatusKey = 'available' | 'pending' | 'maintenance';
export type SimulatorBackendFilter = 'mujoco' | 'isaac_lab' | 'isaacsim';
export type CapabilityFilter = 'data_generation' | 'training' | 'evaluation';
export type TaskTemplateCatalogTier = 'primary' | 'experimental';

/** 正式主流程任务模板（主表仅展示这三项） */
export const PRIMARY_TASK_TEMPLATE_IDS = [
  'cable_threading_single_arm',
  'dual_arm_cable_manipulation',
  ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID,
] as const;

/** 实验 / 示例 / 历史任务模板（默认折叠，不计入主统计） */
export const EXPERIMENTAL_TASK_TEMPLATE_IDS = [
  ISAACSIM_FRANKA_PICK_PLACE_TEMPLATE_ID,
] as const;

/** 产品层隐藏、仅作评测/训练适配的内部 templateId */
export const INTERNAL_TASK_TEMPLATE_IDS = [ISAAC_BLOCK_STACKING_TEMPLATE_ID] as const;

export function isPrimaryTaskTemplateId(templateId: string): boolean {
  return (PRIMARY_TASK_TEMPLATE_IDS as readonly string[]).includes(templateId);
}

export function isExperimentalTaskTemplateId(templateId: string): boolean {
  return (EXPERIMENTAL_TASK_TEMPLATE_IDS as readonly string[]).includes(templateId);
}

export function isInternalTaskTemplateId(templateId: string): boolean {
  return (
    isFrankStackCubeInternalTemplateId(templateId) ||
    (INTERNAL_TASK_TEMPLATE_IDS as readonly string[]).includes(templateId)
  );
}

export function resolveTaskTemplateCatalogTier(templateId: string): TaskTemplateCatalogTier {
  if (isPrimaryTaskTemplateId(templateId)) return 'primary';
  if (isExperimentalTaskTemplateId(templateId)) return 'experimental';
  return 'experimental';
}

export interface TaskTemplatePresentationMeta {
  shortDescription: string;
  involvedObjects: string;
  robotLabel: string;
  sceneLabel: string;
  expertStrategyLabel: string;
  expertStrategyTooltip?: string;
  defaultGenerationMode?: string;
  dataCapabilityTags: string[];
  trainingLabel: string;
  evaluationLabel: string;
  trajectoryQualityLabel: string;
  trajectoryQualitySeverity: TrajectoryQualitySeverity | 'pending';
  defaultCollectionRounds?: string;
  advancedExperimentNote?: string;
  runtimeBackendLabel?: string;
}

const PRESENTATION_BY_ID: Record<string, TaskTemplatePresentationMeta> = {
  cable_threading_single_arm: {
    shortDescription: '机械臂完成线缆穿过目标杆的仿真操作任务。',
    involvedObjects: '柔性线缆、目标杆',
    robotLabel: 'Panda / UR5e',
    sceneLabel: 'MuJoCo 穿杆工作台',
    expertStrategyLabel: '专家策略',
    expertStrategyTooltip: '基于几何约束与 scripted 控制自动生成穿杆轨迹。',
    dataCapabilityTags: ['HDF5', '回放', '失败追溯'],
    trainingLabel: 'Robomimic BC',
    evaluationLabel: '模型评测',
    trajectoryQualityLabel: '可用，失败可追溯',
    trajectoryQualitySeverity: 'passed',
    defaultCollectionRounds: '50–200 轮',
  },
  dual_arm_cable_manipulation: {
    shortDescription: '双臂协同完成线缆整理、拖拽与形态控制的仿真操作任务。',
    involvedObjects: '柔性线缆、固定点',
    robotLabel: 'Dual FR3',
    sceneLabel: 'MuJoCo 线缆整理工位',
    expertStrategyLabel: '双臂控制策略',
    expertStrategyTooltip: '双臂协同 scripted 策略，支持 episode 稳定性采集。',
    dataCapabilityTags: ['HDF5', '双臂轨迹', '回放'],
    trainingLabel: 'torch_bc',
    evaluationLabel: '稳定性评测 / 模型评测',
    trajectoryQualityLabel: '未生成',
    trajectoryQualitySeverity: 'pending',
    defaultCollectionRounds: '100+ episode',
  },
  [ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID]: {
    shortDescription: FRANKA_STACK_CUBE_PRODUCT_DESCRIPTION,
    involvedObjects: 'cube_1 / cube_2 / cube_3',
    robotLabel: 'Franka Panda',
    sceneLabel: 'Isaac Lab Stack Cube 场景',
    expertStrategyLabel: 'Mimic + Seed Demonstration',
    expertStrategyTooltip:
      '基于 Isaac Lab Mimic 与 seed demonstration 的真实专家策略生成轨迹；运行依赖 Isaac Sim / PhysX 环境。',
    defaultGenerationMode: 'mimic_auto',
    dataCapabilityTags: ['HDF5', 'Zarr', '回放'],
    trainingLabel: 'Isaac Robomimic BC',
    evaluationLabel: FRANKA_STACK_CUBE_EVAL_CAPABILITY_LABEL,
    trajectoryQualityLabel: 'Mimic 专家策略可用',
    trajectoryQualitySeverity: 'passed',
    defaultCollectionRounds: '1–50 demo',
    runtimeBackendLabel: '运行依赖 Isaac Sim / PhysX',
    advancedExperimentNote:
      '评测依赖 Isaac Lab 运行时，当前通过 isaac_block_stacking adapter 执行 rollout；底层 templateId 尚未完全合并。',
  },
  [ISAAC_BLOCK_STACKING_TEMPLATE_ID]: {
    shortDescription: FRANKA_STACK_CUBE_PRODUCT_DESCRIPTION,
    involvedObjects: 'cube_1 / cube_2 / cube_3',
    robotLabel: 'Franka Panda',
    sceneLabel: 'Isaac Lab Stack Cube 场景',
    expertStrategyLabel: '—',
    expertStrategyTooltip: '内部评测/训练适配入口，数据生成请使用物块堆叠主入口。',
    dataCapabilityTags: [],
    trainingLabel: 'Isaac Robomimic BC',
    evaluationLabel: FRANKA_STACK_CUBE_EVAL_CAPABILITY_LABEL,
    trajectoryQualityLabel: '—',
    trajectoryQualitySeverity: 'pending',
    runtimeBackendLabel: '运行依赖 Isaac Sim / PhysX',
    advancedExperimentNote: '内部 templateId，不在产品 UI 并列展示。',
  },
  [ISAACSIM_FRANKA_PICK_PLACE_TEMPLATE_ID]: {
    shortDescription: 'Franka 机械臂在 Isaac Sim 中完成官方 pick-and-place 物体搬运任务。',
    involvedObjects: 'cube、目标放置区域',
    robotLabel: 'Franka Panda',
    sceneLabel: 'Isaac Sim Franka Pick Place 官方场景',
    expertStrategyLabel: '官方 FrankaPickPlace controller',
    expertStrategyTooltip:
      '调用 NVIDIA Isaac Sim 官方 FrankaPickPlace 状态机控制器生成专家轨迹。',
    dataCapabilityTags: ['Episode Manifest', 'Metrics', 'Video', '回放'],
    trainingLabel: '—',
    evaluationLabel: '—',
    trajectoryQualityLabel: '可用',
    trajectoryQualitySeverity: 'passed',
    defaultCollectionRounds: '1–5 episode',
  },
};

export interface TaskTemplateAssetRow {
  template: TaskTemplateDto;
  templateId: string;
  registryId: string;
  catalogTier: TaskTemplateCatalogTier;
  name: string;
  shortDescription: string;
  taskTypeLabel: string;
  simulatorBackend: SimulatorBackendFilter;
  simulatorLabel: string;
  simulatorSubtitle: string;
  expertStrategyLabel: string;
  expertStrategyTooltip?: string;
  expertStrategyReady: boolean;
  dataCapabilityTags: string[];
  trainingLabel: string;
  evaluationLabel: string;
  trajectoryQualityLabel: string;
  trajectoryQualitySeverity: TrajectoryQualitySeverity | 'pending';
  statusLabel: string;
  statusKey: TaskTemplateStatusKey;
  supportsDataGeneration: boolean;
  supportsTraining: boolean;
  supportsEvaluation: boolean;
  meta: TaskTemplatePresentationMeta;
}

function resolvePresentationMeta(template: TaskTemplateDto): TaskTemplatePresentationMeta {
  const known = PRESENTATION_BY_ID[template.id];
  if (known) return known;
  return {
    shortDescription: template.description || '—',
    involvedObjects: '—',
    robotLabel: template.supportedRobotTypes?.join(' / ') || '—',
    sceneLabel: template.defaultSceneId ?? '—',
    expertStrategyLabel: '—',
    dataCapabilityTags: template.replayAvailable ? ['回放'] : [],
    trainingLabel: template.supportedPolicyTypes?.join(' / ') || '—',
    evaluationLabel: template.supportedEvaluationModes?.join(' / ') || '—',
    trajectoryQualityLabel: '未生成',
    trajectoryQualitySeverity: 'pending',
  };
}

function resolveSimulatorBackend(template: TaskTemplateDto): SimulatorBackendFilter {
  if (template.simulatorBackend === 'isaacsim') {
    return 'isaacsim';
  }
  if (template.simulatorBackend === 'isaac_lab' || template.simulatorType === 'isaac') {
    return 'isaac_lab';
  }
  return 'mujoco';
}

function resolveStatusKey(template: TaskTemplateDto): TaskTemplateStatusKey {
  if (template.status === 'maintenance') return 'maintenance';
  if (template.supportsDatasetGeneration === 'planned' || template.status === 'pending') {
    return 'pending';
  }
  return 'available';
}

function resolvePrimaryDisplayName(template: TaskTemplateDto): string {
  switch (template.id) {
    case 'cable_threading_single_arm':
      return CABLE_THREADING_DISPLAY_NAME;
    case 'dual_arm_cable_manipulation':
      return DUAL_ARM_CABLE_DISPLAY_NAME;
    case ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID:
      return FRANKA_STACK_CUBE_PRODUCT_NAME;
    default:
      return template.name;
  }
}

function resolveStatusLabel(key: TaskTemplateStatusKey): string {
  switch (key) {
    case 'maintenance':
      return '维护中';
    case 'pending':
      return '待配置';
    default:
      return '已接入';
  }
}

export function buildTaskTemplateAssetRow(template: TaskTemplateDto): TaskTemplateAssetRow {
  const meta = resolvePresentationMeta(template);
  const catalogTier = resolveTaskTemplateCatalogTier(template.id);
  const simulatorBackend = resolveSimulatorBackend(template);
  const statusKey = resolveStatusKey(template);
  const isPrimary = catalogTier === 'primary';

  return {
    template,
    templateId: template.id,
    registryId: template.registryTaskConfigId ?? template.id,
    catalogTier,
    name: isPrimary ? resolvePrimaryDisplayName(template) : template.name,
    shortDescription: meta.shortDescription,
    taskTypeLabel: template.taskType,
    simulatorBackend,
    simulatorLabel: simulatorBackend === 'isaacsim'
      ? 'Isaac Sim'
      : simulatorBackend === 'isaac_lab'
        ? 'Isaac Lab'
        : 'MuJoCo',
    simulatorSubtitle:
      simulatorBackend === 'isaacsim'
        ? 'NVIDIA Isaac Sim 官方示例'
        : simulatorBackend === 'isaac_lab'
          ? '高保真物理仿真'
          : '轻量物理仿真',
    expertStrategyLabel: meta.expertStrategyLabel,
    expertStrategyTooltip: meta.expertStrategyTooltip,
    expertStrategyReady: template.hasExpertPolicy ?? (statusKey === 'available' && template.supportsDatasetGeneration === true),
    dataCapabilityTags: meta.dataCapabilityTags,
    trainingLabel: meta.trainingLabel,
    evaluationLabel: meta.evaluationLabel,
    trajectoryQualityLabel: meta.trajectoryQualityLabel,
    trajectoryQualitySeverity: meta.trajectoryQualitySeverity,
    statusLabel: resolveStatusLabel(statusKey),
    statusKey,
    supportsDataGeneration: template.supportsDataGeneration ?? template.supportsDatasetGeneration === true,
    supportsTraining: (template.supportedPolicyTypes?.length ?? 0) > 0,
    supportsEvaluation:
      template.supportsEvaluation ?? (template.supportedEvaluationModes?.length ?? 0) > 0,
    meta,
  };
}

export interface TaskTemplateOverviewStats {
  connectedCount: number;
  backendCount: number;
  backendSummary: string;
  trainableCount: number;
  evaluableCount: number;
}

export function partitionTaskTemplateRows(rows: TaskTemplateAssetRow[]): {
  primary: TaskTemplateAssetRow[];
  experimental: TaskTemplateAssetRow[];
} {
  const primary: TaskTemplateAssetRow[] = [];
  const experimental: TaskTemplateAssetRow[] = [];
  for (const row of rows) {
    if (isInternalTaskTemplateId(row.templateId)) {
      continue;
    }
    if (row.catalogTier === 'primary') {
      primary.push(row);
    } else {
      experimental.push(row);
    }
  }
  return { primary, experimental };
}

export function computeTaskTemplateOverviewStats(rows: TaskTemplateAssetRow[]): TaskTemplateOverviewStats {
  const { primary } = partitionTaskTemplateRows(rows);
  return {
    connectedCount: primary.filter((r) => r.statusKey === 'available').length,
    backendCount: 2,
    backendSummary: 'MuJoCo / Isaac Lab',
    trainableCount: primary.filter((r) => r.supportsTraining).length,
    evaluableCount: primary.filter((r) => r.supportsEvaluation).length,
  };
}

export function filterTaskTemplateRows(
  rows: TaskTemplateAssetRow[],
  opts: {
    search: string;
    backendFilter: '' | SimulatorBackendFilter;
    statusFilter: '' | TaskTemplateStatusKey;
    capabilityFilter: '' | CapabilityFilter;
  }
): TaskTemplateAssetRow[] {
  const q = opts.search.trim().toLowerCase();
  return rows.filter((row) => {
    if (opts.backendFilter && row.simulatorBackend !== opts.backendFilter) return false;
    if (opts.statusFilter && row.statusKey !== opts.statusFilter) return false;
    if (opts.capabilityFilter === 'data_generation' && !row.supportsDataGeneration) return false;
    if (opts.capabilityFilter === 'training' && !row.supportsTraining) return false;
    if (opts.capabilityFilter === 'evaluation' && !row.supportsEvaluation) return false;
    if (!q) return true;
    const haystack = [
      row.name,
      row.shortDescription,
      row.taskTypeLabel,
      row.simulatorLabel,
      row.expertStrategyLabel,
      row.trainingLabel,
      row.evaluationLabel,
      row.dataCapabilityTags.join(' '),
    ]
      .join(' ')
      .toLowerCase();
    return haystack.includes(q);
  });
}

export function trajectoryQualityBadgeStyle(
  severity: TrajectoryQualitySeverity | 'pending'
): { backgroundColor: string; color: string; borderColor: string } {
  switch (severity) {
    case 'passed':
      return { backgroundColor: '#ecfdf5', color: '#047857', borderColor: '#a7f3d0' };
    case 'mild':
      return { backgroundColor: '#fffbeb', color: '#b45309', borderColor: '#fde68a' };
    case 'motion':
      return { backgroundColor: '#fef2f2', color: '#b91c1c', borderColor: '#fecaca' };
    case 'failed':
      return { backgroundColor: '#fef2f2', color: '#991b1b', borderColor: '#fca5a5' };
    default:
      return { backgroundColor: '#f3f4f6', color: '#6b7280', borderColor: '#e5e7eb' };
  }
}

export function statusBadgeStyle(
  statusKey: TaskTemplateStatusKey
): { backgroundColor: string; color: string; borderColor: string } {
  switch (statusKey) {
    case 'available':
      return { backgroundColor: '#ecfdf5', color: '#047857', borderColor: '#a7f3d0' };
    case 'pending':
      return { backgroundColor: '#fffbeb', color: '#b45309', borderColor: '#fde68a' };
    default:
      return { backgroundColor: '#f3f4f6', color: '#6b7280', borderColor: '#e5e7eb' };
  }
}
