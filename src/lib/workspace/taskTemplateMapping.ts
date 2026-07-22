import type { Dataset } from '@/types/benchmark';
import type { ModelAsset } from '@/types/benchmark';
import type { TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import { resolveDatasetSourceLabel } from '@/lib/workspace/datasetDisplay';
import { resolveModelAssetBackendType } from '@/lib/workspace/evaluationModelBackendCompatibility';
import {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  getTaskDisplayName,
  getTaskTemplateDisplayName,
  LEGACY_CABLE_THREADING_LABELS,
  normalizeTaskDisplayName,
  TASK_TEMPLATE_DISPLAY_NAMES,
} from '@/lib/workspace/taskDisplayNames';

export {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  getTaskDisplayName,
  getTaskTemplateDisplayName,
  normalizeTaskDisplayName,
  TASK_TEMPLATE_DISPLAY_NAMES,
} from '@/lib/workspace/taskDisplayNames';

export const PRECISION_ASSEMBLY_FAMILY = 'precision_assembly';

/** 平台 TaskTemplate.id → 后端 task_type */
export const TEMPLATE_ID_TO_BACKEND_TASK_TYPE: Record<string, string> = {
  cable_threading_single_arm: 'cable_threading',
  nut_assembly_single_arm: 'nut_assembly',
  dual_arm_cable_manipulation: 'dual_arm_cable_manipulation',
  isaac_block_stacking: 'block_stacking',
  isaacsim_franka_pick_place: 'isaacsim_franka_pick_place',
};

/** registry task config asset_id */
export const REGISTRY_TASK_CONFIG_IDS: Record<string, string> = {
  cable_threading: 'task_cable_threading_v1',
  nut_assembly: 'task_nut_assembly_v1',
  dual_arm_cable_manipulation: 'task_dual_arm_cable_manipulation_v1',
  block_stacking: 'task_isaac_block_stacking_v1',
  isaacsim_franka_pick_place: 'task_isaacsim_franka_pick_place_v1',
};

const REGISTRY_ID_TO_TEMPLATE_ID: Record<string, string> = {
  task_cable_threading_v1: 'cable_threading_single_arm',
  task_nut_assembly_v1: 'nut_assembly_single_arm',
  task_dual_arm_cable_manipulation_v1: 'dual_arm_cable_manipulation',
  task_isaac_block_stacking_v1: 'isaac_block_stacking',
  task_isaacsim_franka_pick_place_v1: 'isaacsim_franka_pick_place',
};

export const CABLE_MANIPULATION_FAMILY = 'cable_manipulation';
export const MANIPULATION_CORE_FAMILY = 'manipulation_core';

/** 平台 TaskTemplate.id → 用户可见展示名（不改 template id） */

export function formatTaskTemplateDisplayName(templateId: string | null | undefined): string | null {
  return getTaskTemplateDisplayName(templateId);
}

export function resolveDatasetSourceTaskLabel(dataset: Dataset): string {
  if (dataset.taskDisplayName?.trim()) {
    return getTaskDisplayName(dataset.taskDisplayName);
  }
  if (dataset.taskType?.trim()) {
    return getTaskDisplayName(dataset.taskType);
  }
  if (dataset.taskTemplateId) {
    const fromTemplate = formatTaskTemplateDisplayName(dataset.taskTemplateId);
    if (fromTemplate) return fromTemplate;
  }
  if (
    dataset.sourceJobId.startsWith('data_gen_') ||
    dataset.taskTemplateId === 'isaacsim_franka_pick_place'
  ) {
    return TASK_TEMPLATE_DISPLAY_NAMES.isaacsim_franka_pick_place;
  }
  if (
    dataset.sourceJobId.startsWith('data_gen_') ||
    dataset.taskTemplateId === 'isaacsim_franka_pick_place'
  ) {
    return TASK_TEMPLATE_DISPLAY_NAMES.isaacsim_franka_pick_place;
  }
  if (
    dataset.sourceJobId.startsWith('isaac_import_') ||
    dataset.sourceJobId.startsWith('isaac_gen_')
  ) {
    return TASK_TEMPLATE_DISPLAY_NAMES.isaac_block_stacking;
  }
  if (dataset.sourceJobId.startsWith('dac_gen_')) {
    return TASK_TEMPLATE_DISPLAY_NAMES.dual_arm_cable_manipulation;
  }
  if (dataset.sourceJobId.startsWith('ct_gen_')) {
    return TASK_TEMPLATE_DISPLAY_NAMES.cable_threading_single_arm;
  }
  const registryTemplateId = dataset.sourceTaskTemplateId
    ? REGISTRY_ID_TO_TEMPLATE_ID[dataset.sourceTaskTemplateId]
    : undefined;
  const fromTemplate = formatTaskTemplateDisplayName(registryTemplateId ?? null);
  if (fromTemplate) return fromTemplate;
  if (dataset.sourceTaskTemplateId && LEGACY_CABLE_THREADING_LABELS.has(dataset.sourceTaskTemplateId)) {
    return TASK_TEMPLATE_DISPLAY_NAMES.cable_threading_single_arm;
  }
  return dataset.sourceTaskTemplateId ?? '—';
}

export function resolveBackendTaskType(templateId: string): string | null {
  return TEMPLATE_ID_TO_BACKEND_TASK_TYPE[templateId] ?? null;
}

export function resolveTemplateIdFromBackendTaskType(taskType: string): string | null {
  for (const [id, backend] of Object.entries(TEMPLATE_ID_TO_BACKEND_TASK_TYPE)) {
    if (backend === taskType) return id;
  }
  return null;
}

/** 从模型资产字段推断平台 TaskTemplate.id（评测预填） */
export function resolveTaskTemplateIdFromModelAsset(asset: ModelAsset): string {
  const taskTemplateId = (asset.taskTemplateId ?? '').trim();
  if (taskTemplateId) {
    const fromRegistry = REGISTRY_ID_TO_TEMPLATE_ID[taskTemplateId];
    if (fromRegistry) return fromRegistry;
    if (TEMPLATE_ID_TO_BACKEND_TASK_TYPE[taskTemplateId]) return taskTemplateId;
  }

  const frameworkKey = (asset.framework ?? '').trim().toLowerCase();
  const modelTypeKey = (asset.modelType ?? '').trim().toLowerCase();
  if (modelTypeKey.includes('isaac') || frameworkKey.includes('isaac')) {
    return 'isaac_block_stacking';
  }

  const jobId = (asset.sourceTrainingJobId ?? '').trim();
  if (jobId.startsWith('dac_')) {
    return 'dual_arm_cable_manipulation';
  }

  return 'cable_threading_single_arm';
}

export function resolveTemplateIdFromLegacyName(name: string): string | null {
  if (
    name.includes('线缆穿杆') ||
    name === CABLE_THREADING_DISPLAY_NAME ||
    name === '线缆穿杆任务' ||
    name === '单臂线缆穿杆' ||
    name === '单臂线缆穿杆任务' ||
    name === '线缆穿杆（单臂）'
  ) {
    return 'cable_threading_single_arm';
  }
  if (
    name.includes('线缆操控') ||
    name.includes('双臂') ||
    name.includes('DualArm')
  ) {
    return 'dual_arm_cable_manipulation';
  }
  if (
    name.includes('物块堆叠') ||
    name.includes('Franka Stack Cube') ||
    name.includes('Stack Cube') ||
    name.includes('isaaclab_franka')
  ) {
    return 'isaaclab_franka_stack_cube';
  }
  if (name.includes('Stack') || name.includes('isaac_block')) {
    return 'isaac_block_stacking';
  }
  return null;
}

export function datasetMatchesTaskTemplate(dataset: Dataset, template: TaskTemplateDto): boolean {
  const backendType = resolveBackendTaskType(template.id);
  if (!backendType) return false;

  if (dataset.sourceTaskTemplateId === template.registryTaskConfigId) {
    return true;
  }
  if (backendType === 'cable_threading' && dataset.sourceJobId.startsWith('ct_gen_')) {
    return true;
  }
  if (backendType === 'dual_arm_cable_manipulation' && dataset.sourceJobId.startsWith('dac_gen_')) {
    return true;
  }
  if (backendType === 'isaacsim_franka_pick_place' && dataset.sourceJobId.startsWith('data_gen_')) {
    return true;
  }
  return false;
}

export function modelAssetMatchesTaskTemplate(asset: ModelAsset, template: TaskTemplateDto): boolean {
  if (asset.taskTemplateId === template.registryTaskConfigId) {
    return true;
  }
  const backendType = resolveBackendTaskType(template.id);
  if (!backendType) return false;

  if (backendType === 'dual_arm_cable_manipulation') {
    const assetTask = asset.taskTemplateId ?? '';
    const okTemplates = new Set([
      'dual_arm_cable_manipulation',
      'task_dual_arm_cable_manipulation_v1',
    ]);
    if (assetTask && !okTemplates.has(assetTask) && assetTask !== template.id) {
      return false;
    }
    return resolveModelAssetBackendType(asset) === 'torch_bc';
  }

  if (backendType === 'cable_threading') {
    if (resolveModelAssetBackendType(asset) === 'torch_bc') {
      return false;
    }
    return true;
  }

  if (backendType === 'block_stacking') {
    const assetTask = asset.taskTemplateId ?? '';
    const okTemplates = new Set(['isaac_block_stacking', 'task_isaac_block_stacking_v1']);
    if (assetTask && !okTemplates.has(assetTask) && assetTask !== template.id) {
      return false;
    }
    return (
      asset.framework === 'Robomimic BC' ||
      asset.framework === 'Isaac Robomimic BC' ||
      asset.framework === 'isaac_robomimic_bc' ||
      asset.framework === 'robomimic_bc'
    );
  }

  return false;
}

export const EVALUATION_MODE_LABELS: Record<string, string> = {
  expert_policy_evaluation: '专家策略评测',
  trained_model_evaluation: '训练模型评测',
  episode_stability: 'Episode 稳定性评测',
};

export function formatDatasetSourceType(
  sourceType: Dataset['sourceType'],
  dataset?: Pick<Dataset, 'simulatorBackend' | 'sourceJobId'>
): string {
  return resolveDatasetSourceLabel({
    sourceType,
    simulatorBackend: dataset?.simulatorBackend ?? null,
    sourceJobId: dataset?.sourceJobId ?? '',
  });
}

export function formatDatasetFormat(format: Dataset['format'] | string | null | undefined): string {
  const normalized = String(format ?? '')
    .trim()
    .toLowerCase();
  switch (normalized) {
    case 'hdf5':
    case 'robomimic_hdf5':
    case 'platform_hdf5':
      return 'HDF5';
    case 'npz':
      return 'NPZ';
    case 'zarr':
      return 'Zarr';
    case 'manifest':
      return 'Manifest';
    case 'lerobot':
      return 'LeRobot';
    case 'unknown':
    case '':
      return '—';
    default:
      if (normalized.endsWith('.hdf5') || normalized.endsWith('.h5')) return 'HDF5';
      if (normalized.includes('hdf5')) return 'HDF5';
      return String(format);
  }
}

export function resolveDatasetFormatLabel(
  dataset: Pick<Dataset, 'format' | 'datasetFormat' | 'sourceFormat' | 'datasetFile'>
): string {
  const raw = dataset.format ?? dataset.datasetFormat ?? dataset.sourceFormat;
  if (raw && String(raw).toLowerCase() !== 'unknown') {
    return formatDatasetFormat(raw);
  }
  const filePath = dataset.datasetFile;
  if (filePath && /\.hdf5?$/i.test(filePath)) {
    return 'HDF5';
  }
  if (!raw) return '—';
  return formatDatasetFormat(raw);
}
