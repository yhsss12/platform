import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import { formatTrainingRecipeLabel } from '@/lib/workspace/trainingRecipe';
import { CABLE_THREADING_DEFAULTS } from '@/lib/workspace/cableThreading';
import { DUAL_ARM_CABLE_DEFAULTS } from '@/lib/workspace/dualArmCable';

const INVALID_TASK_NAMES = new Set(['未知任务', 'unknown', 'Unknown']);

export function isInvalidTrainingTaskDisplayName(name?: string | null): boolean {
  const value = (name ?? '').trim();
  if (!value) return true;
  if (INVALID_TASK_NAMES.has(value)) return true;
  return value.toLowerCase() === 'unknown';
}

function isActJointSpaceJob(trainingBackend?: string | null, modelType?: string | null): boolean {
  const backend = String(trainingBackend ?? '').toLowerCase();
  const model = String(modelType ?? '').toLowerCase();
  return backend === 'act' || model === 'act';
}

function isDpJointSpaceJob(trainingBackend?: string | null, modelType?: string | null): boolean {
  const backend = String(trainingBackend ?? '').toLowerCase();
  const model = String(modelType ?? '').toLowerCase();
  return backend === 'diffusion_policy' || model.includes('diffusion');
}

export function normalizeJointSpaceTrainingDisplayName(
  name: string,
  options?: { trainingBackend?: string | null; modelType?: string | null }
): string {
  const value = (name ?? '').trim();
  if (!value) return value;
  if (isActJointSpaceJob(options?.trainingBackend, options?.modelType) && value.startsWith('Joint-Space DP')) {
    return value.replace(/^Joint-Space DP/, 'ACT Joint-Space');
  }
  return value;
}

export function resolveTrainingTaskDisplayName(options: {
  taskName?: string | null;
  metaTaskName?: string | null;
  trainConfigTaskName?: string | null;
  datasetName?: string | null;
  trainingBackend?: string | null;
  modelType?: string | null;
  jobId?: string | null;
}): string {
  const normalize = (candidate?: string | null) =>
    normalizeJointSpaceTrainingDisplayName(String(candidate ?? '').trim(), {
      trainingBackend: options.trainingBackend,
      modelType: options.modelType,
    });

  const candidates = [options.taskName, options.metaTaskName, options.trainConfigTaskName];

  for (const candidate of candidates) {
    const value = normalize(candidate);
    if (!isInvalidTrainingTaskDisplayName(value)) return value;
  }

  const dataset = normalize(options.datasetName);
  if (dataset) {
    if (isActJointSpaceJob(options.trainingBackend, options.modelType)) {
      const suffix = dataset.includes('·') ? dataset.split('·').slice(-1)[0]?.trim() : dataset;
      if (suffix) return `ACT Joint-Space · ${suffix}`;
    }
    if (isDpJointSpaceJob(options.trainingBackend, options.modelType) && !dataset.startsWith('Joint-Space DP')) {
      const suffix = dataset.includes('·') ? dataset.split('·').slice(-1)[0]?.trim() : dataset;
      if (suffix) return `Joint-Space DP · ${suffix}`;
    }
    const model = formatTrainingRecipeLabel(options.trainingBackend, options.modelType);
    return model && model !== 'Unknown' ? `${dataset} · ${model}` : dataset;
  }

  const jobId = (options.jobId ?? '').trim();
  return jobId || '未命名任务';
}

export interface TrainingDatasetCardInfo {
  sourceTask: string;
  dataScale: string;
  simEnvironment: string;
  robotType: string;
  datasetCount?: number;
  totalTrajectories?: number;
}

export function resolveTrainingDatasetCardInfo(
  option: TrainingDatasetOption,
  dataCenterItems: WorkspaceDataItem[] = [],
  selectedOptions: TrainingDatasetOption[] = []
): TrainingDatasetCardInfo {
  const options = selectedOptions.length > 0 ? selectedOptions : [option];
  const totalCount = options.reduce((sum, item) => sum + (item.sampleCount || 0), 0);
  const primary = options[0] ?? option;
  const single = resolveTrainingDatasetCardInfoSingle(primary, dataCenterItems);
  if (options.length <= 1) {
    return single;
  }
  return {
    ...single,
    datasetCount: options.length,
    totalTrajectories: totalCount,
    dataScale: `${options.length} 个数据集 · 共 ${totalCount} 条成功轨迹`,
  };
}

function resolveTrainingDatasetCardInfoSingle(
  option: TrainingDatasetOption,
  dataCenterItems: WorkspaceDataItem[] = []
): TrainingDatasetCardInfo {
  const item = dataCenterItems.find(
    (row) => (row.datasetId ?? row.id) === option.id || row.id === option.id
  );

  const count = option.sampleCount;
  const dataScale = count > 0 ? `${count} 条成功轨迹` : '—';

  let simEnvironment = (item?.simBackend ?? item?.scene ?? '').trim();
  let robotType = (item?.robot ?? '').trim();

  if (option.taskType === 'isaac_block_stacking' || option.simulatorBackend === 'isaac_lab') {
    return {
      sourceTask: option.taskName || '物块堆叠',
      dataScale: count > 0 ? `${count} Episodes` : '—',
      simEnvironment: option.taskEnv || 'Isaac-Stack-Cube-Franka-IK-Rel-v0',
      robotType: 'Franka',
    };
  }

  if (option.taskType === 'dual_arm_cable_manipulation') {
    if (!simEnvironment) simEnvironment = 'MuJoCo';
    if (!robotType) robotType = DUAL_ARM_CABLE_DEFAULTS.robot ?? '双臂协作机器人';
  } else if (option.taskType === 'nut_assembly') {
    if (!simEnvironment) simEnvironment = 'MuJoCo';
    if (!robotType) robotType = 'Panda';
  } else if (option.taskType === 'cable_threading' || option.taskName.includes('单臂')) {
    if (!simEnvironment) simEnvironment = 'MuJoCo';
    if (!robotType) robotType = CABLE_THREADING_DEFAULTS.robot;
  }

  return {
    sourceTask: option.taskName || '—',
    dataScale,
    simEnvironment: simEnvironment || '—',
    robotType: robotType || '—',
  };
}

export function resolveTrainingInitWeightLabel(
  trainConfig?: { pretrained?: { modelAssetName?: string; modelAssetId?: string } | null } | null
): string {
  const pretrained = trainConfig?.pretrained;
  if (pretrained?.modelAssetName?.trim()) return pretrained.modelAssetName.trim();
  if (pretrained?.modelAssetId?.trim()) return pretrained.modelAssetId.trim();
  return '随机初始化';
}
