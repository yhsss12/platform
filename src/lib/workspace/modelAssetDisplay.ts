import type { ModelAsset } from '@/types/benchmark';
import type { TrainingTaskRow } from '@/lib/mock/workspaceTrainingMock';
import {
  TRAINING_ALGORITHM_LABELS,
  formatTrainingRecipeLabel,
} from '@/lib/workspace/trainingRecipe';
import { normalizeDatasetDisplayName, isInvalidDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import {
  getTaskDisplayName,
  getTaskTemplateDisplayName,
} from '@/lib/workspace/taskDisplayNames';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';

const INTERNAL_ID_PATTERN =
  /^(?:model|train)_[0-9]{8}_[0-9]{6}(?:_[0-9a-f]{4})?$/i;
const JOB_ID_IN_NAME_PATTERN =
  /(?:^|_)(?:ct_gen|dac_gen|isaac_import|isaac_gen|isaac_ds)_[0-9]{8}_[0-9]{6}/i;
const SNAKE_CASE_INTERNAL_PATTERN = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;
const TASK_TEMPLATE_ID_PATTERN = /^task_[a-z0-9_]+$/i;
const DATE_TIME_SUFFIX_PATTERN =
  /^\d{4}\/\d{2}\/\d{2}\s+\d{2}:\d{2}$|^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?$/;

const INTERNAL_NAME_MARKERS = [
  'isaac_block_stacking',
  'isaac_stack_bc_smoke',
  'task_cable_threading_v1',
  'task_dual_arm_cable_manipulation_v1',
  'task_isaac_block_stacking_v1',
  'generated_dataset',
  'dataset.hdf5',
] as const;

function containsCjk(value: string): boolean {
  return /[\u4e00-\u9fff]/.test(value);
}

export function isInternalContextLabel(value?: string | null): boolean {
  const text = value?.trim();
  if (!text) return true;
  const lowered = text.toLowerCase();
  if (INTERNAL_ID_PATTERN.test(text)) return true;
  if (JOB_ID_IN_NAME_PATTERN.test(text)) return true;
  if (text.startsWith('train_') || text.startsWith('model_')) return true;
  if (TASK_TEMPLATE_ID_PATTERN.test(text)) return true;
  if (INTERNAL_NAME_MARKERS.some((marker) => lowered.includes(marker) || text.includes(marker))) {
    return true;
  }
  if (text.startsWith('isaac_stack') || text.startsWith('isaac_block')) return true;
  if (SNAKE_CASE_INTERNAL_PATTERN.test(text) && !containsCjk(text)) return true;
  return false;
}

export function isInternalModelAssetName(name?: string | null): boolean {
  const text = name?.trim();
  if (!text) return true;
  if (isInternalContextLabel(text)) return true;

  const parts = text
    .split('·')
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) return true;
  if (parts.length === 1) return isInternalContextLabel(parts[0]);
  if (INTERNAL_ID_PATTERN.test(parts[0]) || parts[0].startsWith('model_')) return true;
  if (parts[0].startsWith('train_')) return true;
  if (isInternalContextLabel(parts[0])) return true;
  const tail = parts[parts.length - 1]?.toLowerCase();
  if (tail === 'bc' || tail === 'diffusion_policy') return true;
  return false;
}

export function isFriendlyModelAssetDisplayName(name?: string | null): boolean {
  const text = name?.trim();
  if (!text || isInternalModelAssetName(text)) return false;

  const parts = text
    .split('·')
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length < 3) return false;
  if (isInternalContextLabel(parts[0])) return false;

  const recipe = parts[1]?.toLowerCase();
  if (!recipe || recipe === 'bc' || recipe === 'diffusion_policy' || recipe === 'unknown') {
    return false;
  }
  return DATE_TIME_SUFFIX_PATTERN.test(parts[parts.length - 1] ?? '') || parts[parts.length - 1] === '—';
}

/** 模型类型 + 框架合并为英文专业名 */
export function formatModelAssetRecipeLabel(
  asset: Pick<ModelAsset, 'framework' | 'modelType'>
): string {
  const framework = (asset.framework ?? '').trim();
  const modelType = (asset.modelType ?? '').trim();

  const fromFramework = formatTrainingRecipeLabel(framework, modelType);
  if (
    fromFramework &&
    fromFramework !== 'Unknown' &&
    fromFramework.toLowerCase() !== 'bc'
  ) {
    return fromFramework;
  }

  const loweredFramework = framework.toLowerCase();
  const loweredModelType = modelType.toLowerCase();

  if (loweredModelType === 'diffusion_policy' || loweredFramework === 'diffusion_policy') {
    return TRAINING_ALGORITHM_LABELS.diffusionPolicy;
  }

  if (loweredModelType === 'bc' || loweredFramework === 'bc') {
    if (loweredFramework.includes('torch') || loweredFramework === 'torch_bc') {
      return TRAINING_ALGORITHM_LABELS.torchBc;
    }
    if (
      loweredFramework.includes('robomimic') ||
      loweredFramework === 'isaac_robomimic_bc' ||
      loweredFramework === 'robomimic_bc'
    ) {
      return TRAINING_ALGORITHM_LABELS.robomimicBc;
    }
    return TRAINING_ALGORITHM_LABELS.robomimicBc;
  }

  return formatTrainingRecipeLabel(framework, modelType);
}

function normalizeContextLabel(
  asset: ModelAsset,
  trainingRow?: Pick<
    TrainingTaskRow,
    'name' | 'datasetName' | 'relatedTask' | 'taskType'
  > | null
): string {
  const trainingTaskName = trainingRow?.name?.trim();
  if (trainingTaskName && !isInternalContextLabel(trainingTaskName)) {
    return trainingTaskName;
  }

  const datasetCandidates = [
    trainingRow?.datasetName,
    trainingRow?.relatedTask,
    asset.sourceDatasetId,
  ];
  for (const candidate of datasetCandidates) {
    const dataset = candidate?.trim();
    if (!dataset) continue;

    const normalized = normalizeDatasetDisplayName({
      displayName: dataset,
      name: dataset,
      taskType: trainingRow?.taskType ?? asset.taskTemplateId,
      sourceJobId: asset.sourceDatasetId,
      taskDisplayName: getTaskTemplateDisplayName(asset.taskTemplateId) ?? undefined,
    });
    if (normalized && !isInternalContextLabel(normalized)) {
      return normalized;
    }
    if (!isInternalContextLabel(dataset)) {
      return dataset;
    }
  }

  const templateLabel = getTaskTemplateDisplayName(asset.taskTemplateId);
  if (templateLabel && !isInternalContextLabel(templateLabel)) {
    return templateLabel.endsWith('数据') ? templateLabel : `${templateLabel}数据`;
  }

  const taskLabel = getTaskDisplayName(asset.taskTemplateId);
  if (taskLabel && taskLabel !== '—' && !isInternalContextLabel(taskLabel)) {
    return taskLabel.endsWith('数据') ? taskLabel : `${taskLabel}数据`;
  }

  return '未命名模型资产';
}

export function buildModelAssetDisplayName(
  asset: ModelAsset,
  trainingRow?: Pick<
    TrainingTaskRow,
    'name' | 'datasetName' | 'relatedTask' | 'taskType'
  > | null
): string {
  const context = normalizeContextLabel(asset, trainingRow);
  const recipe = formatModelAssetRecipeLabel(asset);
  const createdLabel = formatDateTimeMinuteYmdSlash(asset.createdAt);
  return `${context} · ${recipe} · ${createdLabel}`;
}

export function resolveModelAssetColumnLabel(
  asset: ModelAsset,
  trainingRow?: Pick<TrainingTaskRow, 'name' | 'datasetName' | 'relatedTask' | 'taskType'> | null
): string {
  const stored = asset.displayName?.trim() || asset.name?.trim();
  if (stored && (isCheckpointAssetDisplayName(stored) || stored.includes(' · Final') || stored.includes(' · Epoch') || stored.includes(' · Best'))) {
    return stored;
  }

  const context = normalizeContextLabel(asset, trainingRow);
  const kind = (asset.checkpointKind ?? '').toLowerCase();
  if (kind === 'final') return `${context} · Final`;
  if (kind === 'best') {
    const metric = asset.checkpointMetricName?.trim() || 'Loss';
    return `${context} · Best ${metric}`;
  }
  if (kind === 'epoch' && asset.checkpointEpoch != null) {
    return `${context} · Epoch ${asset.checkpointEpoch}`;
  }
  return resolveModelAssetDisplayName(asset, trainingRow);
}

function isCheckpointAssetDisplayName(name: string): boolean {
  const parts = name.split('·').map((part) => part.trim()).filter(Boolean);
  if (parts.length !== 2) return false;
  const suffix = parts[1];
  return (
    suffix === 'Final' ||
    suffix.startsWith('Best ') ||
    suffix.startsWith('Epoch ') ||
    suffix.startsWith('Step ')
  );
}

export function resolveModelAssetDisplayName(
  asset: ModelAsset,
  trainingRow?: Pick<
    TrainingTaskRow,
    'name' | 'datasetName' | 'relatedTask' | 'taskType'
  > | null
): string {
  const explicit = asset.displayName?.trim();
  if (explicit && !isInternalModelAssetName(explicit)) {
    if (isFriendlyModelAssetDisplayName(explicit) || !isInternalContextLabel(explicit)) {
      return explicit;
    }
  }

  const stored = asset.name?.trim();
  if (stored && isFriendlyModelAssetDisplayName(stored)) {
    return stored;
  }
  if (stored && !isInternalModelAssetName(stored) && !stored.includes('·')) {
    return stored;
  }

  return buildModelAssetDisplayName(asset, trainingRow);
}

export { isInvalidDatasetDisplayName } from '@/lib/workspace/datasetNaming';

export function resolveModelAssetSourceLabel(asset: ModelAsset): string {
  if (asset.assetSource === 'imported' || asset.checkpointKind === 'imported') {
    return '外部导入';
  }
  if (asset.sourceTrainingJobId === 'model_asset_import_hub') {
    return '外部导入';
  }
  return '训练生成';
}

export function resolveModelAssetDatasetLabel(
  asset: ModelAsset,
  trainingRow?: Pick<TrainingTaskRow, 'datasetName' | 'relatedTask' | 'taskType'> | null
): string {
  const assetDataset = (asset as ModelAsset & { datasetDisplayName?: string | null }).datasetDisplayName;
  const candidates = [
    trainingRow?.datasetName,
    trainingRow?.relatedTask,
    assetDataset,
    asset.sourceDatasetId,
  ];

  for (const candidate of candidates) {
    const raw = candidate?.trim();
    if (!raw) continue;
    const normalized = normalizeDatasetDisplayName({
      displayName: raw,
      name: raw,
      taskType: trainingRow?.taskType ?? asset.taskTemplateId,
      sourceJobId: asset.sourceDatasetId,
      taskDisplayName: getTaskTemplateDisplayName(asset.taskTemplateId) ?? undefined,
    });
    if (normalized && !isInvalidDatasetDisplayName(normalized)) {
      return normalized;
    }
    if (!isInvalidDatasetDisplayName(raw)) {
      return raw;
    }
  }

  return '未知数据集';
}

export function resolveModelAssetTaskDatasetColumnLabel(
  asset: ModelAsset,
  trainingRow?: Pick<TrainingTaskRow, 'datasetName' | 'relatedTask' | 'taskType' | 'name'> | null
): string {
  const importMeta = asset.importMetadata as { taskLabel?: string; referenceDatasetName?: string } | null | undefined;
  if (asset.assetSource === 'imported' || asset.sourceTrainingJobId === 'model_asset_import_hub') {
    const task = importMeta?.taskLabel || asset.taskType || '—';
    const dataset = importMeta?.referenceDatasetName || asset.datasetDisplayName || resolveModelAssetDatasetLabel(asset, trainingRow);
    return `${task} · ${dataset}`;
  }
  const dataset = resolveModelAssetDatasetLabel(asset, trainingRow);
  const task = resolveModelAssetTrainingTaskLabel(asset, trainingRow);
  if (dataset === '—' || dataset === '未知数据集') return task;
  return `${dataset}`;
}

export function resolveModelAssetTrainingTaskLabel(
  asset: ModelAsset,
  trainingRow?: Pick<TrainingTaskRow, 'name'> | null
): string {
  const taskName = trainingRow?.name?.trim();
  if (taskName && !isInternalContextLabel(taskName)) {
    return taskName;
  }
  return asset.sourceTrainingJobId;
}

export function shortenJobId(jobId: string, max = 28): string {
  const value = jobId.trim();
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}…`;
}
