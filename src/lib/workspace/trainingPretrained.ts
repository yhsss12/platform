import type { TrainingBackendRequest, TrainingCapabilities } from '@/lib/api/trainingClient';
import type { ModelAssetCheckpointOption } from '@/lib/api/modelAssetsClient';
import type { ModelAsset } from '@/types/benchmark';
import type { TrainingPretrainedOptions } from '@/lib/mock/workspaceTrainingMock';
import {
  defaultTrainingRecipeForContext,
  getTrainingRecipe,
} from '@/lib/workspace/trainingRecipe';
import {
  type DatasetStructureSignature,
  extractDatasetStructureSignatureFromOption,
} from '@/lib/workspace/trainingDatasetCompat';
import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import { formatDateTimeMinuteYmdSlash } from '@/utils/format';
import { formatModelAssetRecipeLabel } from '@/lib/workspace/modelAssetDisplay';

const DUAL_ARM_TASK_TEMPLATE_IDS = new Set([
  'dual_arm_cable_manipulation',
  'task_dual_arm_cable_manipulation_v1',
]);

const SINGLE_ARM_TASK_TEMPLATE_IDS = new Set([
  'cable_threading_single_arm',
  'task_cable_threading_v1',
]);

const ISAAC_TASK_TEMPLATE_IDS = new Set(['isaac_block_stacking', 'task_isaac_block_stacking_v1']);

export const PRETRAINED_STRUCTURE_MISMATCH_HINT = '模型结构不匹配，无法用于当前任务';

function normKeys(keys: unknown): string[] {
  if (!Array.isArray(keys)) return [];
  return [...new Set(keys.map((key) => String(key).trim()).filter(Boolean))].sort();
}

export function normalizeAssetFramework(framework: string | null | undefined): TrainingBackendRequest | null {
  const value = (framework || '').trim().toLowerCase();
  if (!value) return null;
  if (value === 'torch_bc') return 'torch_bc';
  if (value === 'robomimic_bc' || value === 'robomimic' || value === 'robomimic bc') return 'robomimic_bc';
  if (value === 'isaac_robomimic_bc' || value.includes('isaac robomimic')) return 'isaac_robomimic_bc';
  if (value === 'diffusion_policy' || value === 'diffusion policy') return 'diffusion_policy';
  if (value === 'act') return 'act';
  return null;
}

export function isIsaacModelAsset(asset: ModelAsset): boolean {
  if (asset.taskTemplateId && ISAAC_TASK_TEMPLATE_IDS.has(asset.taskTemplateId)) {
    return true;
  }
  return normalizeAssetFramework(asset.framework) === 'isaac_robomimic_bc';
}

export function isDualArmModelAsset(asset: ModelAsset): boolean {
  if (asset.taskTemplateId && DUAL_ARM_TASK_TEMPLATE_IDS.has(asset.taskTemplateId)) {
    return true;
  }
  return normalizeAssetFramework(asset.framework) === 'torch_bc';
}

export function isSingleArmModelAsset(asset: ModelAsset): boolean {
  if (isIsaacModelAsset(asset) || isDualArmModelAsset(asset)) {
    return false;
  }
  if (asset.taskTemplateId && SINGLE_ARM_TASK_TEMPLATE_IDS.has(asset.taskTemplateId)) {
    return true;
  }
  const framework = normalizeAssetFramework(asset.framework);
  return framework === 'robomimic_bc' || framework === 'diffusion_policy' || framework === 'act';
}

export function trainingRecipeDomainMismatch(
  _trainingRecipeId: TrainingBackendRequest,
  _context: { isDualArm: boolean; isIsaac?: boolean }
): boolean {
  return false;
}

/** 预训练过滤使用与当前数据集匹配的真实训练后端 */
export function effectivePretrainedTrainingBackend(
  trainingRecipeId: TrainingBackendRequest,
  context: {
    isDualArm: boolean;
    isIsaac?: boolean;
    capabilities: TrainingCapabilities | null | undefined;
  }
): TrainingBackendRequest {
  const selected = getTrainingRecipe(trainingRecipeId);
  if (
    selected?.trainability === 'real' &&
    !trainingRecipeDomainMismatch(trainingRecipeId, context)
  ) {
    return trainingRecipeId;
  }
  return defaultTrainingRecipeForContext({ capabilities: context.capabilities });
}

function extractAssetStructureSignature(asset: ModelAsset): Partial<DatasetStructureSignature> {
  const structure = (asset.structureConfig ?? {}) as Record<string, unknown>;
  const resolved = (asset.resolvedModelParams ?? {}) as Record<string, unknown>;
  const input = (structure.input ?? resolved.input ?? {}) as Record<string, unknown>;
  const output = (structure.output ?? resolved.output ?? {}) as Record<string, unknown>;

  const imageKeys = normKeys(
    input.image_keys ?? input.camera_keys ?? input.imageKeys ?? resolved.image_keys
  );
  const lowDimKeys = normKeys(
    input.low_dim_keys ?? input.state_keys ?? input.lowDimKeys ?? resolved.low_dim_keys
  );

  return {
    taskType: String(structure.taskType ?? resolved.taskType ?? asset.taskTemplateId ?? '').trim(),
    actionDim:
      output.action_dim != null
        ? Number(output.action_dim)
        : resolved.action_dim != null
          ? Number(resolved.action_dim)
          : null,
    imageKeys,
    lowDimKeys,
    imageSize:
      input.image_size != null
        ? Number(input.image_size)
        : resolved.image_size != null
          ? Number(resolved.image_size)
          : null,
  };
}

export function modelAssetStructureMatchesDataset(
  asset: ModelAsset,
  datasetSignature: DatasetStructureSignature
): boolean {
  const assetSig = extractAssetStructureSignature(asset);
  if (assetSig.actionDim != null && datasetSignature.actionDim != null) {
    if (assetSig.actionDim !== datasetSignature.actionDim) return false;
  }
  if (assetSig.imageKeys?.length) {
    if (assetSig.imageKeys.join('|') !== datasetSignature.imageKeys.join('|')) return false;
  }
  if (assetSig.lowDimKeys?.length) {
    if (assetSig.lowDimKeys.join('|') !== datasetSignature.lowDimKeys.join('|')) return false;
  }
  if (assetSig.imageSize != null && datasetSignature.imageSize != null) {
    if (assetSig.imageSize !== datasetSignature.imageSize) return false;
  }
  return true;
}

export function modelAssetHorizonWarning(
  asset: ModelAsset,
  datasetSignature: DatasetStructureSignature
): string | null {
  void datasetSignature;
  const resolved = (asset.resolvedModelParams ?? {}) as Record<string, unknown>;
  const horizon = resolved.horizon;
  const nActionSteps = resolved.n_action_steps ?? resolved.nActionSteps;
  if (horizon == null && nActionSteps == null) return null;
  return `horizon/n_action_steps 可能与当前任务默认值不同（checkpoint: horizon=${horizon ?? '—'}, n_action_steps=${nActionSteps ?? '—'}），仍可加载但行为可能变化`;
}

export function modelAssetMatchesTrainingContext(
  asset: ModelAsset,
  context: {
    isDualArm: boolean;
    isIsaac?: boolean;
    trainingBackend: TrainingBackendRequest;
    datasetSignature?: DatasetStructureSignature | null;
  }
): boolean {
  const framework = normalizeAssetFramework(asset.framework);
  if (!framework || framework !== context.trainingBackend) {
    return false;
  }

  if (context.isIsaac) {
    if (!isIsaacModelAsset(asset)) return false;
  } else if (context.isDualArm) {
    if (!isDualArmModelAsset(asset)) return false;
  } else if (!isSingleArmModelAsset(asset)) {
    return false;
  }

  if (context.datasetSignature) {
    return modelAssetStructureMatchesDataset(asset, context.datasetSignature);
  }
  return true;
}

export function formatInitWeightOptionLines(asset: ModelAsset): { titleLine: string; subtitleLine: string; title: string } {
  const metricValue = asset.checkpointMetricValue;
  const lossText =
    metricValue != null && Number.isFinite(Number(metricValue))
      ? `Loss ${Number(metricValue).toFixed(4)}`
      : 'Loss —';
  const recipe = formatModelAssetRecipeLabel(asset);
  const created = formatDateTimeMinuteYmdSlash(asset.createdAt) || asset.createdAt || '—';
  const name = asset.displayName?.trim() || asset.name?.trim() || asset.id;
  const kindSuffix =
    asset.checkpointKind === 'final'
      ? 'Final'
      : asset.checkpointKind === 'best'
        ? 'Best'
        : asset.assetSource === 'imported'
          ? 'Imported'
          : asset.sourceTrainingJobId
            ? 'Final'
            : '';
  const alreadyHasKindSuffix =
    kindSuffix &&
    (name.endsWith(` · ${kindSuffix}`) ||
      name.endsWith(`· ${kindSuffix}`) ||
      name.endsWith(` · Final`) ||
      name.endsWith(`· Final`));
  const titleLine = kindSuffix && !alreadyHasKindSuffix ? `${name} · ${kindSuffix}` : name;
  const subtitleLine = `${recipe} · ${lossText} · ${created}`;
  return {
    titleLine,
    subtitleLine,
    title: `${titleLine}\n${subtitleLine}`,
  };
}

export function formatPretrainedModelOptionLabel(asset: ModelAsset): string {
  const lines = formatInitWeightOptionLines(asset);
  return `${lines.titleLine} · ${lines.subtitleLine}`;
}

export function buildTrainingPretrainedPayload(
  option: ModelAssetCheckpointOption,
  assets: ModelAsset[]
): TrainingPretrainedOptions {
  const asset = assets.find((item) => item.id === option.modelAssetId);
  return {
    modelAssetId: option.modelAssetId,
    checkpointPath: option.checkpointPath ?? undefined,
    modelAssetName: asset?.name ?? option.label,
    sourceTrainJobId: asset?.sourceTrainingJobId ?? option.trainJobId,
  };
}

export function resolvePrimaryDatasetSignature(
  selectedDatasetOption: TrainingDatasetOption | undefined
): DatasetStructureSignature | null {
  if (!selectedDatasetOption) return null;
  return extractDatasetStructureSignatureFromOption(selectedDatasetOption);
}
