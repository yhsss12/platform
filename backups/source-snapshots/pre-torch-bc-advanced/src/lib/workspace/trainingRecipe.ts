// Historical snapshot retained from before the advanced torch-BC refactor.
import type { TrainingBackendRequest, TrainingCapabilities } from '@/lib/api/trainingClient';
import type { DownstreamModelType } from '@/lib/workspace/trainingCapabilityUi';
import {
  hasRobomimicBackend,
  hasTorchBcBackend,
  type TrainingTrainability,
} from '@/lib/workspace/trainingCapabilityUi';

export type TrainingRecipeAdvancedFamily = 'robomimic' | 'act' | 'dp';

export interface TrainingRecipeDefinition {
  id: TrainingBackendRequest;
  label: string;
  description: string;
  downstreamModelType: DownstreamModelType;
  trainability: TrainingTrainability;
  advancedFamily: TrainingRecipeAdvancedFamily | null;
}

export const TRAINING_RECIPE_CATALOG: Record<string, TrainingRecipeDefinition> = {
  robomimic_bc: {
    id: 'robomimic_bc',
    label: '行为克隆 · Robomimic',
    description: '单臂低维观测，Robomimic BC 训练与评测',
    downstreamModelType: 'Robomimic',
    trainability: 'real',
    advancedFamily: 'robomimic',
  },
  torch_bc: {
    id: 'torch_bc',
    label: '行为克隆 · PyTorch',
    description: '双臂低维观测，平台 torch_bc 训练与 rollout 评测',
    downstreamModelType: 'Robomimic',
    trainability: 'real',
    advancedFamily: null,
  },
  act: {
    id: 'act',
    label: 'ACT',
    description: '视觉模仿学习（训练后端待接入）',
    downstreamModelType: 'ACT',
    trainability: 'placeholder',
    advancedFamily: 'act',
  },
  diffusion_policy: {
    id: 'diffusion_policy',
    label: 'Diffusion Policy',
    description: '扩散策略训练（训练后端待接入）',
    downstreamModelType: 'Diffusion Policy',
    trainability: 'placeholder',
    advancedFamily: 'dp',
  },
  dt: {
    id: 'dt',
    label: 'Decision Transformer',
    description: 'DT 训练（训练后端待接入）',
    downstreamModelType: 'DT',
    trainability: 'placeholder',
    advancedFamily: null,
  },
};

/** 训练方案下拉固定展示顺序（含待接入项，便于后续扩展） */
const TRAINING_RECIPE_ORDER: TrainingBackendRequest[] = [
  'robomimic_bc',
  'torch_bc',
  'act',
  'diffusion_policy',
  'dt',
];

function resolveRecipeTrainability(
  recipeId: TrainingBackendRequest,
  isDualArm: boolean,
  capabilities: TrainingCapabilities | null | undefined
): TrainingTrainability {
  if (recipeId === 'robomimic_bc') {
    return !isDualArm && hasRobomimicBackend(capabilities) ? 'real' : 'placeholder';
  }
  if (recipeId === 'torch_bc') {
    return isDualArm && hasTorchBcBackend(capabilities) ? 'real' : 'placeholder';
  }
  return 'placeholder';
}

export function getTrainingRecipe(
  recipeId: TrainingBackendRequest | string | null | undefined
): TrainingRecipeDefinition | undefined {
  const key = (recipeId ?? '').trim();
  return key ? TRAINING_RECIPE_CATALOG[key] : undefined;
}

export function listAvailableTrainingRecipes(options: {
  isDualArm: boolean;
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingRecipeDefinition[] {
  const { isDualArm, capabilities } = options;

  return TRAINING_RECIPE_ORDER.map((recipeId) => {
    const base = TRAINING_RECIPE_CATALOG[recipeId];
    return {
      ...base,
      trainability: resolveRecipeTrainability(recipeId, isDualArm, capabilities),
    };
  });
}

export function defaultTrainingRecipeForContext(options: {
  isDualArm: boolean;
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingBackendRequest {
  const recipes = listAvailableTrainingRecipes(options);
  const preferred = recipes.find((recipe) => recipe.trainability === 'real');
  return (preferred ?? recipes[0])?.id ?? 'robomimic_bc';
}

export function formatTrainingRecipeLabel(
  trainingBackend?: string | null,
  downstreamModelType?: string | null
): string {
  const recipe = getTrainingRecipe(trainingBackend ?? undefined);
  if (recipe) return recipe.label;

  const downstream = (downstreamModelType ?? '').trim();
  if (trainingBackend === 'torch_bc') return TRAINING_RECIPE_CATALOG.torch_bc.label;
  if (downstream === 'ACT') return 'ACT';
  if (downstream === 'Diffusion Policy') return 'Diffusion Policy';
  if (downstream === 'DT') return 'Decision Transformer';
  if (downstream.toLowerCase().includes('robomimic') || trainingBackend === 'robomimic_bc') {
    return TRAINING_RECIPE_CATALOG.robomimic_bc.label;
  }
  return downstream || trainingBackend || '未知方案';
}

export function normalizeTrainingRecipeFilterValue(row: {
  trainingBackend?: string | null;
  modelType?: string | null;
}): string {
  return formatTrainingRecipeLabel(row.trainingBackend, row.modelType);
}

export function recipeToSubmitFields(
  recipeId: TrainingBackendRequest,
  context?: {
    isDualArm: boolean;
    capabilities: TrainingCapabilities | null | undefined;
  }
): {
  downstreamModelType: DownstreamModelType;
  trainingBackend: TrainingBackendRequest;
  trainability: TrainingTrainability;
} {
  const recipe = getTrainingRecipe(recipeId) ?? TRAINING_RECIPE_CATALOG.robomimic_bc;
  const trainability = context
    ? resolveRecipeTrainability(recipe.id, context.isDualArm, context.capabilities)
    : recipe.trainability;
  return {
    downstreamModelType: recipe.downstreamModelType,
    trainingBackend: recipe.id,
    trainability,
  };
}
