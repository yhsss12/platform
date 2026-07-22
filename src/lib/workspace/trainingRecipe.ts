import type { TrainingBackendRequest, TrainingCapabilities } from '@/lib/api/trainingClient';
import type { DownstreamModelType } from '@/lib/workspace/trainingCapabilityUi';
import {
  hasRobomimicBackend,
  hasTorchBcBackend,
  hasDiffusionPolicyBackend,
  hasIsaacRobomimicBackend,
  hasActBackend,
  type TrainingTrainability,
} from '@/lib/workspace/trainingCapabilityUi';

export type TrainingRecipeAdvancedFamily = 'robomimic' | 'torch_bc' | 'act' | 'dp';

/** 用户可见的模型/算法展示名（与内部 trainingBackend 解耦） */
export const TRAINING_ALGORITHM_LABELS = {
  robomimicBc: 'Robomimic BC',
  torchBc: 'BC (PyTorch)',
  act: 'ACT',
  diffusionPolicy: 'Diffusion Policy',
  dt: 'Decision Transformer',
} as const;

export interface TrainingRecipeDefinition {
  id: TrainingBackendRequest;
  label: string;
  description: string;
  downstreamModelType: DownstreamModelType;
  trainability: TrainingTrainability;
  advancedFamily: TrainingRecipeAdvancedFamily | null;
}

/**
 * 内部训练后端注册表。
 *
 * `robomimic_bc` 与 `isaac_robomimic_bc` 均为 Robomimic 行为克隆，但 runner 不同：
 * - robomimic_bc → CableThreadingMVP/train_bc.py（MuJoCo 线缆等）
 * - isaac_robomimic_bc → Isaac Lab isaaclab.sh + robomimic/train.py（物块堆叠）
 *
 * 用户界面只展示算法名「Robomimic BC」；具体 runner 由数据集 manifest 自动选择。
 */
export const TRAINING_RECIPE_CATALOG: Record<string, TrainingRecipeDefinition> = {
  robomimic_bc: {
    id: 'robomimic_bc',
    label: TRAINING_ALGORITHM_LABELS.robomimicBc,
    description: '单臂低维行为克隆（MuJoCo / Robomimic HDF5）',
    downstreamModelType: 'Robomimic',
    trainability: 'real',
    advancedFamily: 'robomimic',
  },
  isaac_robomimic_bc: {
    id: 'isaac_robomimic_bc',
    label: TRAINING_ALGORITHM_LABELS.robomimicBc,
    description: '单臂低维行为克隆（Isaac Lab 物块堆叠适配器，内部自动路由）',
    downstreamModelType: 'Robomimic',
    trainability: 'real',
    advancedFamily: null,
  },
  torch_bc: {
    id: 'torch_bc',
    label: TRAINING_ALGORITHM_LABELS.torchBc,
    description: '双臂低维 PyTorch 行为克隆',
    downstreamModelType: 'Robomimic',
    trainability: 'real',
    advancedFamily: 'torch_bc',
  },
  act: {
    id: 'act',
    label: TRAINING_ALGORITHM_LABELS.act,
    description: 'Action Chunking Transformer（图像 + proprio，后端自动适配）',
    downstreamModelType: 'ACT',
    trainability: 'real',
    advancedFamily: 'act',
  },
  diffusion_policy: {
    id: 'diffusion_policy',
    label: TRAINING_ALGORITHM_LABELS.diffusionPolicy,
    description: '扩散策略训练（低维 / 图像观测由后端自动适配）',
    downstreamModelType: 'Diffusion Policy',
    trainability: 'real',
    advancedFamily: 'dp',
  },
  dt: {
    id: 'dt',
    label: TRAINING_ALGORITHM_LABELS.dt,
    description: 'DT 训练（该训练后端暂未开放）',
    downstreamModelType: 'DT',
    trainability: 'placeholder',
    advancedFamily: null,
  },
};

/** 通用模型列表展示顺序（不含 isaac 专用后端；该后端由数据集自动注入） */
const TRAINING_RECIPE_ORDER: TrainingBackendRequest[] = [
  'robomimic_bc',
  'torch_bc',
  'act',
  'diffusion_policy',
  'dt',
];

const LEGACY_FRAMEWORK_LABELS: Record<string, string> = {
  'Isaac Robomimic BC': TRAINING_ALGORITHM_LABELS.robomimicBc,
  isaac_robomimic_bc: TRAINING_ALGORITHM_LABELS.robomimicBc,
  robomimic_bc: TRAINING_ALGORITHM_LABELS.robomimicBc,
  robomimic: TRAINING_ALGORITHM_LABELS.robomimicBc,
  '行为克隆 · Robomimic': TRAINING_ALGORITHM_LABELS.robomimicBc,
  Robomimic: TRAINING_ALGORITHM_LABELS.robomimicBc,
  torch_bc: TRAINING_ALGORITHM_LABELS.torchBc,
  'BC (torch)': TRAINING_ALGORITHM_LABELS.torchBc,
  '行为克隆 · PyTorch': TRAINING_ALGORITHM_LABELS.torchBc,
  diffusion_policy: TRAINING_ALGORITHM_LABELS.diffusionPolicy,
  'Diffusion Policy': TRAINING_ALGORITHM_LABELS.diffusionPolicy,
  act: TRAINING_ALGORITHM_LABELS.act,
  ACT: TRAINING_ALGORITHM_LABELS.act,
  dt: TRAINING_ALGORITHM_LABELS.dt,
  DT: TRAINING_ALGORITHM_LABELS.dt,
  'Decision Transformer': TRAINING_ALGORITHM_LABELS.dt,
};

function platformTrainability(
  recipeId: TrainingBackendRequest,
  capabilities: TrainingCapabilities | null | undefined
): TrainingTrainability {
  if (recipeId === 'robomimic_bc') {
    return hasRobomimicBackend(capabilities) ? 'real' : 'placeholder';
  }
  if (recipeId === 'isaac_robomimic_bc') {
    return hasIsaacRobomimicBackend(capabilities) ? 'real' : 'placeholder';
  }
  if (recipeId === 'torch_bc') {
    return hasTorchBcBackend(capabilities) ? 'real' : 'placeholder';
  }
  if (recipeId === 'diffusion_policy') {
    return hasDiffusionPolicyBackend(capabilities) ? 'real' : 'placeholder';
  }
  if (recipeId === 'act') {
    return hasActBackend(capabilities) ? 'real' : 'placeholder';
  }
  return TRAINING_RECIPE_CATALOG[recipeId]?.trainability ?? 'placeholder';
}

/** 展示平台已实现的模型架构，不按数据集兼容性过滤。 */
export function listCreatableTrainingRecipes(options: {
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingRecipeDefinition[] {
  if (!options.capabilities) {
    return TRAINING_RECIPE_ORDER.map((recipeId) => TRAINING_RECIPE_CATALOG[recipeId]).filter(
      (recipe) => recipe.trainability === 'real'
    );
  }

  const recipes: TrainingRecipeDefinition[] = TRAINING_RECIPE_ORDER.map((recipeId) => {
    const base = TRAINING_RECIPE_CATALOG[recipeId];
    return {
      ...base,
      trainability: platformTrainability(recipeId, options.capabilities),
    };
  }).filter((recipe) => recipe.trainability === 'real');

  if (hasIsaacRobomimicBackend(options.capabilities)) {
    const isaacRecipe = TRAINING_RECIPE_CATALOG.isaac_robomimic_bc;
    if (!recipes.some((item) => item.id === 'isaac_robomimic_bc')) {
      recipes.push({ ...isaacRecipe, trainability: 'real' });
    }
  }

  return recipes;
}

export function defaultTrainingRecipeForContext(options: {
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingBackendRequest {
  const recipes = listCreatableTrainingRecipes(options);
  return recipes[0]?.id ?? 'robomimic_bc';
}

export function getTrainingRecipe(
  recipeId: TrainingBackendRequest | string | null | undefined
): TrainingRecipeDefinition | undefined {
  const key = (recipeId ?? '').trim();
  return key ? TRAINING_RECIPE_CATALOG[key] : undefined;
}

/** 用户所选 recipe 即提交的后端；不再按数据集改写。 */
export function resolveTrainingBackendForDataset(options: {
  selectedRecipeId: TrainingBackendRequest;
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingBackendRequest {
  const recipe = getTrainingRecipe(options.selectedRecipeId);
  if (recipe && platformTrainability(recipe.id, options.capabilities) === 'real') {
    return recipe.id;
  }
  return defaultTrainingRecipeForContext({ capabilities: options.capabilities });
}

export function listAvailableTrainingRecipes(options: {
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingRecipeDefinition[] {
  return listCreatableTrainingRecipes(options);
}

/** @deprecated 保留兼容；与 listCreatableTrainingRecipes 相同，不再按数据集过滤。 */
export function listCreatableTrainingRecipesLegacy(options: {
  isDualArm: boolean;
  isIsaac?: boolean;
  capabilities: TrainingCapabilities | null | undefined;
}): TrainingRecipeDefinition[] {
  return listCreatableTrainingRecipes({ capabilities: options.capabilities });
}

export function formatTrainingRecipeLabel(
  trainingBackend?: string | null,
  downstreamModelType?: string | null
): string {
  const backend = (trainingBackend ?? '').trim();
  if (backend && LEGACY_FRAMEWORK_LABELS[backend]) {
    return LEGACY_FRAMEWORK_LABELS[backend];
  }

  const recipe = getTrainingRecipe(backend);
  if (recipe) return recipe.label;

  const downstream = (downstreamModelType ?? '').trim();
  if (downstream && LEGACY_FRAMEWORK_LABELS[downstream]) {
    return LEGACY_FRAMEWORK_LABELS[downstream];
  }

  if (backend === 'torch_bc') return TRAINING_ALGORITHM_LABELS.torchBc;
  if (backend === 'isaac_robomimic_bc' || backend === 'robomimic_bc') {
    return TRAINING_ALGORITHM_LABELS.robomimicBc;
  }
  if (downstream === 'ACT') return TRAINING_ALGORITHM_LABELS.act;
  if (downstream === 'Diffusion Policy') return TRAINING_ALGORITHM_LABELS.diffusionPolicy;
  if (downstream === 'DT') return TRAINING_ALGORITHM_LABELS.dt;
  if (downstream.toLowerCase().includes('robomimic')) {
    return TRAINING_ALGORITHM_LABELS.robomimicBc;
  }
  return downstream || backend || 'Unknown';
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
    isIsaac?: boolean;
    capabilities: TrainingCapabilities | null | undefined;
  }
): {
  downstreamModelType: DownstreamModelType;
  trainingBackend: TrainingBackendRequest;
  trainability: TrainingTrainability;
} {
  const recipe = getTrainingRecipe(recipeId) ?? TRAINING_RECIPE_CATALOG.robomimic_bc;
  const trainability = context
    ? platformTrainability(recipe.id, context.capabilities)
    : recipe.trainability;
  return {
    downstreamModelType: recipe.downstreamModelType,
    trainingBackend: recipe.id,
    trainability,
  };
}
