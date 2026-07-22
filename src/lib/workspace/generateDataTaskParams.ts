import type { AugmentationAlgorithm, GenerationPath } from '@/lib/workspace/generateDataTypes';
import { NUT_ASSEMBLY_DEFAULTS } from '@/lib/workspace/nutAssembly';
import { resolveNutAssemblyEffectiveSourceDemoDatasetId } from '@/lib/workspace/nutAssemblySeedDatasets';

/** 螺母装配 robosuite_rollout 真实专家策略 key */
export const NUT_ASSEMBLY_EXPERT_POLICY_KEY = 'partial_scripted';

export const NUT_ASSEMBLY_EXPERT_POLICY_OPTIONS = [
  { value: NUT_ASSEMBLY_EXPERT_POLICY_KEY, label: '脚本专家策略' },
] as const;

export interface NutAssemblyPathParamDefaults {
  generationPath: GenerationPath;
  generationCount: number;
  maxSteps: number;
  expertPolicy: string;
  successFilter: boolean;
  keepFailedTrajectories: boolean;
  sourceDemoDatasetId: string;
  augmentationAlgorithm: AugmentationAlgorithm;
  targetCount: number;
  seedGenerationCount: number;
  seedKeepCount: number;
  autoSelectBestSeeds: boolean;
  replayValidation: boolean;
  useExistingSeedDataset: boolean;
}

export const NUT_ASSEMBLY_PATH_DEFAULTS: NutAssemblyPathParamDefaults = {
  generationPath: 'expert_seed_then_augmentation',
  generationCount: NUT_ASSEMBLY_DEFAULTS.episodes,
  maxSteps: NUT_ASSEMBLY_DEFAULTS.horizon,
  expertPolicy: NUT_ASSEMBLY_EXPERT_POLICY_KEY,
  successFilter: true,
  keepFailedTrajectories: false,
  sourceDemoDatasetId: '',
  augmentationAlgorithm: 'mimicgen',
  targetCount: 100,
  seedGenerationCount: 20,
  seedKeepCount: 5,
  autoSelectBestSeeds: true,
  replayValidation: true,
  useExistingSeedDataset: false,
};

export function defaultNutAssemblyPathParams(): NutAssemblyPathParamDefaults {
  return { ...NUT_ASSEMBLY_PATH_DEFAULTS };
}

export interface NutAssemblyGenerateValidationInput {
  datasetName: string;
  generationPath: GenerationPath | null | undefined;
  sourceDemoDatasetId: string;
  seedGenerationCount: number;
  seedKeepCount: number;
  targetCount: number;
  generationCount: number;
  maxSteps: number;
  enablePinnRepair: boolean;
  supportsPinnRepair: boolean;
  useExistingSeedDataset: boolean;
  requiresSourceDemo: boolean;
}

export function validateNutAssemblyGenerateInput(
  input: NutAssemblyGenerateValidationInput
): string | null {
  if (!input.datasetName.trim()) {
    return '请填写数据名称';
  }
  if (!input.generationPath) {
    return '请选择生成路径';
  }
  if (input.maxSteps <= 0) {
    return '最大步数须大于 0';
  }
  if (input.generationPath === 'expert_policy' && input.generationCount <= 0) {
    return '生成条数须大于 0';
  }
  if (input.generationPath === 'demo_augmentation') {
    const effectiveSourceDemoId = resolveNutAssemblyEffectiveSourceDemoDatasetId(
      'demo_augmentation',
      input.sourceDemoDatasetId
    );
    if (!effectiveSourceDemoId) {
      return '请选择源示范数据集。';
    }
    if (input.targetCount <= 0) {
      return '目标生成条数须大于 0';
    }
  }
  if (input.generationPath === 'expert_seed_then_augmentation') {
    if (input.seedGenerationCount <= 0 || input.seedKeepCount <= 0 || input.targetCount <= 0) {
      return '种子与扩增条数须大于 0';
    }
    if (input.seedKeepCount > input.seedGenerationCount) {
      return '种子保留条数不能大于种子生成条数';
    }
    if (input.useExistingSeedDataset && !input.sourceDemoDatasetId.trim()) {
      return '请选择源示范数据集。';
    }
  }
  if (input.requiresSourceDemo && !input.sourceDemoDatasetId.trim()) {
    return '请选择源示范数据集。';
  }
  if (input.enablePinnRepair && !input.supportsPinnRepair) {
    return '当前任务模板不支持物理增强模型';
  }
  return null;
}
