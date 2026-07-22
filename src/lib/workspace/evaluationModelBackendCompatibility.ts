import type { ModelAsset } from '@/types/benchmark';
import type { WorkspaceEvaluationMode } from '@/lib/api/taskTemplatesClient';
import { isFrankStackCubeEvalTask } from '@/lib/workspace/isaacStackCubeProduct';
import { FRANKA_STACK_CUBE_PRODUCT_NAME } from '@/lib/workspace/isaacStackCubeProduct';

/** 评测任务 → 允许的 modelAsset backendType（小写 canonical key） */
export const EVALUATION_MODEL_BACKEND_COMPATIBILITY: Record<string, readonly string[]> = {
  block_stacking: ['isaac_robomimic_bc'],
  cable_threading: [
    'expert_policy',
    'robomimic_bc',
    'robomimic',
    'bc',
    'act',
    'pi0',
    'diffusion_policy',
  ],
  dual_arm_cable_manipulation: ['torch_bc', 'act', 'diffusion_policy', 'bc'],
  nut_assembly: ['robomimic_bc', 'robomimic', 'bc'],
};

export interface EvaluationModelCompatibilityContext {
  taskTemplateId?: string | null;
  taskType?: string | null;
  evaluationMode?: WorkspaceEvaluationMode | null;
}

const BACKEND_ALIASES: Record<string, string> = {
  act: 'act',
  bc: 'bc',
  pi0: 'pi0',
  robomimic: 'robomimic_bc',
  robomimic_bc: 'robomimic_bc',
  'robomimic bc': 'robomimic_bc',
  isaac_robomimic_bc: 'isaac_robomimic_bc',
  'isaac robomimic bc': 'isaac_robomimic_bc',
  torch_bc: 'torch_bc',
  diffusion_policy: 'diffusion_policy',
  'diffusion policy': 'diffusion_policy',
  diffusion: 'diffusion_policy',
  expert_policy: 'expert_policy',
};

export function normalizeModelAssetBackendType(value: string | null | undefined): string {
  const raw = String(value ?? '').trim();
  if (!raw) return '';
  const lowered = raw.toLowerCase().replace(/\s+/g, ' ');
  const underscored = lowered.replace(/\s+/g, '_');
  return BACKEND_ALIASES[lowered] ?? BACKEND_ALIASES[underscored] ?? underscored;
}

export function resolveModelAssetBackendType(asset: Pick<
  ModelAsset,
  'backendType' | 'trainingBackend' | 'framework' | 'modelType' | 'baseAlgorithm'
>): string {
  for (const candidate of [
    asset.backendType,
    asset.trainingBackend,
    asset.framework,
    asset.modelType,
    asset.baseAlgorithm,
  ]) {
    const normalized = normalizeModelAssetBackendType(candidate);
    if (normalized) return normalized;
  }
  return '';
}

export function resolveEvaluationCompatibilityTaskKey(
  context: Pick<EvaluationModelCompatibilityContext, 'taskTemplateId' | 'taskType'>
): string | null {
  const templateId = String(context.taskTemplateId ?? '').trim();
  const taskType = String(context.taskType ?? '').trim();

  if (
    isFrankStackCubeEvalTask(templateId) ||
    templateId === 'isaac_block_stacking' ||
    taskType === 'block_stacking' ||
    taskType === 'isaac_block_stacking' ||
    taskType === 'isaaclab_franka_stack_cube'
  ) {
    return 'block_stacking';
  }
  if (templateId === 'cable_threading_single_arm' || taskType === 'cable_threading') {
    return 'cable_threading';
  }
  if (templateId === 'dual_arm_cable_manipulation' || taskType === 'dual_arm_cable_manipulation') {
    return 'dual_arm_cable_manipulation';
  }
  if (templateId === 'nut_assembly_single_arm' || taskType === 'nut_assembly') {
    return 'nut_assembly';
  }
  return null;
}

export function getAllowedEvaluationBackendTypes(
  context: EvaluationModelCompatibilityContext
): readonly string[] {
  const key = resolveEvaluationCompatibilityTaskKey(context);
  if (!key) return [];
  return EVALUATION_MODEL_BACKEND_COMPATIBILITY[key] ?? [];
}

export function isModelAssetCompatibleWithEvaluationTask(
  asset: ModelAsset,
  context: EvaluationModelCompatibilityContext
): boolean {
  if (context.evaluationMode !== 'trained_model_evaluation') {
    return true;
  }

  const taskKey = resolveEvaluationCompatibilityTaskKey(context);
  if (!taskKey) return true;

  const allowed = EVALUATION_MODEL_BACKEND_COMPATIBILITY[taskKey] ?? [];
  if (allowed.length === 0) return true;

  const backendType = resolveModelAssetBackendType(asset);
  if (!backendType) return false;

  return allowed.includes(backendType);
}

export function getModelAssetIncompatibilityMessage(
  asset: ModelAsset,
  context: EvaluationModelCompatibilityContext
): string {
  const taskKey = resolveEvaluationCompatibilityTaskKey(context);
  const actual = resolveModelAssetBackendType(asset) || String(asset.framework ?? '未知');

  if (taskKey === 'block_stacking') {
    return `${FRANKA_STACK_CUBE_PRODUCT_NAME}评测当前仅支持 Isaac Robomimic BC 模型，当前模型类型为 ${actual.toUpperCase()}，无法评测。`;
  }

  return '当前模型资产与所选评测任务不兼容，请选择匹配的模型资产。';
}

export function getNoCompatibleModelAssetsHint(context: EvaluationModelCompatibilityContext): string {
  const taskKey = resolveEvaluationCompatibilityTaskKey(context);
  if (taskKey === 'block_stacking') {
    return `${FRANKA_STACK_CUBE_PRODUCT_NAME}评测当前仅支持 Isaac Robomimic BC 模型，请先训练或导入 backendType=isaac_robomimic_bc 的模型资产。`;
  }
  if (taskKey === 'dual_arm_cable_manipulation') {
    return '暂无可兼容的已训练模型，请先完成 torch_bc / ACT 训练。';
  }
  if (taskKey === 'nut_assembly') {
    return '暂无可兼容的螺母装配模型，请先完成 Robomimic BC 训练。';
  }
  return '暂无兼容的已训练模型，请先完成训练或选择其他评测任务。';
}

export function parseApiErrorMessage(detail: unknown, fallback = '请求失败'): string {
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (detail && typeof detail === 'object') {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === 'string' && message.trim()) return message;
  }
  return fallback;
}
