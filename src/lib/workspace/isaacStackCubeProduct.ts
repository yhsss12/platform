import type { TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import type { Dataset } from '@/types/benchmark';
import { ISAAC_BLOCK_STACKING_TEMPLATE_ID } from '@/lib/workspace/isaacBlockStacking';
import {
  ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID,
  isIsaacLabFrankaStackCubeDataset,
  isIsaacLabFrankaStackCubeTask,
} from '@/lib/workspace/isaaclabFrankaStackCube';

/** 产品层统一任务名（用户可见） */
export const FRANKA_STACK_CUBE_PRODUCT_NAME = '物块堆叠';

/** 中文说明 / 副标题 */
export const FRANKA_STACK_CUBE_PRODUCT_SUBTITLE = '基于 Isaac Lab 的 Franka 方块堆叠任务';

/** 带中文说明的展示名 */
export const FRANKA_STACK_CUBE_PRODUCT_DISPLAY_WITH_SUBTITLE = `${FRANKA_STACK_CUBE_PRODUCT_NAME}（基于 Isaac Lab 的 Franka 方块堆叠任务）`;

/** 数据生成主入口 templateId */
export const FRANKA_STACK_CUBE_DATA_GENERATION_TEMPLATE_ID = ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID;

/** 评测 / 训练适配入口 templateId */
export const FRANKA_STACK_CUBE_EVALUATION_TEMPLATE_ID = ISAAC_BLOCK_STACKING_TEMPLATE_ID;

export const FRANKA_STACK_CUBE_PRODUCT_DESCRIPTION =
  '基于 Isaac Lab 的 Franka 方块堆叠任务，支持 Mimic 专家数据生成、Robomimic BC 训练与 Isaac Lab rollout 评测。';

export const FRANKA_STACK_CUBE_EVAL_CAPABILITY_LABEL =
  '已接入 Isaac Lab rollout 评测，success 来自环境 success_term';

export const FRANKA_STACK_CUBE_DATA_CAPABILITY_LABEL =
  '已接入真实专家策略 / Mimic 数据生成';

export const FRANKA_STACK_CUBE_INTERNAL_TEMPLATE_IDS = new Set<string>([
  ISAAC_BLOCK_STACKING_TEMPLATE_ID,
  'task_isaac_block_stacking_v1',
  'block_stacking',
]);

const FRANKA_STACK_CUBE_LEGACY_LABELS = new Set([
  FRANKA_STACK_CUBE_PRODUCT_NAME,
  FRANKA_STACK_CUBE_PRODUCT_DISPLAY_WITH_SUBTITLE,
  'Franka Stack Cube',
  'Isaac Lab Franka Stack Cube',
  'Franka 物块堆叠',
  'Franka 方块堆叠',
  'Stack Cube',
  'stack cube',
  'Block Stacking',
  'block stacking',
  '物块堆叠',
  '物块堆叠任务',
  ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID,
  ISAAC_BLOCK_STACKING_TEMPLATE_ID,
  'stacking',
]);

export function isFrankStackCubeProductTask(templateOrId: string | null | undefined): boolean {
  if (!templateOrId) return false;
  const value = templateOrId.trim();
  if (FRANKA_STACK_CUBE_INTERNAL_TEMPLATE_IDS.has(value)) return true;
  if (isIsaacLabFrankaStackCubeTask(value)) return true;
  return FRANKA_STACK_CUBE_LEGACY_LABELS.has(value);
}

export function isFrankStackCubeInternalTemplateId(templateId: string | null | undefined): boolean {
  if (!templateId) return false;
  return FRANKA_STACK_CUBE_INTERNAL_TEMPLATE_IDS.has(templateId.trim());
}

export function resolveFrankStackCubeProductDisplayName(): string {
  return FRANKA_STACK_CUBE_PRODUCT_NAME;
}

export function formatFrankStackCubeProductLabel(options?: { withSubtitle?: boolean }): string {
  if (options?.withSubtitle) {
    return FRANKA_STACK_CUBE_PRODUCT_DISPLAY_WITH_SUBTITLE;
  }
  return FRANKA_STACK_CUBE_PRODUCT_NAME;
}

/** 创建评测时：产品 templateId → 实际评测 adapter templateId */
export function resolveFrankStackCubeEvaluationTemplateId(
  templateId: string | null | undefined
): string {
  if (!templateId?.trim()) return FRANKA_STACK_CUBE_EVALUATION_TEMPLATE_ID;
  const value = templateId.trim();
  if (isFrankStackCubeProductTask(value)) {
    return FRANKA_STACK_CUBE_EVALUATION_TEMPLATE_ID;
  }
  return value;
}

export function isFrankStackCubeEvalTask(templateId: string | null | undefined): boolean {
  return isFrankStackCubeProductTask(templateId);
}

export function shouldShowFrankStackCubeInEvalDropdown(template: TaskTemplateDto): boolean {
  if (isFrankStackCubeInternalTemplateId(template.id)) return false;
  if (template.id === FRANKA_STACK_CUBE_DATA_GENERATION_TEMPLATE_ID) {
    return (template.supportedEvaluationModes?.length ?? 0) > 0;
  }
  return (template.supportedEvaluationModes?.length ?? 0) > 0;
}

export function shouldShowFrankStackCubeInDataGenOptions(templateLabel: string): boolean {
  if (isFrankStackCubeInternalTemplateId(templateLabel)) return false;
  if (isFrankStackCubeProductTask(templateLabel)) {
    return templateLabel === FRANKA_STACK_CUBE_PRODUCT_NAME || isIsaacLabFrankaStackCubeTask(templateLabel);
  }
  return false;
}

export function isFrankStackCubeDataset(
  dataset: Pick<Dataset, 'taskType' | 'taskTemplateId' | 'sourceJobId' | 'simulatorBackend' | 'replayBackend' | 'sourceType'>
): boolean {
  if (isIsaacLabFrankaStackCubeDataset(dataset)) return true;
  const jobId = dataset.sourceJobId?.trim() ?? '';
  if (
    dataset.sourceType === 'imported_demo' ||
    jobId.startsWith('isaac_import_') ||
    jobId.startsWith('isaac_gen_')
  ) {
    return true;
  }
  if (
    jobId.startsWith('data_gen_') &&
    (isIsaacLabFrankaStackCubeTask(dataset.taskType) ||
      isIsaacLabFrankaStackCubeTask(dataset.taskTemplateId))
  ) {
    return true;
  }
  return false;
}

export function resolveFrankStackCubeEvaluationUiBindingTemplateId(
  templateId: string | null | undefined
): string {
  return resolveFrankStackCubeEvaluationTemplateId(templateId);
}
