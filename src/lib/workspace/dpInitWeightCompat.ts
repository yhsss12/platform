import type { ModelAsset } from '@/types/benchmark';
import type { DatasetStructureSignature } from '@/lib/workspace/trainingDatasetCompat';

export const DP_INIT_INCOMPATIBLE_ACTION_SPACE_HINT = '不兼容：动作空间不同';

export interface DpInitSchema {
  family: 'joint' | 'eef' | '';
  actionKey: string;
  gripperActionKey?: string | null;
  actionDim: number | null;
  trainedActionMode?: string | null;
  evalExecutor: string;
  controllerType: string;
  imageKeys: string[];
  lowDimKeys: string[];
  lowDimDim?: number | null;
  imageSize?: number | null;
}

function normKeys(keys: unknown): string[] {
  if (!Array.isArray(keys)) return [];
  return [...new Set(keys.map((key) => String(key).trim()).filter(Boolean))].sort();
}

function normToken(value: unknown): string {
  return String(value ?? '').trim().toLowerCase();
}

function normController(value: unknown): string {
  return String(value ?? '').trim().toUpperCase();
}

const JOINT_ACTION_KEYS = new Set(['joint_actions']);
const JOINT_ACTION_MODES = new Set(['joint_delta', 'joint_delta_derived']);
const JOINT_LOW_DIM_KEYS = new Set(['robot0_joint_pos', 'robot0_joint_pos_rel']);

function inferFamily(schema: Partial<DpInitSchema>): DpInitSchema['family'] {
  const actionKey = schema.actionKey ?? '';
  const evalExecutor = normToken(schema.evalExecutor);
  const controllerType = normController(schema.controllerType);
  const trainedMode = normToken(schema.trainedActionMode);
  const lowDimKeys = normKeys(schema.lowDimKeys);

  if (
    evalExecutor === 'joint_position' ||
    JOINT_ACTION_KEYS.has(actionKey) ||
    JOINT_ACTION_MODES.has(trainedMode) ||
    controllerType === 'JOINT_POSITION' ||
    lowDimKeys.some((key) => JOINT_LOW_DIM_KEYS.has(key))
  ) {
    return 'joint';
  }
  if (actionKey === 'actions' || evalExecutor === 'osc_pose' || controllerType === 'OSC_POSE') {
    return 'eef';
  }
  return '';
}

export function extractDpInitSchemaFromAsset(asset: ModelAsset): DpInitSchema {
  const dpInitSchema = (asset as ModelAsset & { dpInitSchema?: Partial<DpInitSchema> }).dpInitSchema;
  if (dpInitSchema?.actionKey) {
    return {
      family: (dpInitSchema.family as DpInitSchema['family']) || inferFamily(dpInitSchema),
      actionKey: String(dpInitSchema.actionKey),
      gripperActionKey: dpInitSchema.gripperActionKey ?? null,
      actionDim: dpInitSchema.actionDim != null ? Number(dpInitSchema.actionDim) : null,
      trainedActionMode: dpInitSchema.trainedActionMode ?? null,
      evalExecutor: String(dpInitSchema.evalExecutor ?? ''),
      controllerType: String(dpInitSchema.controllerType ?? ''),
      imageKeys: normKeys(dpInitSchema.imageKeys),
      lowDimKeys: normKeys(dpInitSchema.lowDimKeys),
      lowDimDim: dpInitSchema.lowDimDim != null ? Number(dpInitSchema.lowDimDim) : null,
      imageSize: dpInitSchema.imageSize != null ? Number(dpInitSchema.imageSize) : null,
    };
  }

  const structure = (asset.structureConfig ?? {}) as Record<string, unknown>;
  const input = (structure.input ?? {}) as Record<string, unknown>;
  const output = (structure.output ?? {}) as Record<string, unknown>;
  const partial: Partial<DpInitSchema> = {
    actionKey: String(asset.actionKey ?? output.action_key ?? ''),
    gripperActionKey: String(asset.gripperActionKey ?? output.gripper_action_key ?? '') || null,
    actionDim:
      asset.actionDim != null
        ? Number(asset.actionDim)
        : output.action_dim != null
          ? Number(output.action_dim)
          : null,
    trainedActionMode: asset.trainedActionMode ?? asset.actionMode ?? null,
    evalExecutor: String(asset.evalExecutor ?? ''),
    controllerType: String(asset.controllerType ?? ''),
    imageKeys: normKeys(input.image_keys ?? input.imageKeys),
    lowDimKeys: normKeys(input.low_dim_keys ?? input.lowDimKeys),
    imageSize: input.image_size != null ? Number(input.image_size) : null,
  };
  return {
    family: inferFamily(partial),
    actionKey: partial.actionKey ?? '',
    gripperActionKey: partial.gripperActionKey ?? null,
    actionDim: partial.actionDim ?? null,
    trainedActionMode: partial.trainedActionMode ?? null,
    evalExecutor: partial.evalExecutor ?? '',
    controllerType: partial.controllerType ?? '',
    imageKeys: partial.imageKeys ?? [],
    lowDimKeys: partial.lowDimKeys ?? [],
    lowDimDim: null,
    imageSize: partial.imageSize ?? null,
  };
}

function isJointLegacyActionsAlias(source: DpInitSchema, target: DpInitSchema): boolean {
  if (source.family !== 'joint' || target.family !== 'joint') return false;
  if (source.actionKey !== 'actions' || target.actionKey !== 'joint_actions') return false;
  const trainedMode = normToken(source.trainedActionMode);
  const controllerType = normController(source.controllerType);
  return trainedMode === 'joint_delta' || trainedMode === 'joint_delta_derived' || controllerType === 'JOINT_POSITION';
}

export function resolveDpInitTargetFromDatasetManifest(
  manifest: Record<string, unknown> | null | undefined,
  datasetSignature: DatasetStructureSignature | null | undefined
): DpInitSchema | null {
  if (!datasetSignature) return null;

  const availableActionKeys = normKeys(
    manifest?.availableActionKeys ?? manifest?.available_action_keys
  );
  const observationKeys = normKeys(
    manifest?.observationKeys ?? manifest?.obsKeys ?? manifest?.observation_keys
  );
  const lowDimKeys = normKeys(
    observationKeys.length
      ? observationKeys.filter((key) => !String(key).includes('image'))
      : datasetSignature.lowDimKeys.length
        ? datasetSignature.lowDimKeys
        : []
  );
  const imageKeys = normKeys(datasetSignature.imageKeys);
  const actionSchema = String(manifest?.actionSchema ?? '').toLowerCase();
  const observationSchema = String(manifest?.observationSchema ?? '').toLowerCase();

  const jointAvailable = Boolean(
    manifest?.joint_action_available ||
      availableActionKeys.includes('joint_actions') ||
      actionSchema.includes('joint') ||
      observationSchema.includes('joint') ||
      lowDimKeys.some((key) => JOINT_LOW_DIM_KEYS.has(key))
  );

  if (jointAvailable) {
    return {
      family: 'joint',
      actionKey: 'joint_actions',
      gripperActionKey: 'gripper_actions',
      actionDim: 8,
      trainedActionMode: 'joint_delta',
      evalExecutor: 'joint_position',
      controllerType: 'JOINT_POSITION',
      imageKeys,
      lowDimKeys: lowDimKeys.length
        ? lowDimKeys.filter((key) => !key.includes('image'))
        : ['robot0_joint_pos', 'robot0_gripper_qpos'],
      lowDimDim: 9,
      imageSize: datasetSignature.imageSize,
    };
  }

  return {
    family: 'eef',
    actionKey: 'actions',
    gripperActionKey: null,
    actionDim: datasetSignature.actionDim ?? 7,
    trainedActionMode: 'osc_pose_delta_eef',
    evalExecutor: 'osc_pose',
    controllerType: 'OSC_POSE',
    imageKeys,
    lowDimKeys: lowDimKeys.length
      ? lowDimKeys.filter((key) => !key.includes('image'))
      : ['robot0_eef_pos', 'robot0_gripper_qpos'],
    lowDimDim: 9,
    imageSize: datasetSignature.imageSize,
  };
}

export function dpInitWeightsCompatible(
  source: DpInitSchema,
  target: DpInitSchema
): { ok: boolean; reason?: string } {
  const jointLegacyAlias = isJointLegacyActionsAlias(source, target);

  if (source.family && target.family && source.family !== target.family) {
    if (source.family === 'eef' && target.family === 'joint') {
      return {
        ok: false,
        reason:
          '该 checkpoint 为 EEF/OSC Diffusion Policy，当前训练任务为 Joint-Space Diffusion Policy，action schema 不一致，不能作为初始化权重。',
      };
    }
    return {
      ok: false,
      reason:
        '该 checkpoint 为 Joint-Space Diffusion Policy，当前训练任务为 EEF/OSC Diffusion Policy，action schema 不一致，不能作为初始化权重。',
    };
  }

  if (target.actionKey && source.actionKey && source.actionKey !== target.actionKey && !jointLegacyAlias) {
    if (target.actionKey === 'joint_actions' && source.actionKey === 'actions' && source.family === 'eef') {
      return {
        ok: false,
        reason:
          '该 checkpoint 为 EEF/OSC Diffusion Policy，当前训练任务为 Joint-Space Diffusion Policy，action schema 不一致，不能作为初始化权重。',
      };
    }
    return {
      ok: false,
      reason: `${DP_INIT_INCOMPATIBLE_ACTION_SPACE_HINT}（${source.actionKey} ≠ ${target.actionKey}）`,
    };
  }

  const sourceExecutor = normToken(source.evalExecutor);
  const targetExecutor = normToken(target.evalExecutor);
  if (sourceExecutor && targetExecutor && sourceExecutor !== targetExecutor) {
    return {
      ok: false,
      reason: `模型 eval_executor=${sourceExecutor} 与当前任务 ${targetExecutor} 不一致，不能作为初始化权重。`,
    };
  }

  const sourceController = normController(source.controllerType);
  const targetController = normController(target.controllerType);
  if (sourceController && targetController && sourceController !== targetController) {
    return {
      ok: false,
      reason: `模型 controller_type=${sourceController} 与当前任务 ${targetController} 不一致，不能作为初始化权重。`,
    };
  }

  if (
    source.actionDim != null &&
    target.actionDim != null &&
    Number(source.actionDim) !== Number(target.actionDim)
  ) {
    return { ok: false, reason: '模型与当前任务结构不匹配，无法作为初始化权重。（action_dim 不一致）' };
  }

  if (source.imageKeys.length && target.imageKeys.length && source.imageKeys.join('|') !== target.imageKeys.join('|')) {
    return { ok: false, reason: '模型与当前任务结构不匹配，无法作为初始化权重。（imageKeys 不一致）' };
  }

  if (source.lowDimKeys.length && target.lowDimKeys.length && source.lowDimKeys.join('|') !== target.lowDimKeys.join('|')) {
    return { ok: false, reason: '模型与当前任务结构不匹配，无法作为初始化权重。（lowDimKeys 不一致）' };
  }

  return { ok: true };
}

export function modelAssetDpInitCompatible(
  asset: ModelAsset,
  target: DpInitSchema | null
): { ok: boolean; reason?: string } {
  if (!target) return { ok: true };
  return dpInitWeightsCompatible(extractDpInitSchemaFromAsset(asset), target);
}
