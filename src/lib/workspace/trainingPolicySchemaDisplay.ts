export interface PolicySchemaDisplay {
  policyLabel: string;
  observationKeys: string[];
  actionKey: string;
  actionDim: number | null;
  actionDescription: string;
  controllerType: string;
  evalExecutor: string;
  note?: string;
}

function normKeys(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function isJointSpaceSchema(input: {
  evalExecutor?: string | null;
  controllerType?: string | null;
  actionDim?: number | null;
  lowDimKeys?: string[] | null;
}): boolean {
  const executor = String(input.evalExecutor ?? '').toLowerCase();
  const controller = String(input.controllerType ?? '').toUpperCase();
  const keys = new Set(normKeys(input.lowDimKeys));
  return (
    executor === 'joint_position' ||
    controller === 'JOINT_POSITION' ||
    (input.actionDim === 8 && keys.has('robot0_joint_pos') && keys.has('robot0_gripper_qpos'))
  );
}

export function resolvePolicySchemaDisplay(options: {
  trainingBackend?: string | null;
  modelType?: string | null;
  trainConfig?: Record<string, unknown> | null;
  modelAsset?: Record<string, unknown> | null;
}): PolicySchemaDisplay | null {
  const backend = String(options.trainingBackend ?? '').toLowerCase();
  const modelType = String(options.modelType ?? '').toLowerCase();
  const trainConfig = options.trainConfig ?? {};
  const asset = options.modelAsset ?? {};

  const isPi0 = backend === 'pi0' || modelType === 'pi0';
  if (isPi0) {
    const actionDimRaw =
      asset.actionDim ?? trainConfig.actionDim ?? trainConfig.action_dim ?? null;
    const actionDim = actionDimRaw != null ? Number(actionDimRaw) : null;
    const datasetFormat = String(trainConfig.datasetFormat ?? trainConfig.dataFormat ?? 'lerobot');
    const taskInstruction = String(
      trainConfig.taskInstruction ?? trainConfig.task_instruction ?? ''
    ).trim();
    const lowDimKeys = normKeys(
      asset.lowDimKeys ?? trainConfig.lowDimKeys ?? trainConfig.low_dim_keys
    );
    return {
      policyLabel: 'pi0 / openpi',
      observationKeys: [
        ...normKeys(asset.imageKeys ?? trainConfig.imageKeys ?? trainConfig.image_keys),
        ...(lowDimKeys.length > 0 ? lowDimKeys : ['robot0_joint_pos', 'robot0_gripper_qpos']),
      ],
      actionKey: 'normalized joint delta + gripper',
      actionDim,
      actionDescription: 'normalized joint delta + gripper',
      controllerType: String(
        asset.controllerType ?? trainConfig.controllerType ?? trainConfig.controller_type ?? 'JOINT_POSITION'
      ),
      evalExecutor: '不可评测',
      note: [
        `Dataset format: ${datasetFormat.toLowerCase() === 'lerobot' ? 'LeRobot' : datasetFormat}`,
        taskInstruction ? `Task instruction: ${taskInstruction}` : null,
        'Evaluation: pi0 eval adapter not ready',
      ]
        .filter(Boolean)
        .join('\n'),
    };
  }

  const actConfig =
    trainConfig.actConfig && typeof trainConfig.actConfig === 'object'
      ? (trainConfig.actConfig as Record<string, unknown>)
      : {};
  const dpConfig =
    trainConfig.dpConfig && typeof trainConfig.dpConfig === 'object'
      ? (trainConfig.dpConfig as Record<string, unknown>)
      : {};

  const evalExecutor = String(
    asset.evalExecutor ?? trainConfig.evalExecutor ?? actConfig.eval_executor ?? dpConfig.eval_executor ?? ''
  );
  const controllerType = String(
    asset.controllerType ?? trainConfig.controllerType ?? actConfig.controller_type ?? dpConfig.controller_type ?? ''
  );
  const actionDimRaw =
    asset.actionDim ?? trainConfig.actionDim ?? actConfig.action_dim ?? dpConfig.action_dim ?? null;
  const actionDim = actionDimRaw != null ? Number(actionDimRaw) : null;
  const actionKey = String(
    asset.actionKey ?? actConfig.action_key ?? dpConfig.action_key ?? trainConfig.actionKey ?? 'actions'
  );
  const lowDimKeys = normKeys(
    asset.lowDimKeys ?? actConfig.low_dim_keys ?? dpConfig.low_dim_keys ?? trainConfig.lowDimKeys
  );
  const imageKeys = normKeys(
    asset.imageKeys ??
      actConfig.image_keys ??
      dpConfig.image_keys ??
      trainConfig.imageKeys ?? ['agentview_image', 'robot0_eye_in_hand_image']
  );

  const joint = isJointSpaceSchema({ evalExecutor, controllerType, actionDim, lowDimKeys });
  if (!joint) return null;

  const isAct = backend === 'act' || modelType === 'act';
  const policyLabel = isAct ? 'ACT' : 'Diffusion Policy';

  return {
    policyLabel,
    observationKeys: [...imageKeys, ...lowDimKeys.filter((key) => !key.includes('image'))],
    actionKey,
    actionDim,
    actionDescription: 'action[0:7] = normalized joint delta；action[7] = gripper command',
    controllerType: controllerType || 'JOINT_POSITION',
    evalExecutor: evalExecutor || 'joint_position',
    note:
      policyLabel === 'ACT'
        ? 'ACT joint-space 与 DP joint-space 使用相同 obs/action/controller schema，但模型结构不同。'
        : undefined,
  };
}
