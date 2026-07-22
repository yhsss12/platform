import { describe, expect, it } from 'vitest';
import type { ModelAsset } from '@/types/benchmark';
import {
  dpInitWeightsCompatible,
  extractDpInitSchemaFromAsset,
  modelAssetDpInitCompatible,
  resolveDpInitTargetFromDatasetManifest,
} from '@/lib/workspace/dpInitWeightCompat';
import { formatInitWeightOptionLines } from '@/lib/workspace/trainingPretrained';
import type { DatasetStructureSignature } from '@/lib/workspace/trainingDatasetCompat';

const JOINT_TARGET = resolveDpInitTargetFromDatasetManifest(
  {
    joint_action_available: true,
    availableActionKeys: ['joint_actions', 'actions'],
    observationKeys: ['robot0_joint_pos', 'robot0_gripper_qpos', 'agentview_image'],
    actionSchema: 'joint_state_obs_joint_action',
  },
  {
    taskType: 'cable_threading',
    robotType: 'Panda',
    simulatorBackend: 'mujoco',
    imageKeys: ['agentview_image', 'robot0_eye_in_hand_image'],
    lowDimKeys: ['robot0_joint_pos', 'robot0_gripper_qpos'],
    actionDim: 7,
    imageSize: 84,
  } satisfies DatasetStructureSignature
)!;

function asset(partial: Partial<ModelAsset>): ModelAsset {
  return {
    id: 'model_test',
    name: 'test',
    sourceTrainingJobId: 'train_test',
    sourceDatasetId: null,
    taskTemplateId: null,
    modelType: 'diffusion_policy',
    framework: 'Diffusion Policy',
    checkpointPath: '/tmp/model_final.pt',
    manifestPath: '',
    version: 'v1',
    status: 'available',
    createdAt: '2026-06-25T09:00:00Z',
    updatedAt: '2026-06-25T09:00:00Z',
    ...partial,
  };
}

describe('dpInitWeightCompat', () => {
  it('resolves joint-space target from dataset manifest', () => {
    expect(JOINT_TARGET.actionKey).toBe('joint_actions');
    expect(JOINT_TARGET.evalExecutor).toBe('joint_position');
  });

  it('accepts joint_actions checkpoint for joint-space training', () => {
    const source = extractDpInitSchemaFromAsset(
      asset({
        actionKey: 'joint_actions',
        evalExecutor: 'joint_position',
        controllerType: 'JOINT_POSITION',
        actionDim: 8,
      })
    );
    expect(modelAssetDpInitCompatible(asset({ actionKey: 'joint_actions', actionDim: 8 }), JOINT_TARGET).ok).toBe(true);
    expect(dpInitWeightsCompatible(source, JOINT_TARGET).ok).toBe(true);
  });

  it('accepts legacy joint-space checkpoint with actions key for joint_actions training', () => {
    const legacy = asset({
      actionKey: 'actions',
      controllerType: 'JOINT_POSITION',
      actionDim: 8,
      trainedActionMode: 'joint_delta_derived',
      dpInitSchema: {
        family: 'joint',
        actionKey: 'actions',
        evalExecutor: 'joint_position',
        controllerType: 'JOINT_POSITION',
        actionDim: 8,
        trainedActionMode: 'joint_delta_derived',
        lowDimKeys: ['robot0_gripper_qpos', 'robot0_joint_pos'],
        imageKeys: ['agentview_image', 'robot0_eye_in_hand_image'],
      },
    });
    expect(modelAssetDpInitCompatible(legacy, JOINT_TARGET).ok).toBe(true);
  });

  it('rejects EEF checkpoint for joint-space training', () => {
    const eef = asset({
      actionKey: 'actions',
      evalExecutor: 'osc_pose',
      controllerType: 'OSC_POSE',
      actionDim: 7,
      dpInitSchema: {
        family: 'eef',
        actionKey: 'actions',
        evalExecutor: 'osc_pose',
        controllerType: 'OSC_POSE',
        actionDim: 7,
      },
    });
    const result = modelAssetDpInitCompatible(eef, JOINT_TARGET);
    expect(result.ok).toBe(false);
  });

  it('does not duplicate Final in init weight option label', () => {
    const lines = formatInitWeightOptionLines(
      asset({
        displayName: '线缆穿杆数据_20260625_0929 · Final',
        checkpointKind: 'final',
      })
    );
    expect(lines.titleLine).toBe('线缆穿杆数据_20260625_0929 · Final');
    expect(lines.titleLine).not.toContain('Final · Final');
  });
});
