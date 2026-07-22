import { describe, expect, it } from 'vitest';

import {
  canEvaluateModelAsset,
  getModelAssetEvalDisabledReason,
  isTrainingJobInProgressForAssets,
} from '@/lib/workspace/modelAssetRules';

describe('modelAssetRules training job detail', () => {
  it('blocks evaluation for all assets while job is in progress', () => {
    const epochAsset = {
      status: 'ready',
      checkpointPath: '/tmp/model_epoch_10.pth',
      checkpointKind: 'epoch',
      canEvaluate: true,
      displayStatus: 'ready',
    };
    expect(
      canEvaluateModelAsset(epochAsset, { jobInProgress: isTrainingJobInProgressForAssets('running') })
    ).toBe(false);
  });

  it('allows evaluation for ready current-job asset when job completed', () => {
    const finalAsset = {
      status: 'ready',
      checkpointPath: '/tmp/model_final.pt',
      checkpointKind: 'final',
      canEvaluate: true,
      displayStatus: 'ready',
      modelType: 'diffusion_policy',
    };
    expect(
      canEvaluateModelAsset(finalAsset, {
        jobInProgress: isTrainingJobInProgressForAssets('completed'),
      })
    ).toBe(true);
  });

  it('allows evaluation for ready ACT joint-space asset when job completed', () => {
    const finalAsset = {
      status: 'ready',
      checkpointPath: '/tmp/model_final.pt',
      checkpointKind: 'final',
      canEvaluate: true,
      displayStatus: 'ready',
      modelType: 'act',
      evalExecutor: 'joint_position',
      controllerType: 'JOINT_POSITION',
      actionDim: 8,
    };
    expect(
      canEvaluateModelAsset(finalAsset, {
        jobInProgress: isTrainingJobInProgressForAssets('completed'),
      })
    ).toBe(true);
  });

  it('shows specific disable reason when evalExecutor missing', () => {
    const asset = {
      status: 'ready',
      checkpointPath: '/tmp/model_final.pt',
      checkpointKind: 'final',
      canEvaluate: false,
      canEvaluateReason: 'evalExecutor 缺失，无法确定 joint-space 评测执行器',
      displayStatus: 'ready',
      modelType: 'act',
    };
    expect(canEvaluateModelAsset(asset, { jobInProgress: false })).toBe(false);
    expect(getModelAssetEvalDisabledReason(asset, { jobInProgress: false })).toContain('evalExecutor');
  });
});
