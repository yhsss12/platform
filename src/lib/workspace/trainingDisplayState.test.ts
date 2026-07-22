import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { resolveTrainingDisplayState } from '@/lib/workspace/trainingDisplayState';

describe('resolveTrainingDisplayState', () => {
  it('launching shows waiting subLabel instead of misleading running message', () => {
    const state = resolveTrainingDisplayState({
      backendStatus: 'running',
      currentEpoch: 0,
      totalEpochs: 5,
      message: '训练进行中（diffusion_policy）',
    });
    assert.equal(state.phase, 'launching');
    assert.equal(state.badgeLabel, '正在启动');
    assert.equal(state.subLabel, '等待训练进程启动');
    assert.notEqual(state.subLabel, '训练进行中（diffusion_policy）');
  });

  it('created maps training job created message to friendly subLabel', () => {
    const state = resolveTrainingDisplayState({
      backendStatus: 'starting',
      currentEpoch: 0,
      totalEpochs: 5,
      message: 'training job created',
    });
    assert.equal(state.phase, 'created');
    assert.equal(state.badgeLabel, '正在启动');
    assert.match(state.subLabel ?? '', /等待 runner 启动/);
  });

  it('launching progress uses waiting label instead of Epoch 0/5', () => {
    const state = resolveTrainingDisplayState({
      backendStatus: 'running',
      currentEpoch: 0,
      totalEpochs: 5,
      message: 'training job created',
    });
    assert.equal(state.progressLabel, '等待启动');
    assert.notEqual(state.progressLabel, 'Epoch 0/5');
    assert.equal(state.progressIndeterminate, true);
    assert.equal(state.showLossChart, false);
  });

  it('running shows epoch progress after activity', () => {
    const state = resolveTrainingDisplayState({
      backendStatus: 'running',
      currentEpoch: 2,
      totalEpochs: 5,
      message: '训练进行中（diffusion_policy）',
      lossSeries: [{ epoch: 2, trainLoss: 0.4 }],
    });
    assert.equal(state.phase, 'running');
    assert.equal(state.badgeLabel, '训练中');
    assert.equal(state.subLabel, 'Epoch 2/5');
    assert.equal(state.progressLabel, 'Epoch 2/5');
    assert.equal(state.showLossChart, true);
  });

  it('completed shows full progress and final loss sections', () => {
    const state = resolveTrainingDisplayState({
      backendStatus: 'completed',
      currentEpoch: 5,
      totalEpochs: 5,
    });
    assert.equal(state.phase, 'completed');
    assert.equal(state.progressPercent, 100);
    assert.equal(state.showFinalLoss, true);
    assert.equal(state.showGeneratedAssets, true);
  });
});
