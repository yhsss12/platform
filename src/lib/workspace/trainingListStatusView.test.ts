import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { buildTrainingListStatusView } from '@/components/workspace/training/TrainingStatusCell';
import { resolveTrainingDisplayState } from '@/lib/workspace/trainingDisplayState';

describe('training list status column', () => {
  it('launching list view only exposes badge label with tooltip, not inline subLabel', () => {
    const row = {
      backendStatus: 'running',
      status: '正在启动' as const,
      currentEpoch: 0,
      totalEpochs: 5,
      progressPercent: 0,
      message: 'training job created',
    };
    const display = resolveTrainingDisplayState({
      backendStatus: row.backendStatus,
      status: row.status,
      currentEpoch: row.currentEpoch,
      totalEpochs: row.totalEpochs,
      message: row.message,
    });
    const listView = buildTrainingListStatusView({
      ...row,
      id: 't1',
      trainJobId: 't1',
      source: 'real',
      name: 'test',
      relatedTask: 'ds',
      modelType: 'dp',
      dataset: 'ds',
      dataVolume: '—',
      trainability: 'ready',
      checkpoint: null,
      checkpointExists: false,
      hasModelManifest: false,
      createdAt: '2026-07-05',
      batchSize: 16,
      learningRate: 0.0001,
      seed: 1,
    });

    assert.equal(listView.badgeLabel, '正在启动');
    assert.match(listView.tooltip ?? '', /等待 runner 启动/);
    assert.notEqual(listView.badgeLabel, display.subLabel);
  });

  it('failed list view keeps error in tooltip only', () => {
    const listView = buildTrainingListStatusView({
      id: 't2',
      trainJobId: 't2',
      source: 'real',
      name: 'test',
      relatedTask: 'ds',
      modelType: 'dp',
      dataset: 'ds',
      dataVolume: '—',
      status: '失败',
      trainability: 'ready',
      backendStatus: 'failed',
      currentEpoch: 0,
      totalEpochs: 5,
      progressPercent: 0,
      message: '训练进程退出，return code=1',
      checkpoint: null,
      checkpointExists: false,
      hasModelManifest: false,
      createdAt: '2026-07-05',
      batchSize: 16,
      learningRate: 0.0001,
      seed: 1,
    });

    assert.equal(listView.badgeLabel, '失败');
    assert.equal(listView.tooltip, '训练进程退出，return code=1');
  });

  it('display state logic unchanged for detail drawer', () => {
    const display = resolveTrainingDisplayState({
      backendStatus: 'failed',
      currentEpoch: 0,
      totalEpochs: 5,
      message: '训练进程退出，return code=1',
    });
    assert.equal(display.badgeLabel, '失败');
    assert.equal(display.subLabel, '训练进程退出，return code=1');
  });
});
