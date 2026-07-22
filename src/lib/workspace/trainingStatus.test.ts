import { describe, expect, it } from 'vitest';

import { normalizedTrainingMetrics } from '@/lib/workspace/normalizedTrainingMetrics';
import { normalizeTrainingJobStatus, resolveTrainingDisplayStatus, trainingProgressPercent } from '@/lib/workspace/trainingStatus';

describe('trainingStatus sync', () => {
  it('shows waiting sync when remote running without epoch/log', () => {
    expect(
      resolveTrainingDisplayStatus({
        backendStatus: 'running',
        currentEpoch: 0,
        message: '本地 SSH 轮询中断，请稍后自动同步',
      })
    ).toBe('等待同步');
  });

  it('shows training only after epoch/log activity', () => {
    expect(
      resolveTrainingDisplayStatus({
        backendStatus: 'running',
        currentEpoch: 2,
        totalEpochs: 5,
      })
    ).toBe('训练中');
  });

  it('treats completed badge as running when epoch behind max', () => {
    const normalized = normalizeTrainingJobStatus({
      backendStatus: 'completed',
      currentEpoch: 5,
      totalEpochs: 200,
      progress: 1,
    });
    expect(normalized.backendStatus).toBe('running');
    expect(normalized.displayStatus).toBe('训练中');
    expect(normalized.inProgress).toBe(true);
  });

  it('computes progress from epoch ratio not stale progress=1', () => {
    const pct = trainingProgressPercent({
      backendStatus: 'running',
      epoch: 5,
      totalEpochs: 200,
      progress: 1,
    });
    expect(pct).toBe(3);
  });
});

describe('normalizedTrainingMetrics running', () => {
  it('hides finalLoss while running', () => {
    const metrics = normalizedTrainingMetrics({
      row: {
        backendStatus: 'running',
        currentEpoch: 5,
        totalEpochs: 200,
        loss: 0.23,
      },
      metrics: {
        finalLoss: 0.231848,
        bestLoss: 0.22,
        progress: 1,
      },
      log: 'command: --init-checkpoint /data/model_final.pt\nEpoch 5 Loss: 0.23\n',
    });
    expect(metrics.finalLoss).toBeNull();
    expect(metrics.progressPercent).toBeLessThan(5);
    expect(metrics.loss).toBe(0.23);
  });
});
