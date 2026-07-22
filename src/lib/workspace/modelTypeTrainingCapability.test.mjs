import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

function modelTypeTrainingCapabilityLabel(item) {
  if (item.trainingReadinessStatus === 'pending') return '环境检测中';
  if (item.trainingReady) return '可训练';
  if (item.disabledReason?.trim()) return item.disabledReason.trim();
  return '该训练后端暂未开放';
}

function modelTypeHasPendingReadiness(items) {
  return items.some((item) => item.trainingReadinessStatus === 'pending');
}

describe('modelTypeTrainingCapability', () => {
  it('shows pending label for pi0 probe in progress', () => {
    assert.equal(
      modelTypeTrainingCapabilityLabel({ trainingReady: false, trainingReadinessStatus: 'pending' }),
      '环境检测中'
    );
  });

  it('shows disabledReason when training is unavailable', () => {
    assert.equal(
      modelTypeTrainingCapabilityLabel({
        trainingReady: false,
        trainingReadinessStatus: 'unavailable',
        disabledReason: '未检测到可用 openpi 环境',
      }),
      '未检测到可用 openpi 环境'
    );
  });

  it('detects pending readiness in list', () => {
    assert.equal(
      modelTypeHasPendingReadiness([
        { trainingReadinessStatus: 'ready' },
        { trainingReadinessStatus: 'pending' },
      ]),
      true
    );
  });
});
