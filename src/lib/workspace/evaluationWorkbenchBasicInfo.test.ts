import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { resolveEvaluationWorkbenchBasicInfo } from './evaluationWorkbenchBasicInfo';

describe('resolveEvaluationWorkbenchBasicInfo', () => {
  it('uses workbenchBasicInfo from status even when policy suggests expert', () => {
    const info = resolveEvaluationWorkbenchBasicInfo({
      status: {
        status: 'completed',
        taskType: 'cable_threading',
        workbenchBasicInfo: {
          taskName: '线缆穿杆评测_20260625_814',
          evaluationTypeLabel: '模型评测',
          evaluationObjectLabel: '已训练模型',
          simulationPlatform: 'MuJoCo',
          statusLabel: '已完成',
          robotType: 'Panda',
          modelAssetName: '线缆穿杆 · Final',
          associatedTaskName: '线缆穿杆',
        },
        live: { policy: 'scripted' },
        metrics: { policy: 'scripted' },
      },
    });

    assert.equal(info.taskName, '线缆穿杆评测_20260625_814');
    assert.equal(info.evaluationTypeLabel, '模型评测');
    assert.equal(info.evaluationObjectLabel, '已训练模型');
    assert.equal(info.modelAssetName, '线缆穿杆 · Final');
  });

  it('prefers trained_model over policy=expert in fallback path', () => {
    const info = resolveEvaluationWorkbenchBasicInfo({
      status: {
        status: 'completed',
        taskType: 'cable_threading',
        evaluationObject: 'trained_model',
        evaluationTypeLabel: '模型评测',
        taskName: '线缆穿杆评测_20260625_814',
        modelAssetName: '线缆穿杆 · Final',
        live: { policy: 'expert' },
        metrics: { policy: 'expert' },
      },
    });

    assert.equal(info.evaluationTypeLabel, '模型评测');
    assert.equal(info.evaluationObjectLabel, '已训练模型');
  });

  it('does not use associated task display name as task name', () => {
    const info = resolveEvaluationWorkbenchBasicInfo({
      evalJobId: 'ct_eval_20260625_090114_3c3a',
      status: { status: 'completed', taskType: 'cable_threading' },
      listItem: {
        taskName: '线缆穿杆评测_20260625_814',
        evaluationTypeLabel: '模型评测',
        evaluationObject: 'trained_model',
      },
      fallbackTaskName: '线缆穿杆',
    });

    assert.equal(info.taskName, '线缆穿杆评测_20260625_814');
    assert.equal(info.associatedTaskName, '线缆穿杆');
  });
});
