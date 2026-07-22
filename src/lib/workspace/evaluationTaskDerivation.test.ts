import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  CABLE_THREADING_AVAILABLE_METRIC_IDS,
  CABLE_THREADING_RUNTIME_METRIC_IDS,
  DUAL_ARM_AVAILABLE_METRIC_IDS,
  DEFAULT_MAX_EVALUATION_EPISODES,
  deriveEvaluationConfigFromTask,
  deriveMetricDefinitionsFromTask,
  isSuccessRateMetric,
} from './evaluationTaskDerivation';

describe('deriveMetricDefinitionsFromTask success rate', () => {
  it('includes common runtime metrics for dual arm cable manipulation', () => {
    const result = deriveMetricDefinitionsFromTask({
      id: 'dual_arm_cable_manipulation',
      name: '线缆整理',
      taskType: 'dual_arm_cable_manipulation',
      defaultMetricIds: ['metric_success_rate_v1'],
    } as never);

    assert.ok(result.availableMetrics.some((m) => m.label === '成功率'));
    assert.equal(result.availableMetrics.length, DUAL_ARM_AVAILABLE_METRIC_IDS.length);
    assert.ok(result.availableMetrics.some((m) => m.key === 'metric_runtime_mean_steps_v1'));
    assert.ok(result.availableMetrics.some((m) => m.key === 'metric_runtime_mean_sim_time_sec_v1'));
    assert.ok(result.defaultSelectedMetricKeys.includes('metric_success_rate_v1'));
    assert.equal(
      result.availableMetrics.filter((m) => isSuccessRateMetric(m)).length,
      1
    );
  });

  it('excludes hidden and non-computable metrics from cable threading selection', () => {
    const result = deriveMetricDefinitionsFromTask({
      id: 'cable_threading_single_arm',
      name: '线缆穿杆',
      taskType: 'cable_threading',
      defaultMetricIds: ['metric_cable_success_rate_v1'],
    } as never);

    assert.equal(result.availableMetrics.length, CABLE_THREADING_AVAILABLE_METRIC_IDS.length);
    assert.ok(!result.availableMetrics.some((m) => m.key === 'metric_runtime_smoothness_v1'));
    assert.ok(!result.availableMetrics.some((m) => m.key === 'metric_runtime_max_action_norm_v1'));
    assert.ok(!result.availableMetrics.some((m) => m.key === 'metric_runtime_ee_path_length_v1'));
    assert.ok(!result.availableMetrics.some((m) => m.key === 'metric_runtime_mean_joint_speed_v1'));
  });

  it('filters legacy registry metrics down to computable cable threading set', () => {
    const result = deriveMetricDefinitionsFromTask(
      {
        id: 'cable_threading_single_arm',
        name: '线缆穿杆',
        taskType: 'cable_threading',
        defaultMetricIds: ['metric_cable_success_rate_v1'],
      } as never,
      {
        assetId: 'task_cable_threading_v1',
        metrics: [
          'metric_cable_success_rate_v1',
          'metric_runtime_mean_steps_v1',
          'metric_runtime_smoothness_v1',
          'metric_runtime_mean_joint_speed_v1',
        ],
      } as never
    );

    assert.equal(result.availableMetrics.length, CABLE_THREADING_AVAILABLE_METRIC_IDS.length);
    assert.ok(result.availableMetrics.some((metric) => metric.key === 'metric_runtime_mean_steps_v1'));
    assert.ok(!result.availableMetrics.some((metric) => metric.key === 'metric_runtime_smoothness_v1'));
  });

  it('includes success rate for isaac block stacking', () => {
    const result = deriveMetricDefinitionsFromTask({
      id: 'isaac_block_stacking',
      name: '物块堆叠',
      taskType: 'block_stacking',
      defaultMetricIds: [
        'isaac_stack_success_rate_v1',
        'isaac_stack_mean_reward_v1',
        'isaac_stack_failure_count_v1',
      ],
    } as never);

    assert.ok(result.availableMetrics.some((m) => m.label === '成功率'));
    assert.ok(result.availableMetrics.some((m) => m.label === '平均奖励'));
    assert.ok(result.defaultSelectedMetricKeys.includes('isaac_stack_success_rate_v1'));
  });

  it('fallback injects success rate when task has no metrics', () => {
    const result = deriveMetricDefinitionsFromTask({
      id: 'unknown_task',
      name: '未知',
    } as never);

    assert.equal(result.availableMetrics[0]?.label, '成功率');
    assert.equal(result.defaultSelectedMetricKeys[0], 'metric_cable_success_rate_v1');
  });
});

describe('deriveEvaluationConfigFromTask episodes bounds', () => {
  it('allows dual arm cable manipulation episodes up to 100 by default', () => {
    const result = deriveEvaluationConfigFromTask({
      id: 'dual_arm_cable_manipulation',
      name: '线缆整理',
      taskType: 'dual_arm_cable_manipulation',
    } as never);

    assert.equal(result.episodes, 1);
    assert.equal(result.episodesMin, 1);
    assert.equal(result.episodesMax, DEFAULT_MAX_EVALUATION_EPISODES);
  });

  it('reads episodes bounds from registry evaluationConfig', () => {
    const result = deriveEvaluationConfigFromTask(
      {
        id: 'dual_arm_cable_manipulation',
        name: '线缆整理',
        taskType: 'dual_arm_cable_manipulation',
      } as never,
      {
        metadata: {
          evaluationConfig: {
            episodes: { min: 1, max: 100, default: 1 },
          },
        },
      } as never
    );

    assert.equal(result.episodesMin, 1);
    assert.equal(result.episodesMax, 100);
    assert.equal(result.episodes, 1);
  });
});
