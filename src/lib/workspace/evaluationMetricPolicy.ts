/** Shared evaluation metric visibility and per-task computability policy (mirrors backend metric_policy). */

export const DEPRECATED_METRIC_IDS = new Set<string>(['metric_runtime_max_runtime_sec_v1']);

export const DEPRECATED_HIDDEN_METRIC_IDS = new Set<string>([
  'metric_runtime_ee_path_length_v1',
  'metric_runtime_smoothness_v1',
  'metric_runtime_max_action_norm_v1',
]);

export const REPORT_BODY_EXCLUDED_METRIC_IDS = new Set<string>([
  ...DEPRECATED_METRIC_IDS,
  ...DEPRECATED_HIDDEN_METRIC_IDS,
  'metric_episode_stability_v1',
]);

export const CABLE_THREADING_COMPUTABLE_METRIC_IDS = [
  'metric_cable_success_rate_v1',
  'metric_runtime_mean_steps_v1',
  'metric_runtime_max_steps_v1',
  'metric_runtime_video_fps_v1',
  'metric_runtime_control_frequency_v1',
  'metric_runtime_mean_sim_time_sec_v1',
] as const;

export const DUAL_ARM_COMPUTABLE_METRIC_IDS = [
  'metric_success_rate_v1',
  'metric_runtime_mean_steps_v1',
  'metric_runtime_max_steps_v1',
  'metric_runtime_video_fps_v1',
  'metric_runtime_control_frequency_v1',
  'metric_runtime_mean_sim_time_sec_v1',
  'metric_runtime_mean_joint_speed_v1',
  'metric_runtime_max_joint_speed_v1',
  'metric_runtime_mean_joint_acceleration_v1',
  'metric_runtime_max_joint_acceleration_v1',
] as const;

export const ISAAC_STACK_COMPUTABLE_METRIC_IDS = [
  'isaac_stack_success_rate_v1',
  'isaac_stack_mean_reward_v1',
  'isaac_stack_mean_episode_length_v1',
  'isaac_stack_failure_count_v1',
  'isaac_stack_timeout_rate_v1',
] as const;

const TASK_COMPUTABLE_METRIC_IDS: Record<string, readonly string[]> = {
  cable_threading: CABLE_THREADING_COMPUTABLE_METRIC_IDS,
  dual_arm_cable_manipulation: DUAL_ARM_COMPUTABLE_METRIC_IDS,
  block_stacking: ISAAC_STACK_COMPUTABLE_METRIC_IDS,
  isaaclab_franka_stack_cube: ISAAC_STACK_COMPUTABLE_METRIC_IDS,
  stacking: ISAAC_STACK_COMPUTABLE_METRIC_IDS,
};

export function isReportBodyMetricId(metricId: string): boolean {
  return !REPORT_BODY_EXCLUDED_METRIC_IDS.has(metricId);
}

export function filterMetricIdsForDisplay(metricIds: string[]): string[] {
  return metricIds.filter(isReportBodyMetricId);
}

export function resolveTaskTypeForMetricPolicy(templateId: string, taskType: string): string {
  if (templateId === 'cable_threading_single_arm' || taskType === 'cable_threading') {
    return 'cable_threading';
  }
  if (templateId === 'dual_arm_cable_manipulation' || taskType === 'dual_arm_cable_manipulation') {
    return 'dual_arm_cable_manipulation';
  }
  if (
    templateId === 'isaac_block_stacking' ||
    templateId === 'isaaclab_franka_stack_cube' ||
    taskType === 'block_stacking' ||
    taskType === 'isaaclab_franka_stack_cube' ||
    taskType === 'stacking'
  ) {
    return taskType === 'isaaclab_franka_stack_cube' ? 'isaaclab_franka_stack_cube' : 'block_stacking';
  }
  return taskType;
}

export function filterMetricIdsForTaskSelection(
  metricIds: string[],
  templateId: string,
  taskType: string
): string[] {
  const policyTaskType = resolveTaskTypeForMetricPolicy(templateId, taskType);
  const allowed = TASK_COMPUTABLE_METRIC_IDS[policyTaskType];
  const seen = new Set<string>();
  const result: string[] = [];

  for (const rawId of metricIds) {
    const metricId = rawId.trim();
    if (!metricId || seen.has(metricId)) continue;
    if (!isReportBodyMetricId(metricId)) continue;
    if (allowed && !allowed.includes(metricId)) continue;
    seen.add(metricId);
    result.push(metricId);
  }

  return result;
}
