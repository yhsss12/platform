export const ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID = 'isaaclab_franka_stack_cube';
export const ISAACLAB_FRANKA_STACK_CUBE_TASK_TYPE = 'isaaclab_franka_stack_cube';
/** 与 isaacStackCubeProduct.FRANKA_STACK_CUBE_PRODUCT_NAME 保持一致，避免循环依赖 */
export const ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME = '物块堆叠';
export const ISAACLAB_FRANKA_STACK_CUBE_DEFAULT_ENV = 'Isaac-Stack-Cube-Franka-IK-Rel-v0';

export const ISAACLAB_FRANKA_STACK_CUBE_DEFAULTS = {
  episodes: 1,
  seed: 0,
  saveVideo: true,
  saveTrajectory: true,
  headless: true,
  generationMode: 'scripted_expert',
} as const;

const ISAACLAB_FRANKA_STACK_CUBE_LABELS = new Set([
  ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME,
  'Franka Stack Cube',
  'Franka Stack Cube（Isaac Lab 物块堆叠）',
  'isaaclab_franka_stack_cube',
  'stacking',
  '物块堆叠',
  '物块堆叠任务',
]);

export function isIsaacLabFrankaStackCubeTask(templateOrId: string | null | undefined): boolean {
  if (!templateOrId) return false;
  const value = templateOrId.trim();
  return (
    value === ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID ||
    value === ISAACLAB_FRANKA_STACK_CUBE_TASK_TYPE ||
    ISAACLAB_FRANKA_STACK_CUBE_LABELS.has(value)
  );
}

export function isIsaacLabFrankaStackCubeReplayMode(taskType: string | null | undefined): boolean {
  return (taskType ?? '').trim() === ISAACLAB_FRANKA_STACK_CUBE_TASK_TYPE;
}

export function resolveIsaacLabFrankaStackCubeRuntimePath(jobId: string): string {
  return `runs/data_generation/jobs/${jobId}`;
}

export function buildIsaacLabFrankaStackCubeConsoleHref(params: {
  jobId: string;
  mode?: 'data-generation' | 'replay';
}): string {
  const search = new URLSearchParams({
    mode: params.mode ?? 'data-generation',
    taskType: ISAACLAB_FRANKA_STACK_CUBE_TASK_TYPE,
    jobId: params.jobId,
  });
  return `/workspace/simulation/console?${search.toString()}`;
}

export function buildIsaacLabFrankaStackCubeReplayHref(params: { jobId: string }): string {
  const search = new URLSearchParams({
    replayType: 'dataset',
    taskType: ISAACLAB_FRANKA_STACK_CUBE_TASK_TYPE,
    jobId: params.jobId,
  });
  return `/workspace/replay?${search.toString()}`;
}

export function isIsaacLabFrankaStackCubeDataset(dataset: {
  taskType?: string | null;
  taskTemplateId?: string | null;
}): boolean {
  if (isIsaacLabFrankaStackCubeTask(dataset.taskType)) return true;
  return isIsaacLabFrankaStackCubeTask(dataset.taskTemplateId);
}
