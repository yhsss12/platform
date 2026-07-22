export const ISAAC_BLOCK_STACKING_TEMPLATE_ID = 'isaac_block_stacking';
export const ISAAC_BLOCK_STACKING_TASK_TYPE = 'isaac_block_stacking';
/** 与 isaacStackCubeProduct.FRANKA_STACK_CUBE_PRODUCT_NAME 保持一致，避免循环依赖 */
export const ISAAC_BLOCK_STACKING_DISPLAY_NAME = '物块堆叠';
export const ISAAC_BLOCK_STACKING_DEFAULT_ENV = 'Isaac-Stack-Cube-Franka-IK-Rel-v0';

export function isIsaacBlockStackingReplayMode(taskType: string | null | undefined): boolean {
  return (taskType ?? '').trim() === ISAAC_BLOCK_STACKING_TASK_TYPE;
}

/** 与 MuJoCo 任务一致：统一进入 /workspace/simulation/console */
export function buildIsaacBlockStackingConsoleHref(params: {
  jobId: string;
  dataId?: string;
  mode?: 'data-generation' | 'replay';
}): string {
  const search = new URLSearchParams({
    mode: params.mode ?? 'data-generation',
    taskType: ISAAC_BLOCK_STACKING_TASK_TYPE,
    jobId: params.jobId,
  });
  if (params.dataId) search.set('dataId', params.dataId);
  return `/workspace/simulation/console?${search.toString()}`;
}

export function buildIsaacBlockStackingReplayConsoleHref(params: { jobId: string }): string {
  return buildIsaacBlockStackingConsoleHref({ jobId: params.jobId, mode: 'replay' });
}

/** 与 MuJoCo 任务一致：统一进入 /workspace/replay */
export function buildIsaacBlockStackingReplayHref(params: {
  jobId?: string;
  datasetId?: string;
  replayJobId?: string;
  evalId?: string;
}): string {
  const isEval = Boolean(params.evalId?.trim());
  const search = new URLSearchParams({
    replayType: isEval ? 'evaluation' : 'dataset',
    taskType: ISAAC_BLOCK_STACKING_TASK_TYPE,
  });
  if (params.jobId) search.set('jobId', params.jobId);
  if (params.datasetId) search.set('datasetId', params.datasetId);
  if (params.replayJobId) search.set('replayJobId', params.replayJobId);
  if (params.evalId) search.set('evalId', params.evalId);
  return `/workspace/replay?${search.toString()}`;
}

export function generateIsaacBlockStackingEvalTaskName(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const seq = String(Math.floor(Math.random() * 999) + 1).padStart(3, '0');
  return `${ISAAC_BLOCK_STACKING_DISPLAY_NAME}评测_${date}_${seq}`;
}

export function buildIsaacEvalReplayHref(params: { evalJobId: string; episode?: number }): string {
  return buildIsaacBlockStackingReplayHref({ evalId: params.evalJobId });
}

export function buildIsaacEvalReportHref(params: { evalJobId: string }): string {
  return `/workspace/evaluation/report?evalId=${encodeURIComponent(params.evalJobId)}&taskType=${ISAAC_BLOCK_STACKING_TASK_TYPE}`;
}

export function isIsaacEvalRow(row: { id?: string; taskType?: string | null }): boolean {
  return (
    row.id?.startsWith('isaac_eval_') === true ||
    row.taskType === ISAAC_BLOCK_STACKING_TASK_TYPE
  );
}

const ISAAC_BLOCK_STACKING_LABELS = new Set([
  'isaac_block_stacking',
  'block_stacking',
  'task_isaac_block_stacking_v1',
]);

/** 仅匹配评测/训练适配内部 templateId，不匹配产品展示名「物块堆叠」 */
export function isIsaacBlockStackingTask(templateOrId: string | null | undefined): boolean {
  if (!templateOrId) return false;
  const value = templateOrId.trim();
  return (
    value === ISAAC_BLOCK_STACKING_TEMPLATE_ID ||
    value === 'block_stacking' ||
    ISAAC_BLOCK_STACKING_LABELS.has(value)
  );
}

export function resolveIsaacBlockStackingTemplateLabel(templateId?: string | null): string {
  if (templateId && isIsaacBlockStackingTask(templateId)) {
    return ISAAC_BLOCK_STACKING_DISPLAY_NAME;
  }
  return ISAAC_BLOCK_STACKING_DISPLAY_NAME;
}
