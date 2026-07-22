import { TASK_TEMPLATE_DISPLAY_NAMES } from '@/lib/workspace/taskDisplayNames';

export const ISAACSIM_FRANKA_PICK_PLACE_TEMPLATE_ID = 'isaacsim_franka_pick_place';
export const ISAACSIM_FRANKA_PICK_PLACE_TASK_TYPE = 'isaacsim_franka_pick_place';
export const ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME =
  TASK_TEMPLATE_DISPLAY_NAMES.isaacsim_franka_pick_place ?? 'Franka 物体搬运';

export const ISAACSIM_FRANKA_PICK_PLACE_DEFAULTS = {
  episodes: 1,
  seed: 0,
  saveVideo: true,
  saveTrajectory: true,
  headless: true,
} as const;

const ISAACSIM_FRANKA_PICK_PLACE_LABELS = new Set([
  ISAACSIM_FRANKA_PICK_PLACE_DISPLAY_NAME,
  'Franka 物体搬运',
  'isaacsim_franka_pick_place',
  'pick_and_place',
]);

export function isIsaacSimFrankaPickPlaceTask(templateOrId: string | null | undefined): boolean {
  if (!templateOrId) return false;
  const value = templateOrId.trim();
  return (
    value === ISAACSIM_FRANKA_PICK_PLACE_TEMPLATE_ID ||
    value === ISAACSIM_FRANKA_PICK_PLACE_TASK_TYPE ||
    ISAACSIM_FRANKA_PICK_PLACE_LABELS.has(value)
  );
}

export function isIsaacSimFrankaPickPlaceReplayMode(taskType: string | null | undefined): boolean {
  return (taskType ?? '').trim() === ISAACSIM_FRANKA_PICK_PLACE_TASK_TYPE;
}

export function buildIsaacSimFrankaPickPlaceConsoleHref(params: {
  jobId: string;
  dataId?: string;
  mode?: 'data-generation' | 'replay';
}): string {
  const search = new URLSearchParams({
    mode: params.mode ?? 'data-generation',
    taskType: ISAACSIM_FRANKA_PICK_PLACE_TASK_TYPE,
    jobId: params.jobId,
  });
  if (params.dataId) search.set('dataId', params.dataId);
  return `/workspace/simulation/console?${search.toString()}`;
}

export function buildIsaacSimFrankaPickPlaceReplayHref(params: {
  jobId?: string;
  datasetId?: string;
}): string {
  const search = new URLSearchParams({
    replayType: 'dataset',
    taskType: ISAACSIM_FRANKA_PICK_PLACE_TASK_TYPE,
  });
  if (params.jobId) search.set('jobId', params.jobId);
  if (params.datasetId) search.set('datasetId', params.datasetId);
  return `/workspace/replay?${search.toString()}`;
}

export function buildIsaacSimFrankaPickPlaceVideoApiPath(jobId: string, episode = 'ep_000001'): string {
  return `/api/workspace/isaacsim-franka-pick-place/jobs/${encodeURIComponent(jobId)}/video?episode=${encodeURIComponent(episode)}`;
}

export function resolveIsaacSimFrankaPickPlaceRuntimePath(jobId: string): string {
  return `runs/data_generation/jobs/${jobId}`;
}

export function isIsaacSimFrankaPickPlaceDataset(dataset: {
  taskType?: string | null;
  taskTemplateId?: string | null;
  sourceJobId?: string | null;
  simulatorBackend?: string | null;
}): boolean {
  if (isIsaacSimFrankaPickPlaceTask(dataset.taskType)) return true;
  if (isIsaacSimFrankaPickPlaceTask(dataset.taskTemplateId)) return true;
  if (dataset.simulatorBackend === 'isaacsim') return true;
  return false;
}

export function resolveIsaacSimVideoStatusLabel(input: {
  videoAvailable?: boolean | null;
  video_status?: string | null;
  videoStatus?: string | null;
}): string {
  const status =
    input.video_status ??
    input.videoStatus ??
    (input.videoAvailable ? 'available' : 'pending');
  switch (status) {
    case 'available':
      return '可用';
    case 'failed':
      return '生成失败';
    case 'partial':
      return '部分可用';
    case 'pending':
    default:
      return '生成中';
  }
}

export function resolveIsaacSimVideoPlaceholderMessage(input: {
  videoAvailable?: boolean | null;
  video_status?: string | null;
  videoStatus?: string | null;
}): string {
  const status =
    input.video_status ??
    input.videoStatus ??
    (input.videoAvailable ? 'available' : 'pending');
  switch (status) {
    case 'pending':
      return '回放视频生成中';
    case 'failed':
      return '回放视频生成失败，结构化数据可用';
    case 'partial':
      return '部分 episode 视频可用';
    case 'available':
      return input.videoAvailable === false ? '当前 episode 暂无回放视频' : '回放视频生成中';
    default:
      return '当前 episode 暂无回放视频';
  }
}
