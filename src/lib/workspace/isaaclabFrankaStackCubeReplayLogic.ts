export const STACK_CUBE_IN_PROGRESS_STATUSES = new Set([
  'running',
  'pending',
  'queued',
  'processing',
  'generating',
]);

export function isStackCubeJobInProgress(status?: string | null): boolean {
  const normalized = (status ?? '').trim().toLowerCase();
  return STACK_CUBE_IN_PROGRESS_STATUSES.has(normalized);
}

export function isStackCubeTaskIdsConsistent(input: {
  taskIdValidated?: boolean | null;
  episodeTaskId?: string | null;
  datasetTaskId?: string | null;
  datasetSourceTaskId?: string | null;
}): boolean {
  if (input.taskIdValidated === false) return false;
  const expected = 'isaaclab_franka_stack_cube';
  if (!input.episodeTaskId || input.episodeTaskId !== expected) return false;
  const datasetTask = input.datasetTaskId || input.datasetSourceTaskId;
  return datasetTask === expected;
}

export type StackCubeReplayOutcome =
  | { kind: 'in_progress'; message: string }
  | { kind: 'asset_validation_failed'; message: string }
  | { kind: 'video_available' }
  | {
      kind: 'hdf5_ready_no_video';
      message: string;
      datasetHdf5Path?: string;
      episodeCount?: number;
      successfulEpisodes?: number;
    }
  | { kind: 'video_pending'; message: string };

export function buildStackCubeReplayModeNotice(replayMode?: string | null): string | null {
  if (replayMode === 'state_based') {
    return '当前视频为基于 HDF5 状态轨迹生成的严格回放。';
  }
  if (replayMode === 'open_loop_preview') {
    return '当前视频为 open-loop 预览，可能与 HDF5 状态轨迹存在偏差。';
  }
  return null;
}

export interface StackCubeReplayOutcomeInput {
  jobStatus?: string | null;
  progress?: number | null;
  validationError?: string | null;
  taskIdValidated?: boolean | null;
  episodeTaskId?: string | null;
  datasetTaskId?: string | null;
  datasetSourceTaskId?: string | null;
  forbiddenVideoPath?: boolean;
  videoExists?: boolean;
  videoStatus?: string | null;
  datasetHdf5Path?: string | null;
  episodeCount?: number | null;
  successfulEpisodes?: number | null;
}

export function resolveStackCubeReplayOutcome(
  input: StackCubeReplayOutcomeInput
): StackCubeReplayOutcome {
  if (isStackCubeJobInProgress(input.jobStatus)) {
    const message =
      typeof input.progress === 'number' && input.progress >= 45
        ? '正在生成回放预览，请稍后刷新。'
        : '数据生成中，回放资产尚未就绪。';
    return { kind: 'in_progress', message };
  }

  if (input.validationError || input.taskIdValidated === false || input.forbiddenVideoPath) {
    return { kind: 'asset_validation_failed', message: '任务资产校验失败，无法播放视频' };
  }

  const taskIdsConsistent = isStackCubeTaskIdsConsistent(input);
  if (!taskIdsConsistent) {
    if (input.jobStatus === 'failed') {
      return { kind: 'asset_validation_failed', message: '任务资产校验失败，无法播放视频' };
    }
    if (input.jobStatus === 'completed') {
      return { kind: 'asset_validation_failed', message: '任务资产校验失败，无法播放视频' };
    }
    return { kind: 'in_progress', message: '数据生成中，回放资产尚未就绪。' };
  }

  const videoStatus = (input.videoStatus ?? '').trim().toLowerCase();
  if ((videoStatus === 'available' || videoStatus === 'partial') && input.videoExists) {
    return { kind: 'video_available' };
  }

  if (input.jobStatus === 'completed') {
    return {
      kind: 'hdf5_ready_no_video',
      message: '该数据集已生成 HDF5 数据，但当前未生成视频回放资产。',
      datasetHdf5Path: input.datasetHdf5Path ?? 'datasets/dataset.hdf5',
      episodeCount:
        typeof input.episodeCount === 'number' ? input.episodeCount : undefined,
      successfulEpisodes:
        typeof input.successfulEpisodes === 'number' ? input.successfulEpisodes : undefined,
    };
  }

  return { kind: 'video_pending', message: '回放视频生成中或不可用' };
}

export function buildStackCubeHdf5PreviewNotice(
  outcome: Extract<StackCubeReplayOutcome, { kind: 'hdf5_ready_no_video' }>,
  sourceJobId: string
): string {
  const lines = [
    outcome.message,
    '',
    `dataset.hdf5: ${outcome.datasetHdf5Path ?? 'datasets/dataset.hdf5'}`,
  ];
  if (outcome.episodeCount != null) {
    lines.push(`episode_count: ${outcome.episodeCount}`);
  }
  if (outcome.successfulEpisodes != null) {
    lines.push(`successfulEpisodes: ${outcome.successfulEpisodes}`);
  }
  lines.push(`sourceJobId: ${sourceJobId}`);
  return lines.join('\n');
}
