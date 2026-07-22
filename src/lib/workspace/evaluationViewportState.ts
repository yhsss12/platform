export type EvaluationViewportState =
  | { kind: 'running_live'; evalJobId: string }
  | { kind: 'running'; message: string }
  | { kind: 'video'; videoJobId: string; partialFailure?: boolean; partialMessage?: string }
  | { kind: 'completed_no_video'; title: string; message: string }
  | { kind: 'failed_no_video'; title: string; message: string }
  | { kind: 'failed_partial_video'; videoJobId: string; message: string };

export function resolveEvaluationViewportState(params: {
  jobStatus: string;
  evalJobId: string;
  evalVideoExists: boolean;
  hasValidLiveFrame?: boolean;
}): EvaluationViewportState {
  const { jobStatus, evalJobId, evalVideoExists, hasValidLiveFrame = false } = params;
  const normalized = jobStatus.toLowerCase();

  if ((normalized === 'running' || normalized === 'queued') && hasValidLiveFrame) {
    return { kind: 'running_live', evalJobId };
  }

  if (normalized === 'running' || normalized === 'queued') {
    return {
      kind: 'running',
      message: '正在初始化 MuJoCo 场景…',
    };
  }

  if (evalVideoExists) {
    if (normalized === 'failed') {
      return {
        kind: 'failed_partial_video',
        videoJobId: evalJobId,
        message: '评测任务失败，但已生成部分回放画面。',
      };
    }
    return { kind: 'video', videoJobId: evalJobId };
  }

  if (normalized === 'failed') {
    if (evalVideoExists) {
      return {
        kind: 'failed_partial_video',
        videoJobId: evalJobId,
        message: '评测任务失败，但已生成部分回放画面。',
      };
    }
    if (hasValidLiveFrame) {
      return { kind: 'running_live', evalJobId };
    }
    return {
      kind: 'failed_no_video',
      title: '评测画面未生成',
      message:
        '当前评测任务执行失败，未生成可回放视频。请查看右侧失败诊断或评测日志。',
    };
  }

  if (normalized === 'completed') {
    if (hasValidLiveFrame) {
      return { kind: 'video', videoJobId: evalJobId };
    }
    return {
      kind: 'completed_no_video',
      title: '评测画面未生成',
      message: '评测已完成，但未生成回放视频。',
    };
  }

  return {
    kind: 'running',
    message: '正在初始化 MuJoCo 场景…',
  };
}
