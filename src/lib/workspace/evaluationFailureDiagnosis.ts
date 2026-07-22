export interface EvalFailureDiagnosis {
  failedStage?: string | null;
  failureReason?: string | null;
  errorMessage?: string | null;
  error?: string | null;
  logPaths?: {
    stdout?: string | null;
    stderr?: string | null;
    run?: string | null;
  };
}

const FAILED_STAGE_LABELS: Record<string, string> = {
  model_loading: '模型加载',
  env_start: '环境启动',
  obs_validation: '观测校验',
  rollout: 'Rollout 执行',
  video_generation: '视频生成',
  aggregation: '结果聚合',
  unknown: '未知阶段',
};

const FAILURE_REASON_LABELS: Record<string, string> = {
  obs_key_mismatch: '观测键不匹配',
  obs_dim_mismatch: '观测维度不匹配',
  action_dim_mismatch: '动作维度不匹配',
  model_load_failed: '模型加载失败',
  runner_exception: 'Runner 异常',
  video_generation_failed: '视频生成失败',
  unknown_error: '未知错误',
};

export function evalFailedStageLabel(stage?: string | null): string {
  if (!stage) return '未知阶段';
  return FAILED_STAGE_LABELS[stage] ?? stage;
}

export function evalFailureReasonLabel(reason?: string | null): string {
  if (!reason) return '未知错误';
  return FAILURE_REASON_LABELS[reason] ?? reason;
}

export function resolveEvalFailureDiagnosis(
  status: Record<string, unknown> | null | undefined
): EvalFailureDiagnosis | null {
  if (!status || status.status !== 'failed') return null;

  const live = (status.live ?? {}) as Record<string, unknown>;
  const failedStage = (status.failedStage ?? live.failedStage) as string | undefined;
  const failureReason = (status.failureReason ?? live.failureReason) as string | undefined;
  const errorMessage = (status.errorMessage ?? live.errorMessage) as string | undefined;
  const error = (status.error ?? live.error) as string | undefined;
  const logPaths = (status.logPaths ?? live.logPaths) as EvalFailureDiagnosis['logPaths'];

  if (!failedStage && !failureReason && !errorMessage && !error) return null;

  return {
    failedStage,
    failureReason,
    errorMessage,
    error,
    logPaths,
  };
}
