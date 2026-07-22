import type { TrainingTaskStatus } from '@/lib/mock/workspaceTrainingMock';

const COMPLETION_LOG_MARKERS = [
  'training completed',
  'finished training',
  'saved final model',
  'saved checkpoint:',
  '训练完成',
];

/** 将接口 / 文件状态归一为内部 backend status */
export function normalizeTrainingBackendStatus(raw?: string | null): string {
  const value = (raw ?? '').trim().toLowerCase();
  if (!value) return 'unknown';

  const aliases: Record<string, string> = {
    queued: 'pending',
    created: 'starting',
    training: 'running',
    success: 'completed',
    succeeded: 'completed',
    finished: 'completed',
    done: 'completed',
    error: 'failed',
    backend_unavailable: 'failed',
    cancelled: 'canceled',
  };

  return aliases[value] ?? value;
}

export type TrainingJobStatusInput = {
  backendStatus?: string | null;
  status?: TrainingTaskStatus | string | null;
  currentEpoch?: number;
  totalEpochs?: number;
  progress?: number | null;
  progressPercent?: number | null;
  checkpointExists?: boolean;
  log?: string | null;
  message?: string | null;
};

function logIndicatesTrainingCompleted(log?: string | null): boolean {
  if (!log?.trim()) return false;
  const body = log
    .split('\n')
    .filter((line) => !line.trim().toLowerCase().startsWith('command:'))
    .join('\n')
    .toLowerCase();
  return COMPLETION_LOG_MARKERS.some((marker) => body.includes(marker));
}

function isEpochBehindMax(currentEpoch: number, totalEpochs: number): boolean {
  return totalEpochs > 0 && currentEpoch > 0 && currentEpoch < totalEpochs;
}

function logHasTrainingActivity(log?: string | null): boolean {
  if (!log?.trim()) return false;
  return /epoch\s+\d+/i.test(log) || /loss\s*[:=]/i.test(log);
}

/** @internal exported for display-state derivation */
export function trainingLogHasActivity(log?: string | null): boolean {
  return logHasTrainingActivity(log);
}

/** 结合 backend 状态、message、epoch/log 推断用户可见的细粒度状态 */
export function resolveTrainingDisplayStatus(input: TrainingJobStatusInput): TrainingTaskStatus {
  const backend = normalizeTrainingBackendStatus(input.backendStatus ?? input.status);
  const message = (input.message ?? '').trim();
  const messageLower = message.toLowerCase();
  const epoch = Math.max(0, Number(input.currentEpoch ?? 0));
  const hasActivity = epoch > 0 || logHasTrainingActivity(input.log);

  if (backend === 'failed') return '失败';
  if (backend === 'canceled') return '已取消';
  if (backend === 'completed') return '已完成';

  if (backend === 'running') {
    if (!hasActivity) {
      if (
        messageLower.includes('ssh') ||
        messageLower.includes('同步') ||
        messageLower.includes('轮询中断') ||
        messageLower.includes('连接中断')
      ) {
        return '等待同步';
      }
      return '正在启动';
    }
    return '训练中';
  }

  if (backend === 'starting') {
    if (
      messageLower.includes('ssh') ||
      messageLower.includes('同步') ||
      messageLower.includes('轮询中断') ||
      messageLower.includes('连接中断')
    ) {
      return '等待同步';
    }
    return '正在启动';
  }

  if (backend === 'queued' || backend === 'pending') {
    if (
      messageLower.includes('gpu') ||
      messageLower.includes('节点') ||
      messageLower.includes('空闲') ||
      messageLower.includes('忙碌')
    ) {
      return '等待节点';
    }
    return '排队中';
  }

  return '等待中';
}

/** 统一训练任务完成判定（列表 / 详情 / 模型资产共用） */
export function normalizeTrainingJobStatus(input: TrainingJobStatusInput): {
  backendStatus: string;
  displayStatus: TrainingTaskStatus;
  inProgress: boolean;
  completed: boolean;
} {
  let raw = normalizeTrainingBackendStatus(input.backendStatus ?? input.status);

  if (raw === 'failed' || raw === 'canceled') {
    return {
      backendStatus: raw,
      displayStatus: resolveTrainingDisplayStatus(input),
      inProgress: false,
      completed: false,
    };
  }

  const currentEpoch = Math.max(0, Number(input.currentEpoch ?? 0));
  const totalEpochs = Math.max(0, Number(input.totalEpochs ?? 0));

  if (raw === 'completed' && isEpochBehindMax(currentEpoch, totalEpochs)) {
    raw = 'running';
  }

  if (raw === 'completed') {
    return {
      backendStatus: 'completed',
      displayStatus: '已完成',
      inProgress: false,
      completed: true,
    };
  }

  const progressPercent = trainingProgressPercent({
    backendStatus: raw,
    epoch: currentEpoch,
    totalEpochs,
    progress: input.progress ?? undefined,
  });

  const epochComplete = totalEpochs > 0 && currentEpoch >= totalEpochs;
  const progressComplete = progressPercent >= 100 && (totalEpochs === 0 || currentEpoch >= totalEpochs);
  const logComplete = logIndicatesTrainingCompleted(input.log);

  if (epochComplete && (logComplete || !input.log)) {
    return {
      backendStatus: 'completed',
      displayStatus: '已完成',
      inProgress: false,
      completed: true,
    };
  }

  if (progressComplete && epochComplete) {
    return {
      backendStatus: 'completed',
      displayStatus: '已完成',
      inProgress: false,
      completed: true,
    };
  }

  const inProgress =
    raw === 'running' ||
    raw === 'starting' ||
    raw === 'pending' ||
    raw === 'queued' ||
    isEpochBehindMax(currentEpoch, totalEpochs);
  const displayRaw = inProgress && raw === 'completed' ? 'running' : raw;
  const displayStatus = resolveTrainingDisplayStatus({
    ...input,
    backendStatus: displayRaw,
    currentEpoch,
    totalEpochs,
  });
  return {
    backendStatus: displayRaw,
    displayStatus,
    inProgress,
    completed: false,
  };
}

/** 列表 / 详情展示用中文状态（不依赖 checkpoint） */
export function mapTrainingStatusToDisplay(raw?: string | null): TrainingTaskStatus {
  const status = normalizeTrainingBackendStatus(raw);

  switch (status) {
    case 'pending':
      return '等待中';
    case 'queued':
      return '排队中';
    case 'starting':
      return '正在启动';
    case 'running':
      return '训练中';
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
    case 'canceled':
      return '已取消';
    default:
      return '等待中';
  }
}

export function isTrainingJobInProgress(displayStatus: TrainingTaskStatus): boolean {
  return (
    displayStatus === '训练中' ||
    displayStatus === '正在启动' ||
    displayStatus === '等待同步' ||
    displayStatus === '等待节点' ||
    displayStatus === '等待中' ||
    displayStatus === '排队中'
  );
}

export function isTrainingJobInProgressFromSignals(input: TrainingJobStatusInput): boolean {
  return normalizeTrainingJobStatus(input).inProgress;
}

export function trainingProgressPercent(options: {
  backendStatus?: string | null;
  epoch?: number;
  totalEpochs?: number;
  progress?: number;
}): number {
  const status = normalizeTrainingBackendStatus(options.backendStatus);
  const epoch = Math.max(0, options.epoch ?? 0);
  const totalEpochs = Math.max(0, options.totalEpochs ?? 0);

  if (status === 'completed' && totalEpochs > 0 && epoch >= totalEpochs) {
    return 100;
  }
  if (totalEpochs > 0 && epoch >= totalEpochs) {
    return 100;
  }

  if (totalEpochs > 0 && epoch > 0) {
    return Math.min(99, Math.round((epoch / totalEpochs) * 100));
  }

  if (status === 'completed') {
    return 100;
  }

  if (options.progress != null && Number.isFinite(options.progress)) {
    const p = options.progress;
    const fraction = p <= 1 ? p : p / 100;
    if (totalEpochs > 0 && epoch > 0 && epoch < totalEpochs) {
      return Math.min(99, Math.round((epoch / totalEpochs) * 100));
    }
    return Math.min(99, Math.round(fraction * 100));
  }
  return 0;
}
