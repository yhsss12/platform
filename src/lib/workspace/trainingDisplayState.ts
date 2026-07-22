import type { TrainingTaskStatus } from '@/lib/mock/workspaceTrainingMock';
import type { TrainingMetricPoint } from '@/lib/workspace/trainingLogParser';
import {
  normalizeTrainingJobStatus,
  type TrainingJobStatusInput,
  trainingLogHasActivity,
  trainingProgressPercent,
} from '@/lib/workspace/trainingStatus';

export type TrainingDisplayPhase =
  | 'created'
  | 'launching'
  | 'running'
  | 'completed'
  | 'failed'
  | 'canceled'
  | 'waiting';

export interface TrainingDisplayState {
  phase: TrainingDisplayPhase;
  badgeLabel: TrainingTaskStatus;
  subLabel: string | null;
  progressLabel: string;
  progressHint: string | null;
  progressPercent: number | null;
  showProgressBar: boolean;
  progressIndeterminate: boolean;
  showLossChart: boolean;
  showFinalLoss: boolean;
  showGeneratedAssets: boolean;
}

export interface TrainingDisplayStateInput extends TrainingJobStatusInput {
  lossSeries?: TrainingMetricPoint[];
  executionMode?: string | null;
}

function hasTrainingActivity(input: TrainingDisplayStateInput): boolean {
  const epoch = Math.max(0, Number(input.currentEpoch ?? 0));
  const seriesLen = input.lossSeries?.length ?? 0;
  return epoch > 0 || seriesLen > 0 || trainingLogHasActivity(input.log);
}

function isMisleadingRunningMessage(message: string): boolean {
  const trimmed = message.trim();
  if (!trimmed) return false;
  return /训练进行中/i.test(trimmed) || /training in progress/i.test(trimmed);
}

function resolveLaunchingSubLabel(input: TrainingDisplayStateInput): string {
  const message = (input.message ?? '').trim();
  const messageLower = message.toLowerCase();
  const executionMode = (input.executionMode ?? '').trim().toLowerCase();

  if (messageLower.includes('training job created') || messageLower === 'job created') {
    return '训练任务已创建，等待 runner 启动';
  }
  if (
    executionMode === 'remote_ssh' ||
    messageLower.includes('ssh') ||
    messageLower.includes('远端') ||
    messageLower.includes('remote')
  ) {
    if (message && !isMisleadingRunningMessage(message)) return message;
    return '正在连接远端训练节点…';
  }
  if (messageLower.includes('等待首批') || messageLower.includes('等待 runner')) {
    return message;
  }
  if (isMisleadingRunningMessage(message)) {
    return '等待训练进程启动';
  }
  if (message) return message;
  return '等待训练进程启动';
}

function resolveRunningSubLabel(input: TrainingDisplayStateInput): string | null {
  const totalEpochs = Math.max(0, Number(input.totalEpochs ?? 0));
  const currentEpoch = Math.max(0, Number(input.currentEpoch ?? 0));
  if (totalEpochs > 0 && currentEpoch > 0) {
    return `Epoch ${currentEpoch}/${totalEpochs}`;
  }
  if (currentEpoch > 0) {
    return `Epoch ${currentEpoch}`;
  }
  return null;
}

export function resolveTrainingDisplayState(input: TrainingDisplayStateInput): TrainingDisplayState {
  const normalized = normalizeTrainingJobStatus(input);
  const badgeLabel = normalized.displayStatus;
  const backend = normalized.backendStatus;
  const hasActivity = hasTrainingActivity(input);
  const totalEpochs = Math.max(0, Number(input.totalEpochs ?? 0));
  const currentEpoch = Math.max(0, Number(input.currentEpoch ?? 0));
  const messageLower = (input.message ?? '').trim().toLowerCase();

  let phase: TrainingDisplayPhase;
  if (backend === 'failed') {
    phase = 'failed';
  } else if (backend === 'canceled') {
    phase = 'canceled';
  } else if (normalized.completed || badgeLabel === '已完成') {
    phase = 'completed';
  } else if (badgeLabel === '训练中' && hasActivity) {
    phase = 'running';
  } else if (
    badgeLabel === '正在启动' ||
    badgeLabel === '等待同步' ||
    ((backend === 'running' || backend === 'starting') && !hasActivity)
  ) {
    phase =
      messageLower.includes('training job created') || messageLower === 'job created'
        ? 'created'
        : 'launching';
  } else if (badgeLabel === '排队中' || badgeLabel === '等待中' || badgeLabel === '等待节点') {
    phase = 'waiting';
  } else if (badgeLabel === '训练中') {
    phase = 'running';
  } else {
    phase = 'waiting';
  }

  let subLabel: string | null = null;
  if (phase === 'created' || phase === 'launching') {
    subLabel = resolveLaunchingSubLabel(input);
  } else if (phase === 'running') {
    subLabel = resolveRunningSubLabel(input);
  } else if (phase === 'waiting' && input.message?.trim()) {
    subLabel = input.message.trim();
  } else if (phase === 'failed' && input.message?.trim()) {
    subLabel = input.message.trim();
  }

  let progressLabel = '—';
  let progressHint: string | null = null;
  let progressPercent: number | null = null;
  let showProgressBar = false;
  let progressIndeterminate = false;

  if (phase === 'completed') {
    progressLabel =
      totalEpochs > 0
        ? `Epoch ${Math.max(currentEpoch, totalEpochs)}/${totalEpochs}`
        : '已完成';
    progressPercent = 100;
    showProgressBar = true;
  } else if (phase === 'running') {
    progressLabel =
      totalEpochs > 0 ? `Epoch ${currentEpoch}/${totalEpochs}` : `Epoch ${currentEpoch}`;
    progressPercent =
      input.progressPercent ??
      trainingProgressPercent({
        backendStatus: backend,
        epoch: currentEpoch,
        totalEpochs,
        progress: input.progress ?? undefined,
      });
    showProgressBar = true;
  } else if (phase === 'created' || phase === 'launching') {
    progressLabel = '等待启动';
    progressHint = '任务已创建，训练进程尚未开始写入 epoch/loss。';
    progressPercent = 0;
    showProgressBar = true;
    progressIndeterminate = true;
  } else if (phase === 'failed') {
    if (currentEpoch > 0 && totalEpochs > 0) {
      progressLabel = `Epoch ${currentEpoch}/${totalEpochs}`;
      progressPercent = trainingProgressPercent({
        backendStatus: backend,
        epoch: currentEpoch,
        totalEpochs,
      });
      showProgressBar = true;
    } else {
      progressLabel = '训练失败';
      progressHint = input.message?.trim() || null;
    }
  } else if (phase === 'waiting') {
    progressLabel = badgeLabel;
    progressHint = input.message?.trim() || null;
  }

  return {
    phase,
    badgeLabel,
    subLabel,
    progressLabel,
    progressHint,
    progressPercent,
    showProgressBar,
    progressIndeterminate,
    showLossChart: phase === 'running' || phase === 'completed',
    showFinalLoss: phase === 'completed',
    showGeneratedAssets: phase === 'completed',
  };
}
