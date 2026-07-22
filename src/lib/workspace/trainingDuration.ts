import { normalizeTrainingBackendStatus } from '@/lib/workspace/trainingStatus';

/**
 * 解析训练开始时间（DB started_at 等为真实 UTC；status.json 无时区字符串按本地墙钟解析）。
 */
export function parseTrainingStartMs(value?: string | null): number | null {
  if (value == null) return null;
  const raw = String(value).trim();
  if (!raw) return null;

  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(raw)) {
    const ms = new Date(raw.replace(' ', 'T')).getTime();
    return Number.isFinite(ms) ? ms : null;
  }

  const text = raw.replace(' ', 'T');
  if (!/[zZ]$|[+-]\d{2}:?\d{2}$/.test(text)) {
    const ms = new Date(text).getTime();
    return Number.isFinite(ms) ? ms : null;
  }

  const ms = Date.parse(text);
  return Number.isFinite(ms) ? ms : null;
}

/**
 * 解析训练结束时间。status.json 的 updatedAt 常以 +00:00 导出本地墙钟，需按本地解析避免 +8h 偏差。
 */
export function parseTrainingEndMs(value?: string | null): number | null {
  if (value == null) return null;
  const raw = String(value).trim();
  if (!raw) return null;

  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(raw)) {
    const ms = new Date(raw.replace(' ', 'T')).getTime();
    return Number.isFinite(ms) ? ms : null;
  }

  const text = raw.replace(' ', 'T');
  const mislabeledLocal = text.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\+00:00|Z)$/);
  if (mislabeledLocal) {
    const ms = new Date(mislabeledLocal[1]).getTime();
    return Number.isFinite(ms) ? ms : null;
  }

  if (!/[zZ]$|[+-]\d{2}:?\d{2}$/.test(text)) {
    const ms = new Date(text).getTime();
    return Number.isFinite(ms) ? ms : null;
  }

  const ms = Date.parse(text);
  return Number.isFinite(ms) ? ms : null;
}

/** @deprecated 使用 parseTrainingStartMs / parseTrainingEndMs */
export function parseApiInstantMs(value?: string | null): number | null {
  return parseTrainingStartMs(value);
}

export interface TrainingDurationInput {
  status?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  completedAt?: string | null;
  endedAt?: string | null;
  metrics?: Record<string, unknown> | null;
  /** 运行中任务传入当前时间戳，用于实时计时 */
  nowMs?: number;
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'canceled', 'cancelled', 'succeeded']);
const RUNNING_STATUSES = new Set(['running', 'training', 'pending', 'queued']);

function normalizeStatus(status?: string | null): string {
  return normalizeTrainingBackendStatus(status);
}

export function isTrainingDurationRunning(status?: string | null): boolean {
  return RUNNING_STATUSES.has(normalizeStatus(status));
}

function pickMetricString(metrics: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const raw = metrics[key];
    if (raw == null || raw === '') continue;
    const text = String(raw).trim();
    if (text) return text;
  }
  return null;
}

function pickStartMs(options: TrainingDurationInput, metrics: Record<string, unknown>): number | null {
  const candidates = [
    options.startedAt,
    pickMetricString(metrics, ['startedAt', 'started_at', 'trainStartedAt', 'train_started_at']),
  ];
  for (const value of candidates) {
    const ms = parseTrainingStartMs(value);
    if (ms != null) return ms;
  }
  return null;
}

function pickEndMs(options: TrainingDurationInput, metrics: Record<string, unknown>): number | null {
  const candidates = [
    pickMetricString(metrics, ['updatedAt', 'updated_at', 'finishedAt', 'finished_at']),
    options.finishedAt,
    options.completedAt,
    options.endedAt,
    pickMetricString(metrics, ['completedAt', 'completed_at', 'endedAt', 'ended_at']),
  ];
  for (const value of candidates) {
    const ms = parseTrainingEndMs(value);
    if (ms != null) return ms;
  }
  return null;
}

export function formatTrainingDurationFromMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';

  const totalSeconds = Math.floor(ms / 1000);
  if (totalSeconds <= 0) return '0 秒';

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours >= 1) {
    return minutes > 0 ? `${hours} 小时 ${String(minutes).padStart(2, '0')} 分` : `${hours} 小时`;
  }

  if (minutes >= 1) {
    return `${minutes} 分 ${String(seconds).padStart(2, '0')} 秒`;
  }

  return `${seconds} 秒`;
}

/** 计算训练耗时毫秒；缺少 startedAt 或结束时间时返回 null */
export function resolveTrainingDurationMs(options: TrainingDurationInput): number | null {
  const metrics = options.metrics ?? {};
  const status = normalizeStatus(options.status ?? (metrics.status as string | undefined));

  const startMs = pickStartMs(options, metrics);
  if (startMs == null) return null;

  let endMs: number | null = null;

  if (isTrainingDurationRunning(status)) {
    endMs = options.nowMs ?? Date.now();
  } else if (TERMINAL_STATUSES.has(status)) {
    endMs = pickEndMs(options, metrics);
  } else {
    endMs = pickEndMs(options, metrics);
    if (endMs == null && isTrainingDurationRunning(metrics.status as string | undefined)) {
      endMs = options.nowMs ?? Date.now();
    }
  }

  if (endMs == null) return null;

  const diff = endMs - startMs;
  if (diff < 0) return null;
  return diff;
}

export function resolveTrainingDurationLabel(options: TrainingDurationInput): string {
  const ms = resolveTrainingDurationMs(options);
  if (ms == null) return '—';
  return formatTrainingDurationFromMs(ms);
}

export function buildTrainingDurationInput(options: {
  status?: string | null;
  jobDetail?: {
    startedAt?: string | null;
    finishedAt?: string | null;
    metrics?: Record<string, unknown> | null;
  } | null;
  nowMs?: number;
}): TrainingDurationInput {
  const metrics = options.jobDetail?.metrics ?? null;
  return {
    status: options.status ?? null,
    startedAt: options.jobDetail?.startedAt ?? null,
    finishedAt: options.jobDetail?.finishedAt ?? null,
    completedAt:
      (metrics?.completedAt as string | undefined) ??
      (metrics?.completed_at as string | undefined) ??
      null,
    endedAt:
      (metrics?.endedAt as string | undefined) ??
      (metrics?.ended_at as string | undefined) ??
      null,
    metrics,
    nowMs: options.nowMs,
  };
}

/** 训练任务 / 模型资产详情共用的耗时格式化入口 */
export function formatTrainingDuration(options: {
  status?: string | null;
  jobDetail?: {
    startedAt?: string | null;
    finishedAt?: string | null;
    metrics?: Record<string, unknown> | null;
  } | null;
  nowMs?: number;
}): string {
  return resolveTrainingDurationLabel(buildTrainingDurationInput(options));
}
