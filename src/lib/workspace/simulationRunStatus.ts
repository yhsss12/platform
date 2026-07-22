/** 仿真/运行控制台状态 — 纯逻辑，供 lib 与 UI 共用，避免 lib→component 循环依赖。 */

export type SimRunDisplayStatus = 'queued' | 'running' | 'completed' | 'failed';

export function normalizeSimRunStatus(status: string): SimRunDisplayStatus {
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'queued') return 'queued';
  return 'running';
}

export function runStatusLabel(status: SimRunDisplayStatus): string {
  switch (status) {
    case 'queued':
      return '等待中';
    case 'running':
      return '运行中';
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
  }
}

export function runStatusBadgeStatus(
  status: SimRunDisplayStatus
): 'running' | 'completed' | 'failed' | 'draft' {
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'running') return 'running';
  return 'draft';
}

export function runStatusHint(status: SimRunDisplayStatus): string {
  switch (status) {
    case 'running':
    case 'queued':
      return '仿真任务正在执行，画面将自动刷新。';
    case 'completed':
      return '仿真任务已完成，可查看回放或返回数据中心查看记录。';
    case 'failed':
      return '仿真任务执行失败，请查看日志。';
  }
}
