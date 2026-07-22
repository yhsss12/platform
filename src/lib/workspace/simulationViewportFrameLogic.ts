import type { SimRunDisplayStatus } from '@/lib/workspace/simulationRunStatus';

export type SimulationViewportFramePhase =
  | 'initializing'
  | 'warming_up'
  | 'waiting'
  | 'live'
  | 'final'
  | 'failed';

export function formatSimulationRoundPart(
  current: string | number | null | undefined,
  total: string | number
): string {
  const currentLabel = current != null && current !== '' ? String(current) : '—';
  return `第 ${currentLabel}/${total} 轮`;
}

export function buildSimulationFrameStatusLine(params: {
  displayStatus: SimRunDisplayStatus;
  episodePart: string;
  framePhase: SimulationViewportFramePhase;
  completedSuffix?: string;
}): { line: string; accent: string } {
  const { displayStatus, episodePart, framePhase, completedSuffix } = params;

  if (displayStatus === 'failed' || framePhase === 'failed') {
    return { line: '画面中断 · 执行失败', accent: '#b91c1c' };
  }

  if (displayStatus === 'completed' || framePhase === 'final') {
    const suffix = completedSuffix ? ` · ${completedSuffix}` : '';
    return { line: `最终帧 · ${episodePart} · 已完成${suffix}`, accent: '#047857' };
  }

  if (framePhase === 'live') {
    return { line: `实时刷新 · ${episodePart} · 运行中`, accent: '#2563eb' };
  }

  if (framePhase === 'warming_up') {
    return { line: '正在初始化 · 画面预热中', accent: '#6b7280' };
  }

  return { line: '正在初始化 · 等待有效画面', accent: '#6b7280' };
}

export function resolveCableThreadingFramePhase(
  displayStatus: SimRunDisplayStatus,
  live: Record<string, unknown>,
  hasLoadedFrame: boolean
): SimulationViewportFramePhase {
  if (displayStatus === 'failed') return 'failed';
  if (displayStatus === 'completed') return hasLoadedFrame ? 'final' : 'waiting';
  if (hasLoadedFrame) return 'live';

  const frameStatus = String(live.frameStatus ?? '');
  if (frameStatus === 'warming_up') return 'warming_up';
  if (frameStatus === 'waiting_valid_frame' || frameStatus === 'waiting') return 'waiting';
  if (live.hasValidFrame === true || Number(live.frameCount ?? 0) > 0) return 'live';
  return 'initializing';
}

export function resolveDualArmFramePhase(
  displayStatus: SimRunDisplayStatus,
  liveFrameExists: boolean | undefined,
  hasLoadedFrame: boolean
): SimulationViewportFramePhase {
  if (displayStatus === 'failed') return 'failed';
  if (displayStatus === 'completed') return hasLoadedFrame || liveFrameExists ? 'final' : 'waiting';
  if (hasLoadedFrame || liveFrameExists) return 'live';
  return 'initializing';
}

export function resolveIsaacBlockStackingFramePhase(
  displayStatus: SimRunDisplayStatus,
  ctx: {
    initWaiting: boolean;
    shouldPollLive: boolean;
    frameLoaded: boolean;
    liveFrameBlack?: boolean;
  }
): SimulationViewportFramePhase {
  if (displayStatus === 'failed' || ctx.liveFrameBlack) return 'failed';
  if (displayStatus === 'completed') return ctx.frameLoaded ? 'final' : 'waiting';
  if (ctx.frameLoaded || ctx.shouldPollLive) return 'live';
  if (ctx.initWaiting) return 'warming_up';
  return 'initializing';
}
