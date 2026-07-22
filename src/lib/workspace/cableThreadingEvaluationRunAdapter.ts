import type { CreateEvaluationPayload } from '@/components/workspace/evaluation/CreateEvaluationModal';
import type { CableThreadingJobStatusResponse } from '@/lib/api/cableThreadingClient';
import type { ConsoleRunContext } from '@/components/workspace/simulation/SimulationConsoleSections';
import {
  CABLE_THREADING_EVAL_STRATEGY_LABELS,
  CABLE_THREADING_TASK_DISPLAY_NAME,
  CABLE_THREADING_TASK_NAME,
  resolveCableThreadingHasValidLiveFrame,
} from '@/lib/workspace/cableThreading';
import type {
  CurrentSimulation,
  SimulationEventLog,
  SimulationRunStatus,
} from '@/lib/mock/workspaceSimulationMock';
import type { EvaluationMetricsInput } from '@/lib/workspace/evaluationLiveMetrics';
import { evaluationMetricsInputFromCableStatus } from '@/lib/workspace/evaluationLiveMetrics';

const PHASE_TO_STEP: Record<string, string> = {
  init: '环境初始化',
  initialize: '环境初始化',
  approach: '接近线缆末端',
  approach_cable: '接近线缆末端',
  connect: '建立线缆连接',
  grasp: '建立线缆连接',
  thread: '牵引线缆穿过杆间隙',
  threading: '牵引线缆穿过杆间隙',
  pull_through: '牵引线缆穿过杆间隙',
  release: '释放并等待稳定',
  stabilize: '释放并等待稳定',
  success_check: '成功条件判定',
  evaluate: '成功条件判定',
  done: '任务完成',
  complete: '任务完成',
};

export interface CableThreadingEvalSummaryOverride {
  modeTitle?: string;
  taskName: string;
  statusLabel: string;
  progressText: string;
  showProgressBar: boolean;
  progressPercent: number;
  metaItems: { label: string; value: string }[];
}

export interface CableThreadingEvalRunDetailRow {
  label: string;
  value: string;
  mono?: boolean;
}

export interface CableThreadingEvalResultsView {
  successRate: string;
  everSuccessRate: string;
  evalCsvPath?: string;
  resultsJsonPath?: string;
  failuresJsonPath?: string;
  logPath?: string;
}

export interface CableThreadingEvalConsoleViewModel {
  sim: CurrentSimulation;
  runStatus: SimulationRunStatus;
  context: ConsoleRunContext;
  summary: CableThreadingEvalSummaryOverride;
  evaluationMetrics: EvaluationMetricsInput;
  runDetailRows: CableThreadingEvalRunDetailRow[];
  progressRows: CableThreadingEvalRunDetailRow[];
  results: CableThreadingEvalResultsView | null;
  viewport: {
    hasLiveFrame: boolean;
    frameJobId: string;
    frameStatus: 'running' | 'completed' | 'failed';
    waitingMessage: string;
    frameCount: number;
    evalVideoExists: boolean;
    evalVideoStatus: string | null;
    replayHref: string;
    jobStatus: string;
  };
  logEvents: SimulationEventLog[];
  reportHref: string;
  replayHref: string;
  recordsHref: string;
  viewEvaluationEnabled: boolean;
}

function cableModelLabel(model: string): string {
  if (model === 'composite_cable') return '复合线缆模型';
  return model;
}

function formatPercent(rate: number | null | undefined): string {
  if (rate == null || Number.isNaN(rate)) return '—';
  return `${Math.round(rate * 1000) / 10}%`;
}

function mapJobStatusToRunStatus(status: string | undefined): SimulationRunStatus {
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'queued' || status === 'running') return 'running';
  return 'running';
}

function mapStatusLabel(status: string | undefined): string {
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'queued') return '排队中';
  if (status === 'running') return '运行中';
  return '运行中';
}

function resolveStrategyLabel(payload: CreateEvaluationPayload | undefined): string {
  if (payload?.cableThreadingEvalStrategy === 'checkpoint') {
    const asset = payload.cableThreadingCheckpointAssetId;
    return asset ? `已训练模型 ${asset}` : '已训练模型 Robomimic';
  }
  return '专家策略 scripted';
}

function resolveCurrentPhase(live: Record<string, unknown>): string {
  const phase = live.phase;
  if (typeof phase === 'string' && phase.trim()) {
    const mapped = PHASE_TO_STEP[phase.trim().toLowerCase()];
    return mapped ?? phase;
  }
  return '评测执行中';
}

function resolveProgress(
  status: string | undefined,
  live: Record<string, unknown>,
  episodesFallback: number
): { progressText: string; showProgressBar: boolean; progressPercent: number } {
  if (status === 'completed') {
    return { progressText: '评测完成', showProgressBar: true, progressPercent: 100 };
  }
  if (status === 'failed') {
    const partial = Number(live.progressPercent ?? 0);
    if (partial > 0) {
      return { progressText: `评测中断 ${partial}%`, showProgressBar: true, progressPercent: partial };
    }
    return { progressText: '评测失败', showProgressBar: false, progressPercent: 0 };
  }

  const fromLive = Number(live.progressPercent ?? 0);
  if (fromLive > 0) {
    return {
      progressText: `评测进度 ${fromLive}%`,
      showProgressBar: true,
      progressPercent: Math.min(99, fromLive),
    };
  }

  const completed = Number(live.completedEpisodes ?? 0);
  const total = Number(live.episodes ?? episodesFallback);
  if (total > 0 && completed > 0) {
    const percent = Math.min(99, Math.round((completed / total) * 100));
    return {
      progressText: `评测进度 ${percent}%`,
      showProgressBar: true,
      progressPercent: percent,
    };
  }

  return { progressText: '评测运行中', showProgressBar: false, progressPercent: 0 };
}

function logTailToEvents(logTail: string): SimulationEventLog[] {
  if (!logTail.trim()) return [];
  const lines = logTail.split('\n').filter((line) => line.trim());
  return lines.map((line, index) => ({
    id: `ct-eval-log-${index}`,
    time: '—',
    type: '评测',
    content: line.trim(),
    status: /error|fail/i.test(line) ? 'error' : /success|saved_/i.test(line) ? 'success' : 'info',
  }));
}

export function adaptCableThreadingEvalJobToConsoleView(params: {
  evalJobId: string;
  status: CableThreadingJobStatusResponse | null;
  payload?: CreateEvaluationPayload;
  logTail?: string;
}): CableThreadingEvalConsoleViewModel {
  const { evalJobId, status, payload, logTail = '' } = params;
  const live = (status?.live ?? {}) as Record<string, unknown>;
  const jobStatus = status?.status ?? 'running';
  const runStatus = mapJobStatusToRunStatus(jobStatus);
  const episodes = Number(live.episodes ?? payload?.evalRounds ?? 10);
  const horizon = Number(live.horizon ?? payload?.cableThreadingHorizon ?? 600);
  const seed = payload?.seed ?? 0;
  const robot = payload?.cableThreadingRobot ?? 'Panda';
  const cableModel = payload?.cableThreadingCableModel ?? 'composite_cable';
  const difficulty = payload?.cableThreadingDifficulty ?? 'easy';
  const strategyLabel = resolveStrategyLabel(payload);
  const currentPhase = resolveCurrentPhase(live);
  const progress = resolveProgress(jobStatus, live, episodes);
  const evaluationMetrics: EvaluationMetricsInput = evaluationMetricsInputFromCableStatus(status);

  const completedEpisode = Number(live.completedEpisodes ?? live.episode ?? 0);
  const currentEpisodeDisplay =
    jobStatus === 'completed'
      ? String(episodes)
      : completedEpisode > 0
        ? String(Math.min(completedEpisode, episodes))
        : '—';

  const successRateRaw = status?.metrics.successRate;
  const everSuccessRateRaw = status?.metrics.everSuccessRate;

  const results =
    jobStatus === 'completed'
      ? {
          successRate: formatPercent(successRateRaw),
          everSuccessRate: formatPercent(everSuccessRateRaw),
          evalCsvPath: status?.paths.evalCsv?.path,
          resultsJsonPath: status?.paths.resultsJson?.path,
          failuresJsonPath: status?.paths.failuresJson?.path,
          logPath: status?.paths.log?.path,
        }
      : null;

  const hasLiveFrame = resolveCableThreadingHasValidLiveFrame(status);
  const evalVideoExists = Boolean(
    status?.evalVideoExists ?? status?.paths.evalVideo?.exists
  );
  const evalVideoStatus =
    (status?.evalVideoStatus as string | undefined) ??
    (typeof live.evalVideoStatus === 'string' ? live.evalVideoStatus : null);

  const replayHref = `/workspace/replay?replayType=evaluation&taskType=cable_threading&evalId=${encodeURIComponent(evalJobId)}`;

  const viewportState = (() => {
    if (jobStatus === 'failed' && !evalVideoExists && !hasLiveFrame) {
      return {
        waitingMessage:
          '当前评测任务执行失败，未生成可回放视频。请查看右侧失败诊断或评测日志。',
      };
    }
    if (jobStatus === 'completed') {
      if (evalVideoExists || hasLiveFrame) return { waitingMessage: '' };
      return { waitingMessage: '评测已完成，但未生成回放视频。' };
    }
    if (jobStatus === 'failed' && evalVideoExists) {
      return { waitingMessage: '评测任务失败，但已生成部分回放画面。' };
    }
    if (hasLiveFrame) return { waitingMessage: '' };
    if (jobStatus === 'running' || jobStatus === 'queued') {
      return { waitingMessage: '正在初始化 MuJoCo 场景…' };
    }
    return { waitingMessage: '评测任务运行中，等待后端输出评测画面。' };
  })();

  const viewportWaitingMessage = viewportState.waitingMessage;

  const sim: CurrentSimulation = {
    id: evalJobId,
    taskName: CABLE_THREADING_TASK_DISPLAY_NAME,
    scene: '桌面双杆穿线工位',
    robot,
    policy: strategyLabel,
    status: runStatus,
    runDuration: '—',
    progressPercent: progress.progressPercent,
    currentStepLabel: currentPhase,
    engine: 'MuJoCo',
    simTime: '—',
    frame: Number(live.frameCount ?? 0),
    objectsInScene: ['复合线缆', '穿杆工位'],
  };

  const context: ConsoleRunContext = {
    mode: 'evaluation',
    modelVersion: strategyLabel,
    evalRounds: `${episodes} 次`,
    simEnvironment: 'MuJoCo',
  };

  const summary: CableThreadingEvalSummaryOverride = {
    taskName: CABLE_THREADING_TASK_DISPLAY_NAME,
    statusLabel: mapStatusLabel(jobStatus),
    progressText: progress.progressText,
    showProgressBar: progress.showProgressBar,
    progressPercent: progress.progressPercent,
    metaItems: [
      { label: '评测任务', value: CABLE_THREADING_TASK_DISPLAY_NAME },
      { label: '仿真后端', value: 'MuJoCo' },
      { label: '机器人', value: robot },
      { label: '对象模型', value: cableModelLabel(cableModel) },
      { label: '策略', value: strategyLabel },
      { label: '评测轮次', value: `${episodes} 次` },
      { label: '当前阶段', value: currentPhase },
    ],
  };

  const runDetailRows: CableThreadingEvalRunDetailRow[] = [
    { label: '任务名称', value: CABLE_THREADING_TASK_DISPLAY_NAME },
    { label: '评测类型', value: '策略评测' },
    { label: '仿真后端', value: 'MuJoCo' },
    { label: '机器人', value: robot },
    { label: '对象模型', value: cableModelLabel(cableModel) },
    { label: '难度', value: difficulty },
    { label: '策略', value: strategyLabel },
  ];

  const progressRows: CableThreadingEvalRunDetailRow[] = [
    { label: '状态', value: mapStatusLabel(jobStatus) },
    { label: 'Episodes', value: String(episodes) },
    { label: '当前 Episode', value: currentEpisodeDisplay },
    { label: 'Horizon', value: String(horizon) },
    { label: 'Seed', value: String(seed) },
    {
      label: '进度',
      value: progress.showProgressBar ? `${progress.progressPercent}%` : progress.progressText,
    },
  ];

  return {
    sim,
    runStatus,
    context,
    summary,
    evaluationMetrics,
    runDetailRows,
    progressRows,
    results,
    viewport: {
      hasLiveFrame,
      frameJobId: evalJobId,
      frameStatus:
        jobStatus === 'completed' ? 'completed' : jobStatus === 'failed' ? 'failed' : 'running',
      waitingMessage: viewportWaitingMessage,
      frameCount: Number(live.frameCount ?? 0),
      evalVideoExists,
      evalVideoStatus,
      replayHref,
      jobStatus,
    },
    logEvents: logTailToEvents(logTail),
    reportHref: `/workspace/evaluation/report?taskType=cable_threading&evalId=${encodeURIComponent(evalJobId)}`,
    replayHref: `/workspace/replay?replayType=evaluation&taskType=cable_threading&evalId=${encodeURIComponent(evalJobId)}`,
    recordsHref: '/workspace/evaluation',
    viewEvaluationEnabled: jobStatus === 'completed',
  };
}
