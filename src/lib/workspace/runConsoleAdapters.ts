import type { CableThreadingJobStatusResponse } from '@/lib/api/cableThreadingClient';
import type { DualArmCableJobStatusResponse } from '@/lib/api/dualArmCableClient';
import type { IsaacLabRunJobStatus } from '@/lib/api/isaacLabClient';
import type { IsaacLabFrankaStackCubeJobStatusResponse } from '@/lib/api/isaaclabFrankaStackCubeClient';
import type { NutAssemblyJobStatusResponse } from '@/lib/api/nutAssemblyClient';
import type { CableThreadingGenerateRun } from '@/lib/mock/workspaceMockFlowStore';
import {
  buildSimulationFrameStatusLine,
  formatSimulationRoundPart,
  resolveCableThreadingFramePhase,
  resolveDualArmFramePhase,
  resolveIsaacBlockStackingFramePhase,
  type SimulationViewportFramePhase,
} from '@/components/workspace/simulation/SimulationViewport';
import {
  normalizeSimRunStatus,
  type SimRunDisplayStatus,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import {
  CABLE_THREADING_TASK_DISPLAY_NAME,
} from '@/lib/workspace/cableThreading';
import {
  DUAL_ARM_CABLE_DEFAULTS,
  DUAL_ARM_CABLE_TASK_NAME,
  formatDualArmMetric,
  releaseModeLabel,
  stretchModeLabel,
} from '@/lib/workspace/dualArmCable';
import {
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
} from '@/lib/workspace/isaacBlockStacking';
import {
  ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME,
} from '@/lib/workspace/isaaclabFrankaStackCube';
import {
  NUT_ASSEMBLY_TASK_DISPLAY_NAME,
  buildNutAssemblyReplayHref,
  buildNutAssemblyVideoApiPath,
  formatNutAssemblyGenerationMode,
  formatNutAssemblyPolicyMode,
} from '@/lib/workspace/nutAssembly';
import {
  isValidCableThreadingGenerateJobId,
  isValidDualArmGenerateJobId,
  isValidIsaacGenerateJobId,
  isValidIsaacReplayJobId,
  isValidNutAssemblyGenerateJobId,
} from '@/lib/workspace/backendJobIds';
import {
  runConsoleFileReadyLabel,
  runConsoleResultStatusLabel,
  type RunConsoleDisplayStatus,
  type RunConsoleInfoRow,
  type RunConsoleViewModel,
} from '@/lib/workspace/runConsoleViewModel';

export type RunConsoleKind =
  | 'cable_threading'
  | 'dual_arm_cable'
  | 'isaac_block_stacking'
  | 'nut_assembly'
  | 'evaluation';

export function resolveRunConsoleKind(
  jobId: string | null | undefined,
  taskType?: string | null
): RunConsoleKind | null {
  if (!jobId) return null;
  if (jobId.startsWith('na_gen_') || isValidNutAssemblyGenerateJobId(jobId)) {
    return 'nut_assembly';
  }
  const normalizedTask = (taskType ?? '').trim();
  if (
    normalizedTask === 'nut_assembly' ||
    normalizedTask === 'nut_assembly_single_arm' ||
    normalizedTask === 'task_nut_assembly_v1'
  ) {
    return 'nut_assembly';
  }
  if (jobId.startsWith('ct_gen_') || isValidCableThreadingGenerateJobId(jobId)) {
    return 'cable_threading';
  }
  if (jobId.startsWith('dac_gen_') || isValidDualArmGenerateJobId(jobId)) {
    return 'dual_arm_cable';
  }
  if (
    jobId.startsWith('isaac_gen_') ||
    jobId.startsWith('isaac_replay_') ||
    isValidIsaacGenerateJobId(jobId) ||
    isValidIsaacReplayJobId(jobId)
  ) {
    return 'isaac_block_stacking';
  }
  if (jobId.startsWith('eval_') || jobId.startsWith('isaac_eval_') || jobId.startsWith('ct_eval_')) {
    return 'evaluation';
  }
  if (normalizedTask === 'cable_threading' || normalizedTask === 'task_cable_threading_v1') {
    return 'cable_threading';
  }
  if (normalizedTask === 'dual_arm_cable_manipulation' || normalizedTask === 'task_dual_arm_cable_manipulation_v1') {
    return 'dual_arm_cable';
  }
  if (normalizedTask === 'isaac_block_stacking' || normalizedTask === 'task_isaac_block_stacking_v1') {
    return 'isaac_block_stacking';
  }
  return null;
}

export function getRunConsoleKindDisplayName(kind: RunConsoleKind | null): string {
  switch (kind) {
    case 'nut_assembly':
      return NUT_ASSEMBLY_TASK_DISPLAY_NAME;
    case 'dual_arm_cable':
      return DUAL_ARM_CABLE_TASK_NAME;
    case 'isaac_block_stacking':
      return ISAAC_BLOCK_STACKING_DISPLAY_NAME;
    case 'cable_threading':
      return CABLE_THREADING_TASK_DISPLAY_NAME;
    default:
      return '任务';
  }
}

export function isValidRunConsoleJobId(kind: RunConsoleKind, jobId: string): boolean {
  switch (kind) {
    case 'nut_assembly':
      return isValidNutAssemblyGenerateJobId(jobId);
    case 'cable_threading':
      return isValidCableThreadingGenerateJobId(jobId);
    case 'dual_arm_cable':
      return isValidDualArmGenerateJobId(jobId);
    case 'isaac_block_stacking':
      return isValidIsaacGenerateJobId(jobId) || isValidIsaacReplayJobId(jobId);
    default:
      return false;
  }
}

function cablePhaseLabel(phase: unknown): string {
  if (phase == null || phase === '') return '—';
  const map: Record<string, string> = {
    rollout: '轨迹采集',
    encoding: '视频合成',
    completed: '已完成',
    failed: '失败',
  };
  return map[String(phase)] ?? String(phase);
}

function dualArmPhaseLabel(phase?: string | null): string {
  const map: Record<string, string> = {
    queued: '排队中',
    initializing: '初始化',
    scene_ready: '场景就绪',
    perception: '视觉感知',
    grasp_planning: '抓取规划',
    manipulation: '双臂操控',
    recording: '视频生成',
    completed: '已完成',
    failed: '失败',
  };
  return phase ? map[phase] ?? phase : '—';
}

function isaacPhaseLabel(phase?: string | null): string {
  const map: Record<string, string> = {
    queued: '排队中',
    annotate: '初始化',
    generate: '生成数据',
    done: '登记数据集',
    postprocess: '后处理',
    register_dataset: '登记数据集',
    replay_preview: '回放预览',
    completed: '已完成',
    failed: '失败',
  };
  if (!phase) return '—';
  return map[phase] ?? phase;
}

function isaacSeedSourceLabel(source?: string | null): string {
  const map: Record<string, string> = {
    default_seed: '平台默认 Seed',
    dataset_registry: '数据集 Registry',
    manual_path: '手动指定路径',
  };
  if (!source) return '—';
  return map[source] ?? source;
}

function isaacVisualModeLabel(mode?: string | null): string {
  const map: Record<string, string> = {
    single_env: '单环境预览',
    parallel_overview: '并行环境总览',
    replay_preview: '单环境回放预览',
  };
  if (!mode) return '—';
  return map[mode] ?? mode;
}

function computeCableProgress(
  displayStatus: RunConsoleDisplayStatus,
  live: Record<string, unknown>,
  payload: CableThreadingGenerateRun['payload'] | undefined,
  hasLiveStatus: boolean,
  finalSuccessRate?: number | null
): number {
  if (displayStatus === 'completed') return 100;
  if (live.finalSuccessRate != null || finalSuccessRate != null) return 100;
  if (!hasLiveStatus) return 0;

  const episodes = Number(live.episodes ?? payload?.episodes ?? 0);
  const horizon = Number(live.horizon ?? payload?.cableThreadingHorizon ?? 600);
  const totalSteps = episodes * horizon;
  if (totalSteps <= 0) return 0;

  const completedEpisodeCount = Math.max(0, Number(live.episode ?? 0));
  const currentStep = Math.max(0, Number(live.step ?? 0));
  const completedSteps = completedEpisodeCount * horizon + currentStep;
  const rollout = Math.min(100, Math.round((completedSteps / totalSteps) * 100));
  if (displayStatus === 'failed') return rollout;
  if (displayStatus === 'running') return Math.min(rollout, 99);
  return rollout;
}

const DUAL_ARM_PHASE_PROGRESS: Record<string, number> = {
  queued: 5,
  initializing: 10,
  scene_ready: 15,
  perception: 25,
  grasp_planning: 40,
  manipulation: 60,
  recording: 85,
  completed: 100,
  failed: 0,
};

function computeDualArmProgress(
  status: string,
  phase: string | null | undefined,
  progress: number | null | undefined
): number {
  if (status === 'completed') return 100;
  if (typeof progress === 'number' && Number.isFinite(progress)) {
    const pct = progress <= 1 ? Math.round(progress * 100) : Math.round(progress);
    return Math.max(0, Math.min(99, pct));
  }
  if (status === 'failed') return DUAL_ARM_PHASE_PROGRESS[phase ?? ''] ?? 0;
  return DUAL_ARM_PHASE_PROGRESS[phase ?? ''] ?? 5;
}

function computeIsaacProgress(status: RunConsoleDisplayStatus, jobStatus: IsaacLabRunJobStatus | null): number {
  if (status === 'completed') return 100;
  if (status === 'failed') return 0;
  if (status === 'queued') return 5;
  const phase = jobStatus?.phase ?? '';
  if (phase === 'annotate') return 25;
  if (phase === 'generate') return 55;
  if (phase === 'replay_preview') return 85;
  if (phase === 'done' || phase === 'register_dataset') return 95;
  return 15;
}

function buildSceneStatus(
  displayStatus: RunConsoleDisplayStatus,
  episodePart: string | null,
  framePhase: SimulationViewportFramePhase,
  completedSuffix?: string
): { line: string; accent: string } {
  if (!episodePart) {
    if (displayStatus === 'failed' || framePhase === 'failed') {
      return { line: '画面中断 · 执行失败', accent: '#b91c1c' };
    }
    if (displayStatus === 'completed' || framePhase === 'final') {
      const suffix = completedSuffix ? ` · ${completedSuffix}` : '';
      return { line: `最终帧 · 已完成${suffix}`, accent: '#047857' };
    }
    if (framePhase === 'live') {
      return { line: '实时刷新 · 运行中', accent: '#2563eb' };
    }
    if (framePhase === 'warming_up') {
      return { line: '正在初始化 · 画面预热中', accent: '#6b7280' };
    }
    return { line: '正在初始化 · 等待有效画面', accent: '#6b7280' };
  }

  return buildSimulationFrameStatusLine({
    displayStatus,
    episodePart,
    framePhase,
    completedSuffix,
  });
}

function isIsaacInitPhase(
  jobStatus: IsaacLabRunJobStatus | null,
  displayStatus: RunConsoleDisplayStatus,
  frameLoaded: boolean
): boolean {
  if (displayStatus === 'failed' || displayStatus === 'completed') return false;
  if (frameLoaded) return false;
  const phase = jobStatus?.phase ?? '';
  if (displayStatus === 'queued') return true;
  if (phase === 'annotate') return true;
  if (
    displayStatus === 'running' &&
    phase !== 'generate' &&
    jobStatus?.liveFrameAvailable !== true &&
    (jobStatus?.visualPhase === 'none' || !jobStatus?.visualPhase)
  ) {
    return true;
  }
  return false;
}

export function buildCableThreadingRunConsoleViewModel(input: {
  jobId: string;
  backendJobId: string;
  dataId?: string;
  localRunId?: string;
  jobStatus: CableThreadingJobStatusResponse | null;
  run?: CableThreadingGenerateRun | null;
  frameLoaded: boolean;
  canViewReplay: boolean;
}): RunConsoleViewModel {
  const { jobId, backendJobId, dataId, localRunId, jobStatus, run, frameLoaded, canViewReplay } = input;
  const payload = run?.payload;
  const result = run?.result;
  const live = (jobStatus?.live ?? {}) as Record<string, unknown>;
  const displayStatus = normalizeSimRunStatus(jobStatus?.status ?? run?.status ?? 'running');
  const hasLiveStatus = jobStatus !== null;

  const episodeDisplay = live.episode != null ? Number(live.episode) + 1 : null;
  const episodesTotal = String(live.episodes ?? payload?.episodes ?? '—');
  const successfulEpisodes =
    (live.successfulEpisodes as number | undefined) ??
    result?.successfulEpisodes ??
    jobStatus?.metrics?.successfulEpisodes;

  const generateVideoStatus = String(result?.generateVideoStatus ?? live.generateVideoStatus ?? '');
  const generateVideoExists =
    result?.generateVideoExists === true ||
    live.generateVideoExists === true ||
    jobStatus?.generateVideoExists === true ||
    jobStatus?.paths.generateVideo?.exists === true;
  const processVideoEnabled = payload?.cableThreadingSaveProcessVideo !== false;

  const progress = computeCableProgress(
    displayStatus,
    live,
    payload,
    hasLiveStatus,
    jobStatus?.metrics?.finalSuccessRate
  );

  const episodePart = formatSimulationRoundPart(episodeDisplay, episodesTotal);
  const framePhase = resolveCableThreadingFramePhase(displayStatus, live, frameLoaded);
  const completedSuffix =
    successfulEpisodes != null ? `成功 ${successfulEpisodes}/${episodesTotal}` : undefined;
  const frameStatus = buildSceneStatus(displayStatus, episodePart, framePhase, completedSuffix);

  const successRateLabel =
    displayStatus === 'completed' && result?.successRate != null
      ? `${result.successRate}%`
      : displayStatus === 'completed' && jobStatus?.metrics?.finalSuccessRate != null
        ? `${Math.round(Number(jobStatus.metrics.finalSuccessRate) * 100)}%`
        : '—';

  const datasetReady =
    jobStatus?.paths.npz?.exists ||
    jobStatus?.paths.hdf5?.exists ||
    Boolean(result?.npzPath || result?.hdf5Path);

  const jobRoot = `runs/cable_threading/jobs/${backendJobId}`;

  const viewportMode =
    displayStatus === 'failed' ? 'failed' : framePhase === 'live' || framePhase === 'final' ? 'live' : 'init';

  return {
    jobId: backendJobId,
    taskName: CABLE_THREADING_TASK_DISPLAY_NAME,
    taskTypeLabel: '数据生成',
    taskKindLabel: 'cable_threading',
    simulatorBackend: 'MuJoCo',
    status: displayStatus,
    progress,
    scene: {
      title: '仿真画面',
      backendLabel: 'MuJoCo',
      initializingText: '正在初始化 MuJoCo 场景。',
      frameStatusLine: frameStatus.line,
      frameStatusAccent: frameStatus.accent,
      viewportMode,
      failedMessage: '仿真画面不可用，请查看日志。',
      liveFrame:
        displayStatus === 'failed'
          ? undefined
          : {
              kind: 'cable_threading',
              jobId: backendJobId,
              pollEnabled: true,
              status: displayStatus === 'completed' ? 'completed' : 'running',
              frameCount: Number(live.frameCount ?? 0),
            },
    },
    sections: {
      basicInfo: [
        { label: '任务名称', value: CABLE_THREADING_TASK_DISPLAY_NAME },
        { label: '后端标识', value: 'cable_threading' },
        { label: '仿真后端', value: 'MuJoCo' },
        { label: '随机种子', value: String(payload?.seed ?? 0) },
        { label: '当前阶段', value: cablePhaseLabel(live.phase) },
      ],
      assetConfig: [
        { label: '机器人', value: payload?.cableThreadingRobot ?? 'Panda' },
        { label: '线缆模型', value: payload?.cableThreadingCableModel ?? 'composite_cable' },
        { label: '难度', value: payload?.cableThreadingDifficulty ?? 'easy' },
        { label: '采集轮次', value: String(payload?.episodes ?? '—') },
        { label: '最大步数', value: String(payload?.cableThreadingHorizon ?? 600) },
        {
          label: '数据格式',
          value: payload?.dataFormat ?? (payload?.cableThreadingSaveHdf5 ? 'hdf5' : 'npz'),
        },
      ],
      results: [
        { label: '状态', value: runConsoleResultStatusLabel(displayStatus) },
        { label: '成功率', value: successRateLabel },
        {
          label: '成功轨迹',
          value: successfulEpisodes != null ? `${successfulEpisodes} / ${episodesTotal}` : '—',
        },
        {
          label: '结果文件',
          value: runConsoleFileReadyLabel(datasetReady, displayStatus === 'running'),
        },
        {
          label: '视频文件',
          value: !processVideoEnabled
            ? '未启用'
            : generateVideoExists
              ? '已生成'
              : generateVideoStatus === 'encoding'
                ? '合成中'
                : runConsoleFileReadyLabel(false, displayStatus === 'running'),
        },
        {
          label: '日志文件',
          value: displayStatus === 'queued' ? '等待生成' : '已生成',
        },
      ],
      debug: [
        { label: 'jobId', value: backendJobId },
        ...(localRunId ? [{ label: '本地运行 ID', value: localRunId }] : []),
        ...(dataId ? [{ label: 'dataId', value: dataId }] : []),
        { label: 'runtimePath', value: jobRoot },
        { label: 'status.json', value: `${jobRoot}/status.json` },
        { label: 'live_status.json', value: jobStatus?.paths.liveStatus?.path ?? `${jobRoot}/live_status.json` },
        { label: 'generate.mp4', value: result?.generateVideoPath ?? jobStatus?.generateVideoPath ?? `${jobRoot}/videos/generate.mp4` },
        { label: 'run.log', value: result?.logPath ?? jobStatus?.paths.log?.path ?? `${jobRoot}/logs/run.log` },
        { label: 'npz', value: result?.npzPath ?? jobStatus?.paths.npz?.path ?? '—' },
        { label: 'hdf5', value: result?.hdf5Path ?? jobStatus?.paths.hdf5?.path ?? '—' },
        { label: 'manifest', value: result?.manifestPath ?? jobStatus?.paths.manifest?.path ?? '—' },
        { label: 'frameCount', value: String(live.frameCount ?? '—') },
        { label: 'frame API', value: `/api/workspace/cable-threading/jobs/${backendJobId}/frame` },
        ...(run?.errorMessage ? [{ label: 'error', value: run.errorMessage }] : []),
      ],
    },
    actions: {
      backToDataCenterHref: '/workspace/data',
      canViewReplay,
      showViewDataRecord: displayStatus === 'completed',
    },
  };
}

export function buildDualArmCableRunConsoleViewModel(input: {
  jobId: string;
  dataId?: string;
  jobStatus: DualArmCableJobStatusResponse | null;
  payload?: CableThreadingGenerateRun['payload'];
  frameLoaded: boolean;
  canViewReplay: boolean;
}): RunConsoleViewModel {
  const { jobId, dataId, jobStatus, payload, frameLoaded, canViewReplay } = input;
  const displayStatus = normalizeSimRunStatus(jobStatus?.status ?? 'running');
  const phase = jobStatus?.phase;
  const maxCables = jobStatus?.maxCables ?? payload?.dualArmMaxCables ?? 1;
  const succeededCables = jobStatus?.succeededCables;
  const episodeSuccessValue =
    displayStatus === 'completed' || displayStatus === 'failed'
      ? formatDualArmMetric(jobStatus?.episodeSuccess)
      : '—';

  const progress = computeDualArmProgress(displayStatus, phase, jobStatus?.progress);
  const framePhase = resolveDualArmFramePhase(displayStatus, jobStatus?.liveFrameExists, frameLoaded);
  const episodePart = formatSimulationRoundPart(1, maxCables);
  const completedSuffix =
    succeededCables != null
      ? `成功 ${succeededCables}/${maxCables}`
      : episodeSuccessValue !== '—'
        ? episodeSuccessValue
        : undefined;
  const frameStatus = buildSceneStatus(displayStatus, episodePart, framePhase, completedSuffix);

  const jobRoot = jobStatus?.runtimePath?.replace(/\/$/, '');
  const viewportMode =
    displayStatus === 'failed' ? 'failed' : framePhase === 'live' || framePhase === 'final' ? 'live' : 'init';

  return {
    jobId,
    taskName: DUAL_ARM_CABLE_TASK_NAME,
    taskTypeLabel: '数据生成',
    taskKindLabel: 'dual_arm_cable_manipulation',
    simulatorBackend: 'MuJoCo',
    status: displayStatus,
    progress,
    scene: {
      title: '仿真画面',
      backendLabel: 'MuJoCo',
      initializingText: '正在初始化 MuJoCo 场景。',
      frameStatusLine: frameStatus.line,
      frameStatusAccent: frameStatus.accent,
      viewportMode,
      failedMessage: '仿真画面不可用，请查看日志。',
      liveFrame:
        displayStatus === 'failed'
          ? undefined
          : {
              kind: 'dual_arm_cable',
              jobId,
              pollEnabled: true,
              status: displayStatus === 'queued' ? 'queued' : displayStatus,
              phase,
            },
    },
    sections: {
      basicInfo: [
        { label: '任务名称', value: DUAL_ARM_CABLE_TASK_NAME },
        { label: '后端标识', value: 'dual_arm_cable_manipulation' },
        { label: '仿真后端', value: 'MuJoCo' },
        { label: '随机种子', value: String(payload?.seed ?? 42) },
        { label: '当前阶段', value: dualArmPhaseLabel(phase) },
      ],
      assetConfig: [
        { label: '机器人', value: DUAL_ARM_CABLE_DEFAULTS.robot },
        { label: '夹爪', value: DUAL_ARM_CABLE_DEFAULTS.endEffector },
        { label: '操作对象', value: '杂乱柔性线缆' },
        { label: '线缆数量', value: String(payload?.dualArmMaxCables ?? maxCables) },
        { label: '拉伸模式', value: stretchModeLabel(payload?.dualArmStretchMode) },
        { label: '释放策略', value: releaseModeLabel(payload?.dualArmReleaseMode) },
      ],
      results: [
        { label: '状态', value: runConsoleResultStatusLabel(displayStatus) },
        { label: '成功', value: episodeSuccessValue },
        {
          label: '成功线缆',
          value: succeededCables != null ? `${succeededCables} / ${maxCables}` : '—',
        },
        {
          label: '结果文件',
          value: runConsoleFileReadyLabel(Boolean(jobStatus?.resultPath), displayStatus === 'running'),
        },
        {
          label: '视频文件',
          value: jobStatus?.videoExists
            ? '已生成'
            : jobStatus?.phase === 'recording'
              ? '合成中'
              : runConsoleFileReadyLabel(false, displayStatus === 'running'),
        },
        {
          label: '日志文件',
          value: displayStatus === 'queued' ? '等待生成' : '已生成',
        },
      ],
      debug: [
        { label: 'jobId', value: jobId },
        ...(dataId ? [{ label: 'dataId', value: dataId }] : []),
        { label: 'runtimePath', value: jobRoot ?? '等待后端分配' },
        { label: 'status.json', value: jobRoot ? `${jobRoot}/status.json` : '—' },
        { label: 'episode_result.json', value: jobStatus?.resultPath ?? (jobRoot ? `${jobRoot}/results/episode_result.json` : '—') },
        { label: 'generate.mp4', value: jobStatus?.videoPath ?? (jobRoot ? `${jobRoot}/videos/generate.mp4` : '—') },
        { label: 'run.log', value: jobStatus?.logPath ?? (jobRoot ? `${jobRoot}/logs/run.log` : '—') },
        { label: 'latest_grasp.json', value: jobRoot ? `${jobRoot}/results/steps/step_00/grasp_output/latest_grasp.json` : '—' },
        { label: 'frame API', value: `/api/workspace/dual-arm-cable/jobs/${jobId}/frame` },
      ],
    },
    actions: {
      backToDataCenterHref: '/workspace/data',
      canViewReplay,
      showViewDataRecord: displayStatus === 'completed',
    },
  };
}

export function buildIsaacBlockStackingRunConsoleViewModel(input: {
  jobId: string;
  jobStatus: IsaacLabRunJobStatus | null;
  frameLoaded: boolean;
  canViewReplay: boolean;
}): RunConsoleViewModel {
  const { jobId, jobStatus, frameLoaded, canViewReplay } = input;
  const displayStatus = normalizeSimRunStatus(jobStatus?.status ?? 'queued');
  const enableCameras = jobStatus?.enableCameras !== false;
  const liveFrameAvailable = jobStatus?.liveFrameAvailable === true;
  const liveFrameBlack = jobStatus?.liveFrameBlack === true;
  const numDemos = jobStatus?.numDemos;
  const currentDemo =
    typeof jobStatus?.stackEnvMatches === 'number' && jobStatus.stackEnvMatches > 0
      ? jobStatus.stackEnvMatches
      : null;

  const progress = computeIsaacProgress(displayStatus, jobStatus);
  const initWaiting = isIsaacInitPhase(jobStatus, displayStatus, frameLoaded);
  const shouldPollLive = enableCameras && liveFrameAvailable && !liveFrameBlack && displayStatus !== 'failed';

  const framePhase = resolveIsaacBlockStackingFramePhase(
    displayStatus,
    { initWaiting, shouldPollLive, frameLoaded, liveFrameBlack },
  );

  const episodePart =
    numDemos != null && currentDemo != null
      ? formatSimulationRoundPart(currentDemo, numDemos)
      : numDemos != null && displayStatus === 'running'
        ? formatSimulationRoundPart('—', numDemos)
        : null;

  const successfulTrajectories =
    displayStatus === 'completed' && numDemos != null
      ? jobStatus?.datasetAvailable
        ? `${numDemos} / ${numDemos}`
        : '—'
      : '—';

  const frameStatus = buildSceneStatus(displayStatus, episodePart, framePhase);

  const artifacts = jobStatus?.artifactStatus;
  const paths = jobStatus?.paths ?? {};
  const datasetReady = Boolean(jobStatus?.datasetAvailable ?? artifacts?.datasetHdf5);
  const previewReady = Boolean(jobStatus?.previewVideoAvailable || jobStatus?.videoAvailable);

  let viewportMode: RunConsoleViewModel['scene']['viewportMode'] = 'init';
  if (displayStatus === 'failed') viewportMode = 'failed';
  else if (!enableCameras) viewportMode = 'cameras_disabled';
  else if (shouldPollLive || frameLoaded) viewportMode = 'live';
  else if (initWaiting) viewportMode = 'init';

  return {
    jobId,
    taskName: ISAAC_BLOCK_STACKING_DISPLAY_NAME,
    taskTypeLabel: '数据生成',
    taskKindLabel: 'isaac_block_stacking',
    simulatorBackend: 'Isaac Lab',
    status: displayStatus,
    progress,
    scene: {
      title: '仿真画面',
      backendLabel: 'Isaac Lab',
      initializingText: '正在初始化 Isaac Lab 场景。',
      frameStatusLine: frameStatus.line,
      frameStatusAccent: frameStatus.accent,
      viewportMode,
      failedMessage: '仿真画面不可用，请查看日志。',
      liveFrame: shouldPollLive
          ? {
              kind: 'isaac_lab',
              jobId,
              pollEnabled: true,
              status: displayStatus,
            }
          : undefined,
    },
    sections: {
      basicInfo: [
        { label: '任务名称', value: ISAAC_BLOCK_STACKING_DISPLAY_NAME },
        { label: '后端标识', value: 'isaac_block_stacking' },
        { label: '仿真后端', value: 'Isaac Lab' },
        { label: '随机种子', value: '—' },
        { label: '当前阶段', value: isaacPhaseLabel(jobStatus?.phase) },
      ],
      assetConfig: [
        { label: '机器人', value: 'Franka Panda' },
        { label: '场景', value: 'Stack Cube' },
        { label: '数据格式', value: 'HDF5' },
        { label: '采集轮次', value: String(numDemos ?? '—') },
        { label: '生成方式', value: 'Mimic 自动生成' },
      ],
      results: [
        { label: '状态', value: runConsoleResultStatusLabel(displayStatus) },
        { label: '成功轨迹', value: successfulTrajectories },
        {
          label: '结果文件',
          value: runConsoleFileReadyLabel(datasetReady, displayStatus === 'running'),
        },
        {
          label: '视频文件',
          value: runConsoleFileReadyLabel(previewReady, displayStatus === 'running'),
        },
        {
          label: '日志文件',
          value: displayStatus === 'queued' ? '等待生成' : '已生成',
        },
      ],
      debug: [
        { label: 'jobId', value: jobId },
        { label: 'taskId', value: jobStatus?.taskId ?? '—' },
        { label: 'generationMode', value: jobStatus?.generationMode ?? 'mimic_auto' },
        { label: 'enableCameras', value: enableCameras ? 'true' : 'false' },
        { label: 'visualPhase', value: jobStatus?.visualPhase ?? 'none' },
        { label: 'visualMode', value: isaacVisualModeLabel(jobStatus?.visualMode) },
        { label: 'visualEnvIndex', value: String(jobStatus?.visualEnvIndex ?? 0) },
        { label: 'parallelNumEnvs', value: String(jobStatus?.parallelNumEnvs ?? numDemos ?? '—') },
        { label: 'seedSource', value: isaacSeedSourceLabel(jobStatus?.seedSource) },
        { label: 'seed.hdf5', value: runConsoleFileReadyLabel(artifacts?.seedHdf5, displayStatus === 'running') },
        { label: 'annotated.hdf5', value: runConsoleFileReadyLabel(artifacts?.annotatedHdf5, displayStatus === 'running') },
        { label: 'dataset.hdf5', value: runConsoleFileReadyLabel(datasetReady, displayStatus === 'running') },
        { label: 'generation_manifest.json', value: runConsoleFileReadyLabel(artifacts?.generationManifest, displayStatus === 'running') },
        { label: 'metrics.json', value: runConsoleFileReadyLabel(artifacts?.metricsJson, displayStatus === 'running') },
        { label: 'preview.mp4', value: runConsoleFileReadyLabel(previewReady, displayStatus === 'running') },
        { label: 'datasetId', value: jobStatus?.datasetId ?? '—' },
        { label: 'datasetFile', value: jobStatus?.datasetFile ?? paths.datasetHdf5 ?? '—' },
        { label: 'runtimeRoot', value: paths.jobRoot ?? '—' },
        { label: 'stdoutLog', value: paths.stdoutLog ?? '—' },
        { label: 'stderrLog', value: paths.stderrLog ?? '—' },
        { label: 'previewVideo', value: paths.previewVideo ?? '—' },
        { label: 'message', value: jobStatus?.message ?? '—' },
        ...(jobStatus?.exitCode != null ? [{ label: 'exitCode', value: String(jobStatus.exitCode) }] : []),
      ],
    },
    actions: {
      backToDataCenterHref: '/workspace/data',
      canViewReplay,
      showViewDataRecord: displayStatus === 'completed' && Boolean(jobStatus?.datasetAvailable && jobStatus?.datasetId),
    },
  };
}


export function nutAssemblyArtifactStageLabel(stage: string): string {
  const map: Record<string, string> = {
    prepare_source: '准备 source demo',
    mimicgen_generate: 'MimicGen 生成中',
    write_manifest: '写入 manifest',
    write_summary: '写入 summary',
    robosuite_rollout: 'robosuite rollout',
    completed: '已完成',
    failed: '失败',
    stalled: '失败',
    queued: '排队中',
  };
  return (map[stage] ?? stage) || '—';
}

function nutAssemblyArtifactRunningLabel(value: unknown, running: boolean): string {
  if (value === true) return '是';
  if (value === false) return '否';
  if (running) return '运行中 / 待生成';
  return '—';
}

export function buildNutAssemblyRunConsoleViewModel(input: {
  jobId: string;
  jobStatus: NutAssemblyJobStatusResponse | null;
  jobResult?: Record<string, unknown> | null;
  logTail?: string;
}): RunConsoleViewModel {
  const { jobId, jobStatus, jobResult, logTail = '' } = input;
  const live = (jobStatus?.live ?? {}) as Record<string, unknown>;
  const summary = (jobResult?.summary ?? jobStatus?.metrics?.summary ?? {}) as Record<string, unknown>;
  const displayStatus = normalizeSimRunStatus(jobStatus?.status ?? 'running');
  const generationMode = String(jobStatus?.generationMode ?? live.generationMode ?? summary.generationMode ?? '—');
  const episodesRequested = Number(summary.episodesRequested ?? live.episodesRequested ?? live.episodes ?? 0);
  const episodesGenerated = Number(summary.episodesGenerated ?? live.episodesGenerated ?? live.episode ?? 0);
  const successEpisodes = Number(summary.successEpisodes ?? jobStatus?.metrics?.successEpisodes ?? 0);
  const successRateRaw = summary.successRate ?? jobStatus?.successRate ?? live.successRate;
  const successRateLabel =
    successRateRaw != null && successRateRaw !== ''
      ? `${Math.round(Number(successRateRaw) * 1000) / 10}%`
      : '—';
  const videoFileExists = Boolean(jobStatus?.generateVideoExists ?? jobStatus?.paths?.generateVideo?.exists);
  const videoUrl = jobStatus?.videoUrl ?? (videoFileExists ? buildNutAssemblyVideoApiPath(jobId) : null);
  const replayHref = buildNutAssemblyReplayHref({ jobId });
  const progress =
    displayStatus === 'completed'
      ? 100
      : episodesRequested > 0
        ? Math.min(99, Math.round((episodesGenerated / episodesRequested) * 100))
        : 0;
  const episodePart = formatSimulationRoundPart(
    episodesGenerated > 0 ? episodesGenerated : null,
    episodesRequested || '—'
  );
  const frameStatus = buildSceneStatus(displayStatus, episodePart, 'warming_up');
  return {
    jobId,
    taskName: NUT_ASSEMBLY_TASK_DISPLAY_NAME,
    taskTypeLabel: '数据生成',
    taskKindLabel: 'Nut Assembly',
    simulatorBackend: 'MuJoCo',
    status: displayStatus,
    progress,
    scene: {
      title: '生成任务数据',
      backendLabel: formatNutAssemblyGenerationMode(generationMode),
      initializingText: '正在进行 NutAssembly 数据生成…',
      frameStatusLine: frameStatus.line,
      frameStatusAccent: frameStatus.accent,
      viewportMode: videoUrl ? 'video' : displayStatus === 'completed' ? 'hdf5_replay' : 'init',
      previewVideoApiPath: videoUrl,
      hdf5ReplayHref: displayStatus === 'completed' ? replayHref : null,
    },
    sections: {
      basicInfo: [
        { label: 'Job ID', value: jobId },
        { label: '生成模式', value: formatNutAssemblyGenerationMode(generationMode) },
        { label: '策略模式', value: formatNutAssemblyPolicyMode(String(jobStatus?.policyMode ?? live.policyMode ?? '—')) },
        { label: '请求轮数', value: String(episodesRequested || '—') },
        { label: '已生成轮数', value: String(episodesGenerated || '—') },
      ],
      assetConfig: [{ label: '随机种子', value: String(live.seed ?? summary.seed ?? '—') }],
      results: [
        { label: '成功轮数', value: String(successEpisodes) },
        { label: '成功率', value: successRateLabel },
        { label: '状态', value: runConsoleResultStatusLabel(displayStatus) },
      ],
      debug: [{ label: 'logTail', value: logTail.slice(-200) || '—' }],
    },
    actions: {
      backToDataCenterHref: '/workspace/data',
      canViewReplay: displayStatus === 'completed' || displayStatus === 'failed',
      showViewDataRecord: displayStatus === 'completed',
    },
  };
}

export function buildIsaacLabFrankaStackCubeRunConsoleViewModel(input: {
  jobId: string;
  jobStatus: IsaacLabFrankaStackCubeJobStatusResponse | null;
  frameLoaded: boolean;
  canViewReplay: boolean;
}): RunConsoleViewModel {
  const { jobId, jobStatus, frameLoaded, canViewReplay } = input;
  const displayStatus = normalizeSimRunStatus(jobStatus?.status ?? 'queued');
  const total = jobStatus?.totalEpisodes ?? null;
  const completed = jobStatus?.completedEpisodes ?? null;
  const enableCameras = jobStatus?.enableCameras !== false;
  const liveFrameAvailable = jobStatus?.liveFrameAvailable === true;
  const liveFrameBlack = jobStatus?.liveFrameBlack === true;
  const shouldPollLive =
    enableCameras && liveFrameAvailable && !liveFrameBlack && displayStatus !== 'failed';
  const framePhase = resolveIsaacBlockStackingFramePhase(displayStatus, {
    initWaiting: displayStatus === 'running' && !frameLoaded,
    shouldPollLive,
    frameLoaded,
    liveFrameBlack,
  });
  const episodePart = formatSimulationRoundPart(
    completed != null && completed > 0 ? completed : null,
    total ?? '—'
  );
  const frameStatus = buildSceneStatus(displayStatus, episodePart, framePhase);
  const progress =
    typeof jobStatus?.progress === 'number'
      ? Math.max(0, Math.min(100, jobStatus.progress))
      : displayStatus === 'completed'
        ? 100
        : displayStatus === 'failed'
          ? 0
          : displayStatus === 'queued'
            ? 5
            : 20;

  let viewportMode: RunConsoleViewModel['scene']['viewportMode'] = 'init';
  if (displayStatus === 'failed') viewportMode = 'failed';
  else if (!enableCameras) viewportMode = 'cameras_disabled';
  else if (shouldPollLive || frameLoaded) viewportMode = 'live';

  return {
    jobId,
    taskName: ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME,
    taskTypeLabel: '数据生成',
    taskKindLabel: 'isaaclab_franka_stack_cube',
    simulatorBackend: 'Isaac Lab',
    status: displayStatus,
    progress,
    scene: {
      title: '仿真画面',
      backendLabel: 'Isaac Lab',
      initializingText: '正在初始化 Isaac Lab 场景。',
      frameStatusLine: frameStatus.line,
      frameStatusAccent: frameStatus.accent,
      viewportMode,
      failedMessage: jobStatus?.errorSummary ?? jobStatus?.message ?? '仿真画面不可用，请查看日志。',
      liveFrame: shouldPollLive
          ? {
              kind: 'isaac_lab',
              jobId,
              pollEnabled: true,
              status: displayStatus,
            }
          : undefined,
    },
    sections: {
      basicInfo: [
        { label: '任务名称', value: ISAACLAB_FRANKA_STACK_CUBE_DISPLAY_NAME },
        { label: '仿真后端', value: 'Isaac Lab' },
        { label: '当前阶段', value: jobStatus?.phaseLabel ?? jobStatus?.phase ?? '—' },
        { label: '生成模式', value: jobStatus?.generationMode ?? '—' },
      ],
      assetConfig: [
        { label: '机器人', value: 'Franka Panda' },
        { label: '数据格式', value: 'HDF5' },
        { label: '采集轮次', value: total != null ? String(total) : '—' },
      ],
      results: [
        { label: '状态', value: runConsoleResultStatusLabel(displayStatus) },
        {
          label: '成功轮次',
          value:
            jobStatus?.successEpisodes != null && total != null
              ? `${jobStatus.successEpisodes} / ${total}`
              : '—',
        },
        {
          label: '视频文件',
          value: runConsoleFileReadyLabel(jobStatus?.videoExists, displayStatus === 'running'),
        },
        { label: '进度说明', value: jobStatus?.progressMessage ?? jobStatus?.message ?? '—' },
      ],
      debug: [
        { label: 'jobId', value: jobId },
        { label: 'taskId', value: jobStatus?.taskId ?? '—' },
        { label: 'datasetId', value: jobStatus?.datasetId ?? '—' },
        { label: 'runtimeMode', value: jobStatus?.runtimeMode ?? '—' },
        { label: 'manifestPath', value: jobStatus?.manifestPath ?? '—' },
      ],
    },
    actions: {
      backToDataCenterHref: '/workspace/data',
      canViewReplay,
      showViewDataRecord: displayStatus === 'completed' && Boolean(jobStatus?.datasetId),
    },
  };
}
