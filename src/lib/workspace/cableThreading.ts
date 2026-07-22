import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import type { CreateEvaluationPayload } from '@/components/workspace/evaluation/CreateEvaluationModal';
import type {
  CableThreadingEvaluateResponse,
  CableThreadingGenerateResponse,
  CableThreadingJobStatusResponse,
  CableThreadingVideoResponse,
} from '@/lib/api/cableThreadingClient';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { isPendingLocalJobId, isValidCableThreadingGenerateJobId } from '@/lib/workspace/backendJobIds';

import {
  CABLE_THREADING_DISPLAY_NAME,
  matchesCableThreadingDisplayName,
} from '@/lib/workspace/taskDisplayNames';

/** 用户可见任务展示名（不改 template id / 内部 taskType） */
export const CABLE_THREADING_TASK_DISPLAY_NAME = CABLE_THREADING_DISPLAY_NAME;
export const CABLE_THREADING_TASK_NAME = CABLE_THREADING_TASK_DISPLAY_NAME;

export const CABLE_THREADING_ROBOTS = ['Panda', 'UR5e'] as const;
export const CABLE_THREADING_CABLE_MODELS = [
  'composite_cable',
  'composite_soft',
  'rmb',
  'flex',
] as const;
export const CABLE_THREADING_DIFFICULTIES = ['easy', 'medium', 'hard'] as const;
export const CABLE_THREADING_POLICIES = ['scripted', 'random', 'robomimic'] as const;

export type CableThreadingEvalStrategy = 'scripted' | 'checkpoint';

export const CABLE_THREADING_EVAL_STRATEGY_LABELS: Record<CableThreadingEvalStrategy, string> = {
  scripted: '专家策略',
  checkpoint: '已训练模型',
};

export const CABLE_THREADING_DEFAULTS = {
  robot: 'Panda',
  cableModel: 'composite_cable',
  difficulty: 'easy',
  horizon: 600,
  seed: 0,
  policy: 'scripted',
  generateEpisodes: 10,
  evalEpisodes: 10,
  videoEpisodes: 1,
  saveHdf5: true,
} as const;

/** Robomimic BC 线缆穿杆评测环境可提供的观测键（与 HDF5 训练数据一致） */
export const CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS = [
  'robot0_eef_pos',
  'robot0_eef_quat',
  'robot0_gripper_qpos',
  'agentview_image',
  'robot0_eye_in_hand_image',
] as const;

export function modelAssetSupportsCableThreadingEvalObs(asset: {
  framework?: string | null;
  trainingBackend?: string | null;
  backendType?: string | null;
  modelType?: string | null;
  modelTypeId?: string | null;
  baseAlgorithm?: string | null;
  observationSchema?: string | null;
}): boolean {
  const framework = String(
    asset.framework ?? asset.trainingBackend ?? asset.backendType ?? asset.modelType ?? ''
  ).toLowerCase();
  const modelTypeId = String(asset.modelTypeId ?? '').toLowerCase();
  const baseAlgorithm = String(asset.baseAlgorithm ?? '').toLowerCase();
  const modelType = String(asset.modelType ?? '').toLowerCase();

  const supportsCableThreadingPolicy =
    framework.includes('robomimic') ||
    framework.includes('diffusion') ||
    framework.includes('diffusion_policy') ||
    framework === 'act' ||
    modelType === 'act' ||
    baseAlgorithm === 'act' ||
    modelTypeId === 'act' ||
    framework === 'pi0' ||
    modelType === 'pi0' ||
    baseAlgorithm === 'pi0' ||
    modelTypeId === 'pi0';
  if (!supportsCableThreadingPolicy) {
    return false;
  }
  const schema = asset.observationSchema?.trim();
  if (schema && schema !== 'cable_threading_robomimic_v1') {
    return false;
  }
  return true;
}

export function isCableThreadingTask(taskName: string | undefined | null): boolean {
  return matchesCableThreadingDisplayName(taskName);
}

/** 运行/评测控制台：仅当后端确认有有效 live 帧时才展示画面 */
export function resolveCableThreadingHasValidLiveFrame(
  status: CableThreadingJobStatusResponse | null | undefined
): boolean {
  if (!status) return false;
  const live = (status.live ?? {}) as Record<string, unknown>;
  if (live.hasValidFrame === true) return true;
  if (live.hasValidFrame === false) return false;
  const frameStatus = String(live.frameStatus ?? '');
  return Boolean(status.paths?.liveFrame?.exists) && frameStatus === 'ready';
}

/** 从数据中心记录解析真实后端 jobId（ct_gen_*） */
export function resolveCableThreadingBackendJobId(item: {
  id: string;
  jobId?: string | null;
  backendJobId?: string | null;
  sourceJobId?: string | null;
  simulationId?: string | null;
}): string | undefined {
  const candidates = [item.backendJobId, item.jobId, item.sourceJobId, item.simulationId, item.id];
  for (const candidate of candidates) {
    if (candidate && isValidCableThreadingGenerateJobId(candidate)) return candidate;
  }
  return undefined;
}

export function resolveCableThreadingConsoleJobId(item: WorkspaceDataItem): string | undefined {
  if (item.staleLocalPending) return undefined;
  const realJobId = resolveCableThreadingBackendJobId(item);
  if (!realJobId || isPendingLocalJobId(realJobId)) return undefined;
  return realJobId;
}

function nowLabel() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function formatBytes(sizeBytes: number | null | undefined): string {
  if (sizeBytes == null || sizeBytes <= 0) return '—';
  if (sizeBytes >= 1024 * 1024) return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
  if (sizeBytes >= 1024) return `${Math.round(sizeBytes / 1024)} KB`;
  return `${sizeBytes} B`;
}

function generateVideoFieldsFromStatus(
  status: CableThreadingJobStatusResponse,
  jobId: string
): {
  generateVideoPath?: string;
  generateVideoExists?: boolean;
  generateVideoSizeBytes?: number;
  videoJobId?: string;
} {
  const live = status.live ?? {};
  const exists =
    status.generateVideoExists === true ||
    live.generateVideoExists === true ||
    status.paths.generateVideo?.exists === true;
  if (!exists) return {};
  const path =
    status.generateVideoPath ??
    (live.generateVideo as string | undefined) ??
    status.paths.generateVideo?.path;
  const sizeBytes =
    status.generateVideoSizeBytes ??
    (live.generateVideoSizeBytes as number | undefined) ??
    status.paths.generateVideo?.sizeBytes ??
    undefined;
  return {
    generateVideoPath: path,
    generateVideoExists: true,
    generateVideoSizeBytes: sizeBytes ?? undefined,
    videoJobId: jobId,
  };
}

function formatPercent(rate: number | null | undefined): number | null {
  if (rate == null || Number.isNaN(rate)) return null;
  return Math.round(rate * 1000) / 10;
}

export function cableThreadingDataItemFromGenerate(
  response: CableThreadingGenerateResponse,
  payload: GenerateDataPayload
): WorkspaceDataItem {
  const successRate = formatPercent(response.metrics.finalSuccessRate);
  const episodes = response.metrics.episodes ?? payload.episodes;
  const successful = response.metrics.successfulEpisodes ?? 0;
  const hdf5Exists = response.paths.hdf5?.exists;
  const npzExists = response.paths.npz?.exists;

  return {
    id: response.jobId,
    name: payload.outputName?.trim() || generateDefaultCableThreadingDataName(),
    taskId: 'cable_threading',
    taskName: CABLE_THREADING_TASK_NAME,
    simulationId: response.jobId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: `${episodes} 条`,
    size: hdf5Exists
      ? `${Math.round((response.paths.hdf5.sizeBytes ?? 0) / (1024 * 1024))} MB`
      : npzExists
        ? `${Math.round((response.paths.npz.sizeBytes ?? 0) / 1024)} KB`
        : '—',
    status: response.status === 'completed' ? 'completed' : 'failed',
    generatedAt: nowLabel(),
    creator: '当前用户',
    scene: '桌面双杆穿线工位',
    robot: payload.cableThreadingRobot ?? CABLE_THREADING_DEFAULTS.robot,
    simBackend: 'MuJoCo',
    contents: [
      ...(payload.saveTrajectory ? ['轨迹'] : []),
      ...(hdf5Exists ? ['图像'] : []),
      ...(npzExists ? ['NPZ'] : []),
      ...(hdf5Exists ? ['HDF5'] : []),
      ...(payload.saveProcessVideo ? ['过程视频'] : []),
    ],
    frameOrTrajectoryCount: `${successful}/${episodes} 成功 · seed ${payload.seed ?? CABLE_THREADING_DEFAULTS.seed}`,
    taskType: 'cable_threading',
    cableModel: payload.cableThreadingCableModel ?? CABLE_THREADING_DEFAULTS.cableModel,
    difficulty: payload.cableThreadingDifficulty ?? CABLE_THREADING_DEFAULTS.difficulty,
    horizon: payload.cableThreadingHorizon ?? CABLE_THREADING_DEFAULTS.horizon,
    successRate: successRate ?? undefined,
    successfulEpisodes: successful,
    hdf5Path: response.paths.hdf5?.path,
    npzPath: response.paths.npz?.path,
    manifestPath: response.paths.manifest?.path,
    collectCsvPath: response.paths.collectCsv?.path,
    failuresPath: response.paths.failures?.path,
    backendJobStatus: response.status,
    backendCommand: response.command,
  };
}

export function cableThreadingEvalRowFromEvaluate(
  response: CableThreadingEvaluateResponse,
  payload: CreateEvaluationPayload
): EvaluationTaskRow {
  const successRate = formatPercent(response.metrics.successRate);
  const everSuccessRate = formatPercent(response.metrics.everSuccessRate);
  const episodes = response.metrics.numEpisodes ?? payload.evalRounds;

  return {
    id: response.jobId,
    name: payload.name.trim() || `${CABLE_THREADING_TASK_NAME} · ${payload.cableThreadingPolicy ?? 'scripted'} 评测`,
    evaluationMode: '策略评测',
    relatedTask: CABLE_THREADING_TASK_NAME,
    checkpoint: payload.cableThreadingPolicy ?? CABLE_THREADING_DEFAULTS.policy,
    modelType: payload.cableThreadingPolicy ?? CABLE_THREADING_DEFAULTS.policy,
    dataVolume: `${episodes} 条`,
    evalBackend: 'MuJoCo',
    evalRounds: episodes,
    status: response.status === 'completed' ? '已完成' : '失败',
    successRate,
    createdAt: nowLabel(),
    metrics: ['成功率', '失败案例', '视频回放'],
    resultSummary:
      successRate != null
        ? `${episodes} 次 scripted 评测完成，成功率 ${successRate}%。`
        : '评测已完成，请查看报告与失败案例。',
    taskType: 'cable_threading',
    cableModel: payload.cableThreadingCableModel ?? CABLE_THREADING_DEFAULTS.cableModel,
    difficulty: payload.cableThreadingDifficulty ?? CABLE_THREADING_DEFAULTS.difficulty,
    policy: payload.cableThreadingPolicy ?? CABLE_THREADING_DEFAULTS.policy,
    robot: payload.cableThreadingRobot ?? CABLE_THREADING_DEFAULTS.robot,
    everSuccessRate: everSuccessRate ?? undefined,
    resultPath: response.paths.resultsJson?.path,
    evalCsvPath: response.paths.evalCsv?.path,
    failuresPath: response.paths.failuresJson?.path,
    aggregate: response.metrics.aggregate,
    backendJobStatus: response.status,
    backendCommand: response.command,
  };
}

export function findCableThreadingEvalById(
  evalId: string,
  tasks: EvaluationTaskRow[]
): EvaluationTaskRow | null {
  const row = tasks.find((t) => t.id === evalId && t.taskType === 'cable_threading');
  return row ?? null;
}

export function findCableThreadingEvalByJobId(
  jobId: string,
  tasks: EvaluationTaskRow[]
): EvaluationTaskRow | null {
  const row = tasks.find(
    (t) =>
      t.taskType === 'cable_threading' &&
      (t.id === jobId || t.videoJobId === jobId)
  );
  return row ?? null;
}

export function findCableThreadingDataByJobId(
  jobId: string,
  items: WorkspaceDataItem[]
): WorkspaceDataItem | null {
  const item = items.find((i) => {
    if (i.taskType !== 'cable_threading') return false;
    if (i.id === jobId || i.simulationId === jobId) return true;
    if (i.id === `ct-pending-${jobId}`) return true;
    return false;
  });
  return item ?? null;
}

export function resolveCableThreadingReplayEval(
  params: {
    evalId?: string | null;
    jobId?: string | null;
  },
  tasks: EvaluationTaskRow[]
): EvaluationTaskRow | null {
  const { evalId, jobId } = params;
  if (evalId) {
    const byEval = findCableThreadingEvalById(evalId, tasks);
    if (byEval) return byEval;
  }
  if (jobId) {
    const byJob = findCableThreadingEvalByJobId(jobId, tasks);
    if (byJob) return byJob;
  }
  return null;
}

export function buildCableThreadingReplayHref(params: {
  evalId?: string;
  jobId?: string;
  datasetId?: string;
}): string {
  const isEval = Boolean(params.evalId?.trim());
  const search = new URLSearchParams({
    replayType: isEval ? 'evaluation' : 'dataset',
    taskType: 'cable_threading',
  });
  if (params.evalId) search.set('evalId', params.evalId);
  if (params.jobId) search.set('jobId', params.jobId);
  if (params.datasetId) search.set('datasetId', params.datasetId);
  return `/workspace/replay?${search.toString()}`;
}

export function buildCableThreadingVideoApiPath(videoJobId: string): string {
  return `/api/workspace/cable-threading/jobs/${encodeURIComponent(videoJobId)}/video`;
}

export function buildCableThreadingHdf5TrajectoryMetaApiPath(jobId: string, demoName: string): string {
  return `/api/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/hdf5-trajectory/${encodeURIComponent(demoName)}`;
}

export function buildCableThreadingHdf5TrajectoryFrameApiPath(
  jobId: string,
  demoName: string,
  params: { camera: string; index: number }
): string {
  const search = new URLSearchParams({
    camera: params.camera,
    index: String(params.index),
  });
  return `/api/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/hdf5-trajectory/${encodeURIComponent(demoName)}/frame?${search.toString()}`;
}

export function buildCableThreadingHdf5TrajectoryStepApiPath(
  jobId: string,
  demoName: string,
  index: number
): string {
  const search = new URLSearchParams({ index: String(index) });
  return `/api/workspace/cable-threading/jobs/${encodeURIComponent(jobId)}/hdf5-trajectory/${encodeURIComponent(demoName)}/step?${search.toString()}`;
}

export function isCableThreadingReplayMode(taskType: string | null | undefined): boolean {
  return taskType === 'cable_threading';
}

export function makeCableThreadingLocalRunId(): string {
  const suffix = Math.random().toString(36).slice(2, 6);
  return `ct-run_${Date.now().toString(36)}_${suffix}`;
}

export function generateDefaultCableThreadingDataName(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const seq = String(Math.floor(Math.random() * 999) + 1).padStart(3, '0');
  return `${CABLE_THREADING_TASK_DISPLAY_NAME}数据_${date}_${seq}`;
}

export function createPendingCableThreadingDataItem(
  payload: GenerateDataPayload,
  dataItemKey: string,
  backendJobId?: string
): WorkspaceDataItem {
  const name = payload.outputName?.trim() || generateDefaultCableThreadingDataName();
  const volume = `${payload.episodes} 条`;
  const realJobId = backendJobId && isValidCableThreadingGenerateJobId(backendJobId) ? backendJobId : undefined;
  return {
    id: `ct-pending-${dataItemKey}`,
    name,
    taskId: 'cable_threading',
    taskName: CABLE_THREADING_TASK_NAME,
    jobId: realJobId,
    backendJobId: realJobId,
    sourceJobId: realJobId,
    simulationId: realJobId ?? dataItemKey,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: volume,
    size: '—',
    status: 'generating',
    generatedAt: nowLabel(),
    creator: '当前用户',
    scene: '桌面双杆穿线工位',
    robot: payload.cableThreadingRobot ?? CABLE_THREADING_DEFAULTS.robot,
    simBackend: 'MuJoCo',
    saveTrajectory: payload.saveTrajectory,
    frameOrTrajectoryCount: `${volume} · seed ${payload.seed ?? CABLE_THREADING_DEFAULTS.seed} · 生成中`,
    taskType: 'cable_threading',
    cableModel: payload.cableThreadingCableModel ?? CABLE_THREADING_DEFAULTS.cableModel,
    difficulty: payload.cableThreadingDifficulty ?? CABLE_THREADING_DEFAULTS.difficulty,
    horizon: payload.cableThreadingHorizon ?? CABLE_THREADING_DEFAULTS.horizon,
    backendJobStatus: 'running',
  };
}

export function buildCableThreadingConsoleHref(params: {
  jobId: string;
  dataId?: string;
}): string {
  const search = new URLSearchParams({
    mode: 'data-generation',
    taskType: 'cable_threading',
    jobId: params.jobId,
  });
  if (params.dataId) search.set('dataId', params.dataId);
  return `/workspace/simulation/console?${search.toString()}`;
}

export function buildCableThreadingEvalConsoleHref(params: { evalJobId: string }): string {
  const search = new URLSearchParams({
    mode: 'evaluation',
    taskType: 'cable_threading',
    evalId: params.evalJobId,
  });
  return `/workspace/simulation/console?${search.toString()}`;
}

export function generateCableThreadingEvalTaskName(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const date = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
  const seq = String(Math.floor(Math.random() * 999) + 1).padStart(3, '0');
  return `线缆穿杆评测_${date}_${seq}`;
}

export function cableThreadingEvalRowFromJobStatus(
  status: CableThreadingJobStatusResponse,
  payload: CreateEvaluationPayload
): EvaluationTaskRow {
  const successRate = formatPercent(status.metrics.successRate);
  const everSuccessRate = formatPercent(status.metrics.everSuccessRate);
  const episodes = status.metrics.numEpisodes ?? payload.evalRounds;
  const policy =
    payload.cableThreadingEvalStrategy === 'checkpoint'
      ? 'robomimic'
      : payload.cableThreadingPolicy ?? CABLE_THREADING_DEFAULTS.policy;
  const strategyLabel =
    payload.cableThreadingEvalStrategy === 'checkpoint' ? '已训练模型' : '专家策略';
  const checkpointLabel =
    payload.cableThreadingEvalStrategy === 'checkpoint'
      ? payload.cableThreadingCheckpointAssetId ?? payload.checkpoint ?? '已训练模型'
      : 'scripted';

  return {
    id: status.jobId,
    name: payload.name.trim() || generateCableThreadingEvalTaskName(),
    evaluationMode: '策略评测',
    relatedTask: CABLE_THREADING_TASK_NAME,
    checkpoint: checkpointLabel,
    modelType: strategyLabel,
    dataVolume: `${episodes} 条`,
    evalBackend: 'MuJoCo',
    evalRounds: episodes,
    status: status.status === 'completed' ? '已完成' : status.status === 'failed' ? '失败' : '评测中',
    successRate,
    createdAt: nowLabel(),
    metrics: ['成功率', '失败案例', '视频回放'],
    resultSummary:
      successRate != null
        ? `${episodes} 次策略评测完成，成功率 ${successRate}%。`
        : status.status === 'failed'
          ? '评测失败，请查看运行日志。'
          : '评测任务运行中…',
    taskType: 'cable_threading',
    cableModel: payload.cableThreadingCableModel ?? CABLE_THREADING_DEFAULTS.cableModel,
    difficulty: payload.cableThreadingDifficulty ?? CABLE_THREADING_DEFAULTS.difficulty,
    policy,
    robot: payload.cableThreadingRobot ?? CABLE_THREADING_DEFAULTS.robot,
    everSuccessRate: everSuccessRate ?? undefined,
    resultPath: status.paths.resultsJson?.path,
    evalCsvPath: status.paths.evalCsv?.path,
    failuresPath: status.paths.failuresJson?.path,
    aggregate: status.metrics.aggregate,
    backendJobStatus: status.status,
    backendCommand: status.command,
    evalVideoExists: status.evalVideoExists ?? status.paths.evalVideo?.exists ?? false,
    evalVideoPath: status.evalVideoPath ?? status.paths.evalVideo?.path,
    evalVideoSizeBytes: status.evalVideoSizeBytes ?? status.paths.evalVideo?.sizeBytes ?? undefined,
    videoExists: status.evalVideoExists ?? status.paths.evalVideo?.exists ?? false,
    videoPath: status.evalVideoPath ?? status.paths.evalVideo?.path,
    videoSizeBytes: status.evalVideoSizeBytes ?? status.paths.evalVideo?.sizeBytes ?? undefined,
    videoJobId: status.jobId,
    timelineExists: status.timelineExists ?? false,
    timelinePath: status.timelinePath ?? undefined,
  };
}

export function cableThreadingEvalRunResultFromStatus(
  status: CableThreadingJobStatusResponse
): {
  successRate?: number;
  everSuccessRate?: number;
  evalCsvPath?: string;
  resultPath?: string;
  failuresPath?: string;
  logPath?: string;
  backendCommand?: string;
} {
  const successRate = formatPercent(status.metrics.successRate);
  const everSuccessRate = formatPercent(status.metrics.everSuccessRate);
  return {
    successRate: successRate ?? undefined,
    everSuccessRate: everSuccessRate ?? undefined,
    evalCsvPath: status.paths.evalCsv?.path,
    resultPath: status.paths.resultsJson?.path,
    failuresPath: status.paths.failuresJson?.path,
    logPath: status.paths.log?.path,
    backendCommand: status.command,
  };
}

export function cableThreadingDataItemFromJobStatus(
  status: CableThreadingJobStatusResponse,
  payload: GenerateDataPayload
): WorkspaceDataItem {
  const live = status.live ?? {};
  const successRate = formatPercent(
    (live.finalSuccessRate as number | null | undefined) ??
      status.metrics.finalSuccessRate ??
      null
  );
  const episodes =
    (live.episodes as number | undefined) ??
    status.metrics.episodes ??
    payload.episodes;
  const successful =
    (live.successfulEpisodes as number | undefined) ??
    status.metrics.successfulEpisodes ??
    0;
  const hdf5Exists = status.paths.hdf5?.exists;
  const npzExists = status.paths.npz?.exists;
  const videoFields = generateVideoFieldsFromStatus(status, status.jobId);

  return {
    id: status.jobId,
    name: payload.outputName?.trim() || generateDefaultCableThreadingDataName(),
    taskId: 'cable_threading',
    taskName: CABLE_THREADING_TASK_NAME,
    jobId: status.jobId,
    backendJobId: status.jobId,
    sourceJobId: status.jobId,
    simulationId: status.jobId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: `${episodes} 条`,
    size: hdf5Exists
      ? `${Math.round((status.paths.hdf5?.sizeBytes ?? 0) / (1024 * 1024))} MB`
      : npzExists
        ? `${Math.round((status.paths.npz?.sizeBytes ?? 0) / 1024)} KB`
        : '—',
    status: status.status === 'completed' ? 'completed' : status.status === 'failed' ? 'failed' : 'generating',
    generatedAt: nowLabel(),
    creator: '当前用户',
    scene: '桌面双杆穿线工位',
    robot: payload.cableThreadingRobot ?? CABLE_THREADING_DEFAULTS.robot,
    simBackend: 'MuJoCo',
    contents: [
      ...(payload.saveTrajectory ? ['轨迹'] : []),
      ...(hdf5Exists ? ['图像', 'HDF5'] : []),
      ...(npzExists ? ['NPZ'] : []),
      ...(payload.saveProcessVideo ? ['过程视频'] : []),
    ],
    frameOrTrajectoryCount: `${successful}/${episodes} 成功 · seed ${payload.seed ?? CABLE_THREADING_DEFAULTS.seed}`,
    taskType: 'cable_threading',
    cableModel: payload.cableThreadingCableModel ?? CABLE_THREADING_DEFAULTS.cableModel,
    difficulty: payload.cableThreadingDifficulty ?? CABLE_THREADING_DEFAULTS.difficulty,
    horizon: payload.cableThreadingHorizon ?? CABLE_THREADING_DEFAULTS.horizon,
    successRate: successRate ?? undefined,
    successfulEpisodes: successful,
    hdf5Path: status.paths.hdf5?.path,
    npzPath: status.paths.npz?.path,
    manifestPath: status.paths.manifest?.path,
    collectCsvPath: status.paths.collectCsv?.path,
    failuresPath: status.paths.failures?.path,
    backendJobStatus: status.status,
    backendCommand: status.command,
    datasetBuildSupported:
      status.status === 'completed' &&
      Boolean(status.paths.hdf5?.exists || status.paths.npz?.exists || status.paths.manifest?.exists || successful > 0),
    ...videoFields,
  };
}

export function cableThreadingGenerateRunResultFromStatus(
  status: CableThreadingJobStatusResponse
): {
  successRate?: number;
  successfulEpisodes?: number;
  npzPath?: string;
  hdf5Path?: string;
  manifestPath?: string;
  collectCsvPath?: string;
  failuresPath?: string;
  logPath?: string;
  backendCommand?: string;
  generateVideoPath?: string;
  generateVideoExists?: boolean;
  generateVideoSizeBytes?: number;
  generateVideoStatus?: string;
} {
  const live = status.live ?? {};
  const successRate = formatPercent(
    (live.finalSuccessRate as number | null | undefined) ??
      status.metrics.finalSuccessRate ??
      null
  );
  const videoFields = generateVideoFieldsFromStatus(status, status.jobId);
  return {
    successRate: successRate ?? undefined,
    successfulEpisodes:
      (live.successfulEpisodes as number | undefined) ??
      status.metrics.successfulEpisodes ??
      undefined,
    npzPath: status.paths.npz?.path,
    hdf5Path: status.paths.hdf5?.path,
    manifestPath: status.paths.manifest?.path,
    collectCsvPath: status.paths.collectCsv?.path,
    failuresPath: status.paths.failures?.path,
    logPath: status.paths.log?.path,
    backendCommand: status.command,
    generateVideoStatus: live.generateVideoStatus as string | undefined,
    ...videoFields,
  };
}

export function cableThreadingGenerateRunResultFromResponse(
  response: CableThreadingGenerateResponse
): {
  successRate?: number;
  successfulEpisodes?: number;
  npzPath?: string;
  hdf5Path?: string;
  manifestPath?: string;
  collectCsvPath?: string;
  failuresPath?: string;
  logPath?: string;
  backendCommand?: string;
} {
  const successRate = formatPercent(response.metrics.finalSuccessRate);
  return {
    successRate: successRate ?? undefined,
    successfulEpisodes: response.metrics.successfulEpisodes ?? undefined,
    npzPath: response.paths.npz?.path,
    hdf5Path: response.paths.hdf5?.path,
    manifestPath: response.paths.manifest?.path,
    collectCsvPath: response.paths.collectCsv?.path,
    failuresPath: response.paths.failures?.path,
    logPath: response.paths.log?.path,
    backendCommand: response.command,
  };
}

function formatMetricValue(value: unknown): string {
  if (value == null) return '—';
  if (typeof value === 'number') {
    if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
    return value.toFixed(4);
  }
  return String(value);
}

export function cableThreadingReportSections(row: EvaluationTaskRow) {
  const aggregate = (row.aggregate ?? {}) as Record<string, unknown>;
  return {
    basic: [
      { label: '任务名称', value: CABLE_THREADING_TASK_NAME },
      { label: '仿真后端', value: 'MuJoCo' },
      { label: '机器人', value: row.robot ?? CABLE_THREADING_DEFAULTS.robot },
      { label: '线缆模型', value: row.cableModel ?? CABLE_THREADING_DEFAULTS.cableModel },
      { label: '策略', value: row.policy ?? CABLE_THREADING_DEFAULTS.policy },
      { label: '难度', value: row.difficulty ?? CABLE_THREADING_DEFAULTS.difficulty },
      { label: '评测次数', value: `${row.evalRounds} 次` },
      {
        label: '成功率',
        value: row.successRate != null ? `${row.successRate}%` : '—',
      },
      {
        label: 'ever_success_rate',
        value:
          row.everSuccessRate != null
            ? `${row.everSuccessRate}%`
            : formatMetricValue(aggregate.ever_success_rate),
      },
    ],
    metrics: [
      {
        label: 'final_success_rate',
        value: formatMetricValue(
          aggregate.final_success_rate ??
            (row.successRate != null ? row.successRate / 100 : null)
        ),
      },
      {
        label: 'ever_success_rate',
        value: formatMetricValue(aggregate.ever_success_rate),
      },
      {
        label: 'mean_endpoint_goal_error_final',
        value: formatMetricValue(aggregate.mean_endpoint_goal_error_final),
      },
      {
        label: 'mean_straightness_error_final',
        value: formatMetricValue(aggregate.mean_straightness_error_final),
      },
      {
        label: 'mean_anchor_error_final',
        value: formatMetricValue(aggregate.mean_anchor_error_final),
      },
      {
        label: 'mean_tabletop_spread_final',
        value: formatMetricValue(aggregate.mean_tabletop_spread_final),
      },
      {
        label: 'mean_thread_completion_max',
        value: formatMetricValue(aggregate.mean_thread_completion_max),
      },
    ],
    failures: [
      {
        label: 'failures.json',
        value: row.failuresPath ?? '—',
      },
      {
        label: 'results.json',
        value: row.resultPath ?? '—',
      },
    ],
    video: row.evalVideoExists
      ? [
          { label: '评测过程视频', value: '已生成' },
          {
            label: 'eval.mp4',
            value: row.evalVideoPath ?? row.videoPath ?? '—',
          },
        ]
      : null,
  };
}

export function formatCableThreadingVideoSize(sizeBytes: number | null | undefined): string {
  return formatBytes(sizeBytes);
}

export function applyCableThreadingVideoToEvalRow(
  row: EvaluationTaskRow,
  response: CableThreadingVideoResponse
): EvaluationTaskRow {
  return {
    ...row,
    videoPath: response.paths.video?.path,
    videoSizeBytes: response.videoSizeBytes ?? undefined,
    videoExists: response.videoExists,
    videoJobId: response.jobId,
  };
}
