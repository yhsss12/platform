import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import type { NutAssemblyGenerationMode, NutAssemblyJobStatusResponse } from '@/lib/api/nutAssemblyClient';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import { isValidNutAssemblyGenerateJobId } from '@/lib/workspace/backendJobIds';
import { isTerminalSimJobStatus } from '@/lib/workspace/simulationPolling';
import {
  NUT_ASSEMBLY_DISPLAY_NAME,
  matchesNutAssemblyDisplayName,
} from '@/lib/workspace/taskDisplayNames';

export const NUT_ASSEMBLY_TASK_DISPLAY_NAME = NUT_ASSEMBLY_DISPLAY_NAME;
export const NUT_ASSEMBLY_OFFICIAL_ENV_NAME = 'NutAssembly_D0';
export const NUT_ASSEMBLY_OFFICIAL_SOURCE_REL_PATH =
  'assets/mimicgen/nut_assembly/source/nut_assembly.hdf5';

export const NUT_ASSEMBLY_ROBOT_OPTIONS = [
  { value: 'panda_single_arm', label: 'Panda 单臂机械臂' },
  { value: 'fr3_single_arm', label: 'FR3 单臂机械臂' },
] as const;

export const NUT_ASSEMBLY_DEFAULT_ROBOT = 'panda_single_arm';

export const NUT_ASSEMBLY_DEFAULTS = {
  episodes: 20,
  seed: 0,
  robot: NUT_ASSEMBLY_DEFAULT_ROBOT,
  envName: NUT_ASSEMBLY_OFFICIAL_ENV_NAME,
  outputName: 'nut_assembly_dataset',
  horizon: 500,
  renderVideo: true,
  generationMode: 'mimicgen_datagen' as NutAssemblyGenerationMode,
} as const;

export function nutAssemblyRobotLabel(robotValue: string): string {
  return (
    NUT_ASSEMBLY_ROBOT_OPTIONS.find((option) => option.value === robotValue)?.label ??
    'Panda 单臂机械臂'
  );
}

export const NUT_ASSEMBLY_GENERATION_MODE_OPTIONS: Array<{
  value: NutAssemblyGenerationMode;
  label: string;
  description: string;
}> = [
  {
    value: 'mimicgen_datagen',
    label: 'MimicGen 生成',
    description: '基于示教数据自动生成新的训练数据。',
  },
  {
    value: 'robosuite_rollout',
    label: '规则生成（调试）',
    description: '使用平台规则策略生成，仅用于调试。',
  },
];

export function isNutAssemblyOfficialSourceReady(
  status: {
    officialSourceValidated?: boolean;
    options?: { official?: { exists?: boolean; validationPassed?: boolean } };
  } | null
): boolean {
  const official = status?.options?.official;
  return Boolean(
    status?.officialSourceValidated && official?.exists && official?.validationPassed
  );
}

export function formatNutAssemblyDemoDataSummary(
  status: {
    officialSourceValidated?: boolean;
    options?: { official?: { exists?: boolean; validationPassed?: boolean; demoCount?: number } };
  } | null
): string {
  if (!isNutAssemblyOfficialSourceReady(status)) {
    return '未检测到系统示教数据，请先完成示教数据校验。';
  }
  const count = status?.options?.official?.demoCount;
  return `系统内置示教数据 · 已校验 · ${count ?? 10} 条`;
}

export function isNutAssemblyTask(label: string | null | undefined): boolean {
  return matchesNutAssemblyDisplayName(label);
}

export function buildNutAssemblyConsoleHref(params: {
  jobId: string;
  dataId?: string;
}): string {
  const search = new URLSearchParams({
    mode: 'data-generation',
    taskType: 'nut_assembly',
    jobId: params.jobId,
  });
  if (params.dataId) search.set('dataId', params.dataId);
  return `/workspace/simulation/console?${search.toString()}`;
}

export function buildNutAssemblyVideoApiPath(jobId: string): string {
  return `/api/workspace/nut-assembly/jobs/${encodeURIComponent(jobId)}/video`;
}

export function buildNutAssemblyReplayHref(params: {
  jobId: string;
  datasetId?: string;
}): string {
  const search = new URLSearchParams({
    replayType: 'dataset',
    taskType: 'nut_assembly',
    jobId: params.jobId,
  });
  if (params.datasetId) search.set('datasetId', params.datasetId);
  return `/workspace/replay?${search.toString()}`;
}

export function createPendingNutAssemblyDataItem(
  payload: GenerateDataPayload,
  dataItemId: string,
  backendJobId?: string
): WorkspaceDataItem {
  const now = new Date().toLocaleString('zh-CN', { hour12: false });
  return {
    id: dataItemId,
    name: payload.outputName || NUT_ASSEMBLY_DEFAULTS.outputName,
    taskId: 'nut_assembly_single_arm',
    taskName: NUT_ASSEMBLY_TASK_DISPLAY_NAME,
    simulationId: backendJobId ?? dataItemId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: 'Robomimic',
    dataVolume: `${payload.episodes} 条`,
    size: '生成中',
    status: 'generating',
    generatedAt: now,
    creator: '当前用户',
    simBackend: 'MuJoCo',
    backendJobId: backendJobId ?? undefined,
    jobId: backendJobId ?? undefined,
  };
}

export function nutAssemblyDataItemFromJobStatus(
  status: NutAssemblyJobStatusResponse,
  payload: GenerateDataPayload
): WorkspaceDataItem {
  const hdf5 = status.paths.hdf5;
  const sizeBytes = hdf5?.sizeBytes ?? 0;
  const sizeLabel =
    sizeBytes && sizeBytes > 0 ? `${(sizeBytes / (1024 * 1024)).toFixed(2)} MB` : '—';
  const episodes = Number(status.metrics.episodes ?? payload.episodes ?? NUT_ASSEMBLY_DEFAULTS.episodes);
  return {
    id: `ds_${status.jobId}`,
    name: payload.outputName || NUT_ASSEMBLY_DEFAULTS.outputName,
    taskId: 'nut_assembly_single_arm',
    taskName: NUT_ASSEMBLY_TASK_DISPLAY_NAME,
    simulationId: status.jobId,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: 'Robomimic',
    dataVolume: `${episodes} 条`,
    size: sizeLabel,
    status: 'completed',
    generatedAt: new Date().toLocaleString('zh-CN', { hour12: false }),
    creator: '当前用户',
    simBackend: 'MuJoCo',
    backendJobId: status.jobId,
    jobId: status.jobId,
  };
}

export function makeNutAssemblyLocalRunId(): string {
  return `na-run_${Date.now()}`;
}

export function isNutAssemblyReplayMode(taskType: string | null | undefined): boolean {
  return taskType === 'nut_assembly';
}

export function isNutAssemblyDataset(dataset: {
  taskType?: string | null;
  taskTemplateId?: string | null;
}): boolean {
  return (
    dataset.taskType === 'nut_assembly' ||
    dataset.taskTemplateId === 'nut_assembly_single_arm' ||
    dataset.taskTemplateId === 'task_nut_assembly_v1'
  );
}

export function formatNutAssemblyDatagenSuccessRate(dataset: {
  datagenSuccessRate?: number | null;
  episodesGenerated?: number | null;
  episodesRequested?: number | null;
  generationMode?: string | null;
}): string {
  if (dataset.generationMode !== 'mimicgen_datagen') {
    return '—';
  }
  if (dataset.datagenSuccessRate != null) {
    const pct = Math.round(Number(dataset.datagenSuccessRate) * 1000) / 10;
    return `${pct}%（datagen）`;
  }
  const generated = dataset.episodesGenerated;
  const requested = dataset.episodesRequested;
  if (generated != null && requested != null && requested > 0) {
    const rate = generated / requested;
    const pct = Math.round(rate * 1000) / 10;
    return `${pct}%（datagen）`;
  }
  return '—';
}

export function formatNutAssemblySuccessRate(dataset: {
  successRate?: number | null;
  successStatus?: string | null;
  hasEpisodeMetadata?: boolean | null;
  generationMode?: string | null;
}): string {
  if (
    dataset.successStatus === 'datagen_success_count' ||
    dataset.successStatus === 'not_evaluated'
  ) {
    return '未评测';
  }
  if (dataset.generationMode === 'mimicgen_datagen' && dataset.successRate == null) {
    return '未评测';
  }
  if (!dataset.hasEpisodeMetadata) return '未标注';
  if (dataset.successRate == null) return '未标注';
  const pct = Math.round(Number(dataset.successRate) * 1000) / 10;
  return `${pct}%`;
}

export function nutAssemblyStageStatusLabel(stage: string | null | undefined): string {
  switch (stage) {
    case 'prepare_source':
      return '正在准备源示教数据';
    case 'mimicgen_generate':
      return '正在执行 MimicGen 数据生成';
    case 'write_summary':
    case 'write_manifest':
    case 'write_dataset':
      return '正在写入 HDF5 数据集';
    case 'render_video':
      return '正在生成回放视频';
    case 'robosuite_rollout':
      return 'robosuite rollout 采集中';
    case 'completed':
      return '生成完成';
    case 'failed':
      return '生成失败';
    case 'stalled':
      return '任务可能卡住，请查看日志';
    case 'queued':
      return '任务排队中';
    default:
      return stage || '—';
  }
}

export function formatNutAssemblyElapsedSeconds(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds)) return '—';
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins > 0 ? `${mins} 分 ${secs} 秒` : `${secs} 秒`;
}

export function formatNutAssemblyEpisodesGeneratedDisplay(
  status: string | null | undefined,
  value: number | null | undefined
): string {
  if (!isTerminalSimJobStatus(status)) {
    if (value == null || value <= 0) return '生成中';
    return String(value);
  }
  if (value == null) return '—';
  return String(value);
}

export function formatNutAssemblyDatagenFailedTrialsDisplay(
  status: string | null | undefined,
  value: number | null | undefined
): string {
  if (!isTerminalSimJobStatus(status)) {
    if (value == null || value <= 0) return '待统计';
    return String(value);
  }
  if (value == null) return '—';
  return String(value);
}

export function formatNutAssemblyDatagenSuccessRateDisplay(
  rate: number | null | undefined,
  episodesGenerated?: number | null,
  episodesRequested?: number | null
): string {
  let resolved = rate;
  if (resolved == null && episodesGenerated != null && episodesRequested != null && episodesRequested > 0) {
    resolved = episodesGenerated / episodesRequested;
  }
  if (resolved == null) return '—';
  const pct = Math.round(resolved * 1000) / 10;
  return `${pct}%（数据生成成功率）`;
}

export function mergeNutAssemblyJobWithResult(
  status: NutAssemblyJobStatusResponse,
  result: Record<string, unknown> | null | undefined,
  logTail?: string | null
): NutAssemblyJobStatusResponse {
  const summary =
    result?.summary && typeof result.summary === 'object'
      ? (result.summary as Record<string, unknown>)
      : null;
  const merged: NutAssemblyJobStatusResponse = {
    ...status,
    logTail: logTail?.trim() || status.logTail,
    episodesRequested:
      status.episodesRequested ??
      (typeof result?.episodesRequested === 'number' ? result.episodesRequested : null) ??
      (typeof summary?.episodesRequested === 'number' ? summary.episodesRequested : null),
    episodesGenerated:
      status.episodesGenerated ??
      (typeof result?.episodesGenerated === 'number' ? result.episodesGenerated : null) ??
      (typeof summary?.episodesGenerated === 'number' ? summary.episodesGenerated : null),
    datagenFailedTrials:
      status.datagenFailedTrials ??
      (typeof result?.datagenFailedTrials === 'number' ? result.datagenFailedTrials : null) ??
      (typeof summary?.datagenFailedTrials === 'number' ? summary.datagenFailedTrials : null),
    datagenSuccessRate:
      status.datagenSuccessRate ??
      (typeof result?.datagenSuccessRate === 'number' ? result.datagenSuccessRate : null) ??
      (typeof summary?.datagenSuccessRate === 'number' ? summary.datagenSuccessRate : null),
    generationMode:
      status.generationMode ??
      (typeof result?.generationMode === 'string' ? result.generationMode : null) ??
      (typeof summary?.generationMode === 'string' ? summary.generationMode : null),
    policyMode:
      status.policyMode ??
      (typeof result?.policyMode === 'string' ? result.policyMode : null) ??
      (typeof summary?.policyMode === 'string' ? summary.policyMode : null),
    sourceDemoOrigin:
      status.sourceDemoOrigin ??
      (typeof result?.sourceDemoOrigin === 'string' ? result.sourceDemoOrigin : null) ??
      (typeof summary?.sourceDemoOrigin === 'string' ? summary.sourceDemoOrigin : null),
    hdf5Path:
      status.hdf5Path ??
      (typeof result?.hdf5Path === 'string' ? result.hdf5Path : null) ??
      (typeof summary?.datasetPath === 'string' ? summary.datasetPath : null),
    videoPath:
      status.videoPath ??
      (typeof result?.videoPath === 'string' ? result.videoPath : null) ??
      (typeof summary?.videoPath === 'string' ? summary.videoPath : null),
    videoUrl:
      status.videoUrl ??
      (typeof result?.videoUrl === 'string' ? result.videoUrl : null),
    generateVideoExists:
      status.generateVideoExists ??
      (typeof result?.generateVideoExists === 'boolean' ? result.generateVideoExists : null),
  };
  if (typeof result?.status === 'string' && isTerminalSimJobStatus(result.status)) {
    merged.status = result.status;
  }
  const live = { ...(merged.live ?? {}) } as Record<string, unknown>;
  const pinnFields = [
    'physicsEnhancementEnabled',
    'enhancementMode',
    'pinnModelId',
    'pinnBackend',
    'modelLoaded',
    'modelPath',
    'pipelineVersion',
    'candidateMode',
    'mimicgenGeneratedDemos',
    'rawDemoCount',
    'repairedDemoCount',
    'finalDemoCount',
    'pinnCandidateCount',
    'pinnRepairAttempted',
    'pinnRepairSucceeded',
    'pinnValidationSucceeded',
    'pinnRepairValidationRate',
    'enhancementGain',
    'enhancementStatus',
  ] as const;
  for (const key of pinnFields) {
    const val = result?.[key] ?? summary?.[key];
    if (val != null) live[key] = val;
  }
  if (typeof summary?.finalDemoCount === 'number') {
    merged.episodesGenerated = summary.finalDemoCount;
    live.episodesGenerated = summary.finalDemoCount;
  }
  merged.live = live;
  return merged;
}

export function formatNutAssemblyPolicyMode(mode: string | null | undefined): string {
  if (!mode) return '—';
  const labels: Record<string, string> = {
    scripted_expert: '脚本专家',
    partial_scripted: '规则策略',
    random_rollout: '随机 rollout',
    robosuite_rollout: '规则生成（调试）',
    mimicgen: 'MimicGen',
  };
  return labels[mode] ?? mode;
}

export function formatNutAssemblyGenerationMode(mode: string | null | undefined): string {
  if (!mode) return '—';
  const labels: Record<string, string> = {
    mimicgen_datagen: 'MimicGen 生成',
    robosuite_rollout: '规则生成（调试）',
  };
  return labels[mode] ?? mode;
}

export function isNutAssemblyBackendJobId(jobId: string | null | undefined): boolean {
  return isValidNutAssemblyGenerateJobId(jobId);
}
