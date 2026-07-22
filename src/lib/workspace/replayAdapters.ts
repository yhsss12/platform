import type { Dataset } from '@/types/benchmark';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import { getDualArmCableJobStatus } from '@/lib/api/dualArmCableClient';
import {
  getIsaacLabDatasetReplayContext,
  startIsaacLabReplayFromDataset,
  type IsaacLabDatasetReplayContext,
} from '@/lib/api/isaacLabClient';
import { getWorkspaceJob, getWorkspaceJobArtifacts } from '@/lib/api/workspaceJobClient';
import {
  buildIsaacBlockStackingConsoleHref,
  buildIsaacBlockStackingReplayConsoleHref,
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
} from '@/lib/workspace/isaacBlockStacking';
import {
  buildCableThreadingConsoleHref,
  CABLE_THREADING_TASK_DISPLAY_NAME,
} from '@/lib/workspace/cableThreading';
import {
  buildDualArmCableConsoleHref,
  DUAL_ARM_CABLE_TASK_NAME,
} from '@/lib/workspace/dualArmCable';
import {
  buildUnifiedDatasetReplayHref,
  inferReplayTaskTypeFromJobId,
  resolveDatasetReplayTaskType,
} from '@/lib/workspace/datasetReplayHref';
import { formatIsaacGenerationMode } from '@/lib/workspace/isaacGenerationMode';
import { formatNutAssemblyGenerationMode } from '@/lib/workspace/nutAssembly';
import { isPersistedWorkspaceJobId } from '@/lib/workspace/workspaceJobReplay';
import { resolveDatasetSourceLabel } from '@/lib/workspace/datasetDisplay';
import { resolveNutAssemblyReplay } from '@/lib/workspace/replayNutAssemblyAdapter';

export type ReplayVideoSource =
  | 'replay'
  | 'preview'
  | 'generate'
  | 'episode'
  | 'converted'
  | 'none';

export type ReplayVideoBackend = 'cable_threading' | 'dual_arm_cable' | 'isaac_lab' | 'nut_assembly' | 'none';

export interface ReplayAdapterInput {
  taskType?: string;
  jobId?: string;
  datasetId?: string;
  replayJobId?: string;
  evalId?: string;
}

export interface ReplayAdapterResult {
  taskType: string;
  taskName: string;
  simulatorBackend: string;
  pageTitle: string;
  pageSubtitle: string;
  videoSourceLabel: string;
  previewNotice?: string | null;
  datasetId?: string;
  datasetName?: string;
  sourceJobId?: string;
  replayJobId?: string;
  episodeCount?: number;
  datasetFormat?: string;
  status?: string;
  sourceJobStatus?: string;
  generationMode?: string | null;
  seedSource?: string | null;
  createdAt?: string | null;
  videoSource: ReplayVideoSource;
  videoBackend: ReplayVideoBackend;
  videoJobId?: string;
  videoPath?: string | null;
  codec?: string | null;
  browserCompatible?: boolean;
  transcoded?: boolean;
  videoPlayable: boolean;
  canGenerateReplay: boolean;
  generateReplayDisabledReason?: string;
  replayInProgress: boolean;
  replayFailed: boolean;
  replayFailureMessage?: string | null;
  runConsoleHref?: string;
  replayConsoleHref?: string;
  datasetDetailHref?: string;
  metadata: Record<string, string | number | boolean | null | undefined>;
  error?: string;
  videoTag?: string;
  videoSourceDisplay?: string;
  replayContentKind?: import('@/lib/workspace/replayContentKind').ReplayContentKind;
  trajectories?: string[];
  trajectoryCount?: number | null;
  failureRecords?: import('@/lib/workspace/replayContentKind').ReplayFailureRecord[];
  totalEpisodes?: number | null;
  failedEpisodesCount?: number | null;
  primarySource?: string | null;
  trajectoryQualityLabel?: string | null;
  metricsSummary?: Record<string, unknown> | null;
  metricsAggregate?: Record<string, unknown> | null;
  defaultReplayTab?: import('@/lib/workspace/replayContentKind').ReplayContentKind;
  replayTabs?: Array<{ id: string; label: string }>;
  hasGenerationPreview?: boolean;
  hasHdf5Trajectories?: boolean;
  replayContent?: import('@/lib/workspace/replayContentKind').ReplayContentDetection;
  videoPlaceholderMessage?: string | null;
  datasetSourceLabel?: string | null;
}

function coerceMetricRate(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

const PAGE_SUBTITLE = '查看数据集回放、运行视频、episode 结果与产物状态';

export function replayVideoSourceLabel(source: ReplayVideoSource, transcoded?: boolean): string {
  if (transcoded || source === 'converted') return '视频来源：浏览器兼容转码视频';
  if (source === 'replay') return '视频来源：数据集回放 replay.mp4';
  if (source === 'preview') return '视频来源：数据生成 preview.mp4';
  if (source === 'generate') return '视频来源：数据生成 generate.mp4';
  if (source === 'episode') return '视频来源：Episode 录制视频';
  return '视频来源：暂无';
}

/** 回放页用户可见视频来源标签（评测场景不得显示生成预览文案） */
export function replayVideoSourceUserLabel(
  sourceKind: string,
  replayType: string,
  evaluationMode?: string | null
): string {
  if (sourceKind === 'evaluation' || replayType === 'evaluation') {
    const mode = (evaluationMode ?? '').trim();
    if (mode === 'trained_model_evaluation') return '训练模型 rollout 评测';
    if (mode === 'expert_policy_evaluation') return '专家策略评测';
    if (mode === 'episode_stability') return 'Episode 稳定性评测';
    return '评测回放';
  }
  if (replayType === 'dataset' || sourceKind === 'dataset') return '数据集回放';
  return '生成过程预览';
}

export function replayVideoTag(
  sourceKind: string,
  replayType: string,
  evaluationMode?: string | null
): string {
  if (sourceKind === 'evaluation' || replayType === 'evaluation') {
    const mode = (evaluationMode ?? '').trim();
    if (mode === 'episode_stability') return 'episode_stability.mp4';
    return 'evaluation.mp4';
  }
  return 'generate.mp4';
}

export function formatGenerationMode(mode: string | null | undefined): string | undefined {
  if (!mode) return undefined;
  const nutLabel = formatNutAssemblyGenerationMode(mode);
  if (nutLabel !== '—') return nutLabel;
  const isaacLabel = formatIsaacGenerationMode(mode);
  if (isaacLabel) return isaacLabel;
  return mode;
}

async function findDatasetById(datasetId: string): Promise<Dataset | null> {
  try {
    const response = await listWorkspaceDatasets();
    return response.datasets.find((row) => row.id === datasetId) ?? null;
  } catch {
    return null;
  }
}

function baseFromDataset(dataset: Dataset, taskName: string, simulatorBackend: string): Partial<ReplayAdapterResult> {
  return {
    datasetId: dataset.id,
    datasetName: dataset.name,
    sourceJobId: dataset.sourceJobId || undefined,
    episodeCount: dataset.episodeCount,
    datasetFormat: dataset.format?.toUpperCase() ?? dataset.datasetFormat ?? undefined,
    status: dataset.status === 'available' ? '可用' : dataset.status,
    createdAt: dataset.createdAt,
    taskName,
    simulatorBackend,
    datasetSourceLabel: resolveDatasetSourceLabel(dataset),
    pageSubtitle: PAGE_SUBTITLE,
    datasetDetailHref: `/workspace/data?dataset=${encodeURIComponent(dataset.id)}`,
  };
}

function inferVideoSourceFromArtifacts(
  artifacts: Array<{ filePath?: string | null; label?: string | null; artifactType?: string }>,
  defaultSource: ReplayVideoSource
): ReplayVideoSource {
  const paths = artifacts.map((a) => `${a.filePath ?? ''} ${a.label ?? ''}`.toLowerCase());
  if (paths.some((p) => p.includes('replay.mp4') || p.includes('replay.browser'))) return 'replay';
  if (paths.some((p) => p.includes('episode') && p.includes('.mp4'))) return 'episode';
  if (paths.some((p) => p.includes('preview.mp4') || p.includes('preview.browser'))) return 'preview';
  if (paths.some((p) => p.includes('generate.mp4'))) return 'generate';
  return defaultSource;
}

export async function resolveCableThreadingReplay(
  input: ReplayAdapterInput
): Promise<ReplayAdapterResult> {
  const taskType = 'cable_threading';
  const taskName = CABLE_THREADING_TASK_DISPLAY_NAME;
  const simulatorBackend = 'MuJoCo';
  const dataset =
    (input.datasetId ? await findDatasetById(input.datasetId) : null) ?? undefined;
  const jobId = input.jobId ?? dataset?.sourceJobId ?? input.evalId;

  const base: ReplayAdapterResult = {
    taskType,
    taskName,
    simulatorBackend,
    pageTitle: `${taskName} / ${simulatorBackend} 回放`,
    pageSubtitle: PAGE_SUBTITLE,
    videoSourceLabel: replayVideoSourceLabel('none'),
    videoSource: 'none',
    videoBackend: 'cable_threading',
    videoPlayable: false,
    canGenerateReplay: false,
    generateReplayDisabledReason: '当前任务暂不支持重新生成回放视频',
    replayInProgress: false,
    replayFailed: false,
    metadata: {},
    ...(dataset ? baseFromDataset(dataset, taskName, simulatorBackend) : {}),
  };

  if (!jobId) {
    base.error = '缺少 sourceJobId，无法加载回放';
    return base;
  }

  base.sourceJobId = jobId;
  base.runConsoleHref = buildCableThreadingConsoleHref({ jobId });

  if (!isPersistedWorkspaceJobId(jobId)) {
    return base;
  }

  try {
    const [job, artifactRes] = await Promise.all([
      getWorkspaceJob(jobId),
      getWorkspaceJobArtifacts(jobId),
    ]);
    base.sourceJobStatus = job.status;
    base.createdAt = base.createdAt ?? job.createdAt;
    const artifacts = artifactRes.artifacts ?? [];
    const videoSource = inferVideoSourceFromArtifacts(artifacts, 'generate');
    const hasVideo = job.videoAvailable || artifacts.some((a) => a.artifactType === 'video' && a.filePath);

    if (hasVideo) {
      base.videoSource = videoSource;
      base.videoBackend = 'cable_threading';
      base.videoJobId = jobId;
      base.videoPlayable = true;
      base.videoSourceLabel = replayVideoSourceLabel(videoSource);
    }

    base.metadata = {
      ...base.metadata,
      successRate:
        coerceMetricRate(job.metricsSummary?.successRate) ??
        coerceMetricRate(job.metricsSummary?.finalSuccessRate),
      taskName: job.taskName,
    };
  } catch {
    base.error = '无法加载线缆穿杆运行记录';
  }

  return base;
}

export async function resolveDualArmCableReplay(
  input: ReplayAdapterInput
): Promise<ReplayAdapterResult> {
  const taskType = 'dual_arm_cable_manipulation';
  const taskName = DUAL_ARM_CABLE_TASK_NAME;
  const simulatorBackend = 'MuJoCo';
  const dataset =
    (input.datasetId ? await findDatasetById(input.datasetId) : null) ?? undefined;
  const jobId = input.jobId ?? dataset?.sourceJobId;

  const base: ReplayAdapterResult = {
    taskType,
    taskName,
    simulatorBackend,
    pageTitle: `${taskName} / ${simulatorBackend} 回放`,
    pageSubtitle: PAGE_SUBTITLE,
    videoSourceLabel: replayVideoSourceLabel('none'),
    videoSource: 'none',
    videoBackend: 'dual_arm_cable',
    videoPlayable: false,
    canGenerateReplay: false,
    generateReplayDisabledReason: '当前任务暂不支持重新生成回放视频',
    replayInProgress: false,
    replayFailed: false,
    metadata: {},
    ...(dataset ? baseFromDataset(dataset, taskName, simulatorBackend) : {}),
  };

  if (!jobId?.startsWith('dac_gen_')) {
    base.error = jobId ? '无效的双臂线缆 sourceJobId' : '缺少 sourceJobId，无法加载回放';
    return base;
  }

  base.sourceJobId = jobId;
  base.runConsoleHref = buildDualArmCableConsoleHref({ jobId });

  try {
    const status = await getDualArmCableJobStatus(jobId);
    base.sourceJobStatus = status.status;
    base.metadata = {
      ...base.metadata,
      episodeSuccess: status.metrics?.episode_success ?? null,
      maxCables: status.maxCables ?? null,
      videoPath: status.videoPath ?? null,
    };

    if (status.videoExists) {
      base.videoSource = status.videoPath?.includes('episode') ? 'episode' : 'generate';
      base.videoBackend = 'dual_arm_cable';
      base.videoJobId = jobId;
      base.videoPath = status.videoPath ?? null;
      base.videoPlayable = true;
      base.videoSourceLabel = replayVideoSourceLabel(base.videoSource);
    }
  } catch {
    base.error = '无法加载双臂线缆运行记录';
  }

  return base;
}

function isaacContextToAdapter(ctx: IsaacLabDatasetReplayContext): ReplayAdapterResult {
  const playback = ctx.playback;
  const taskName = ISAAC_BLOCK_STACKING_DISPLAY_NAME;
  const simulatorBackend = 'Isaac Lab';
  const sourceJobId = ctx.sourceJobId ?? undefined;
  const replayJobId = ctx.replayJobId ?? undefined;

  let videoSource: ReplayVideoSource = 'none';
  if (playback?.videoSourceKind === 'replay') videoSource = playback.transcoded ? 'converted' : 'replay';
  else if (playback?.videoSourceKind === 'preview') videoSource = playback.transcoded ? 'converted' : 'preview';
  else if (playback?.videoSourceKind === 'videos') videoSource = playback.transcoded ? 'converted' : 'episode';
  else if (playback?.videoSource === 'converted') videoSource = 'converted';

  const result: ReplayAdapterResult = {
    taskType: 'isaac_block_stacking',
    taskName,
    simulatorBackend,
    pageTitle: `${taskName} / ${simulatorBackend} 回放`,
    pageSubtitle: PAGE_SUBTITLE,
    videoSourceLabel: ctx.videoSourceLabel,
    previewNotice: ctx.usingPreviewFallback
      ? '当前播放生成预览视频；正式数据集回放视频可通过「生成回放视频」创建。'
      : null,
    datasetId: ctx.dataset.id,
    datasetName: ctx.dataset.name,
    sourceJobId,
    replayJobId,
    episodeCount: ctx.dataset.episodeCount,
    datasetFormat: 'HDF5',
    status: ctx.dataset.status === 'available' ? '可用' : ctx.dataset.status,
    sourceJobStatus: ctx.sourceJobStatus?.status,
    generationMode: ctx.sourceJobStatus?.generationMode ?? null,
    seedSource: ctx.sourceJobStatus?.seedSource ?? null,
    createdAt: ctx.dataset.createdAt,
    videoSource,
    videoBackend: playback?.playable ? 'isaac_lab' : 'none',
    videoJobId: playback?.videoJobId ?? undefined,
    videoPath: playback?.browserVideoPath ?? playback?.rawVideoPath ?? null,
    codec: playback?.codec ?? null,
    browserCompatible: playback?.browserCompatible,
    transcoded: playback?.transcoded,
    videoPlayable: playback?.playable === true,
    canGenerateReplay: ctx.hasDatasetFile,
    generateReplayDisabledReason: ctx.hasDatasetFile ? undefined : '缺少 datasetFile，无法生成回放',
    replayInProgress: ctx.replayInProgress,
    replayFailed: ctx.replayFailed,
    replayFailureMessage:
      ctx.replayFailed
        ? ctx.replayJobs.find((j) => j.jobId === replayJobId)?.message ?? '回放任务失败'
        : null,
    runConsoleHref: sourceJobId ? buildIsaacBlockStackingConsoleHref({ jobId: sourceJobId }) : undefined,
    replayConsoleHref: replayJobId
      ? buildIsaacBlockStackingReplayConsoleHref({ jobId: replayJobId })
      : undefined,
    datasetDetailHref: `/workspace/data?dataset=${encodeURIComponent(ctx.dataset.id)}`,
    metadata: {
      numDemos: ctx.sourceJobStatus?.numDemos ?? null,
      datasetFile: ctx.dataset.datasetFile ?? null,
    },
  };

  if (!playback?.playable && !ctx.replayInProgress) {
    result.videoSourceLabel = replayVideoSourceLabel('none');
  }

  return result;
}

export async function resolveIsaacBlockStackingReplay(
  input: ReplayAdapterInput
): Promise<ReplayAdapterResult> {
  const datasetId = input.datasetId;
  if (!datasetId) {
    return {
      taskType: 'isaac_block_stacking',
      taskName: ISAAC_BLOCK_STACKING_DISPLAY_NAME,
      simulatorBackend: 'Isaac Lab',
      pageTitle: `${ISAAC_BLOCK_STACKING_DISPLAY_NAME} / Isaac Lab 回放`,
      pageSubtitle: PAGE_SUBTITLE,
      videoSourceLabel: replayVideoSourceLabel('none'),
      videoSource: 'none',
      videoBackend: 'none',
      videoPlayable: false,
      canGenerateReplay: false,
      replayInProgress: false,
      replayFailed: false,
      metadata: {},
      error: '缺少 datasetId，无法加载 Isaac 回放上下文',
    };
  }

  try {
    const ctx = await getIsaacLabDatasetReplayContext(datasetId);
    return isaacContextToAdapter(ctx);
  } catch (err) {
    return {
      taskType: 'isaac_block_stacking',
      taskName: ISAAC_BLOCK_STACKING_DISPLAY_NAME,
      simulatorBackend: 'Isaac Lab',
      pageTitle: `${ISAAC_BLOCK_STACKING_DISPLAY_NAME} / Isaac Lab 回放`,
      pageSubtitle: PAGE_SUBTITLE,
      videoSourceLabel: replayVideoSourceLabel('none'),
      videoSource: 'none',
      videoBackend: 'none',
      videoPlayable: false,
      canGenerateReplay: false,
      replayInProgress: false,
      replayFailed: false,
      metadata: {},
      datasetId,
      error: err instanceof Error ? err.message : '加载 Isaac 回放上下文失败',
    };
  }
}

export async function resolveReplayAdapter(input: ReplayAdapterInput): Promise<ReplayAdapterResult> {
  const taskType =
    inferReplayTaskTypeFromJobId(input.jobId) ??
    input.taskType ??
    (input.datasetId
      ? resolveDatasetReplayTaskType(
          (await findDatasetById(input.datasetId)) ?? {
            id: input.datasetId,
            sourceJobId: input.jobId ?? '',
          } as Dataset
        )
      : undefined);

  if (taskType === 'isaac_block_stacking') {
    return resolveIsaacBlockStackingReplay(input);
  }
  if (taskType === 'dual_arm_cable_manipulation') {
    return resolveDualArmCableReplay(input);
  }
  if (
    taskType === 'nut_assembly' ||
    taskType === 'nut_assembly_single_arm' ||
    taskType === 'task_nut_assembly_v1'
  ) {
    return resolveNutAssemblyReplay(input);
  }
  return resolveCableThreadingReplay(input);
}

export async function generateReplayForAdapter(
  result: ReplayAdapterResult
): Promise<{ replayJobId: string; reused: boolean; refreshHref: string }> {
  if (result.taskType !== 'isaac_block_stacking' || !result.datasetId) {
    throw new Error(result.generateReplayDisabledReason ?? '当前任务不支持生成回放视频');
  }
  const started = await startIsaacLabReplayFromDataset(result.datasetId);
  return {
    replayJobId: started.jobId,
    reused: started.reused === true,
    refreshHref: buildUnifiedDatasetReplayHref({
      taskType: result.taskType,
      datasetId: result.datasetId,
      sourceJobId: result.sourceJobId,
      replayJobId: started.jobId,
    }),
  };
}
