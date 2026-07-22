import { getNutAssemblyJobResult, getNutAssemblyJobStatus } from '@/lib/api/nutAssemblyClient';
import {
  buildNutAssemblyConsoleHref,
  buildNutAssemblyReplayHref,
  formatNutAssemblyPolicyMode,
  NUT_ASSEMBLY_TASK_DISPLAY_NAME,
} from '@/lib/workspace/nutAssembly';
import type { ReplayAdapterInput, ReplayAdapterResult } from '@/lib/workspace/replayAdapters';
import { buildReplayPageTitle } from '@/lib/workspace/taskDisplayNames';

const PAGE_SUBTITLE = '查看数据生成结果与回放视频';

function formatFailureDistribution(value: unknown): string | null {
  if (!value || typeof value !== 'object') return null;
  const entries = Object.entries(value as Record<string, number>);
  if (!entries.length) return null;
  return entries.map(([k, v]) => `${k}:${v}`).join(', ');
}

export async function resolveNutAssemblyReplay(
  input: ReplayAdapterInput
): Promise<ReplayAdapterResult> {
  const taskType = 'nut_assembly';
  const taskName = NUT_ASSEMBLY_TASK_DISPLAY_NAME;
  const jobId = input.jobId?.trim();

  const base: ReplayAdapterResult = {
    taskType,
    taskName,
    simulatorBackend: 'MuJoCo',
    pageTitle: buildReplayPageTitle(taskName),
    pageSubtitle: PAGE_SUBTITLE,
    videoSourceLabel: '',
    videoTag: 'generate.mp4',
    videoSourceDisplay: '生成视频 generate.mp4',
    videoSource: 'generate',
    videoBackend: 'nut_assembly',
    videoPlayable: false,
    canGenerateReplay: false,
    generateReplayDisabledReason: 'NutAssembly 暂不支持重新生成回放视频',
    replayInProgress: false,
    replayFailed: false,
    metadata: {},
    datasetId: input.datasetId,
  };

  if (!jobId) {
    base.error = '缺少 jobId，无法加载 NutAssembly 回放';
    return base;
  }

  base.sourceJobId = jobId;
  base.runConsoleHref = buildNutAssemblyConsoleHref({ jobId, dataId: input.datasetId });

  try {
    const [status, result] = await Promise.all([
      getNutAssemblyJobStatus(jobId),
      getNutAssemblyJobResult(jobId).catch(() => null),
    ]);
    const summary = (result?.summary ?? status.metrics?.summary ?? {}) as Record<string, unknown>;
    const live = (status.live ?? {}) as Record<string, unknown>;
    const videoExists = Boolean(
      status.generateVideoExists ?? status.paths?.generateVideo?.exists ?? summary.videoStatus === 'available'
    );

    base.status = status.status;
    base.generationMode =
      (typeof result?.generationMode === 'string' ? result.generationMode : null) ??
      status.generationMode ??
      null;
    const policyMode =
      (typeof result?.policyMode === 'string' ? result.policyMode : null) ?? status.policyMode ?? null;

    const successRate = summary.successRate ?? status.successRate ?? live.successRate;
    const validForTraining = summary.validForTrainingEpisodes ?? live.validForTrainingEpisodes;
    const failureDistribution = summary.failureDistribution ?? status.failureDistribution ?? live.failureDistribution;
    const validForTrainingEpisodes =
      typeof validForTraining === 'number' && Number.isFinite(validForTraining)
        ? validForTraining
        : typeof validForTraining === 'string' && validForTraining.trim() && Number.isFinite(Number(validForTraining))
          ? Number(validForTraining)
          : null;

    base.metadata = {
      jobId,
      generationMode: base.generationMode ?? null,
      policyMode: policyMode ? formatNutAssemblyPolicyMode(String(policyMode)) : null,
      successRate: successRate != null ? `${Math.round(Number(successRate) * 1000) / 10}%` : null,
      validForTrainingEpisodes,
      failureDistribution: formatFailureDistribution(failureDistribution),
      hdf5Path:
        status.paths?.hdf5?.path ??
        (typeof summary.datasetPath === 'string' ? summary.datasetPath : null),
      videoPath: typeof summary.videoPath === 'string' ? summary.videoPath : 'videos/generate.mp4',
    };

    if (videoExists) {
      base.videoBackend = 'nut_assembly';
      base.videoJobId = jobId;
      base.videoPlayable = true;
      base.videoSource = 'generate';
      base.videoSourceDisplay = '生成视频 generate.mp4';
      base.videoPath = String(summary.videoPath ?? 'videos/generate.mp4');
      base.hasGenerationPreview = true;
      base.replayContentKind = 'generation_process_preview';
      base.defaultReplayTab = 'generation_process_preview';
      base.replayTabs = [{ id: 'generation_process_preview', label: '生成过程预览' }];
    } else {
      base.videoPlaceholderMessage =
        '未生成 generate.mp4。后续版本将支持 HDF5 / object_poses / action trajectory 回放占位。';
      base.hasHdf5Trajectories = Boolean(status.paths?.hdf5?.exists);
      base.replayContentKind = 'dataset_trajectory_replay';
      base.defaultReplayTab = 'dataset_trajectory_replay';
    }

    base.replayConsoleHref = buildNutAssemblyReplayHref({ jobId, datasetId: input.datasetId });
    if (!videoExists && !status.paths?.hdf5?.exists) {
      base.error = '缺少视频与 HDF5，无法回放';
    }
  } catch (err) {
    base.error = err instanceof Error ? err.message : '无法加载 NutAssembly 运行记录';
  }

  return base;
}
