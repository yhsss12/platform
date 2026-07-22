import { CABLE_OBJECT_MODEL_OPTIONS } from '@/lib/mock/generateDataTaskParams';
import {
  parseDataVolume,
  type WorkspaceDataItem,
} from '@/lib/mock/workspaceDataMock';
import type {
  BuildDatasetPayload,
  DataOrganizationFormat,
  DownstreamModelType,
} from '@/lib/workspace/buildDatasetLegacyTypes';

export type DatasetUsageKey = 'training' | 'evaluation' | 'training_and_evaluation';

export type DatasetQualityStatus = 'ready' | 'missing_image' | 'no_success';

export interface DatasetManifestArtifacts {
  npz?: string;
  hdf5?: string;
  hdf5Paths?: string[];
  manifest?: string;
  collectCsv?: string;
  failures?: string;
  generateVideo?: string;
  timeline?: string;
  runLog?: string;
}

export interface DatasetManifestQuality {
  status: DatasetQualityStatus;
  hasTrajectory: boolean;
  hasImage: boolean;
  hasVideo: boolean;
  hasTimeline: boolean;
  hasSuccessfulEpisodes: boolean;
}

export interface DatasetManifest {
  datasetId: string;
  datasetName: string;
  taskType?: string;
  taskName: string;
  sourceJobId: string;
  sourceRecordName: string;
  backend: string;
  robot?: string;
  objectModel?: string;
  difficulty?: string;
  episodes: number;
  successfulEpisodes: number;
  successRate: number;
  usage: DatasetUsageKey;
  downstreamModelType: DownstreamModelType;
  dataFormat: DataOrganizationFormat;
  trainingView: string;
  mainFormats: string[];
  split: {
    enabled: boolean;
    trainRatio: number;
    valRatio: number;
  };
  artifacts: DatasetManifestArtifacts;
  quality: DatasetManifestQuality;
  createdAt: string;
}

export interface SourceArtifactSnapshot {
  episodes: number;
  successfulEpisodes: number;
  successRate: number;
  backend: string;
  robot?: string;
  objectModel?: string;
  objectModelLabel: string;
  difficulty?: string;
  sourceJobId: string;
  sourceRecordName: string;
  hasNpz: boolean;
  hasHdf5: boolean;
  hasVideo: boolean;
  hasTimeline: boolean;
  hasFailures: boolean;
  hasTrajectory: boolean;
  npzPath?: string;
  hdf5Path?: string;
  manifestPath?: string;
  collectCsvPath?: string;
  failuresPath?: string;
  generateVideoPath?: string;
  timelinePath?: string;
  runLogPath?: string;
}

export interface DatasetBuildQualityOptions {
  dataOrganizationFormat?: DataOrganizationFormat;
  downstreamModelType?: DownstreamModelType;
}

export interface DatasetQualityCheck {
  episodes: number;
  successfulEpisodes: number;
  successRatePercent: number;
  trajectoryStatus: string;
  hdf5Status: string;
  npzStatus: string;
  videoStatus: string;
  failuresStatus: string;
  timelineStatus: string;
  trainability: string;
  canBuild: boolean;
  buildDisabledReason?: string;
  hdf5Required?: boolean;
  npzOnlyNote?: string;
}

const IMAGE_BACKED_MODELS = new Set<DownstreamModelType>([
  'ACT',
  'DT',
  'Diffusion Policy',
  'Robomimic',
]);

export function dataFormatRequiresHdf5(format?: DataOrganizationFormat): boolean {
  if (!format) return false;
  return format.includes('HDF5');
}

export function buildRequiresHdf5Artifact(
  format?: DataOrganizationFormat,
  downstreamModelType?: DownstreamModelType
): boolean {
  if (downstreamModelType === 'Robomimic') return true;
  return dataFormatRequiresHdf5(format);
}

export function isHdf5ArtifactReady(source: SourceArtifactSnapshot): boolean {
  return Boolean(source.hdf5Path?.trim()) && source.hasHdf5;
}

const TRAINING_VIEW_MAP: Record<DownstreamModelType, string> = {
  ACT: 'imitation_action_sequence',
  DT: 'trajectory_sequence_return_to_go',
  'Diffusion Policy': 'action_trajectory_window',
  Robomimic: 'demonstration_episodes',
  LeRobot: 'episode_dataset',
  自定义模型: 'custom',
};

const TRAINING_VIEW_LABEL_MAP: Record<DownstreamModelType, string> = {
  ACT: '模仿学习 / action sequence',
  DT: '轨迹序列 / return-to-go',
  'Diffusion Policy': '动作轨迹窗口',
  Robomimic: 'demonstration episodes',
  LeRobot: 'episode dataset',
  自定义模型: '自定义训练视图',
};

function deriveJobRootFromArtifact(artifactPath?: string): string | undefined {
  if (!artifactPath) return undefined;
  const match = artifactPath.match(/^(.*\/jobs\/[^/]+)\//);
  return match?.[1];
}

export function deriveRunLogPath(artifactPath?: string): string | undefined {
  const root = deriveJobRootFromArtifact(artifactPath);
  return root ? `${root}/logs/run.log` : undefined;
}

export function deriveTimelinePath(artifactPath?: string): string | undefined {
  const root = deriveJobRootFromArtifact(artifactPath);
  return root ? `${root}/live/generate_timeline.json` : undefined;
}

export function formatObjectModelLabel(internalValue?: string): string {
  if (!internalValue) return '—';
  const option = CABLE_OBJECT_MODEL_OPTIONS.find((o) => o.value === internalValue);
  return option?.label ?? internalValue;
}

export function getTrainingViewKey(modelType: DownstreamModelType): string {
  return TRAINING_VIEW_MAP[modelType];
}

export function getTrainingViewLabel(modelType: DownstreamModelType): string {
  return TRAINING_VIEW_LABEL_MAP[modelType];
}

export function purposeToUsageKey(purpose: BuildDatasetPayload['purpose']): DatasetUsageKey {
  switch (purpose) {
    case '训练数据集':
      return 'training';
    case '评测数据集':
      return 'evaluation';
    case '训练与评测':
      return 'training_and_evaluation';
  }
}

export function dataFormatToMainFormats(format: DataOrganizationFormat): string[] {
  switch (format) {
    case 'HDF5':
      return ['hdf5'];
    case 'NPZ':
      return ['npz'];
    case 'LeRobot':
      return ['lerobot'];
    case 'HDF5 + NPZ':
      return ['hdf5', 'npz'];
  }
}

function pad2(n: number) {
  return String(n).padStart(2, '0');
}

export function generateDefaultDatasetName(taskName: string, existingNames: string[] = []): string {
  const now = new Date();
  const datePart = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}`;
  const prefix = `${taskName}数据集_${datePart}_`;
  const sameDay = existingNames.filter((name) => name.startsWith(prefix));
  let seq = sameDay.length + 1;
  let candidate = `${prefix}${String(seq).padStart(3, '0')}`;
  while (existingNames.includes(candidate)) {
    seq += 1;
    candidate = `${prefix}${String(seq).padStart(3, '0')}`;
  }
  return candidate;
}

export function generateDatasetId(taskType?: string): string {
  const now = new Date();
  const stamp = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}_${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`;
  const slug = taskType === 'cable_threading' ? 'cable' : 'ds';
  return `ds_${slug}_${stamp}`;
}

export function resolveSourceArtifacts(item: WorkspaceDataItem): SourceArtifactSnapshot {
  const { count } = parseDataVolume(item.dataVolume);
  const episodes = count ?? item.successfulEpisodes ?? 0;
  const successfulEpisodes =
    item.successfulEpisodes ??
    (episodes > 0 ? Math.max(1, Math.round(episodes * ((item.successRate ?? 90) / 100))) : 0);
  const successRate =
    item.successRate != null
      ? item.successRate / 100
      : episodes > 0
        ? successfulEpisodes / episodes
        : 0;

  const anchorPath = item.npzPath ?? item.hdf5Path ?? item.collectCsvPath ?? item.manifestPath;
  const timelinePath = deriveTimelinePath(anchorPath);
  const runLogPath = deriveRunLogPath(anchorPath);

  const hasNpz = Boolean(item.npzPath);
  const hasHdf5 = Boolean(item.hdf5Path);
  const hasVideo = item.generateVideoExists === true || Boolean(item.generateVideoPath);
  const hasFailures = Boolean(item.failuresPath);
  const hasTimeline = Boolean(timelinePath);
  const hasTrajectory = hasNpz || Boolean(item.saveTrajectory) || item.contents?.includes('轨迹') === true;

  return {
    episodes,
    successfulEpisodes,
    successRate,
    backend: item.simBackend ?? 'MuJoCo',
    robot: item.robot,
    objectModel: item.cableModel,
    objectModelLabel: formatObjectModelLabel(item.cableModel),
    difficulty: item.difficulty,
    sourceJobId: item.simulationId || item.id,
    sourceRecordName: item.name,
    hasNpz,
    hasHdf5,
    hasVideo,
    hasTimeline,
    hasFailures,
    hasTrajectory,
    npzPath: item.npzPath,
    hdf5Path: item.hdf5Path,
    manifestPath: item.manifestPath,
    collectCsvPath: item.collectCsvPath,
    failuresPath: item.failuresPath,
    generateVideoPath: item.generateVideoPath,
    timelinePath,
    runLogPath,
  };
}

export function deriveAutoBuildFlags(source: SourceArtifactSnapshot): Pick<
  BuildDatasetPayload,
  | 'includeTrajectory'
  | 'includeImageObservation'
  | 'includeStateAction'
  | 'includeProcessVideo'
  | 'includeRunLog'
  | 'includeFailures'
  | 'includeTimeline'
> {
  const hasTrajectory = source.hasNpz || source.hasTrajectory;
  return {
    includeTrajectory: hasTrajectory,
    includeImageObservation: source.hasHdf5,
    includeStateAction: hasTrajectory,
    includeProcessVideo: source.hasVideo,
    includeRunLog: Boolean(source.runLogPath),
    includeFailures: source.hasFailures,
    includeTimeline: source.hasTimeline,
  };
}

export interface BuildArtifactSummaryItem {
  label: string;
  status: '已生成' | '未生成';
}

export function buildArtifactSummary(source: SourceArtifactSnapshot): BuildArtifactSummaryItem[] {
  const status = (exists: boolean): '已生成' | '未生成' => (exists ? '已生成' : '未生成');
  return [
    { label: '轨迹数据 NPZ', status: status(source.hasNpz || source.hasTrajectory) },
    { label: '图像观测 HDF5', status: status(source.hasHdf5) },
    { label: '过程视频 MP4', status: status(source.hasVideo) },
    { label: '运行日志', status: status(Boolean(source.runLogPath)) },
    { label: '失败记录', status: status(source.hasFailures) },
    { label: '阶段同步文件', status: status(source.hasTimeline) },
  ];
}

export function buildQualityCheck(
  source: SourceArtifactSnapshot,
  options?: DatasetBuildQualityOptions
): DatasetQualityCheck {
  const successRatePercent = Math.round(source.successRate * 1000) / 10;
  const format = options?.dataOrganizationFormat;
  const downstream = options?.downstreamModelType;
  const hdf5Required = buildRequiresHdf5Artifact(format, downstream);
  const hdf5Ready = isHdf5ArtifactReady(source);
  const npzOnly = format === 'NPZ';

  let trainability = '可构建';
  let canBuild = true;
  let buildDisabledReason: string | undefined;
  let npzOnlyNote: string | undefined;

  if (source.successfulEpisodes <= 0) {
    trainability = '无成功轨迹';
    canBuild = false;
    buildDisabledReason = '当前记录无成功轨迹，不建议构建训练数据集。';
  } else if (!source.hasNpz && !source.hasTrajectory) {
    trainability = '缺少轨迹数据';
    canBuild = false;
    buildDisabledReason = '当前记录缺少轨迹数据，无法构建数据集。';
  } else if (hdf5Required && !hdf5Ready) {
    trainability = '缺少 HDF5';
    canBuild = false;
    if (downstream === 'Robomimic' || format?.includes('HDF5')) {
      buildDisabledReason = '当前记录未生成 HDF5，无法构建 Robomimic 训练数据集。';
    } else if (downstream && IMAGE_BACKED_MODELS.has(downstream)) {
      buildDisabledReason =
        '图像观测未生成，无法用于需要 HDF5 图像数据的训练后端。';
    } else {
      buildDisabledReason = '当前记录未生成 HDF5，无法按所选数据格式构建数据集。';
    }
  } else if (npzOnly) {
    npzOnlyNote = '仅轨迹数据，可用于后续自定义处理。';
    trainability = npzOnlyNote;
  }

  return {
    episodes: source.episodes,
    successfulEpisodes: source.successfulEpisodes,
    successRatePercent,
    trajectoryStatus: source.hasTrajectory || source.hasNpz ? '已生成' : '未生成',
    hdf5Status: source.hasHdf5 ? '已生成' : '未生成',
    npzStatus: source.hasNpz ? '已生成' : '未生成',
    videoStatus: source.hasVideo ? '已生成' : '未生成',
    failuresStatus: source.hasFailures ? '已生成' : '未生成',
    timelineStatus: source.hasTimeline ? '已生成' : '未生成',
    trainability,
    canBuild,
    buildDisabledReason,
    hdf5Required,
    npzOnlyNote,
  };
}

function resolveUsageEpisodeCount(
  source: SourceArtifactSnapshot,
  payload: BuildDatasetPayload
): number {
  switch (payload.usageScope) {
    case '全部成功轨迹':
      return source.successfulEpisodes;
    case '全部轨迹':
      return source.episodes;
    case '自定义数量':
      return Math.min(payload.customEpisodeCount ?? source.successfulEpisodes, source.episodes);
  }
}

function resolveQualityStatus(
  source: SourceArtifactSnapshot,
  includeImageObservation: boolean
): DatasetQualityStatus {
  if (source.successfulEpisodes <= 0) return 'no_success';
  if (includeImageObservation && !source.hasHdf5) return 'missing_image';
  return 'ready';
}

export function validateBuildPayload(
  source: SourceArtifactSnapshot,
  payload: BuildDatasetPayload
): { ok: boolean; reason?: string } {
  const quality = buildQualityCheck(source, {
    dataOrganizationFormat: payload.dataOrganizationFormat,
    downstreamModelType: payload.downstreamModelType,
  });
  if (!quality.canBuild) {
    return { ok: false, reason: quality.buildDisabledReason ?? quality.trainability };
  }
  return { ok: true };
}

export function buildDatasetManifest(
  sourceItem: WorkspaceDataItem,
  payload: BuildDatasetPayload
): DatasetManifest {
  const source = resolveSourceArtifacts(sourceItem);
  const validation = validateBuildPayload(source, payload);
  if (!validation.ok) {
    throw new Error(validation.reason ?? '数据集构建校验未通过');
  }
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const createdAt = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

  const splitEnabled = payload.splitMode === 'train_val_80_20' || payload.splitMode === 'custom';
  const trainRatio = payload.splitMode === 'custom' ? (payload.customTrainRatio ?? 0.8) : 0.8;
  const valRatio = splitEnabled ? Math.max(0, 1 - trainRatio) : 0;

  const usageEpisodes = resolveUsageEpisodeCount(source, payload);
  const needsHdf5 = buildRequiresHdf5Artifact(
    payload.dataOrganizationFormat,
    payload.downstreamModelType
  );
  const qualityStatus = resolveQualityStatus(
    source,
    payload.includeImageObservation || needsHdf5
  );

  const datasetId = generateDatasetId(sourceItem.taskType);
  const datasetName = payload.outputName.trim();

  const manifestPath = source.manifestPath
    ?? (source.npzPath
      ? source.npzPath.replace(/\/datasets\/[^/]+$/, '/datasets/dataset.manifest.json')
      : undefined);
  const manifest: DatasetManifest = {
    datasetId,
    datasetName,
    taskType: sourceItem.taskType,
    taskName: sourceItem.taskName,
    sourceJobId: source.sourceJobId,
    sourceRecordName: source.sourceRecordName,
    backend: source.backend,
    robot: source.robot,
    objectModel: source.objectModel,
    difficulty: source.difficulty,
    episodes: source.episodes,
    successfulEpisodes: source.successfulEpisodes,
    successRate: source.successRate,
    usage: purposeToUsageKey(payload.purpose),
    downstreamModelType: payload.downstreamModelType,
    dataFormat: payload.dataOrganizationFormat,
    trainingView: getTrainingViewKey(payload.downstreamModelType),
    mainFormats: dataFormatToMainFormats(payload.dataOrganizationFormat),
    split: {
      enabled: splitEnabled,
      trainRatio: splitEnabled ? trainRatio : 0,
      valRatio: splitEnabled ? valRatio : 0,
    },
    artifacts: {
      npz: payload.includeTrajectory ? source.npzPath : undefined,
      hdf5:
        source.hdf5Path &&
        (needsHdf5 || payload.includeImageObservation)
          ? source.hdf5Path
          : undefined,
      manifest: manifestPath,
      collectCsv: source.collectCsvPath,
      failures: payload.includeFailures ? source.failuresPath : undefined,
      generateVideo: payload.includeProcessVideo ? source.generateVideoPath : undefined,
      timeline: payload.includeTimeline ? source.timelinePath : undefined,
      runLog: payload.includeRunLog ? source.runLogPath : undefined,
    },
    quality: {
      status: qualityStatus,
      hasTrajectory: source.hasTrajectory || source.hasNpz,
      hasImage: source.hasHdf5,
      hasVideo: source.hasVideo,
      hasTimeline: source.hasTimeline,
      hasSuccessfulEpisodes: source.successfulEpisodes > 0,
    },
    createdAt,
  };

  if (needsHdf5 && !manifest.artifacts.hdf5?.trim()) {
    throw new Error('当前记录未生成 HDF5，无法构建 Robomimic 训练数据集。');
  }

  return manifest;
}

export function formatQualityStatusLabel(status?: DatasetQualityStatus | string): string {
  switch (status) {
    case 'ready':
      return '可构建';
    case 'missing_image':
      return '缺少图像数据';
    case 'no_success':
      return '无成功轨迹';
    default:
      return status ?? '—';
  }
}

export function formatQualitySummaryLine(
  source: SourceArtifactSnapshot,
  quality: DatasetQualityCheck
): { main: string; hint?: string } {
  if (!quality.canBuild) {
    if (source.successfulEpisodes <= 0) {
      return { main: '不可构建 · 当前记录无成功轨迹' };
    }
    return {
      main: `不可构建 · ${quality.buildDisabledReason ?? quality.trainability}`,
      hint: quality.hdf5Required && !isHdf5ArtifactReady(source)
        ? '图像观测未生成，无法用于需要 HDF5 图像数据的训练后端。'
        : undefined,
    };
  }

  const parts = [
    quality.npzOnlyNote ? '可构建（仅轨迹）' : '可构建',
    `${source.successfulEpisodes} / ${source.episodes} 条成功轨迹`,
    `轨迹数据${quality.trajectoryStatus === '已生成' ? '已生成' : '未生成'}`,
  ];
  if (quality.hdf5Status === '已生成') parts.push('HDF5 已生成');
  if (quality.videoStatus === '已生成') parts.push('过程视频已生成');
  if (quality.failuresStatus === '已生成') parts.push('失败记录已生成');

  let hint: string | undefined;
  if (quality.npzOnlyNote) {
    hint = quality.npzOnlyNote;
  } else if (!source.hasHdf5 && quality.hdf5Required) {
    hint = '图像观测未生成，无法用于需要 HDF5 图像数据的训练后端。';
  }

  return { main: parts.join(' · '), hint };
}

export function formatUsageScopeSummary(
  source: SourceArtifactSnapshot,
  usageScope: BuildDatasetPayload['usageScope'],
  customCount?: number
): string {
  switch (usageScope) {
    case '全部成功轨迹':
      return `${source.successfulEpisodes} / ${source.episodes} 条成功轨迹`;
    case '全部轨迹':
      return `${source.episodes} 条轨迹`;
    case '自定义数量':
      return `${customCount ?? source.successfulEpisodes} / ${source.episodes} 条轨迹`;
  }
}
