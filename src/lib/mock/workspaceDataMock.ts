/** 数据中心 mock 数据 — collect_data / process_data 流程（不接后端） */

import type { PhysicsProxyMode } from '@/lib/mock/physicsProxiesMock';
import {
  isPendingLocalJobId,
  isValidCableThreadingGenerateJobId,
  isValidDualArmGenerateJobId,
  isValidDataGenJobId,
  isValidIsaacGenerateJobId,
  isValidNutAssemblyGenerateJobId,
} from '@/lib/workspace/backendJobIds';

export type WorkspaceDataCategory =
  | '示范数据'
  | '训练数据集'
  | '评测数据集'
  | '外部数据'
  | '真实数据';

/** @deprecated 使用 WorkspaceDataCategory */
export type WorkspaceDataType = WorkspaceDataCategory;

export type WorkspaceDataStatus =
  | 'completed'
  | 'generating'
  | 'pending'
  | 'exported'
  | 'failed'
  | 'built';

/** 数据集构建阶段状态（与生成状态独立） */
export type DatasetBuildStatus = 'none' | 'building' | 'built' | 'failed';

export type WorkspaceDataSource = 'MuJoCo 生成' | '外部导入' | '真实采集';

/** Adapter 层保留，不在数据中心 UI 展示 */
export type WorkspaceDataSourceAdapter = WorkspaceDataSource | 'RoboTwin 生成';

export type WorkspaceTargetModelFormat =
  | 'ACT'
  | 'DT'
  | 'DP3'
  | 'Diffusion Policy'
  | 'Robomimic'
  | 'LeRobot'
  | 'OpenVLA'
  | '自定义模型'
  | '通用格式'
  | '—';

export interface WorkspaceDataItem {
  id: string;
  name: string;
  /** 关联任务内部 ID，仅用于 Drawer / 搜索 */
  taskId: string;
  taskName: string;
  /** 内部 run / batch ID，不在列表主视图展示 */
  simulationId: string;
  dataCategory: WorkspaceDataCategory;
  /** @deprecated 使用 dataCategory */
  dataType?: WorkspaceDataCategory;
  source: WorkspaceDataSource;
  targetModelFormat: WorkspaceTargetModelFormat;
  dataVolume: string;
  size: string;
  status: WorkspaceDataStatus;
  generatedAt: string;
  creator: string;
  /** Drawer 详情字段 */
  scene?: string;
  robot?: string;
  policy?: string;
  contents?: string[];
  sampleRate?: string;
  frameOrTrajectoryCount?: string;
  taskConfig?: string;
  simBackend?: string;
  dataPurpose?: string;
  saveVideo?: boolean;
  saveTrajectory?: boolean;
  saveStateLog?: boolean;
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string | null;
  physicsProxyErrorThreshold?: number;
  physicsProxyReviewRatio?: number;
  /** 真实后端任务记录 */
  taskType?: 'cable_threading' | 'dual_arm_cable_manipulation';
  /** 真实后端 jobId（如 dac_gen_* / ct_gen_*） */
  jobId?: string;
  /** 真实后端 jobId（与 jobId 同步，供控制台/回放解析） */
  backendJobId?: string;
  /** 本地 pending 记录已失效（session 重启后无后端绑定） */
  staleLocalPending?: boolean;
  cableModel?: string;
  difficulty?: string;
  horizon?: number;
  successRate?: number;
  successfulEpisodes?: number;
  trajectoryCount?: number;
  successTrajectoryCount?: number;
  episodeCount?: number;
  totalEpisodes?: number;
  frameCount?: number;
  sizeBytes?: number;
  hdf5Path?: string;
  npzPath?: string;
  manifestPath?: string;
  collectCsvPath?: string;
  failuresPath?: string;
  backendJobStatus?: string;
  backendCommand?: string;
  generateVideoPath?: string;
  generateVideoExists?: boolean;
  generateVideoSizeBytes?: number;
  videoJobId?: string;
  /** DualArmCable 真实后端记录 */
  dualArmMaxCables?: number;
  dualArmSeed?: number;
  dualArmStretchMode?: string;
  dualArmReleaseMode?: string;
  dualArmEpisodeSuccess?: boolean;
  dualArmSucceededCables?: number;
  dualArmLeftContact?: boolean;
  dualArmRightContact?: boolean;
  dualArmStretchReached?: boolean;
  dualArmSagM?: number;
  dualArmSpanM?: number;
  dualArmFinalSagM?: number;
  dualArmFinalSpanM?: number;
  episodeResultPath?: string;
  /** 是否支持从该任务数据构建训练数据集 */
  datasetBuildSupported?: boolean;
  /** 已构建数据集资产 */
  isDatasetAsset?: boolean;
  datasetBuildStatus?: DatasetBuildStatus;
  datasetId?: string;
  sourceJobId?: string;
  sourceRecordName?: string;
  downstreamModelType?: string;
  dataOrganizationFormat?: string;
  trainingView?: string;
  datasetUsage?: 'training' | 'evaluation' | 'training_and_evaluation';
  qualityStatus?: string;
  mainFormats?: string[];
  datasetManifestPath?: string;
  trainable?: boolean;
  ilExportReady?: boolean;
  ilExportProbed?: boolean;
  ilExportFailureReason?: string | null;
  lerobotPath?: string | null;
  lerobotTaskInstruction?: string | null;
  lerobotStateDim?: number | null;
  lerobotActionDim?: number | null;
  pi0Ready?: boolean;
  pi0ReadyReason?: string | null;
  lerobotStatsPath?: string | null;
  lerobotReportPath?: string | null;
  trainingBackendPending?: boolean;
}

/** @deprecated 保留类型别名 */
export type WorkspaceDataRow = WorkspaceDataItem;

export interface WorkspaceDataOverviewItem {
  id: string;
  title: string;
  count: number;
  sizeHint: string;
}

export const WORKSPACE_DEMO_DATA_CATEGORY = '示范数据' as const;

export function isDemoDataCategory(category: WorkspaceDataCategory | string): boolean {
  return category === '示范数据' || category === '原始 demonstration 数据';
}

export function isGenerateDataRow(item: WorkspaceDataItem): boolean {
  if (hasPersistedGenerateJobId(item)) return true;
  if (isDemoDataCategory(normalizeDataCategory(item.dataCategory))) return true;
  if (item.dataCategory === '真实数据' && item.source === 'MuJoCo 生成') return true;
  return false;
}

export function isPureDatasetRow(item: WorkspaceDataItem): boolean {
  if (isGenerateDataRow(item)) return false;
  if (item.source === '外部导入' || normalizeDataCategory(item.dataCategory) === '外部数据') {
    return true;
  }
  if (item.isDatasetAsset) return true;
  return (
    normalizeDataCategory(item.dataCategory) === '训练数据集' ||
    normalizeDataCategory(item.dataCategory) === '评测数据集'
  );
}

export function inferDatasetBuildStatus(item: WorkspaceDataItem): DatasetBuildStatus {
  if (item.datasetBuildStatus) return item.datasetBuildStatus;
  if (item.datasetManifestPath || item.datasetId) return 'built';
  if (item.isDatasetAsset) return 'built';
  return 'none';
}

const DATASET_BUILD_STATUS_LABELS: Record<DatasetBuildStatus, string> = {
  none: '未构建',
  building: '构建中',
  built: '已构建',
  failed: '构建失败',
};

export function formatListDisplayName(item: WorkspaceDataItem): string {
  return item.name?.trim() || item.sourceRecordName?.trim() || item.id;
}

export function formatListDatasetBuildStatus(item: WorkspaceDataItem): string {
  return DATASET_BUILD_STATUS_LABELS[inferDatasetBuildStatus(item)] ?? '—';
}

export function listDatasetBuildBadgeStatus(
  item: WorkspaceDataItem
): 'active' | 'running' | 'completed' | 'draft' {
  const status = inferDatasetBuildStatus(item);
  if (status === 'building') return 'running';
  if (status === 'built') return 'completed';
  if (status === 'failed') return 'draft';
  return 'active';
}

export function formatDatasetDisplayModel(item: WorkspaceDataItem): string {
  return item.downstreamModelType?.trim() || item.targetModelFormat?.trim() || '—';
}

export function formatDatasetMainFormat(item: WorkspaceDataItem): string {
  if (item.dataOrganizationFormat?.trim()) return item.dataOrganizationFormat.trim();
  const formats = item.mainFormats?.filter(Boolean);
  if (formats?.length) return formats.join(' / ');
  return '—';
}

export function hasBuiltDataset(item: WorkspaceDataItem): boolean {
  return inferDatasetBuildStatus(item) === 'built';
}

/** 兼容 sessionStorage 中的旧类别值 */
export function normalizeDataCategory(category: string): WorkspaceDataCategory {
  const map: Record<string, WorkspaceDataCategory> = {
    '原始 demonstration 数据': '示范数据',
    '外部导入数据': '外部数据',
    '真实采集数据': '真实数据',
  };
  return (map[category] ?? category) as WorkspaceDataCategory;
}

/** 用户界面展示 — 内部仍保留「示范数据」类别值 */
export function formatDataCategoryLabel(category: WorkspaceDataCategory | string): string {
  const normalized = normalizeDataCategory(category);
  if (normalized === WORKSPACE_DEMO_DATA_CATEGORY) return '任务数据';
  return normalized;
}

export function normalizeDataSource(source: string): WorkspaceDataSource {
  if (source === 'RoboTwin 生成') return 'MuJoCo 生成';
  return source as WorkspaceDataSource;
}

export const workspaceDataCategoryOptions: WorkspaceDataCategory[] = [
  '示范数据',
  '训练数据集',
  '评测数据集',
  '外部数据',
  '真实数据',
];

/** @deprecated 使用 workspaceDataCategoryOptions */
export const workspaceDataTypeOptions = workspaceDataCategoryOptions;

export const workspaceDataTaskFilterOptions = [
  '线缆穿杆',
  '螺母装配',
  '线缆整理',
  '物块堆叠',
] as const;

export const workspaceDataSourceOptions: WorkspaceDataSource[] = [
  'MuJoCo 生成',
  '外部导入',
  '真实采集',
];

export const workspaceTargetModelFormatOptions: WorkspaceTargetModelFormat[] = [
  'ACT',
  'DP3',
  'Diffusion Policy',
  'OpenVLA',
  '通用格式',
  '—',
];

export const generateDataBackendOptions = ['MuJoCo', 'RoboTwin', 'Isaac Sim'] as const;

/** Phase 1 生成数据弹窗 — 仅展示 MuJoCo 与 Isaac Sim 占位（RoboTwin 保留在 adapter mock 中） */
export const generateDataSimEnvironmentUiOptions = ['MuJoCo', 'Isaac Sim'] as const;
export const generateDataConfigOptions = ['default', 'randomized', 'hard', 'custom'] as const;
export const generateDataEpisodeOptions = [10, 50, 100, 200] as const;

export const workspaceDataStatusFilterOptions: {
  value: '' | WorkspaceDataStatus;
  label: string;
}[] = [
  { value: '', label: '全部' },
  { value: 'completed', label: '已完成' },
  { value: 'generating', label: '生成中' },
  { value: 'pending', label: '待生成' },
  { value: 'exported', label: '已导出' },
  { value: 'failed', label: '失败' },
];

export const workspaceDataStatusLabel: Record<WorkspaceDataStatus, string> = {
  completed: '已完成',
  generating: '生成中',
  pending: '待生成',
  exported: '已导出',
  failed: '失败',
  built: '已构建',
};

export const workspaceDataItemsMock: WorkspaceDataItem[] = [
  {
    id: 'demo-cable-threading-001',
    name: 'cable-threading-demo-50',
    taskId: 'task-008',
    taskName: '线缆穿杆',
    simulationId: 'ct_gen_demo_001',
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: '50 条',
    size: '860 MB',
    status: 'completed',
    generatedAt: '2026-05-31 14:22',
    creator: '平台',
    scene: '桌面双杆穿线工位',
    robot: 'Panda',
    taskConfig: 'default',
    simBackend: 'MuJoCo',
    dataPurpose: '训练与评测',
    saveVideo: true,
    saveTrajectory: true,
    taskType: 'cable_threading',
    jobId: 'ct_gen_demo_001',
  },
  {
    id: 'train-dual-arm-cable-001',
    name: 'dual-arm-cable-train-20',
    taskId: 'task-dual-arm-cable',
    taskName: '线缆整理',
    simulationId: 'dac_gen_demo_001',
    dataCategory: '训练数据集',
    source: 'MuJoCo 生成',
    targetModelFormat: 'Robomimic',
    dataVolume: '20 条',
    size: '1.2 GB',
    status: 'completed',
    generatedAt: '2026-06-01 10:05',
    creator: '平台',
    taskType: 'dual_arm_cable_manipulation',
    jobId: 'dac_gen_demo_001',
  },
  {
    id: 'train-nut-assembly-001',
    name: 'nut-assembly-mimicgen-20',
    taskId: 'task-nut-assembly',
    taskName: '螺母装配',
    simulationId: 'na_gen_demo_001',
    dataCategory: '训练数据集',
    source: 'MuJoCo 生成',
    targetModelFormat: 'Robomimic',
    dataVolume: '20 条',
    size: '920 MB',
    status: 'completed',
    generatedAt: '2026-06-02 09:00',
    creator: '平台',
  },
];

/** @deprecated 使用 workspaceDataItemsMock */
export const workspaceDataRecent = workspaceDataItemsMock;

export function listDemonstrationDataItems(
  extra: WorkspaceDataItem[] = []
): WorkspaceDataItem[] {
  const all = [...workspaceDataItemsMock, ...extra.map(normalizeDataItem)];
  return all.filter((i) => isDemoDataCategory(i.dataCategory));
}

export function normalizeDataItem(item: WorkspaceDataItem): WorkspaceDataItem {
  const dataVolume =
    item.dataVolume?.trim() ||
    item.frameOrTrajectoryCount?.trim() ||
    '—';
  const targetModelFormat = formatTargetModelForTable(item.targetModelFormat) as WorkspaceDataItem['targetModelFormat'];
  return {
    ...item,
    dataCategory: normalizeDataCategory(item.dataCategory ?? item.dataType ?? '示范数据'),
    source: normalizeDataSource(item.source),
    simBackend: item.simBackend === 'RoboTwin' ? 'MuJoCo' : item.simBackend,
    dataVolume,
    targetModelFormat,
  };
}

export function workspaceDataSummaryStats(items: WorkspaceDataItem[]) {
  const normalized = items.map(normalizeDataItem);
  const total = normalized.length;
  const demo = normalized.filter((i) => isDemoDataCategory(i.dataCategory)).length;
  const train = normalized.filter((i) => i.dataCategory === '训练数据集').length;
  const evalSet = normalized.filter((i) => i.dataCategory === '评测数据集').length;
  const exportable = normalized.filter(
    (i) => i.dataCategory === '训练数据集' || i.dataCategory === '评测数据集' || i.status === 'exported'
  ).length;
  return { total, demo, train, evalSet, exportable };
}

export const workspaceDataOverview: WorkspaceDataOverviewItem[] = [
  { id: 'total', title: '数据总量', count: 3, sizeHint: '约 3.0 GB（本页）' },
  { id: 'demo', title: '任务数据', count: 1, sizeHint: '约 860 MB' },
  { id: 'train', title: '训练数据集', count: 2, sizeHint: '约 2.1 GB' },
  { id: 'eval', title: '评测数据集', count: 0, sizeHint: '—' },
  { id: 'export', title: '可导出数据集', count: 0, sizeHint: '—' },
];

/** 表格「数据名称」列副文本：仿真 ID + 数据状态 */
export function formatWorkspaceDataNameSubline(item: WorkspaceDataItem): string {
  return `${item.simulationId} · ${workspaceDataStatusLabel[item.status]}`;
}

export function dataStatusBadgeStatus(
  status: WorkspaceDataStatus
): 'active' | 'running' | 'completed' | 'draft' {
  switch (status) {
    case 'completed':
      return 'active';
    case 'generating':
      return 'running';
    case 'pending':
      return 'draft';
    case 'exported':
      return 'completed';
    case 'failed':
      return 'draft';
    case 'built':
      return 'completed';
  }
}

export function hasPersistedGenerateJobId(item: {
  backendJobId?: string | null;
  jobId?: string | null;
}): boolean {
  const ids = [item.backendJobId, item.jobId].filter(Boolean) as string[];
  return ids.some(
    (id) =>
      !isPendingLocalJobId(id) &&
      (isValidCableThreadingGenerateJobId(id) ||
        isValidDualArmGenerateJobId(id) ||
        isValidIsaacGenerateJobId(id) ||
        isValidDataGenJobId(id) ||
        isValidNutAssemblyGenerateJobId(id))
  );
}

/** 解析 dataVolume 内部值，如 "50 episodes" / "100 demos" */
export interface ParsedDataVolume {
  count: number | null;
  unit: string | null;
  raw: string;
}

export function parseDataVolume(dataVolume?: string | null): ParsedDataVolume {
  const raw = (dataVolume ?? '').trim();
  if (!raw || raw === '—' || raw.includes('生成中') || raw.includes('待生成')) {
    return { count: null, unit: null, raw };
  }
  const match = raw.match(/^(\d+)\s+([a-zA-Z]+)/);
  if (match) {
    return { count: Number(match[1]), unit: match[2], raw };
  }
  const numOnly = raw.match(/^(\d+)/);
  if (numOnly) {
    return { count: Number(numOnly[1]), unit: null, raw };
  }
  return { count: null, unit: null, raw };
}

/** 数据列表 — 目标模型展示 */
export function formatTargetModelForTable(format?: string | null): string {
  const value = format?.trim();
  if (!value) return '—';
  return value;
}

/** 数据列表主表 — 统一样本数量展示 */
export function formatSampleCountForTable(dataVolume?: string | null): string {
  const { count, raw } = parseDataVolume(dataVolume);
  if (count == null) return raw || '—';
  return `${count} 条`;
}

/** 数据中心主列表 — 类型列 */
export function formatListTypeLabel(item: WorkspaceDataItem): string {
  if (item.isDatasetAsset) return '数据集资产';
  const category = normalizeDataCategory(item.dataCategory);
  if (isDemoDataCategory(category)) return '任务数据';
  if (category === '训练数据集') return '数据集资产';
  return formatDataCategoryLabel(category);
}

/** 数据中心主列表 — 状态列（仅资产管理语义） */
export function formatListStatusLabel(item: WorkspaceDataItem): string {
  if (item.status === 'generating' || item.status === 'pending') return '生成中';
  if (item.status === 'failed') return '失败';
  if (item.status === 'built' || item.isDatasetAsset) return '已构建';
  if (item.status === 'completed') return '已完成';
  if (item.status === 'exported') return '已导出';
  return workspaceDataStatusLabel[item.status] ?? item.status;
}

export function listStatusBadgeStatus(
  item: WorkspaceDataItem
): 'active' | 'running' | 'completed' | 'draft' {
  if (item.status === 'generating' || item.status === 'pending') return 'running';
  if (item.status === 'failed') return 'draft';
  if (item.status === 'built' || item.isDatasetAsset) return 'completed';
  return dataStatusBadgeStatus(item.status);
}

/** 数据中心主列表 — 数据规模列 */
export function formatListDataScale(item: WorkspaceDataItem): string {
  const { count, raw } = parseDataVolume(item.dataVolume);
  if (count != null) return `${count} 条`;
  if (raw && raw !== '—' && !raw.includes('生成中') && !raw.includes('待生成')) {
    return formatSampleCountForTable(raw);
  }
  return '—';
}

export function formatDataScale(item: WorkspaceDataItem): string {
  if (item.size?.trim() && item.size !== '—') return item.size;
  return formatListDataScale(item);
}

/** 数据中心主列表 — 适配模型列 */
export function formatListAdaptModel(item: WorkspaceDataItem): string {
  if (!item.isDatasetAsset && item.dataCategory !== '训练数据集') {
    return '—';
  }
  const model = formatDatasetDisplayModel(item);
  return model !== '—' ? model : '—';
}

function deriveJobRootFromArtifact(path?: string | null): string | null {
  if (!path) return null;
  const match = path.match(/^(.*\/jobs\/(?:ct_(?:gen|eval|vid)|dac_gen)_[^/]+)/);
  return match?.[1] ?? null;
}

/** 详情抽屉 — 文件清单 */
export function getWorkspaceDataFileEntries(
  item: WorkspaceDataItem
): { label: string; value: string }[] {
  const entries: { label: string; value: string }[] = [];
  const push = (label: string, value?: string | null) => {
    if (value) entries.push({ label, value });
  };

  push('NPZ', item.npzPath);
  push('HDF5', item.hdf5Path);
  push('manifest', item.manifestPath);
  push('collect.csv', item.collectCsvPath);
  push('failures.json', item.failuresPath);
  push('generate.mp4', item.generateVideoPath);

  const jobRoot =
    deriveJobRootFromArtifact(item.npzPath) ??
    deriveJobRootFromArtifact(item.hdf5Path) ??
    deriveJobRootFromArtifact(item.manifestPath) ??
    (item.taskType === 'cable_threading'
      ? `runs/cable_threading/jobs/${item.isDatasetAsset ? item.sourceJobId ?? item.simulationId : item.id}`
      : null) ??
    null;

  if (jobRoot) {
    if (item.taskType === 'dual_arm_cable_manipulation') {
      push('episode_result.json', item.episodeResultPath ?? `${jobRoot}/results/episode_result.json`);
      push('episode_manifest.json', item.manifestPath ?? `${jobRoot}/results/episode_manifest.json`);
      push('run.log', `${jobRoot}/logs/run.log`);
      push('latest.jpg', `${jobRoot}/live/latest.jpg`);
      push('rgb.png', `${jobRoot}/results/steps/step_00/frame/rgb.png`);
      push('depth.npy', `${jobRoot}/results/steps/step_00/frame/depth.npy`);
      push('latest_grasp.json', `${jobRoot}/results/steps/step_00/grasp_output/latest_grasp.json`);
      push('perception_attempt_0.log', `${jobRoot}/results/steps/step_00/perception_attempt_0.log`);
    } else {
      push('generate_timeline.json', `${jobRoot}/live/generate_timeline.json`);
      push('run.log', `${jobRoot}/logs/run.log`);
    }
  }

  if (item.datasetManifestPath) {
    push('dataset_manifest.json', item.datasetManifestPath);
  }

  return entries;
}

/** 详情 Drawer — 样本数量数值 */
export function formatSampleCountForDrawer(dataVolume?: string | null): string {
  const { count, raw } = parseDataVolume(dataVolume);
  if (count == null) return raw || '—';
  return String(count);
}

/** 详情 Drawer — 样本单位（episode / demo 等） */
export function formatSampleUnitForDrawer(dataVolume?: string | null): string | null {
  return parseDataVolume(dataVolume).unit;
}
