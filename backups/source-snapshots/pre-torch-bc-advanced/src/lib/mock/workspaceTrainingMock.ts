/** Historical pre-advanced-torch-BC training mock snapshot. */

import type { CSSProperties } from 'react';
import {
  normalizeDataCategory,
  parseDataVolume,
  hasBuiltDataset,
  isPureDatasetRow,
  type WorkspaceDataItem,
} from '@/lib/mock/workspaceDataMock';
import type { TrainingBackendRequest } from '@/lib/api/trainingClient';
import type { Dataset } from '@/types/benchmark';
import { resolveDatasetSourceTaskLabel } from '@/lib/workspace/taskTemplateMapping';
import {
  isDualArmTrainingBackendPending,
  isDualArmTrainingDatasetOption,
} from '@/lib/workspace/resolveTrainingDatasetManifest';
import { canOpenDatasetTraining } from '@/lib/workspace/datasetTrainingAccess';
import { formatTrainingDeviceLabel, type TrainingDeviceValue } from '@/lib/workspace/trainingDevice';
import { formatTrainingRecipeLabel } from '@/lib/workspace/trainingRecipe';

export type TrainingTaskStatus =
  | '等待中'
  | '排队中'
  | '训练中'
  | '已完成'
  | '失败'
  | '已取消';

export type TrainingTaskSource = 'real' | 'demo';

export interface TrainingDatasetOption {
  id: string;
  taskName: string;
  datasetName: string;
  modelFormat: string;
  dataFormat?: string;
  sampleCount: number;
  sourceJobId?: string;
  qualityStatus?: string;
  isBuiltAsset?: boolean;
  trainingBackendPending?: boolean;
  taskType?: string;
}

export interface TrainingTaskRow {
  id: string;
  trainJobId: string;
  source: TrainingTaskSource;
  name: string;
  relatedTask: string;
  modelType: string;
  dataset: string;
  datasetName?: string;
  datasetManifestPath?: string;
  dataVolume: string;
  status: TrainingTaskStatus;
  trainability?: 'real' | 'placeholder';
  backendStatus?: string;
  taskType?: string;
  runner?: string;
  runtimePath?: string;
  trainingBackend?: string;
  dataFormat?: string;
  deviceLabel?: string;
  currentEpoch: number;
  totalEpochs: number;
  progressPercent: number;
  loss: number | null;
  message: string;
  checkpoint: string | null;
  checkpointExists: boolean;
  hasModelManifest?: boolean;
  checkpointPath: string | null;
  modelAssetId: string | null;
  createdAt: string;
  updatedAt?: string;
  startedAt?: string;
  finishedAt?: string;
  batchSize: number;
  learningRate: number;
  seed: number;
}

export const trainingBackendOptions: { value: TrainingBackendRequest; label: string }[] = [
  { value: 'auto', label: '自动匹配' },
  { value: 'robomimic_bc', label: 'robomimic BC' },
  { value: 'robomimic', label: 'robomimic（兼容）' },
  { value: 'act', label: 'ACT' },
  { value: 'dt', label: 'DT' },
  { value: 'diffusion_policy', label: 'Diffusion Policy' },
];

export const trainingStatusOptions: TrainingTaskStatus[] = [
  '等待中',
  '排队中',
  '训练中',
  '已完成',
  '失败',
  '已取消',
];

export function formatTrainingDatasetLabel(option: TrainingDatasetOption): string {
  const formatPart = option.dataFormat ? ` · ${option.dataFormat}` : '';
  const qualityPart = option.qualityStatus ? ` · ${option.qualityStatus}` : '';
  return `${option.datasetName} · ${option.modelFormat}${formatPart} · ${option.sampleCount} 条成功${qualityPart}`;
}

function formatApiDatasetDataFormat(dataset: Dataset): string {
  if (dataset.format === 'hdf5' || dataset.datasetFormat === 'hdf5') return 'HDF5';
  if (dataset.format === 'npz') return 'NPZ';
  if (dataset.format === 'manifest') return 'Manifest';
  return dataset.format?.toUpperCase() || 'HDF5';
}

export function datasetToTrainingDatasetOption(dataset: Dataset): TrainingDatasetOption | null {
  if (!canOpenDatasetTraining(dataset)) return null;
  const isDualArm = dataset.sourceJobId.startsWith('dac_gen_') || dataset.taskTemplateId === 'dual_arm_cable_manipulation';
  return {
    id: dataset.id,
    taskName: resolveDatasetSourceTaskLabel(dataset),
    datasetName: dataset.name,
    modelFormat: isDualArm ? 'BC (torch)' : 'Robomimic',
    dataFormat: formatApiDatasetDataFormat(dataset),
    sampleCount: dataset.episodeCount,
    sourceJobId: dataset.sourceJobId,
    isBuiltAsset: true,
    taskType: isDualArm ? 'dual_arm_cable_manipulation' : undefined,
  };
}

export function dataItemToTrainingDatasetOption(item: WorkspaceDataItem): TrainingDatasetOption | null {
  const built = hasBuiltDataset(item);
  const legacyPureDataset =
    isPureDatasetRow(item) &&
    !built &&
    normalizeDataCategory(item.dataCategory) === '训练数据集' &&
    (item.status === 'built' || item.isDatasetAsset || item.status === 'completed');
  if (!built && !legacyPureDataset) return null;
  const sampleCount = item.successfulEpisodes ?? parseDataVolume(item.dataVolume).count ?? 0;
  const modelFormat = item.downstreamModelType ?? item.targetModelFormat ?? '通用格式';
  const dataFormat = resolveDatasetItemFormat(item, modelFormat);
  return {
    id: item.datasetId ?? item.id,
    taskName: item.taskName,
    datasetName: item.name,
    modelFormat,
    dataFormat,
    sampleCount,
    sourceJobId: item.sourceJobId ?? item.jobId ?? item.backendJobId,
    qualityStatus: item.qualityStatus,
    isBuiltAsset: built || Boolean(item.isDatasetAsset),
    trainingBackendPending: item.trainingBackendPending,
    taskType: item.taskType,
  };
}

/** 合并数据中心行与 API 登记的可训练数据集 */
export function listMergedTrainingDatasetOptions(
  dataCenterItems: WorkspaceDataItem[] = [],
  apiDatasets: Dataset[] = []
): TrainingDatasetOption[] {
  const fromItems = dataCenterItems
    .map(dataItemToTrainingDatasetOption)
    .filter((d): d is TrainingDatasetOption => d != null);
  const fromApi = apiDatasets
    .map(datasetToTrainingDatasetOption)
    .filter((d): d is TrainingDatasetOption => d != null);
  const seen = new Set<string>();
  const merged: TrainingDatasetOption[] = [];
  for (const option of [...fromItems, ...fromApi]) {
    if (seen.has(option.id)) continue;
    seen.add(option.id);
    merged.push(option);
  }
  return merged;
}

export function findTrainingDatasetOption(
  datasetId: string,
  dataCenterItems: WorkspaceDataItem[] = [],
  apiDatasets: Dataset[] = []
): TrainingDatasetOption | undefined {
  return listMergedTrainingDatasetOptions(dataCenterItems, apiDatasets).find((d) => d.id === datasetId);
}

function readDatasetFormatField(item: WorkspaceDataItem, key: string): string | undefined {
  const value = (item as unknown as Record<string, unknown>)[key];
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function resolveDatasetItemFormat(item: WorkspaceDataItem, modelFormat: string): string {
  const direct =
    readDatasetFormatField(item, 'format') ??
    readDatasetFormatField(item, 'dataFormat') ??
    item.dataOrganizationFormat ??
    readDatasetFormatField(item, 'storageFormat') ??
    readDatasetFormatField(item, 'dataset_format') ??
    readDatasetFormatField(item, 'artifactFormat') ??
    readDatasetFormatField(item, 'organizationFormat');

  if (direct?.trim()) return direct.trim();

  if (item.hdf5Path && item.npzPath) return 'HDF5 + NPZ';
  if (item.hdf5Path) return 'HDF5';
  if (item.npzPath) return 'NPZ';
  if (item.mainFormats?.length) return item.mainFormats.join(' + ');

  return inferDatasetFormatByModelType(modelFormat);
}

export function inferDatasetFormatByModelType(modelType: string): string {
  switch (modelType) {
    case 'ACT':
    case 'Robomimic':
      return 'HDF5';
    case 'Diffusion Policy':
      return 'Zarr / HDF5';
    case 'DT':
      return 'HDF5 / NPZ';
    default:
      return 'HDF5';
  }
}

export function resolveDatasetDisplayFormat(
  option: TrainingDatasetOption | undefined,
  modelType?: string
): string {
  if (option?.dataFormat?.trim()) return option.dataFormat.trim();
  return inferDatasetFormatByModelType(modelType ?? option?.modelFormat ?? 'Robomimic');
}

export function formatTrainingDatasetLabelById(
  datasetId: string,
  dataCenterItems: WorkspaceDataItem[] = []
): string {
  const option = findTrainingDatasetOption(datasetId, dataCenterItems);
  return option ? formatTrainingDatasetLabel(option) : datasetId;
}

export function trainingStatusColor(status: TrainingTaskStatus): string {
  switch (status) {
    case '等待中':
    case '排队中':
      return '#6b7280';
    case '训练中':
      return '#2563eb';
    case '已完成':
      return '#059669';
    case '失败':
      return '#dc2626';
    case '已取消':
      return '#9ca3af';
    default:
      return '#6b7280';
  }
}

export function trainingTaskStatusBadge(
  status: TrainingTaskStatus
): 'running' | 'completed' | 'failed' | 'draft' | 'paused' {
  switch (status) {
    case '训练中':
      return 'running';
    case '已完成':
      return 'completed';
    case '失败':
      return 'failed';
    case '已取消':
      return 'paused';
    default:
      return 'draft';
  }
}

export function formatTrainingSourceLabel(source: TrainingTaskSource): string {
  return source === 'real' ? '真实' : '示例';
}

export function formatTrainingSourceBadgeStyle(source: TrainingTaskSource): CSSProperties {
  if (source === 'real') {
    return {
      display: 'inline-block',
      marginTop: 4,
      padding: '1px 6px',
      borderRadius: 4,
      fontSize: 10,
      fontWeight: 600,
      color: '#065f46',
      backgroundColor: '#d1fae5',
    };
  }
  return {
    display: 'inline-block',
    marginTop: 4,
    padding: '1px 6px',
    borderRadius: 4,
    fontSize: 10,
    fontWeight: 600,
    color: '#92400e',
    backgroundColor: '#fef3c7',
  };
}

export function isUnknownTrainingModel(modelType: string): boolean {
  const value = (modelType ?? '').trim();
  if (!value || value === '—' || value.toLowerCase() === 'unknown') return true;
  return false;
}

export function formatTrainingModelTypeLabel(modelType: string): string {
  const value = (modelType ?? '').trim();
  if (isUnknownTrainingModel(value)) return '未知模型';
  const lower = value.toLowerCase();
  if (lower.includes('robomimic')) return 'Robomimic';
  if (lower === 'act') return 'ACT';
  if (lower === 'dt') return 'DT';
  if (lower.includes('diffusion')) return 'Diffusion Policy';
  if (lower.includes('dp3')) return 'DP3';
  if (lower.includes('vla') || lower.includes('openvla')) return 'OpenVLA';
  return value;
}

export function normalizeTrainingModelFilterValue(modelType: string): string {
  return formatTrainingModelTypeLabel(modelType);
}

function shortenAssetDisplayName(name: string): string {
  return name.replace(/_\d{8}(?:_\d+)?$/, '').trim();
}

function inferTaskShortName(taskName?: string, datasetName?: string): string {
  const source = (taskName ?? datasetName ?? '').trim();
  if (!source) return '训练';
  if (source.includes('线缆穿杆')) return '线缆穿杆';
  if (source.includes('拧螺丝')) return '拧螺丝';
  if (source.includes('装夹')) return '装夹';
  const shortened = shortenAssetDisplayName(source)
    .replace(/任务数据集$/, '')
    .replace(/数据集$/, '')
    .replace(/任务$/, '')
    .trim();
  return shortened || '训练';
}

/** 训练中心主列表 — 训练任务标题 */
export function formatTrainingTaskTitle(row: TrainingTaskRow): string {
  const taskShort = inferTaskShortName(row.relatedTask, row.datasetName);
  const model = formatTrainingRecipeLabel(row.trainingBackend, row.modelType);
  if (row.name.toLowerCase().includes('最小验证')) {
    return `${taskShort} ${model} 最小验证`;
  }
  return `${taskShort} ${model} 训练`;
}

/** 训练中心主列表 — 训练任务副标题 */
export function formatTrainingTaskSubtitle(row: TrainingTaskRow): string | null {
  const parts: string[] = [row.trainJobId];
  if (row.totalEpochs > 0) parts.push(`${row.currentEpoch}/${row.totalEpochs} epoch`);
  return parts.join(' · ');
}

function resolveDatasetItem(
  row: TrainingTaskRow,
  dataCenterItems: WorkspaceDataItem[]
): WorkspaceDataItem | undefined {
  return dataCenterItems.find(
    (item) => (item.datasetId ?? item.id) === row.dataset || item.id === row.dataset
  );
}

/** 训练中心主列表 — 数据集标题 */
export function formatTrainingDatasetTitle(
  row: TrainingTaskRow,
  dataCenterItems: WorkspaceDataItem[] = []
): string {
  const option = findTrainingDatasetOption(row.dataset, dataCenterItems);
  if (option?.datasetName) return shortenAssetDisplayName(option.datasetName);
  if (row.datasetName) return shortenAssetDisplayName(row.datasetName);
  const item = resolveDatasetItem(row, dataCenterItems);
  if (item?.name) return shortenAssetDisplayName(item.name);
  if (row.dataset.startsWith('ds_') || row.dataset.startsWith('ct_')) return '训练数据集';
  return row.dataset;
}

/** 训练中心主列表 — 数据集副标题 */
export function formatTrainingDatasetSubtitle(
  row: TrainingTaskRow,
  dataCenterItems: WorkspaceDataItem[] = []
): string {
  const option = findTrainingDatasetOption(row.dataset, dataCenterItems);
  const item = resolveDatasetItem(row, dataCenterItems);

  if (row.datasetManifestPath) {
    const parts = row.datasetManifestPath.split('/');
    return parts.slice(-2).join('/') || row.datasetManifestPath;
  }

  if (row.dataset && row.dataset !== '—' && !row.dataset.startsWith('ds_demo')) {
    return row.dataset;
  }

  const formats: string[] = [];
  const dataFormat = (option?.dataFormat ?? item?.dataOrganizationFormat ?? '').toLowerCase();

  if (item?.hdf5Path || dataFormat.includes('hdf5')) formats.push('HDF5');
  if (item?.npzPath || dataFormat.includes('npz')) formats.push('NPZ');
  if (formats.length === 0 && option?.dataFormat) formats.push(option.dataFormat);
  if (formats.length === 0 && item?.mainFormats?.length) {
    formats.push(...item.mainFormats.map((f) => f.toUpperCase()));
  }

  const count =
    option?.sampleCount ??
    item?.successfulEpisodes ??
    parseDataVolume(item?.dataVolume ?? row.dataVolume).count;

  if (formats.length === 1 && formats[0] === 'NPZ' && !item?.hdf5Path) {
    return 'NPZ · 仅轨迹数据';
  }
  if (formats.length > 0 && count != null && count > 0) {
    return `${formats.join(' + ')} · ${count} 条成功轨迹`;
  }
  if (formats.length > 0) return formats.join(' + ');
  if (count != null && count > 0) return `${count} 条成功轨迹`;
  if (!row.dataset || row.dataset === '—') return '未记录';
  return '未记录';
}

export interface TrainingResultDisplay {
  primary: string;
  secondary?: string;
}

/** 训练中心列表 — Checkpoint 列（仅产物状态，不含训练状态） */
export function formatTrainingCheckpointListLabel(row: TrainingTaskRow): string {
  if (row.checkpointExists) return '已生成';
  if (row.hasModelManifest) return '模型清单已生成';
  if (row.status === '训练中') return '生成中';
  return '—';
}

/** @deprecated 使用 formatTrainingCheckpointListLabel */
export function formatTrainingResultListLabel(row: TrainingTaskRow): string {
  return formatTrainingCheckpointListLabel(row);
}

/** 训练中心主列表 — 结果列（详情等场景可用） */
export function formatTrainingResult(row: TrainingTaskRow): TrainingResultDisplay {
  if (row.status === '失败' || row.backendStatus === 'failed') {
    return { primary: '训练未完成：查看日志' };
  }
  if (row.checkpointExists && row.hasModelManifest) {
    return {
      primary: 'checkpoint 已生成',
      secondary: 'model_manifest 已生成',
    };
  }
  if (row.checkpointExists) {
    return {
      primary: 'checkpoint 已生成',
      secondary: row.modelAssetId ?? undefined,
    };
  }
  if (row.hasModelManifest) {
    return { primary: 'model_manifest 已生成' };
  }
  if (row.status === '训练中') {
    const primary =
      row.totalEpochs > 0
        ? `${row.currentEpoch}/${row.totalEpochs} epoch · ${row.progressPercent}%`
        : `训练中 · ${row.progressPercent}%`;
    const secondary = row.loss != null ? `loss ${row.loss.toFixed(4)}` : undefined;
    return { primary, secondary };
  }
  if (row.status === '等待中' || row.status === '排队中') {
    return { primary: '未生成结果' };
  }
  if (row.status === '已完成') {
    return { primary: '未生成结果' };
  }
  return { primary: '未生成结果' };
}

export function formatTrainingDeviceDisplay(row: TrainingTaskRow): { title: string; subtitle?: string } {
  const label = formatTrainingDeviceLabel(row.deviceLabel);
  return {
    title: label,
    subtitle: label.includes('H20') ? '适用于大规模训练任务' : '适用于常规策略训练任务',
  };
}

export function canEnableTrainingEvaluation(row: TrainingTaskRow): boolean {
  if (row.status === '失败' || row.backendStatus === 'failed') return false;
  if (row.status === '已取消' || row.backendStatus === 'canceled') return false;
  return Boolean(row.checkpointExists || row.hasModelManifest);
}

export type TrainingSeedMode = 'random' | 'manual';

export interface TrainingPretrainedOptions {
  modelAssetId: string;
  checkpointPath?: string;
  modelAssetName?: string;
}

export interface RobomimicAdvancedParams {
  actor_hidden_dims: string;
  l2_regularization: number;
}

export interface ActTrainingAdvancedParams {
  chunk_size: number;
  n_action_steps: number;
  kl_weight: number;
  latent_dim: number;
  hidden_dim: number;
}

export interface DiffusionPolicyAdvancedParams {
  n_obs_steps: number;
  horizon: number;
  n_action_steps: number;
  num_inference_steps: number;
  use_ema: boolean;
  ema_decay: number;
  weight_decay: number;
  save_best: boolean;
}

export type TrainingModelAdvancedParams =
  | RobomimicAdvancedParams
  | ActTrainingAdvancedParams
  | DiffusionPolicyAdvancedParams;

export interface CreateTrainingTaskInput {
  dataset: string;
  downstreamModelType: string;
  dataFormat: string;
  trainingBackend: TrainingBackendRequest;
  trainingDevice: TrainingDeviceValue;
  epochs: number;
  batchSize: number;
  learningRate: number;
  device: string;
  seed: number;
  seedMode?: TrainingSeedMode;
  advancedEnabled?: boolean;
  pretrained?: TrainingPretrainedOptions;
  modelParams?: TrainingModelAdvancedParams;
  taskName?: string;
  trainability?: 'real' | 'placeholder';
}
