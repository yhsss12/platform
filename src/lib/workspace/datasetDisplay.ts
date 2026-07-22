import type { Dataset } from '@/types/benchmark';
import { formatFileSize } from '@/utils/format';
import {
  isImportedWorkspaceDataset,
  normalizeImportedDatasetStatus,
} from '@/lib/workspace/datasetImportWorkflow';

type DatasetSourceInput = Pick<
  Dataset,
  'sourceType' | 'simulatorBackend' | 'sourceJobId' | 'dataSourceLabel' | 'generationMode'
>;

type DatasetTrajectoryInput = Pick<
  Dataset,
  | 'id'
  | 'dataCount'
  | 'successfulEpisodes'
  | 'totalEpisodes'
  | 'validTrajectories'
  | 'generationRounds'
  | 'episodeCount'
  | 'fileSizeBytes'
  | 'sourceJobId'
  | 'sourceType'
  | 'simulatorBackend'
  | 'status'
  | 'needsBuild'
  | 'episodeParsed'
  | 'format'
  | 'datasetFormat'
>;

export const DATASET_SOURCE_SIMULATION = '仿真生成';
export const DATASET_SOURCE_EXTERNAL_IMPORT = '外部导入';
export const DATASET_SOURCE_DATA_BUILD = '数据构建';

export type NormalizedDatasetSource =
  | typeof DATASET_SOURCE_SIMULATION
  | typeof DATASET_SOURCE_EXTERNAL_IMPORT
  | typeof DATASET_SOURCE_DATA_BUILD;

export const DATASET_SOURCE_FILTER_OPTIONS: NormalizedDatasetSource[] = [
  DATASET_SOURCE_SIMULATION,
  DATASET_SOURCE_EXTERNAL_IMPORT,
  DATASET_SOURCE_DATA_BUILD,
];

const LEGACY_SOURCE_LABEL_MAP: Record<string, NormalizedDatasetSource> = {
  'mujoco 生成': DATASET_SOURCE_SIMULATION,
  'isaac lab 生成': DATASET_SOURCE_SIMULATION,
  'isaac sim 生成': DATASET_SOURCE_SIMULATION,
  仿真生成: DATASET_SOURCE_SIMULATION,
  仿真导出: DATASET_SOURCE_SIMULATION,
  真实导入: DATASET_SOURCE_EXTERNAL_IMPORT,
  外部导入: DATASET_SOURCE_EXTERNAL_IMPORT,
  导入数据: DATASET_SOURCE_EXTERNAL_IMPORT,
  真机导入: DATASET_SOURCE_EXTERNAL_IMPORT,
  外部公开数据: DATASET_SOURCE_EXTERNAL_IMPORT,
  真实数据构建: DATASET_SOURCE_DATA_BUILD,
  数据构建: DATASET_SOURCE_DATA_BUILD,
};

interface NormalizeDatasetSourceMetadata {
  sourceType?: string | null;
  simulatorBackend?: string | null;
  sourceJobId?: string | null;
  generationMode?: string | null;
  dataSourceLabel?: string | null;
}

function normalizeToken(value: string): string {
  return value.trim().toLowerCase();
}

function includesAnyToken(haystack: string, tokens: string[]): boolean {
  const normalized = normalizeToken(haystack);
  return tokens.some((token) => normalized.includes(token));
}

/** 平台级数据来源：仅返回 仿真生成 / 外部导入 / 数据构建。 */
export function normalizeDatasetSource(
  rawSource?: string | null,
  metadata?: NormalizeDatasetSourceMetadata
): NormalizedDatasetSource {
  const parts: string[] = [];
  if (rawSource?.trim()) parts.push(rawSource.trim());
  if (metadata?.dataSourceLabel?.trim()) parts.push(metadata.dataSourceLabel.trim());
  if (metadata?.sourceType?.trim()) parts.push(metadata.sourceType.trim());
  if (metadata?.simulatorBackend?.trim()) parts.push(metadata.simulatorBackend.trim());
  if (metadata?.sourceJobId?.trim()) parts.push(metadata.sourceJobId.trim());
  if (metadata?.generationMode?.trim()) parts.push(metadata.generationMode.trim());

  for (const part of parts) {
    const mapped = LEGACY_SOURCE_LABEL_MAP[normalizeToken(part)];
    if (mapped) return mapped;
  }

  const combined = parts.join(' ');

  if (includesAnyToken(combined, ['isaac_import_'])) {
    return DATASET_SOURCE_EXTERNAL_IMPORT;
  }

  if (
    includesAnyToken(combined, [
      'real_data_constructed',
      'real_robot_built',
      'dataset_build',
      'manual_build',
      'constructed',
      'standardized',
    ])
  ) {
    return DATASET_SOURCE_DATA_BUILD;
  }

  if (
    includesAnyToken(combined, [
      'real_robot_imported',
      'real_import',
      'uploaded_hdf5',
      'imported_demo',
      'imported',
      'real_collection',
      'public_dataset',
      'simulation_export',
      'external',
      'upload',
      'import',
      'converted',
      'mixed',
    ])
  ) {
    return DATASET_SOURCE_EXTERNAL_IMPORT;
  }

  if (
    includesAnyToken(combined, [
      'simulation_generated',
      'demo_generation',
      'expert_policy',
      'mimicgen',
      'robosuite',
      'ct_gen_',
      'dac_gen_',
      'isaac_gen_',
      'simulation',
      'generated',
      'mujoco',
      'isaac',
    ])
  ) {
    return DATASET_SOURCE_SIMULATION;
  }

  const simulator = normalizeToken(metadata?.simulatorBackend ?? '');
  if (
    simulator === 'mujoco' ||
    simulator === 'isaac_lab' ||
    simulator === 'isaacsim' ||
    simulator === 'isaac_sim'
  ) {
    return DATASET_SOURCE_SIMULATION;
  }

  return DATASET_SOURCE_EXTERNAL_IMPORT;
}

export function resolveDatasetSimulatorBackendLabel(
  simulatorBackend?: string | null
): string | null {
  const normalized = normalizeToken(simulatorBackend ?? '');
  if (!normalized) return null;
  if (normalized === 'mujoco') return 'MuJoCo';
  if (normalized === 'isaac_lab') return 'Isaac Lab';
  if (normalized === 'isaacsim' || normalized === 'isaac_sim') return 'Isaac Sim';
  return simulatorBackend?.trim() || null;
}

export { isImportedWorkspaceDataset, normalizeImportedDatasetStatus } from '@/lib/workspace/datasetImportWorkflow';

export function resolveDatasetStatusLabel(status: string | null | undefined): string {
  const normalized = normalizeImportedDatasetStatus(status);
  if (normalized === 'ready' || normalized === 'available') return '可用';
  if (normalized === 'needs_mapping' || normalized === 'pending_field_mapping') return '需字段映射';
  if (normalized === 'needs_build') return '待构建';
  if (normalized === 'failed' || normalized === 'import_failed') return '导入失败';
  if (normalized === 'parsing') return '解析中';
  return status || '—';
}

export function resolveDatasetSourceLabel(dataset: DatasetSourceInput): string {
  return normalizeDatasetSource(dataset.dataSourceLabel ?? dataset.sourceType, {
    sourceType: dataset.sourceType,
    simulatorBackend: dataset.simulatorBackend,
    sourceJobId: dataset.sourceJobId,
    generationMode: dataset.generationMode,
    dataSourceLabel: dataset.dataSourceLabel,
  });
}

/** 列表「数据数量」列：读取后端 dataCount。 */
export function resolveDatasetCountText(dataset: Pick<Dataset, 'dataCount'>): string {
  const count = positiveInt(dataset.dataCount);
  return count !== null && count > 0 ? String(count) : '—';
}

/** 列表「数据容量」列：读取后端 fileSizeBytes。 */
export function resolveDatasetSizeText(dataset: Pick<Dataset, 'fileSizeBytes'>): string {
  const bytes = positiveInt(dataset.fileSizeBytes);
  if (bytes !== null && bytes > 0) {
    return formatFileSize(bytes);
  }
  return '—';
}

function positiveInt(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
    return Math.trunc(value);
  }
  return null;
}

function isDatasetManifestFormat(dataset: Pick<Dataset, 'format' | 'datasetFormat'>): boolean {
  const format = (dataset.format ?? '').trim().toLowerCase();
  const datasetFormat = (dataset.datasetFormat ?? '').trim().toLowerCase();
  return format === 'manifest' || datasetFormat === 'manifest';
}

/** @deprecated 列表已拆分为 resolveDatasetCountText / resolveDatasetSizeText；详情页仍可用于组合展示。 */
export function resolveDatasetScaleText(
  dataset: Pick<Dataset, 'dataCount' | 'fileSizeBytes'>
): string {
  const count = resolveDatasetCountText(dataset);
  const size = resolveDatasetSizeText(dataset);
  if (count === '—' && size === '—') return '—';
  if (size === '—') return count;
  if (count === '—') return size;
  return `${size} / ${count}`;
}

/** 详情页：有效轨迹 / 总轨迹比例（如 4/5）。 */
export function resolveDatasetValidTrajectoryText(dataset: DatasetTrajectoryInput): string {
  const successfulRaw =
    positiveInt(dataset.successfulEpisodes) ??
    positiveInt(dataset.validTrajectories);
  const totalRaw =
    positiveInt(dataset.totalEpisodes) ??
    positiveInt(dataset.generationRounds);
  const episodeCount = positiveInt(dataset.episodeCount);

  let successful = successfulRaw;
  let total = totalRaw;

  if (successful !== null && total !== null && total > 0) {
    successful = Math.min(successful, total);
    return `${successful}/${total}`;
  }

  if (episodeCount !== null && episodeCount > 0) {
    if (successful !== null) {
      const cappedSuccessful = Math.min(successful, episodeCount);
      return `${cappedSuccessful}/${episodeCount}`;
    }
    return `${episodeCount}/${episodeCount}`;
  }

  if (successful !== null) {
    return `${successful}/${successful}`;
  }

  return '—';
}

export function shouldShowDatasetValidTrajectoryDetail(dataset: DatasetTrajectoryInput): boolean {
  if (isImportedWorkspaceDataset(dataset) || isDatasetManifestFormat(dataset)) {
    return false;
  }
  const ratio = resolveDatasetValidTrajectoryText(dataset);
  return ratio !== '—' && ratio.includes('/');
}
