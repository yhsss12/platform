import type { Dataset } from '@/types/benchmark';
import { normalizeDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import { resolveDatasetSourceTaskLabel } from '@/lib/workspace/taskTemplateMapping';

export const BUILD_SOURCE_DATASET_PICKER_PAGE_SIZE = 10;

export const BUILD_SOURCE_DATASET_PICKER_EMPTY_TEXT = '暂无已导入的 HDF5 数据集';

export const BUILD_SOURCE_DATASET_PICKER_FILTERED_EMPTY_TEXT = '暂无符合条件的已导入 HDF5 数据集';

const SIMULATION_JOB_PREFIXES = ['ct_gen_', 'dac_gen_', 'isaac_gen_', 'isaac_import_'] as const;

function normalizeImportedDatasetStatus(status: string | null | undefined): string {
  const raw = (status ?? '').trim().toLowerCase();
  if (raw === 'available') return 'ready';
  if (raw === 'pending_field_mapping') return 'needs_mapping';
  if (raw === 'import_failed') return 'failed';
  return raw;
}

function isImportedWorkspaceDataset(dataset: Pick<Dataset, 'id' | 'sourceJobId'>): boolean {
  const id = (dataset.id ?? '').trim();
  const jobId = (dataset.sourceJobId ?? '').trim();
  return id.startsWith('ds_import_') || jobId.startsWith('import_ds_import_');
}

function isBuiltWorkspaceDataset(dataset: Pick<Dataset, 'id' | 'sourceJobId'>): boolean {
  const id = (dataset.id ?? '').trim();
  const jobId = (dataset.sourceJobId ?? '').trim();
  return id.startsWith('ds_built_') || jobId.startsWith('built_ds_built_');
}

/** 数据构建可选源：真实导入 HDF5（含不可直接训练的数据），排除仿真/构建后/失败等。 */
export function isBuildSourceImportedHdf5Dataset(dataset: Dataset): boolean {
  if (isBuiltWorkspaceDataset(dataset)) return false;

  const status = normalizeImportedDatasetStatus(dataset.status);
  if (status === 'failed') return false;

  const format = (dataset.format ?? dataset.datasetFormat ?? '').toLowerCase();
  if (format !== 'hdf5') return false;

  const sourceType = (dataset.sourceType ?? '').trim().toLowerCase();
  if (sourceType === 'real_robot_built' || sourceType === 'simulation_generated') return false;

  const dataSourceLabel = (dataset.dataSourceLabel ?? '').trim();
  const isRealImport =
    sourceType === 'real_robot_imported' ||
    dataSourceLabel === '真实导入' ||
    dataSourceLabel === '外部导入' ||
    isImportedWorkspaceDataset(dataset);
  if (!isRealImport) return false;

  const jobId = (dataset.sourceJobId ?? '').trim();
  if (SIMULATION_JOB_PREFIXES.some((prefix) => jobId.startsWith(prefix))) return false;

  const simulator = (dataset.simulatorBackend ?? '').trim().toLowerCase();
  if (simulator === 'mujoco' || simulator === 'isaac_lab' || simulator === 'isaacsim') return false;

  if (dataset.lerobotPath || (dataset.datasetFormat ?? '').toLowerCase() === 'lerobot') return false;

  return true;
}

export function filterBuildSourceDatasets(datasets: Dataset[]): Dataset[] {
  return datasets.filter(isBuildSourceImportedHdf5Dataset);
}

export function resolveBuildSourceDatasetDisplayName(dataset: Dataset): string {
  return normalizeDatasetDisplayName({
    displayName: dataset.displayName,
    name: dataset.name,
    taskType: dataset.taskType,
    createdAt: dataset.createdAt,
    sourceJobId: dataset.sourceJobId,
  });
}

export function filterBuildSourceDatasetsByKeyword(datasets: Dataset[], keyword: string): Dataset[] {
  const token = keyword.trim().toLowerCase();
  if (!token) return datasets;
  return datasets.filter((dataset) => {
    const haystack = [
      resolveBuildSourceDatasetDisplayName(dataset),
      resolveDatasetSourceTaskLabel(dataset),
      dataset.id,
      dataset.taskType,
      dataset.taskDisplayName,
      dataset.sourceJobId,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    return haystack.includes(token);
  });
}

export function getBuildSourceTaskFilterOptions(datasets: Dataset[]): { value: string; label: string }[] {
  const tasks = new Set<string>();
  for (const dataset of datasets) {
    const label = resolveDatasetSourceTaskLabel(dataset);
    if (label && label !== '—') tasks.add(label);
  }
  return [
    { value: 'all', label: '全部任务' },
    ...[...tasks].sort((a, b) => a.localeCompare(b, 'zh-CN')).map((label) => ({
      value: label,
      label,
    })),
  ];
}

export function filterBuildSourceDatasetsByTask(datasets: Dataset[], taskFilter: string): Dataset[] {
  if (!taskFilter || taskFilter === 'all') return datasets;
  return datasets.filter((dataset) => resolveDatasetSourceTaskLabel(dataset) === taskFilter);
}

export function paginateBuildSourceDatasets<T>(
  items: T[],
  page: number,
  pageSize = BUILD_SOURCE_DATASET_PICKER_PAGE_SIZE
): { items: T[]; totalPages: number; page: number } {
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  const start = (safePage - 1) * pageSize;
  return {
    items: items.slice(start, start + pageSize),
    totalPages,
    page: safePage,
  };
}

export function formatBuildSourceDatasetCreatedAt(value?: string | null): string {
  const raw = (value ?? '').trim();
  if (!raw) return '—';
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return raw.length >= 10 ? raw.slice(0, 10).replace(/T/g, ' ').replace(/-/g, '/') : raw;
  }
  return date.toLocaleString('zh-CN', { hour12: false });
}
