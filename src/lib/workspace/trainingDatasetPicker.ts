import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import { isDatasetCompatibleWithSelection } from '@/lib/workspace/trainingDatasetCompat';
import { normalizeDatasetDisplayName } from '@/lib/workspace/datasetNaming';
import { resolveDatasetCountText } from '@/lib/workspace/datasetDisplay';
import { resolveDatasetFormatLabel } from '@/lib/workspace/taskTemplateMapping';
import type { Dataset } from '@/types/benchmark';

export const TRAINING_DATASET_PICKER_PAGE_SIZE = 10;

export const TRAINING_DATASET_PICKER_EMPTY_TEXT = '暂无可用的训练数据集';

export const TRAINING_DATASET_PICKER_FILTERED_EMPTY_TEXT = '暂无符合条件的训练数据集';

export type TrajectoryRangeFilter = 'all' | '1-5' | '6-20' | '21-100' | '100+';

export type DatasetPickerFilterColumn =
  | 'none'
  | 'schema'
  | 'robot'
  | 'format'
  | 'status'
  | 'trajectoryCount';

export type DatasetPickerSingleFilter = {
  column: DatasetPickerFilterColumn;
  value: string;
};

export const DEFAULT_DATASET_PICKER_FILTER: DatasetPickerSingleFilter = {
  column: 'none',
  value: 'all',
};

export const DATASET_FILTER_COLUMN_OPTIONS: { value: DatasetPickerFilterColumn; label: string }[] = [
  { value: 'none', label: '不筛选' },
  { value: 'schema', label: 'Schema' },
  { value: 'robot', label: '机器人' },
  { value: 'format', label: '格式' },
  { value: 'status', label: '状态' },
  { value: 'trajectoryCount', label: '轨迹数' },
];

export const TRAJECTORY_RANGE_FILTER_OPTIONS: { value: TrajectoryRangeFilter; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: '1-5', label: '1–5' },
  { value: '6-20', label: '6–20' },
  { value: '21-100', label: '21–100' },
  { value: '100+', label: '100+' },
];

export interface TrainingDatasetPickerMeta {
  actionSchema?: string | null;
  observationSchema?: string | null;
  createdAt?: string | null;
  robotType?: string | null;
  status?: string | null;
}

export function filterTrainingDatasetOptions(
  options: TrainingDatasetOption[],
  keyword: string
): TrainingDatasetOption[] {
  const token = keyword.trim().toLowerCase();
  if (!token) return options;
  return options.filter((option) => {
    const haystack = [
      option.datasetName,
      option.taskName,
      option.id,
      option.modelFormat,
      option.dataFormat,
      option.taskType,
      option.sourceJobId,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    return haystack.includes(token);
  });
}

export function resetDatasetPickerFilter(): DatasetPickerSingleFilter {
  return { ...DEFAULT_DATASET_PICKER_FILTER };
}

export function datasetPickerFilterWithColumn(column: DatasetPickerFilterColumn): DatasetPickerSingleFilter {
  return { column, value: 'all' };
}

export function getDatasetFilterColumns(): typeof DATASET_FILTER_COLUMN_OPTIONS {
  return DATASET_FILTER_COLUMN_OPTIONS;
}

export function matchesTrajectoryRange(
  sampleCount: number,
  range: TrajectoryRangeFilter
): boolean {
  if (range === 'all') return true;
  const count = Number.isFinite(sampleCount) ? Math.max(0, Math.round(sampleCount)) : 0;
  switch (range) {
    case '1-5':
      return count >= 1 && count <= 5;
    case '6-20':
      return count >= 6 && count <= 20;
    case '21-100':
      return count >= 21 && count <= 100;
    case '100+':
      return count >= 100;
    default:
      return true;
  }
}

function uniqueSortedFilterValues(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value && value !== '—'))].sort(
    (left, right) => left.localeCompare(right, 'zh-CN')
  );
}

export function getDatasetFilterOptions(
  options: TrainingDatasetOption[],
  metaById: Record<string, TrainingDatasetPickerMeta> = {}
): {
  schemaOptions: string[];
  robotOptions: string[];
  formatOptions: string[];
  statusOptions: string[];
} {
  const schemaOptions: string[] = [];
  const robotOptions: string[] = [];
  const formatOptions: string[] = [];
  const statusOptions: string[] = [];

  for (const option of options) {
    const meta = metaById[option.id];
    schemaOptions.push(formatDatasetSchemaLabel(option, meta));
    robotOptions.push(formatDatasetRobotLabel(option, meta));
    formatOptions.push(formatDatasetFormatLabel(option));
    statusOptions.push(formatDatasetStatusLabel(meta?.status));
  }

  return {
    schemaOptions: uniqueSortedFilterValues(schemaOptions),
    robotOptions: uniqueSortedFilterValues(robotOptions),
    formatOptions: uniqueSortedFilterValues(formatOptions),
    statusOptions: uniqueSortedFilterValues(statusOptions),
  };
}

export function getDatasetFilterValues(
  options: TrainingDatasetOption[],
  column: DatasetPickerFilterColumn,
  metaById: Record<string, TrainingDatasetPickerMeta> = {}
): { value: string; label: string }[] {
  if (column === 'none') {
    return [{ value: 'all', label: '全部' }];
  }
  if (column === 'trajectoryCount') {
    return TRAJECTORY_RANGE_FILTER_OPTIONS;
  }

  const grouped = getDatasetFilterOptions(options, metaById);
  let values: string[] = [];
  switch (column) {
    case 'schema':
      values = grouped.schemaOptions;
      break;
    case 'robot':
      values = grouped.robotOptions;
      break;
    case 'format':
      values = grouped.formatOptions;
      break;
    case 'status':
      values = grouped.statusOptions;
      break;
    default:
      values = [];
  }

  return [{ value: 'all', label: '全部' }, ...values.map((value) => ({ value, label: value }))];
}

export function matchesDatasetFilter(
  option: TrainingDatasetOption,
  filter: DatasetPickerSingleFilter,
  metaById: Record<string, TrainingDatasetPickerMeta> = {}
): boolean {
  if (filter.column === 'none' || filter.value === 'all') return true;
  const meta = metaById[option.id];
  switch (filter.column) {
    case 'schema':
      return formatDatasetSchemaLabel(option, meta) === filter.value;
    case 'robot':
      return formatDatasetRobotLabel(option, meta) === filter.value;
    case 'format':
      return formatDatasetFormatLabel(option) === filter.value;
    case 'status':
      return formatDatasetStatusLabel(meta?.status) === filter.value;
    case 'trajectoryCount':
      return matchesTrajectoryRange(option.sampleCount, filter.value as TrajectoryRangeFilter);
    default:
      return true;
  }
}

export function applyDatasetPickerSingleFilter(
  options: TrainingDatasetOption[],
  filter: DatasetPickerSingleFilter,
  metaById: Record<string, TrainingDatasetPickerMeta> = {}
): TrainingDatasetOption[] {
  return options.filter((option) => matchesDatasetFilter(option, filter, metaById));
}

export function filterTrainingDatasets(
  options: TrainingDatasetOption[],
  searchKeyword: string,
  filter: DatasetPickerSingleFilter,
  metaById: Record<string, TrainingDatasetPickerMeta> = {}
): TrainingDatasetOption[] {
  const searched = filterTrainingDatasetOptions(options, searchKeyword);
  return applyDatasetPickerSingleFilter(searched, filter, metaById);
}

export function isDatasetPickerFilterActive(filter: DatasetPickerSingleFilter): boolean {
  return filter.column !== 'none' && filter.value !== 'all';
}

export function getFilteredDatasetSummary(total: number, filtered: number): string | null {
  if (total === 0 || filtered === total) return null;
  return `已筛选 ${filtered} / 共 ${total} 个数据集`;
}

export function paginateTrainingDatasetOptions<T>(
  items: T[],
  page: number,
  pageSize = TRAINING_DATASET_PICKER_PAGE_SIZE
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

export function formatDatasetDisplayName(
  option: TrainingDatasetOption,
  meta?: TrainingDatasetPickerMeta | null
): string {
  return normalizeDatasetDisplayName({
    displayName: option.displayName,
    name: option.datasetName,
    taskType: option.taskType,
    createdAt: meta?.createdAt ?? option.createdAt,
    sourceJobId: option.sourceJobId,
  });
}

function trainingOptionAsDatasetFields(
  option: TrainingDatasetOption
): Pick<Dataset, 'format' | 'datasetFormat' | 'dataCount'> {
  const formatToken = (option.format ?? option.dataFormat ?? 'hdf5').toLowerCase();
  const normalizedFormat =
    formatToken === 'lerobot'
      ? 'lerobot'
      : formatToken === 'npz'
        ? 'npz'
        : formatToken === 'manifest'
          ? 'manifest'
          : 'hdf5';
  return {
    format: normalizedFormat as Dataset['format'],
    datasetFormat: (option.datasetFormat ?? normalizedFormat) as Dataset['datasetFormat'],
    dataCount: option.dataCount ?? option.sampleCount,
  };
}

export function resolveTrainingDatasetTaskLabel(option: TrainingDatasetOption): string {
  return option.taskName?.trim() || '—';
}

export function resolveTrainingDatasetFormatLabel(option: TrainingDatasetOption): string {
  return resolveDatasetFormatLabel(trainingOptionAsDatasetFields(option));
}

export function resolveTrainingDatasetCountLabel(option: TrainingDatasetOption): string {
  return resolveDatasetCountText(trainingOptionAsDatasetFields(option));
}

export function formatDatasetTableCreatedAt(value?: string | null): string {
  if (!value) return '—';
  try {
    const d = new Date(value);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleString('zh-CN', { hour12: false });
    }
  } catch {
    /* ignore */
  }
  return value.slice(0, 19).replace('T', ' ');
}

export function formatDatasetDemoCount(sampleCount: number): string {
  const count = Number.isFinite(sampleCount) ? Math.max(0, Math.round(sampleCount)) : 0;
  return `${count} demo${count === 1 ? '' : 's'}`;
}

function inferSchemaLabelFromOption(option: TrainingDatasetOption): string {
  const name = `${option.datasetName} ${option.id}`.toLowerCase();
  if (name.includes('joint') || name.includes('joint_space')) return 'Joint-Space';
  if (name.includes('eef') || name.includes('osc')) return 'EEF-OSC';
  if (option.taskType === 'isaac_block_stacking') return 'Isaac BC';
  if (option.taskType === 'dual_arm_cable_manipulation') return 'Dual-Arm';
  return option.taskName || '训练数据集';
}

function inferRobotLabel(option: TrainingDatasetOption): string {
  if (option.taskType === 'dual_arm_cable_manipulation') return '双臂协作机器人';
  if (option.taskType === 'isaac_block_stacking') return 'Franka';
  if (option.taskType === 'nut_assembly' || option.sourceJobId?.startsWith('na_gen_')) {
    return 'Panda';
  }
  if (option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_')) {
    return 'Panda';
  }
  return '—';
}

export function formatDatasetSchemaLabel(
  option: TrainingDatasetOption,
  meta?: TrainingDatasetPickerMeta | null
): string {
  return (
    meta?.actionSchema?.trim() ||
    meta?.observationSchema?.trim() ||
    inferSchemaLabelFromOption(option)
  );
}

export function formatDatasetRobotLabel(
  option: TrainingDatasetOption,
  meta?: TrainingDatasetPickerMeta | null
): string {
  return meta?.robotType?.trim() || inferRobotLabel(option) || '—';
}

export function formatDatasetFormatLabel(option: TrainingDatasetOption): string {
  return option.dataFormat || option.modelFormat || 'HDF5';
}

export function formatDatasetStatusLabel(status?: string | null): string {
  const raw = (status ?? '').trim();
  if (!raw) return '—';
  return raw;
}

export function formatDatasetDate(value?: string | null): string {
  const raw = (value ?? '').trim();
  if (!raw) return '—';
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return raw.length >= 10 ? raw.slice(0, 10).replace(/-/g, '/') : raw;
  }
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}/${m}/${d}`;
}

/** @deprecated Use column formatters instead */
export function formatTrainingDatasetPickerSchemaLabel(
  option: TrainingDatasetOption,
  meta?: TrainingDatasetPickerMeta | null
): string {
  return [
    formatDatasetSchemaLabel(option, meta),
    formatDatasetRobotLabel(option, meta),
    formatDatasetFormatLabel(option),
  ]
    .filter(Boolean)
    .join(' · ');
}

/** @deprecated Use formatDatasetDate instead */
export function formatTrainingDatasetPickerCreatedAt(value?: string | null): string {
  return formatDatasetDate(value);
}

export function formatSelectedTrainingDatasetsTriggerLabel(
  selectedIds: string[],
  options: TrainingDatasetOption[]
): string {
  if (selectedIds.length === 0) return '';
  if (selectedIds.length > 1) return `已选择 ${selectedIds.length} 个数据集`;
  const option = options.find((item) => item.id === selectedIds[0]);
  if (!option) return '已选择 1 个数据集';
  return `${formatDatasetDisplayName(option)} · ${option.sampleCount} demo`;
}

export function validateTrainingDatasetSelection(
  selectedIds: string[],
  allOptions: TrainingDatasetOption[]
): { ok: true } | { ok: false; reason: 'incompatible' } {
  if (selectedIds.length <= 1) return { ok: true };
  const [first, ...rest] = selectedIds;
  for (const id of rest) {
    const option = allOptions.find((item) => item.id === id);
    if (!option) continue;
    if (!isDatasetCompatibleWithSelection(option, [first], allOptions)) {
      return { ok: false, reason: 'incompatible' };
    }
  }
  return { ok: true };
}

export function toggleTrainingDatasetDraftSelection(
  draftIds: string[],
  datasetId: string,
  allOptions: TrainingDatasetOption[],
  multiple: boolean
): { nextIds: string[]; error: string | null } {
  const option = allOptions.find((item) => item.id === datasetId);
  if (!option) return { nextIds: draftIds, error: null };

  if (!multiple) {
    return { nextIds: [datasetId], error: null };
  }

  if (draftIds.includes(datasetId)) {
    return { nextIds: draftIds.filter((id) => id !== datasetId), error: null };
  }

  if (!isDatasetCompatibleWithSelection(option, draftIds, allOptions)) {
    return { nextIds: draftIds, error: 'DATASET_MERGE_INCOMPATIBLE' };
  }

  return { nextIds: [...draftIds, datasetId], error: null };
}

export function isDatasetStatusPositive(status?: string | null): boolean {
  const normalized = (status ?? '').trim().toLowerCase();
  return normalized === 'available' || normalized === 'ready' || normalized === 'completed';
}
