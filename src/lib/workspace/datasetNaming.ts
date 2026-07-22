/** 统一数据集用户可见命名（不改 taskType / jobId / 目录名） */

import {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  getTaskDisplayName,
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
} from '@/lib/workspace/taskDisplayNames';

const CANONICAL_DISPLAY_NAME_PATTERN =
  /^(线缆穿杆|线缆整理|线缆操控|螺母装配|物块堆叠|Franka Stack Cube)数据_\d{8}_\d{4}(?:_\d{2})?$/;

const JOB_ID_TIMESTAMP_PATTERN =
  /(?:^|_)(?:ct_gen|dac_gen|na_gen|isaac_import|isaac_gen|isaac_ds)_(\d{8})_(\d{6})/;

const LEGACY_NAME_MARKERS = [
  '单臂线缆穿杆',
  '双臂线缆操控',
  '螺母装配',
  'Isaac Stack Cube',
  'stack cube',
  'cable threading',
  'dual arm cable',
  'generated_dataset',
  'dataset.hdf5',
  'task_cable_threading_v1',
  'task_dual_arm_cable_manipulation_v1',
];

export type DatasetNamingInput = {
  taskType?: string | null;
  displayName?: string | null;
  name?: string | null;
  createdAt?: string | null;
  sourceJobId?: string | null;
  taskDisplayName?: string | null;
};

export function isCanonicalDatasetDisplayName(name: string | null | undefined): boolean {
  if (!name?.trim()) return false;
  return CANONICAL_DISPLAY_NAME_PATTERN.test(name.trim());
}

function looksLikeLegacyDisplayName(name: string): boolean {
  if (isCanonicalDatasetDisplayName(name)) return false;
  const lowered = name.toLowerCase();
  if (name.includes(' · ') && /ct_gen_|dac_gen_|isaac_/.test(name)) return true;
  return LEGACY_NAME_MARKERS.some(
    (marker) => lowered.includes(marker.toLowerCase()) || name.includes(marker)
  );
}

function parseCreatedAt(value: string | null | undefined): Date | null {
  if (!value?.trim()) return null;
  let raw = value.trim();
  if (/^\d{8}T/.test(raw)) {
    raw = `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}${raw.slice(8)}`;
  }
  if (raw.includes(' ') && !raw.includes('T')) {
    raw = raw.replace(' ', 'T');
  }
  const d = new Date(raw);
  return Number.isNaN(d.getTime()) ? null : d;
}

function datetimeFromJobId(jobId: string | null | undefined): Date | null {
  if (!jobId?.trim()) return null;
  const match = jobId.match(JOB_ID_TIMESTAMP_PATTERN);
  if (!match) return null;
  const ymd = match[1];
  const hms = match[2];
  const iso = `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}T${hms.slice(0, 2)}:${hms.slice(2, 4)}:${hms.slice(4, 6)}Z`;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTimestamp(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;
}

function indexSuffix(datasetIndex: number): string {
  if (datasetIndex <= 1) return '';
  return `_${String(datasetIndex).padStart(2, '0')}`;
}

function taskLabelForNaming(taskType: string): string {
  const key = taskType.trim();
  if (!key || key === 'unknown') return '未知';
  const label = getTaskDisplayName(key);
  if (!label || label === '—' || label.toLowerCase() === 'unknown') return '未知';
  return label;
}

export function isInvalidDatasetDisplayName(name?: string | null): boolean {
  const text = name?.trim();
  if (!text) return true;
  if (text === '未知任务数据' || text === '未知数据集') return true;
  if (/^unknown/i.test(text)) return true;
  if (text.startsWith('unknown数据')) return true;
  return false;
}

export function inferTaskTypeFromJobId(jobId: string | null | undefined): string | null {
  if (!jobId) return null;
  let normalized = jobId.trim();
  if (normalized.startsWith('ds_')) {
    normalized = normalized.slice(3);
  }
  if (normalized.startsWith('ct_gen_')) return 'cable_threading';
  if (normalized.startsWith('dac_gen_')) return 'dual_arm_cable_manipulation';
  if (normalized.startsWith('na_gen_')) return 'nut_assembly';
  if (
    normalized.startsWith('isaac_gen_') ||
    normalized.startsWith('isaac_import_') ||
    normalized.startsWith('isaac_ds_')
  ) {
    return 'block_stacking';
  }
  return null;
}

export function buildDatasetDisplayName(input: {
  taskType: string;
  createdAt?: string | Date | null;
  sourceJobId?: string | null;
  datasetIndex?: number;
  generationMode?: string | null;
}): string {
  const label = taskLabelForNaming(input.taskType);
  let dt: Date | null = null;
  if (input.createdAt instanceof Date) {
    dt = input.createdAt;
  } else if (typeof input.createdAt === 'string') {
    dt = parseCreatedAt(input.createdAt);
  }
  dt = dt ?? datetimeFromJobId(input.sourceJobId) ?? new Date();
  const stamp = formatTimestamp(dt);
  const suffix = indexSuffix(input.datasetIndex ?? 1);
  return `${label}数据_${stamp}${suffix}`;
}

export function normalizeDatasetDisplayName(input: DatasetNamingInput): string {
  for (const candidate of [input.displayName, input.name]) {
    if (candidate && isCanonicalDatasetDisplayName(candidate)) {
      return candidate.trim();
    }
  }

  const taskType =
    input.taskType?.trim() ||
    inferTaskTypeFromJobId(input.sourceJobId) ||
    undefined;

  if (taskType && taskType !== 'unknown') {
    for (const candidate of [input.displayName, input.name]) {
      if (candidate && !looksLikeLegacyDisplayName(candidate)) {
        if (isCanonicalDatasetDisplayName(candidate)) return candidate.trim();
      }
    }
    return buildDatasetDisplayName({
      taskType,
      createdAt: input.createdAt,
      sourceJobId: input.sourceJobId,
    });
  }

  if (input.sourceJobId) {
    const inferred = inferTaskTypeFromJobId(input.sourceJobId);
    if (inferred) {
      return buildDatasetDisplayName({
        taskType: inferred,
        createdAt: input.createdAt,
        sourceJobId: input.sourceJobId,
      });
    }
  }

  return '未知数据集';
}

export function resolveDatasetListName(dataset: DatasetNamingInput): string {
  return normalizeDatasetDisplayName(dataset);
}

export function resolveDatasetSourceTaskDisplayName(dataset: DatasetNamingInput): string {
  if (dataset.taskDisplayName?.trim()) {
    return getTaskDisplayName(dataset.taskDisplayName);
  }
  const taskType =
    dataset.taskType?.trim() ||
    inferTaskTypeFromJobId(dataset.sourceJobId) ||
    undefined;
  if (taskType) return getTaskDisplayName(taskType);
  return '—';
}

export {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  ISAAC_BLOCK_STACKING_DISPLAY_NAME,
};
