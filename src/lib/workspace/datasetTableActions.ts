import type { Dataset } from '@/types/benchmark';
import { buildCableThreadingConsoleHref } from '@/lib/workspace/cableThreading';
import { buildDualArmCableConsoleHref } from '@/lib/workspace/dualArmCable';
import { buildNutAssemblyConsoleHref, isNutAssemblyDataset } from '@/lib/workspace/nutAssembly';
import { buildIsaacBlockStackingConsoleHref } from '@/lib/workspace/isaacBlockStacking';
import { buildIsaacSimFrankaPickPlaceConsoleHref } from '@/lib/workspace/isaacsimFrankaPickPlace';
import {
  buildIsaacLabFrankaStackCubeConsoleHref,
  isIsaacLabFrankaStackCubeDataset,
} from '@/lib/workspace/isaaclabFrankaStackCube';
import { isIsaacSimFrankaPickPlaceDataset } from '@/lib/workspace/isaacsimFrankaPickPlace';
import {
  buildUnifiedDatasetReplayHref,
  resolveDatasetReplayTaskType,
} from '@/lib/workspace/datasetReplayHref';

export type DatasetPrimaryActionLabel = '运行' | '回放';

export interface DatasetPrimaryAction {
  label: DatasetPrimaryActionLabel;
  href?: string;
}

const IN_PROGRESS_STATUSES = new Set([
  'running',
  'queued',
  'pending',
  'generating',
  '待生成',
  '生成中',
]);

const FAILED_STATUSES = new Set(['failed', '失败', 'canceled', 'cancelled']);

const AVAILABLE_STATUSES = new Set(['available', 'ready', 'completed', '已完成', 'exported', 'built']);

function normalizeStatus(status: string): string {
  return status.trim().toLowerCase();
}

export function isDatasetAvailable(dataset: Dataset): boolean {
  const normalized = normalizeStatus(dataset.status);
  return AVAILABLE_STATUSES.has(normalized) || AVAILABLE_STATUSES.has(dataset.status);
}

export function isDatasetInProgressOrFailed(dataset: Dataset): boolean {
  const normalized = normalizeStatus(dataset.status);
  if (IN_PROGRESS_STATUSES.has(normalized) || IN_PROGRESS_STATUSES.has(dataset.status)) return true;
  if (FAILED_STATUSES.has(normalized) || FAILED_STATUSES.has(dataset.status)) return true;
  return !isDatasetAvailable(dataset);
}

/**
 * 是否具备可直接进入回放页的资源（视频 / 有效最终帧 / 帧序列）。
 * 优先使用 workspace dataset 索引写入的 replayAvailable；未提供时保守为 false。
 */
export function datasetHasReplayResources(dataset: Dataset): boolean {
  if (!isDatasetAvailable(dataset)) return false;
  return dataset.replayAvailable === true;
}

export function isIsaacLabDataset(dataset: Dataset): boolean {
  return (
    dataset.replayBackend === 'isaac_lab' ||
    dataset.simulatorBackend === 'isaac_lab' ||
    dataset.sourceType === 'imported_demo' ||
    dataset.sourceJobId.startsWith('isaac_import_') ||
    dataset.sourceJobId.startsWith('isaac_gen_')
  );
}

/** Isaac Lab 物块堆叠 registry（isaac_gen_* / isaac_import_*），不含 Franka Stack Cube。 */
export function isLegacyIsaacLabRegistryDataset(dataset: Dataset): boolean {
  const jobId = dataset.sourceJobId?.trim() ?? '';
  return (
    dataset.sourceType === 'imported_demo' ||
    jobId.startsWith('isaac_import_') ||
    jobId.startsWith('isaac_gen_')
  );
}

export function shouldShowIsaacDatasetReplay(dataset: Dataset): boolean {
  if (!isDatasetAvailable(dataset)) return false;
  if (dataset.replayAvailable === true) return true;
  const format = dataset.format === 'hdf5' || dataset.datasetFormat === 'hdf5';
  return format && dataset.simulatorBackend === 'isaac_lab' && Boolean(dataset.datasetFile);
}

export function shouldShowIsaacDatasetRun(dataset: Dataset): boolean {
  const jobId = dataset.sourceJobId?.trim();
  return Boolean(jobId?.startsWith('isaac_gen_'));
}

export function resolveDatasetConsoleHref(dataset: Dataset): string | null {
  const jobId = dataset.sourceJobId?.trim();
  if (!jobId) return null;

  if (jobId.startsWith('na_gen_') || isNutAssemblyDataset(dataset)) {
    return buildNutAssemblyConsoleHref({ jobId });
  }
  if (jobId.startsWith('ct_gen_')) {
    return buildCableThreadingConsoleHref({ jobId });
  }
  if (jobId.startsWith('dac_gen_')) {
    return buildDualArmCableConsoleHref({ jobId });
  }
  if (jobId.startsWith('isaac_gen_')) {
    return buildIsaacBlockStackingConsoleHref({ jobId });
  }
  if (jobId.startsWith('data_gen_')) {
    if (isIsaacLabFrankaStackCubeDataset(dataset)) {
      return buildIsaacLabFrankaStackCubeConsoleHref({ jobId });
    }
    if (isIsaacSimFrankaPickPlaceDataset(dataset)) {
      return buildIsaacSimFrankaPickPlaceConsoleHref({ jobId });
    }
  }
  return null;
}

export function resolveUnifiedDatasetReplayHref(dataset: Dataset): string | null {
  if (!isDatasetAvailable(dataset)) return null;

  const taskType = resolveDatasetReplayTaskType(dataset);
  const sourceJobId = dataset.sourceJobId?.trim();

  if (taskType === 'isaac_block_stacking') {
    if (!shouldShowIsaacDatasetReplay(dataset)) return null;
    return buildUnifiedDatasetReplayHref({
      taskType,
      datasetId: dataset.id,
      sourceJobId: sourceJobId?.startsWith('isaac_gen_') ? sourceJobId : undefined,
    });
  }

  if (taskType === 'isaacsim_franka_pick_place') {
    if (!datasetHasReplayResources(dataset) && !sourceJobId?.startsWith('data_gen_')) return null;
    return buildUnifiedDatasetReplayHref({
      taskType,
      datasetId: dataset.id,
      sourceJobId: sourceJobId || undefined,
    });
  }

  if (taskType === 'isaaclab_franka_stack_cube') {
    if (!datasetHasReplayResources(dataset)) return null;
    return buildUnifiedDatasetReplayHref({
      taskType,
      datasetId: dataset.id,
      sourceJobId: sourceJobId || undefined,
    });
  }

  if (!datasetHasReplayResources(dataset) && !sourceJobId) return null;

  return buildUnifiedDatasetReplayHref({
    taskType,
    datasetId: dataset.id,
    sourceJobId: sourceJobId || undefined,
  });
}

export function resolveIsaacDatasetReplayHref(dataset: Dataset): string | null {
  return resolveUnifiedDatasetReplayHref(dataset);
}

export function resolveDatasetReplayHref(dataset: Dataset): string | null {
  return resolveUnifiedDatasetReplayHref(dataset);
}

/**
 * 数据集列表「运行 / 回放」动态主操作（非 Isaac 数据集）。
 * 可用且有回放资源 → 回放（直达 /workspace/replay）；
 * 有来源 job 但无回放资源 → 运行（控制台）；
 * 无来源 job → 不显示主操作按钮。
 */
export function resolveDatasetPrimaryAction(dataset: Dataset): DatasetPrimaryAction | null {
  const replayHref = resolveUnifiedDatasetReplayHref(dataset);
  const consoleHref = resolveDatasetConsoleHref(dataset);

  if (replayHref) {
    return { label: '回放', href: replayHref };
  }

  if (consoleHref) {
    return { label: '运行', href: consoleHref };
  }

  return null;
}
