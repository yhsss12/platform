import type { Dataset } from '@/types/benchmark';
import type { GenerationPath } from '@/lib/workspace/generateDataTypes';
import { isNutAssemblyDataset } from '@/lib/workspace/nutAssembly';
import { normalizeImportedDatasetStatus } from '@/lib/workspace/datasetImportWorkflow';
import { formatSeedDatasetCreatedAt } from '@/lib/workspace/isaacSeedDatasets';

/** 螺母装配内置默认示范数据集（非 workspace 登记 id，仅用于生成请求） */
export const NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID = 'nut_assembly_default_demo_dataset';

export const NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_LABEL = '螺母装配示范数据（默认）';

export function isNutAssemblyBuiltInDefaultDemoDatasetId(
  datasetId: string | null | undefined
): boolean {
  return (datasetId ?? '').trim() === NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID;
}

export function resolveNutAssemblyEffectiveSourceDemoDatasetId(
  generationPath: GenerationPath,
  sourceDemoDatasetId: string
): string {
  if (generationPath === 'demo_augmentation') {
    const trimmed = sourceDemoDatasetId.trim();
    return trimmed || NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID;
  }
  return sourceDemoDatasetId.trim();
}

const NUT_ASSEMBLY_READY_STATUSES = new Set(['ready', 'available', 'completed', 'built']);

const NUT_ASSEMBLY_HDF5_FORMATS = new Set(['hdf5', 'robomimic_hdf5']);

const MOCK_ID_PREFIXES = ['mock_', 'tpl-', 'demo-mock-'];

function normalizeFormat(dataset: Dataset): string {
  const raw = (dataset.format ?? dataset.datasetFormat ?? '').trim().toLowerCase();
  return raw;
}

function normalizeSimulator(dataset: Dataset): string {
  return (dataset.simulatorBackend ?? '').trim().toLowerCase();
}

function isPandaRobot(dataset: Dataset): boolean {
  const robot = (dataset.robotType ?? '').trim().toLowerCase();
  if (!robot) return true;
  return robot.includes('panda') || robot.includes('panda_single_arm');
}

function resolveDemoCount(dataset: Dataset): number {
  const candidates = [
    dataset.demoCount,
    dataset.successfulEpisodes,
    dataset.successEpisodes,
    dataset.episodeCount,
    dataset.dataCount,
    dataset.finalDemoCount,
    dataset.validForTrainingEpisodes,
  ];
  for (const value of candidates) {
    const n = Number(value);
    if (Number.isFinite(n) && n > 0) return Math.trunc(n);
  }
  return 0;
}

function isMockPlaceholderDataset(dataset: Dataset): boolean {
  const id = (dataset.id ?? '').trim().toLowerCase();
  if (MOCK_ID_PREFIXES.some((prefix) => id.startsWith(prefix))) return true;
  if ((dataset as Dataset & { isPlaceholder?: boolean }).isPlaceholder === true) return true;
  const status = normalizeImportedDatasetStatus(dataset.status);
  if (status === 'generating' || status === 'pending' || status === 'failed') return true;
  return false;
}

export function isNutAssemblySourceDemoDataset(dataset: Dataset): boolean {
  if (!isNutAssemblyDataset(dataset)) return false;
  if (isMockPlaceholderDataset(dataset)) return false;

  const simulator = normalizeSimulator(dataset);
  if (simulator && simulator !== 'mujoco') return false;

  if (!isPandaRobot(dataset)) return false;

  const format = normalizeFormat(dataset);
  if (format && !NUT_ASSEMBLY_HDF5_FORMATS.has(format)) return false;

  const status = normalizeImportedDatasetStatus(dataset.status);
  if (status && !NUT_ASSEMBLY_READY_STATUSES.has(status)) return false;

  if (!dataset.datasetFile?.trim()) return false;

  if (resolveDemoCount(dataset) <= 0) return false;

  return true;
}

export function filterNutAssemblySourceDemoDatasets(datasets: Dataset[]): Dataset[] {
  return datasets
    .filter(isNutAssemblySourceDemoDataset)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function formatNutAssemblySourceDemoOptionLabel(dataset: Dataset): string {
  const count = resolveDemoCount(dataset);
  const episodesLabel = count > 0 ? `${count} 条示范` : '示范数未知';
  const created = formatSeedDatasetCreatedAt(dataset.createdAt);
  return `${dataset.name} · ${episodesLabel} · ${created}`;
}

export function resolveNutAssemblySourceDemoPath(
  datasets: Dataset[],
  datasetId: string
): string | null {
  if (!datasetId.trim() || isNutAssemblyBuiltInDefaultDemoDatasetId(datasetId)) {
    return null;
  }
  const selected = datasets.find((row) => row.id === datasetId);
  const path = selected?.datasetFile?.trim();
  return path || null;
}
