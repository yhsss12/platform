import { ISAAC_BLOCK_STACKING_TEMPLATE_ID } from '@/lib/workspace/isaacBlockStacking';
import type { Dataset } from '@/types/benchmark';

const ISAAC_BLOCK_STACKING_REGISTRY_ID = 'task_isaac_block_stacking_v1';

export function isIsaacBlockStackingSeedDataset(dataset: Dataset): boolean {
  const taskMatch =
    dataset.taskTemplateId === ISAAC_BLOCK_STACKING_TEMPLATE_ID ||
    dataset.sourceTaskTemplateId === ISAAC_BLOCK_STACKING_REGISTRY_ID;

  if (!taskMatch) return false;
  if (dataset.simulatorBackend !== 'isaac_lab') return false;
  if (dataset.format !== 'hdf5') return false;
  if (dataset.status !== 'available') return false;
  if (!dataset.datasetFile?.trim()) return false;

  const sourceOk =
    dataset.sourceType === 'imported_demo' || dataset.replayAvailable === true;
  return sourceOk;
}

export function filterIsaacSeedDatasets(datasets: Dataset[]): Dataset[] {
  return datasets
    .filter(isIsaacBlockStackingSeedDataset)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function formatSeedDatasetOptionLabel(dataset: Dataset): string {
  const episodes =
    dataset.episodeCount > 0 ? `${dataset.episodeCount} demos` : 'demo 数未知';
  const replay = dataset.replayAvailable ? '可回放' : '不可回放';
  const created = formatSeedDatasetCreatedAt(dataset.createdAt);
  return `${dataset.name} · ${episodes} · ${replay} · ${created}`;
}

export function formatSeedDatasetCreatedAt(iso: string): string {
  const parsed = Date.parse(iso);
  if (Number.isNaN(parsed)) return iso;
  return new Date(parsed).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function resolveSeedDatasetFile(
  datasets: Dataset[],
  selectedDatasetId: string,
  manualPath: string
): string {
  if (selectedDatasetId) {
    const selected = datasets.find((row) => row.id === selectedDatasetId);
    if (selected?.datasetFile?.trim()) {
      return selected.datasetFile.trim();
    }
  }
  return manualPath.trim();
}
