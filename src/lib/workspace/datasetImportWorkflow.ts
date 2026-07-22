import type { Dataset } from '@/types/benchmark';
import { isBuildSourceImportedHdf5Dataset } from '@/lib/workspace/buildSourceDatasetPicker';

export { isBuildSourceImportedHdf5Dataset } from '@/lib/workspace/buildSourceDatasetPicker';

export function isImportedWorkspaceDataset(dataset: Pick<Dataset, 'id' | 'sourceJobId'>): boolean {
  const id = (dataset.id ?? '').trim();
  const jobId = (dataset.sourceJobId ?? '').trim();
  return id.startsWith('ds_import_') || jobId.startsWith('import_ds_import_');
}

export function isBuiltWorkspaceDataset(dataset: Pick<Dataset, 'id' | 'sourceJobId'>): boolean {
  const id = (dataset.id ?? '').trim();
  const jobId = (dataset.sourceJobId ?? '').trim();
  return id.startsWith('ds_built_') || jobId.startsWith('built_ds_built_');
}

/** 归一化导入集状态（兼容旧值 available / pending_field_mapping / import_failed）。 */
export function normalizeImportedDatasetStatus(status: string | null | undefined): string {
  const raw = (status ?? '').trim().toLowerCase();
  if (raw === 'available') return 'ready';
  if (raw === 'pending_field_mapping') return 'needs_mapping';
  if (raw === 'import_failed') return 'failed';
  return raw;
}

export function isImportedDatasetDirectTrainable(dataset: Dataset): boolean {
  if (!isImportedWorkspaceDataset(dataset)) return false;
  if (dataset.directTrainable === true) return true;
  const status = normalizeImportedDatasetStatus(dataset.status);
  return status === 'ready' && dataset.trainable === true;
}

/** 可作为数据构建源：真实导入 HDF5，且非 failed。 */
export function isBuildableImportedDataset(dataset: Dataset): boolean {
  return isBuildSourceImportedHdf5Dataset(dataset);
}

export function shouldShowImportedDatasetBuildAction(dataset: Dataset): boolean {
  if (isBuiltWorkspaceDataset(dataset)) return false;
  if (!isBuildableImportedDataset(dataset)) return false;
  const status = normalizeImportedDatasetStatus(dataset.status);
  return status === 'ready' || status === 'needs_mapping' || status === 'needs_build' || dataset.needsBuild === true;
}

/** 数据中心列表：导入 HDF5 展示「构建」（含标准可训练集）；构建后数据集不展示。 */
export function shouldShowImportedDatasetBuildActionInList(dataset: Dataset): boolean {
  return shouldShowImportedDatasetBuildAction(dataset);
}
