import type { Dataset } from '@/types/benchmark';
import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import type { ModelTypeDefinition } from '@/types/modelType';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import { isLerobotSidecarDataset } from '@/lib/workspace/datasetTrainingAccess';

export const PI0_NO_DATASET_HINT = '暂无可用的 pi0 / LeRobot 数据集';

export function isPi0ModelType(modelType: Pick<ModelTypeDefinition, 'modelTypeId' | 'baseAlgorithm' | 'adapterKey'> | null | undefined): boolean {
  if (!modelType) return false;
  return (
    modelType.modelTypeId === 'pi0' ||
    modelType.baseAlgorithm === 'pi0' ||
    modelType.adapterKey === 'pi0_adapter'
  );
}

function readPi0ReadyFromDataItem(item: WorkspaceDataItem): boolean | undefined {
  if (typeof item.pi0Ready === 'boolean') return item.pi0Ready;
  return undefined;
}

function isLerobotSidecarDatasetFromMock(item: WorkspaceDataItem): boolean {
  return item.dataOrganizationFormat === 'LeRobot' || Boolean(item.lerobotPath);
}

function hasLerobotFormatFromDataItem(item: WorkspaceDataItem): boolean {
  const formats = item.mainFormats ?? [];
  if (formats.some((fmt) => String(fmt).toLowerCase() === 'lerobot')) return true;
  return isLerobotSidecarDatasetFromMock(item);
}

export function datasetSupportsPi0Training(dataset: Dataset): boolean {
  const formats = [
    ...(dataset as Dataset & { availableFormats?: string[] }).availableFormats ?? [],
    ...(dataset as Dataset & { mainFormats?: string[] }).mainFormats ?? [],
  ]
    .filter(Boolean)
    .map((value) => String(value).toLowerCase());
  const lerobotReady =
    isLerobotSidecarDataset(dataset) ||
    formats.includes('lerobot') ||
    dataset.dataOrganizationFormat === 'LeRobot';
  return lerobotReady && dataset.pi0Ready === true && (dataset.episodeCount ?? 0) > 0;
}

export function trainingDatasetOptionSupportsPi0(option: TrainingDatasetOption): boolean {
  if (option.pi0Ready !== true) return false;
  const format = String(option.dataFormat ?? '').toLowerCase();
  return format.includes('lerobot') || Boolean(option.lerobotPath);
}

export function filterTrainingDatasetOptionsForModelType(
  options: TrainingDatasetOption[],
  modelType: Pick<ModelTypeDefinition, 'modelTypeId' | 'baseAlgorithm' | 'adapterKey'> | null | undefined,
  dataCenterItems: WorkspaceDataItem[] = [],
  apiDatasets: Dataset[] = []
): TrainingDatasetOption[] {
  if (!isPi0ModelType(modelType)) {
    return options.filter((option) => !trainingDatasetOptionSupportsPi0(option));
  }

  const apiById = new Map(apiDatasets.map((item) => [item.id, item]));
  const itemById = new Map(
    dataCenterItems.map((item) => [item.datasetId ?? item.id, item])
  );

  return options.filter((option) => {
    const apiDataset = apiById.get(option.id);
    if (apiDataset && datasetSupportsPi0Training(apiDataset)) return true;

    const dataItem = itemById.get(option.id);
    if (dataItem) {
      return hasLerobotFormatFromDataItem(dataItem) && readPi0ReadyFromDataItem(dataItem) === true;
    }

    return trainingDatasetOptionSupportsPi0(option);
  });
}
