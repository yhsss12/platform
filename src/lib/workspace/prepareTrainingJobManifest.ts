import type { TrainingBackendRequest } from '@/lib/api/trainingClient';
import type { DatasetManifest } from '@/lib/workspace/datasetManifest';
import type { CreateTrainingTaskInput } from '@/lib/mock/workspaceTrainingMock';
import { trainingDeviceSubmitParams } from '@/lib/workspace/trainingDevice';

export function resolveTrainingDataFormat(manifest: DatasetManifest, input?: CreateTrainingTaskInput): string {
  if (input?.datasetFormat?.trim()) return input.datasetFormat.trim();
  const direct = (manifest.dataFormat ?? '').trim();
  if (direct) return direct;
  if (manifest.mainFormats?.some((f) => f.includes('HDF5'))) {
    if (manifest.mainFormats.some((f) => f.includes('NPZ'))) return 'HDF5 + NPZ';
    return 'HDF5';
  }
  if (manifest.artifacts?.hdf5) return 'HDF5';
  if (manifest.artifacts?.npz) return 'NPZ';
  return 'HDF5';
}

/** 训练任务提交前：以弹窗选择为准，并尽量补全 HDF5 路径。 */
export function prepareTrainingJobManifest(
  manifest: DatasetManifest,
  input: CreateTrainingTaskInput
): DatasetManifest {
  const artifacts = { ...manifest.artifacts };

  if (!artifacts.hdf5) {
    const fromNpz = artifacts.npz?.replace(/dataset\.npz$/i, 'dataset.hdf5');
    if (fromNpz) {
      artifacts.hdf5 = fromNpz;
    }
  }

  return {
    ...manifest,
    downstreamModelType: input.downstreamModelType as DatasetManifest['downstreamModelType'],
    dataFormat: resolveTrainingDataFormat(manifest) as DatasetManifest['dataFormat'],
    artifacts,
  };
}

export function buildTrainingJobRequest(
  manifest: DatasetManifest,
  input: CreateTrainingTaskInput,
  datasetManifests?: DatasetManifest[]
): {
  datasetId: string;
  datasetIds?: string[];
  datasetManifestPath?: string;
  datasetManifest: DatasetManifest;
  datasetManifests?: DatasetManifest[];
  modelTypeId: string;
  downstreamModelType: string;
  trainingBackend: TrainingBackendRequest;
  dataFormat: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  device: string;
  deviceLabel: string;
  trainingNodeId: string;
  seed: number;
  seedMode?: CreateTrainingTaskInput['seedMode'];
  pretrained?: CreateTrainingTaskInput['pretrained'];
  taskName?: string;
  saveFinal?: boolean;
  saveBest?: boolean;
  checkpointIntervalEpochs?: number | null;
} {
  const prepared = prepareTrainingJobManifest(manifest, input);
  const deviceParams = trainingDeviceSubmitParams(input.trainingDevice);
  const request = {
    datasetId: input.datasets[0] ?? input.dataset,
    ...(input.datasets.length > 1
      ? {
          datasetIds: input.datasets,
          datasetManifests: datasetManifests ?? [prepared],
        }
      : {}),
    datasetManifestPath: prepared.artifacts?.manifest,
    datasetManifest: prepared,
    modelTypeId: input.modelTypeId,
    downstreamModelType: input.downstreamModelType,
    trainingBackend: input.trainingBackend as TrainingBackendRequest,
    dataFormat: resolveTrainingDataFormat(prepared, input),
    epochs: input.epochs,
    batchSize: input.batchSize,
    learningRate: input.learningRate,
    device: deviceParams.device,
    deviceLabel: deviceParams.deviceLabel,
    trainingNodeId: deviceParams.trainingNodeId,
    seed: input.seed,
    seedMode: input.seedMode,
    ...(input.pretrained ? { pretrained: input.pretrained } : {}),
    ...(input.taskName?.trim() ? { taskName: input.taskName.trim() } : {}),
    ...(input.maxSteps != null ? { maxSteps: input.maxSteps, smokeSteps: input.maxSteps } : {}),
    ...(input.datasetFormat ? { datasetFormat: input.datasetFormat } : {}),
    ...(input.taskInstruction ? { taskInstruction: input.taskInstruction } : {}),
    saveFinal: input.saveFinal ?? true,
    saveBest: input.saveBest ?? false,
    checkpointIntervalEpochs: input.checkpointIntervalEpochs ?? null,
  };

  return request;
}
