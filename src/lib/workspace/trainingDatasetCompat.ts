import type { DatasetManifest } from '@/lib/workspace/datasetManifest';
import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import { CABLE_THREADING_DEFAULTS, CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS } from '@/lib/workspace/cableThreading';

export interface DatasetStructureSignature {
  taskType: string;
  robotType: string;
  simulatorBackend: string;
  imageKeys: string[];
  lowDimKeys: string[];
  actionDim: number | null;
  imageSize: number | null;
}

export const DATASET_MERGE_INCOMPATIBLE_HINT = '仅支持同任务/同结构数据集合并训练';

function normKeys(keys: unknown): string[] {
  if (!Array.isArray(keys)) return [];
  return [...new Set(keys.map((key) => String(key).trim()).filter(Boolean))].sort();
}

function imageSizeFromManifest(manifest: DatasetManifest & Record<string, unknown>): number | null {
  const direct = manifest.imageSize ?? manifest.image_size;
  if (direct != null && Number(direct) > 0) return Number(direct);
  const shape = manifest.imageShape as Record<string, unknown> | undefined;
  if (shape) {
    for (const key of ['height', 'width', 'h', 'w']) {
      const value = shape[key];
      if (value != null && Number(value) > 0) return Number(value);
    }
  }
  return null;
}

export function extractDatasetStructureSignature(
  manifest: DatasetManifest & Record<string, unknown>
): DatasetStructureSignature {
  const extended = manifest as DatasetManifest & {
    cameraKeys?: string[];
    imageKeys?: string[];
    obsKeys?: string[];
    observationKeys?: string[];
    imageSize?: number;
    image_size?: number;
    imageShape?: Record<string, unknown>;
    robotType?: string;
    simulatorBackend?: string;
    simBackend?: string;
  };
  const cameraKeys = normKeys(
    extended.cameraKeys ??
      extended.imageKeys ??
      extended.obsKeys?.filter((key) => String(key).includes('image'))
  );
  const obsKeys = normKeys(extended.obsKeys ?? extended.observationKeys);
  const lowDimKeys = normKeys(
    extended.observationKeys ??
      obsKeys.filter((key) => !cameraKeys.includes(key) && !String(key).endsWith('_image'))
  );

  let actionDim = manifest.actionDim != null ? Number(manifest.actionDim) : null;
  if (manifest.taskType === 'dual_arm_cable_manipulation' && actionDim == null) {
    actionDim = 14;
  }
  if ((manifest.taskType === 'cable_threading' || manifest.sourceJobId?.startsWith('ct_gen_')) && actionDim == null) {
    actionDim = 7;
  }

  if (cameraKeys.length === 0 && manifest.taskType !== 'dual_arm_cable_manipulation') {
    const cableImageKeys = CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => key.includes('image'));
    if (manifest.sourceJobId?.startsWith('ct_gen_') || manifest.taskType === 'cable_threading') {
      cameraKeys.push(...cableImageKeys);
    }
  }
  if (lowDimKeys.length === 0 && manifest.sourceJobId?.startsWith('ct_gen_')) {
    lowDimKeys.push(
      ...CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => !String(key).includes('image'))
    );
  }

  return {
    taskType: String(manifest.taskType ?? '').trim(),
    robotType: String(manifest.robotType ?? manifest.robot ?? '').trim(),
    simulatorBackend: String(
      manifest.simulatorBackend ?? manifest.backend ?? manifest.simBackend ?? ''
    ).trim(),
    imageKeys: cameraKeys,
    lowDimKeys,
    actionDim,
    imageSize: imageSizeFromManifest(manifest),
  };
}

export function extractDatasetStructureSignatureFromOption(
  option: TrainingDatasetOption
): DatasetStructureSignature {
  let taskType = option.taskType ?? '';
  if (!taskType && option.sourceJobId?.startsWith('ct_gen_')) taskType = 'cable_threading';
  if (!taskType && option.sourceJobId?.startsWith('dac_gen_')) taskType = 'dual_arm_cable_manipulation';
  if (!taskType && option.simulatorBackend === 'isaac_lab') taskType = 'isaac_block_stacking';

  const imageKeys =
    option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_')
      ? CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => key.includes('image'))
      : [];
  const lowDimKeys =
    option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_')
      ? CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => !String(key).includes('image'))
      : [];

  return {
    taskType,
    robotType:
      option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_')
        ? CABLE_THREADING_DEFAULTS.robot
        : option.taskType === 'dual_arm_cable_manipulation'
          ? '双臂协作机器人'
          : option.simulatorBackend === 'isaac_lab'
            ? 'Franka'
            : '',
    simulatorBackend:
      option.simulatorBackend ??
      (option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_')
        ? 'mujoco'
        : option.taskType === 'dual_arm_cable_manipulation'
          ? 'MuJoCo'
          : ''),
    imageKeys,
    lowDimKeys,
    actionDim:
      option.taskType === 'dual_arm_cable_manipulation'
        ? 14
        : option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_')
          ? 7
          : option.simulatorBackend === 'isaac_lab'
            ? 7
            : null,
    imageSize:
      option.taskType === 'cable_threading' || option.sourceJobId?.startsWith('ct_gen_') ? 84 : null,
  };
}

function signatureEqual(left: DatasetStructureSignature, right: DatasetStructureSignature): boolean {
  return (
    left.taskType === right.taskType &&
    left.robotType === right.robotType &&
    left.simulatorBackend === right.simulatorBackend &&
    left.actionDim === right.actionDim &&
    left.imageSize === right.imageSize &&
    left.imageKeys.join('|') === right.imageKeys.join('|') &&
    left.lowDimKeys.join('|') === right.lowDimKeys.join('|')
  );
}

export function isDatasetCompatibleWithSelection(
  option: TrainingDatasetOption,
  selectedIds: string[],
  allOptions: TrainingDatasetOption[]
): boolean {
  if (selectedIds.length === 0) return true;
  const base = allOptions.find((item) => item.id === selectedIds[0]);
  if (!base) return true;
  return signatureEqual(
    extractDatasetStructureSignatureFromOption(base),
    extractDatasetStructureSignatureFromOption(option)
  );
}

export function mergeTrainingManifestsClient(manifests: DatasetManifest[]): DatasetManifest {
  if (manifests.length === 0) {
    throw new Error('至少选择一个数据集');
  }
  const base = manifests[0] as DatasetManifest & Record<string, unknown>;
  const baseSig = extractDatasetStructureSignature(base);
  for (const manifest of manifests.slice(1)) {
    const sig = extractDatasetStructureSignature(manifest as DatasetManifest & Record<string, unknown>);
    if (!signatureEqual(baseSig, sig)) {
      throw new Error('数据集 observation 结构不一致，无法合并训练');
    }
  }

  const datasetIds = manifests.map((item) => item.datasetId).filter(Boolean);
  const datasetNames = manifests.map((item) => item.datasetName).filter(Boolean);
  const sampleCount = manifests.reduce(
    (sum, item) => sum + Number(item.successfulEpisodes || item.episodes || 0),
    0
  );
  const hdf5Paths = manifests
    .map((item) => item.artifacts?.hdf5)
    .filter((path): path is string => Boolean(path?.trim()));

  return {
    ...base,
    datasetId: datasetIds.length === 1 ? datasetIds[0] : datasetIds.join('+'),
    datasetName: datasetNames.length === 1 ? datasetNames[0] : `${datasetIds.length} 个数据集合并`,
    successfulEpisodes: sampleCount,
    episodes: sampleCount,
    artifacts: {
      ...base.artifacts,
      hdf5: hdf5Paths[0] ?? base.artifacts?.hdf5,
      hdf5Paths,
    },
    ...( {
      datasetIds,
      datasetNames,
      mergedDatasetCount: manifests.length,
      structureSignature: baseSig,
    } as Record<string, unknown> ),
  } as DatasetManifest;
}
