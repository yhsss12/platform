import type { Dataset } from '@/types/benchmark';
import {
  isImportedWorkspaceDataset,
  isImportedDatasetDirectTrainable,
  normalizeImportedDatasetStatus,
} from '@/lib/workspace/datasetImportWorkflow';
import { isFrankStackCubeDataset } from '@/lib/workspace/isaacStackCubeProduct';
import { isIsaacLabFrankaStackCubeTask } from '@/lib/workspace/isaaclabFrankaStackCube';

export const IMPORT_DATASET_MAPPING_HINT = '数据集需完成字段映射后才可用于训练';
export const IMPORT_DATASET_BUILD_HINT = '数据集需通过数据构建完成 episode 切分或标准化后才可用于训练';
export const IMPORT_DATASET_FAILED_HINT = '数据集导入失败，请重新上传';
export const IMPORT_DATASET_NOT_TRAINABLE_HINT = '数据集尚未通过导入校验，暂不可用于训练';

export const DUAL_ARM_MANIFEST_TRAIN_DISABLED_HINT =
  '该数据集为过程记录格式，尚未构建为 HDF5 训练数据集。';

export const DUAL_ARM_LEGACY_MANIFEST_HINT =
  '旧版过程记录缺少 step-level actions，无法构建训练数据集。';

export const DUAL_ARM_IL_BUILD_DISABLED_HINT =
  '当前过程数据缺少 step-level actions，无法构建训练数据集。';

export const DUAL_ARM_TRAINING_BACKEND_PENDING_HINT = '该训练后端暂未开放';

export const ISAAC_BLOCK_STACKING_TRAINING_PENDING_HINT =
  'Isaac Robomimic BC 训练未就绪（需配置 Isaac Lab 运行节点）';

const DUAL_ARM_TEMPLATE_IDS = new Set([
  'dual_arm_cable_manipulation',
  'task_dual_arm_cable_manipulation_v1',
]);

export function isIsaacBlockStackingDataset(dataset: Dataset): boolean {
  if (isFrankStackCubeDataset(dataset)) return true;
  if (isIsaacLabFrankaStackCubeTask(dataset.taskType) || isIsaacLabFrankaStackCubeTask(dataset.taskTemplateId)) {
    return true;
  }
  if (dataset.simulatorBackend === 'isaac_lab') return true;
  if (dataset.replayBackend === 'isaac_lab') return true;
  if (dataset.taskTemplateId === 'isaac_block_stacking') return true;
  if (dataset.sourceJobId.startsWith('isaac_gen_') || dataset.sourceJobId.startsWith('isaac_import_')) {
    return true;
  }
  return false;
}

export function isDualArmCableDataset(dataset: Dataset): boolean {
  if (dataset.sourceJobId.startsWith('dac_gen_')) return true;
  if (dataset.taskTemplateId && DUAL_ARM_TEMPLATE_IDS.has(dataset.taskTemplateId)) return true;
  if (dataset.sourceTaskTemplateId === 'task_dual_arm_cable_manipulation_v1') return true;
  return false;
}

export function isLerobotSidecarDataset(
  dataset: Pick<Dataset, 'lerobotPath'> & { datasetFormat?: string | null }
): boolean {
  return dataset.datasetFormat === 'lerobot' || Boolean(dataset.lerobotPath);
}

export function datasetHasBuiltIlArtifacts(dataset: Dataset): boolean {
  if (isLerobotSidecarDataset(dataset)) {
    return Boolean(dataset.lerobotPath || dataset.builtDatasetPath || dataset.datasetFile);
  }
  if (dataset.format === 'hdf5' || dataset.format === 'npz') {
    return Boolean(dataset.builtDatasetPath || dataset.datasetFile);
  }
  return dataset.trainable === true && Boolean(dataset.builtDatasetPath || dataset.datasetFile);
}

export function canOpenDatasetTraining(dataset: Dataset): boolean {
  if (isLerobotSidecarDataset(dataset) && dataset.pi0Ready === true) {
    return datasetHasBuiltIlArtifacts(dataset) && (dataset.episodeCount ?? 0) > 0;
  }
  if (isImportedWorkspaceDataset(dataset)) {
    return (
      isImportedDatasetDirectTrainable(dataset) &&
      datasetHasBuiltIlArtifacts(dataset) &&
      (dataset.episodeCount ?? 0) > 0
    );
  }
  if (isIsaacBlockStackingDataset(dataset)) {
    const hasHdf5 = dataset.format === 'hdf5' || dataset.datasetFormat === 'hdf5';
    return hasHdf5 && Boolean(dataset.datasetFile) && (dataset.episodeCount ?? 0) > 0;
  }
  if (dataset.trainable === true && datasetHasBuiltIlArtifacts(dataset)) {
    return true;
  }
  if (dataset.sourceJobId.startsWith('ct_gen_') && (dataset.format === 'hdf5' || dataset.format === 'npz')) {
    return true;
  }
  if (!isDualArmCableDataset(dataset) && (dataset.format === 'hdf5' || dataset.format === 'npz')) {
    return true;
  }
  return false;
}

export const DUAL_ARM_HDF5_MISSING_HINT = '缺少可训练 HDF5 数据集';

export function datasetTrainingDisabledHint(dataset: Dataset): string | null {
  if (canOpenDatasetTraining(dataset)) {
    return null;
  }
  if (isImportedWorkspaceDataset(dataset)) {
    const status = normalizeImportedDatasetStatus(dataset.status);
    if (status === 'failed') {
      return IMPORT_DATASET_FAILED_HINT;
    }
    if (status === 'needs_mapping' || dataset.needsMapping) {
      return IMPORT_DATASET_MAPPING_HINT;
    }
    if (status === 'needs_build' || dataset.needsBuild) {
      return IMPORT_DATASET_BUILD_HINT;
    }
    if (status === 'parsing') {
      return IMPORT_DATASET_NOT_TRAINABLE_HINT;
    }
    return IMPORT_DATASET_NOT_TRAINABLE_HINT;
  }
  if (dataset.ilExportFailureReason) {
    return dataset.ilExportFailureReason;
  }
  if (isIsaacBlockStackingDataset(dataset)) {
    return '物块堆叠数据集缺少 HDF5 文件或 demo，无法训练';
  }
  if (isDualArmCableDataset(dataset)) {
    if (dataset.format === 'manifest' || dataset.datasetFormat === 'manifest') {
      const reason = dataset.ilExportFailureReason ?? '';
      if (reason.includes('step-level actions') || reason.includes('旧版')) {
        return DUAL_ARM_LEGACY_MANIFEST_HINT;
      }
      return DUAL_ARM_MANIFEST_TRAIN_DISABLED_HINT;
    }
    return DUAL_ARM_HDF5_MISSING_HINT;
  }
  return null;
}

export function shouldShowDatasetTrainingLink(dataset: Dataset): boolean {
  if (isImportedWorkspaceDataset(dataset)) {
    return canOpenDatasetTraining(dataset);
  }
  if (isIsaacBlockStackingDataset(dataset)) {
    return canOpenDatasetTraining(dataset);
  }
  if (dataset.sourceType === 'imported_demo' && !isIsaacBlockStackingDataset(dataset)) return false;
  if (
    !isIsaacBlockStackingDataset(dataset) &&
    (dataset.sourceJobId.startsWith('isaac_import_') || dataset.sourceJobId.startsWith('isaac_gen_'))
  ) {
    return false;
  }
  return canOpenDatasetTraining(dataset);
}

export function shouldShowDatasetTrainingDisabled(dataset: Dataset): boolean {
  if (isImportedWorkspaceDataset(dataset)) {
    return !canOpenDatasetTraining(dataset);
  }
  if (isIsaacBlockStackingDataset(dataset)) {
    return !canOpenDatasetTraining(dataset);
  }
  return Boolean(datasetTrainingDisabledHint(dataset)) && !canOpenDatasetTraining(dataset);
}

export function datasetTrainingDisabledHintForDisplay(dataset: Dataset): string | null {
  if (isIsaacBlockStackingDataset(dataset) && !canOpenDatasetTraining(dataset)) {
    return datasetTrainingDisabledHint(dataset);
  }
  return datasetTrainingDisabledHint(dataset);
}
