import type { Dataset } from '@/types/benchmark';
import type { DatasetManifest } from '@/lib/workspace/datasetManifest';
import { getDatasetManifest } from '@/lib/mock/workspaceMockFlowStore';
import { listWorkspaceDatasets } from '@/lib/api/datasetsClient';
import { canOpenDatasetTraining, isIsaacBlockStackingDataset, isLerobotSidecarDataset } from '@/lib/workspace/datasetTrainingAccess';
import { resolveDatasetSourceTaskLabel } from '@/lib/workspace/taskTemplateMapping';
import {
  CABLE_THREADING_DEFAULTS,
  CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS,
} from '@/lib/workspace/cableThreading';

export function isDualArmApiDataset(dataset: Dataset): boolean {
  if (dataset.sourceJobId.startsWith('dac_gen_')) return true;
  if (dataset.taskTemplateId === 'dual_arm_cable_manipulation') return true;
  if (dataset.sourceTaskTemplateId === 'task_dual_arm_cable_manipulation_v1') return true;
  return false;
}

function resolveDatasetHdf5Path(dataset: Dataset): string {
  const direct = dataset.builtDatasetPath || dataset.datasetFile || '';
  if (direct.trim()) return direct.trim();

  const manifestPath = dataset.manifestPath?.trim();
  if (manifestPath?.endsWith('dataset.manifest.json')) {
    return manifestPath.replace(/dataset\.manifest\.json$/, 'dataset.hdf5');
  }

  const storagePath = dataset.storagePath?.trim().replace(/\/$/, '');
  if (storagePath) {
    if (storagePath.endsWith('datasets')) {
      return `${storagePath}/dataset.hdf5`;
    }
    return `${storagePath}/datasets/dataset.hdf5`;
  }

  return '';
}

export function buildTrainingManifestFromApiDataset(dataset: Dataset): DatasetManifest | null {
  if (!canOpenDatasetTraining(dataset)) return null;
  const hdf5 = resolveDatasetHdf5Path(dataset);
  const isLerobotPi0 = isLerobotSidecarDataset(dataset) && dataset.pi0Ready === true;

  const dualArm = isDualArmApiDataset(dataset);
  const isaac = isIsaacBlockStackingDataset(dataset);
  const nutAssembly =
    dataset.sourceJobId.startsWith('na_gen_') ||
    dataset.taskType === 'nut_assembly' ||
    dataset.taskTemplateId === 'nut_assembly_single_arm' ||
    dataset.taskTemplateId === 'task_nut_assembly_v1';
  const now = new Date().toISOString();

  return {
    datasetId: dataset.id,
    datasetName: dataset.name,
    taskType: isaac
      ? 'isaac_block_stacking'
      : dualArm
        ? 'dual_arm_cable_manipulation'
        : nutAssembly
          ? 'nut_assembly'
          : 'cable_threading',
    taskName: resolveDatasetSourceTaskLabel(dataset),
    sourceJobId: dataset.sourceJobId,
    sourceRecordName: dataset.name,
    backend: dataset.simulatorBackend ?? 'mujoco',
    episodes: dataset.episodeCount,
    successfulEpisodes: dataset.episodeCount,
    successRate: dataset.episodeCount > 0 ? 100 : 0,
    usage: 'training',
    downstreamModelType: isaac || dualArm ? '自定义模型' : 'Robomimic',
    dataFormat: isLerobotPi0 ? 'LeRobot' : 'HDF5',
    trainingView: dualArm ? 'low_dim' : 'default',
    mainFormats: isLerobotPi0 ? ['lerobot'] : ['HDF5'],
    split: { enabled: false, trainRatio: 1, valRatio: 0 },
    artifacts: {
      ...(isLerobotPi0 && dataset.lerobotPath
        ? { lerobot: dataset.lerobotPath, lerobotPath: dataset.lerobotPath }
        : {}),
      ...(hdf5 ? { hdf5 } : {}),
      manifest: dataset.manifestPath || undefined,
    },
    quality: {
      status: 'ready',
      hasTrajectory: true,
      hasImage: isLerobotPi0,
      hasVideo: false,
      hasTimeline: false,
      hasSuccessfulEpisodes: dataset.episodeCount > 0,
    },
    createdAt: dataset.createdAt || now,
    ...(isLerobotPi0
      ? {
          primaryFormat: 'lerobot',
          availableFormats: dataset.availableFormats ?? ['lerobot'],
          format: 'lerobot',
          datasetFormat: 'lerobot',
          pi0Ready: true,
          lerobot: {
            status: 'ready',
            path: dataset.lerobotPath,
            taskInstruction: dataset.lerobotTaskInstruction,
            robot: dataset.robotType ?? 'Panda',
            stateDim: dataset.lerobotStateDim,
            actionDim: dataset.lerobotActionDim,
            pi0Ready: true,
          },
          taskDescription: dataset.lerobotTaskInstruction,
          actionDim: dataset.lerobotActionDim ?? 8,
          state_dim: dataset.lerobotStateDim ?? 9,
        }
      : {}),
    ...(isaac
      ? {
          taskTemplateId: 'isaac_block_stacking',
          simulatorBackend: 'isaac_lab',
          datasetFile: dataset.datasetFile,
          taskEnv: 'Isaac-Stack-Cube-Franka-IK-Rel-v0',
          datasetEnv: 'Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0',
          obsKeys: ['eef_pos', 'eef_quat', 'gripper_pos', 'object'],
          actionDim: 7,
          trainable: true,
          trainingBackends: ['isaac_robomimic_bc'],
        }
      : {}),
    ...(dualArm
      ? {
          taskTemplateId: 'dual_arm_cable_manipulation',
          observationSchema: dataset.observationSchema ?? 'dual_arm_cable_il_v1',
          actionSchema: dataset.actionSchema ?? 'dual_arm_bimanual_action_v1',
          actionDim: 14,
          trainable: true,
          trainingBackends: ['torch_bc'],
        }
      : {}),
    ...(!dualArm && !isaac
      ? {
          taskType: nutAssembly ? 'nut_assembly' : 'cable_threading',
          taskTemplateId: dataset.taskTemplateId ?? (nutAssembly ? 'nut_assembly_single_arm' : 'cable_threading_single_arm'),
          robotType: CABLE_THREADING_DEFAULTS.robot,
          simulatorBackend: dataset.simulatorBackend ?? 'mujoco',
          observationSchema: dataset.observationSchema,
          actionSchema: dataset.actionSchema,
          controllerSchema: dataset.controllerSchema,
          trainedActionMode: dataset.trainedActionMode,
          evalExecutor: dataset.evalExecutor,
          preferredPolicySchemaId: dataset.preferredPolicySchemaId,
          cameraKeys: nutAssembly
            ? []
            : dataset.imageKeys ??
              CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => key.includes('image')),
          imageKeys: nutAssembly
            ? []
            : dataset.imageKeys ??
              CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => key.includes('image')),
          observationKeys:
            dataset.lowDimKeys ??
            (nutAssembly
              ? ['SquareNut_pos', 'SquareNut_quat', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_joint_pos']
              : CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS.filter((key) => !key.includes('image'))),
          obsKeys: dataset.lowDimKeys
            ? [
                ...(dataset.imageKeys ?? []),
                ...(dataset.lowDimKeys ?? []),
              ]
            : nutAssembly
              ? ['SquareNut_pos', 'SquareNut_quat', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_joint_pos']
              : [...CABLE_THREADING_TRAINED_MODEL_EVAL_OBS_KEYS],
          actionDim:
            dataset.actionSchema?.includes('joint') || dataset.trainedActionMode === 'joint_delta'
              ? 8
              : dataset.actionDim ?? 7,
          imageSize: 84,
          jointActionAvailable: (dataset as Dataset & { jointActionAvailable?: boolean })
            .jointActionAvailable,
          policySchemas: (dataset as Dataset & { policySchemas?: Record<string, unknown> }).policySchemas,
          availableActionKeys: (dataset as Dataset & { availableActionKeys?: string[] }).availableActionKeys,
        }
      : {}),
  } as DatasetManifest;
}

export async function resolveTrainingDatasetManifest(datasetId: string): Promise<DatasetManifest | null> {
  const fromStore = getDatasetManifest(datasetId);
  if (fromStore) return fromStore;

  try {
    const response = await listWorkspaceDatasets();
    const dataset = response.datasets.find((d) => d.id === datasetId);
    if (dataset) return buildTrainingManifestFromApiDataset(dataset);
  } catch {
    return null;
  }
  return null;
}

export async function resolveTrainingDatasetManifests(
  datasetIds: string[]
): Promise<{ manifests: DatasetManifest[]; missingIds: string[] }> {
  const manifests: DatasetManifest[] = [];
  const missingIds: string[] = [];
  for (const datasetId of datasetIds) {
    const manifest = await resolveTrainingDatasetManifest(datasetId);
    if (manifest) {
      manifests.push(manifest);
    } else {
      missingIds.push(datasetId);
    }
  }
  return { manifests, missingIds };
}

export function isDualArmTrainingDatasetOption(option: {
  taskType?: string;
  sourceJobId?: string;
}): boolean {
  if (option.taskType === 'dual_arm_cable_manipulation') return true;
  return Boolean(option.sourceJobId?.startsWith('dac_gen_'));
}

export function isIsaacTrainingDatasetOption(option: {
  taskType?: string;
  sourceJobId?: string;
  simulatorBackend?: string;
}): boolean {
  if (option.taskType === 'isaac_block_stacking' || option.taskType === 'isaaclab_franka_stack_cube') {
    return true;
  }
  if (option.simulatorBackend === 'isaac_lab') return true;
  return Boolean(
    option.sourceJobId?.startsWith('isaac_gen_') ||
      option.sourceJobId?.startsWith('isaac_import_') ||
      option.sourceJobId?.startsWith('data_gen_')
  );
}

export function isIsaacTrainingBackendPending(
  option: { taskType?: string; sourceJobId?: string; simulatorBackend?: string },
  capabilities: { supportedTrainingBackends?: string[] } | null | undefined
): boolean {
  if (!isIsaacTrainingDatasetOption(option)) return false;
  return !capabilities?.supportedTrainingBackends?.includes('isaac_robomimic_bc');
}

export function isDualArmTrainingBackendPending(
  option: { taskType?: string; sourceJobId?: string },
  capabilities: { supportedTrainingBackends?: string[] } | null | undefined
): boolean {
  if (!isDualArmTrainingDatasetOption(option)) return false;
  return !capabilities?.supportedTrainingBackends?.includes('torch_bc');
}
