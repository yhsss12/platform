import { isCableThreadingTask } from '@/lib/workspace/cableThreading';
import { isDualArmCableTask } from '@/lib/workspace/dualArmCable';
import { isNutAssemblyTask } from '@/lib/workspace/nutAssembly';
import type { AugmentationAlgorithm, GenerationPath } from '@/lib/workspace/generateDataTypes';
import { NUT_ASSEMBLY_PATH_DEFAULTS } from '@/lib/workspace/generateDataTaskParams';
import {
  ISAAC_BLOCK_STACKING_DEFAULT_ENV,
  ISAAC_BLOCK_STACKING_TEMPLATE_ID,
  isIsaacBlockStackingTask,
} from '@/lib/workspace/isaacBlockStacking';
import {
  FRANKA_STACK_CUBE_PRODUCT_NAME,
  isFrankStackCubeProductTask,
  resolveFrankStackCubeEvaluationUiBindingTemplateId,
} from '@/lib/workspace/isaacStackCubeProduct';
import {
  isIsaacLabFrankaStackCubeTask,
  ISAACLAB_FRANKA_STACK_CUBE_DEFAULT_ENV,
  ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID,
} from '@/lib/workspace/isaaclabFrankaStackCube';
import {
  CABLE_THREADING_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  NUT_ASSEMBLY_DISPLAY_NAME,
} from '@/lib/workspace/taskDisplayNames';

export type SimulatorBackendId = 'mujoco' | 'isaac_lab';
export type PhysicsBackendId = 'mujoco' | 'physx';
export type SupportsDatasetGeneration = boolean | 'planned';

export interface TaskTemplateCapabilityProfile {
  templateId: string;
  simulatorBackend: SimulatorBackendId;
  physicsBackend?: PhysicsBackendId;
  supportsDatasetGeneration: SupportsDatasetGeneration;
  requiresExternalRuntime: boolean;
  replayAvailable: boolean;
  supportsImportedDemoReplay: boolean;
  robotLabel: string;
  defaultEnv?: string;
  datasetFormat?: string;
  /** 数据生成路径 capability（螺母装配等） */
  supportedGenerationPaths?: GenerationPath[];
  defaultGenerationPath?: GenerationPath;
  supportedAugmentationAlgorithms?: AugmentationAlgorithm[];
  supportsPinnRepair?: boolean;
  supportsReplayValidation?: boolean;
  supportsRecordVideo?: boolean;
  outputFormat?: string;
  generationPathDisabledReasons?: Partial<Record<GenerationPath, string>>;
}

const CABLE_THREADING_CAPABILITIES: TaskTemplateCapabilityProfile = {
  templateId: 'cable_threading_single_arm',
  simulatorBackend: 'mujoco',
  physicsBackend: 'mujoco',
  supportsDatasetGeneration: true,
  requiresExternalRuntime: false,
  replayAvailable: true,
  supportsImportedDemoReplay: false,
  robotLabel: 'Panda / UR5e',
};

const DUAL_ARM_CABLE_CAPABILITIES: TaskTemplateCapabilityProfile = {
  templateId: 'dual_arm_cable_manipulation',
  simulatorBackend: 'mujoco',
  physicsBackend: 'mujoco',
  supportsDatasetGeneration: true,
  requiresExternalRuntime: false,
  replayAvailable: true,
  supportsImportedDemoReplay: false,
  robotLabel: 'Dual FR3',
};

const ISAAC_BLOCK_STACKING_CAPABILITIES: TaskTemplateCapabilityProfile = {
  templateId: ISAAC_BLOCK_STACKING_TEMPLATE_ID,
  simulatorBackend: 'isaac_lab',
  physicsBackend: 'physx',
  supportsDatasetGeneration: 'planned',
  requiresExternalRuntime: true,
  replayAvailable: true,
  supportsImportedDemoReplay: true,
  robotLabel: 'Franka Panda',
  defaultEnv: ISAAC_BLOCK_STACKING_DEFAULT_ENV,
  datasetFormat: 'HDF5',
};

const NUT_ASSEMBLY_CAPABILITIES: TaskTemplateCapabilityProfile = {
  templateId: 'nut_assembly_single_arm',
  simulatorBackend: 'mujoco',
  physicsBackend: 'mujoco',
  supportsDatasetGeneration: true,
  requiresExternalRuntime: false,
  replayAvailable: true,
  supportsImportedDemoReplay: true,
  robotLabel: 'Panda 单臂机械臂',
  defaultEnv: 'NutAssembly_D0',
  datasetFormat: 'robomimic_hdf5',
  outputFormat: 'HDF5',
  supportedGenerationPaths: [
    'expert_policy',
    'demo_augmentation',
    'expert_seed_then_augmentation',
  ],
  defaultGenerationPath: NUT_ASSEMBLY_PATH_DEFAULTS.generationPath,
  supportedAugmentationAlgorithms: ['mimicgen'],
  supportsPinnRepair: true,
  supportsReplayValidation: true,
  supportsRecordVideo: true,
};

const ISAACLAB_FRANKA_STACK_CUBE_CAPABILITIES: TaskTemplateCapabilityProfile = {
  templateId: ISAACLAB_FRANKA_STACK_CUBE_TEMPLATE_ID,
  simulatorBackend: 'isaac_lab',
  physicsBackend: 'physx',
  supportsDatasetGeneration: true,
  requiresExternalRuntime: true,
  replayAvailable: true,
  supportsImportedDemoReplay: false,
  robotLabel: 'Franka Panda',
  defaultEnv: ISAACLAB_FRANKA_STACK_CUBE_DEFAULT_ENV,
  datasetFormat: 'HDF5',
};

export function resolveTaskTemplateCapabilities(
  templateLabel: string
): TaskTemplateCapabilityProfile | null {
  if (isIsaacLabFrankaStackCubeTask(templateLabel)) {
    return ISAACLAB_FRANKA_STACK_CUBE_CAPABILITIES;
  }
  if (isIsaacBlockStackingTask(templateLabel)) {
    return ISAAC_BLOCK_STACKING_CAPABILITIES;
  }
  if (isCableThreadingTask(templateLabel)) {
    return CABLE_THREADING_CAPABILITIES;
  }
  if (isDualArmCableTask(templateLabel)) {
    return DUAL_ARM_CABLE_CAPABILITIES;
  }
  if (isNutAssemblyTask(templateLabel)) {
    return NUT_ASSEMBLY_CAPABILITIES;
  }
  return null;
}

export function resolveTaskTemplateCapabilitiesById(
  templateId: string | null | undefined
): TaskTemplateCapabilityProfile | null {
  if (!templateId?.trim()) return null;
  return resolveTaskTemplateCapabilities(templateId.trim());
}

export function isDatasetGenerationEnabled(templateLabel: string): boolean {
  const profile = resolveTaskTemplateCapabilities(templateLabel);
  if (!profile) return false;
  return profile.supportsDatasetGeneration === true;
}

/** 前端已接入真实数据生成后端的任务展示名 */
export const STATIC_GENERATE_DATA_TEMPLATE_LABELS = [
  CABLE_THREADING_DISPLAY_NAME,
  NUT_ASSEMBLY_DISPLAY_NAME,
  DUAL_ARM_CABLE_DISPLAY_NAME,
  FRANKA_STACK_CUBE_PRODUCT_NAME,
] as const;

export function getStaticGenerateDataTemplateOptions(): string[] {
  return STATIC_GENERATE_DATA_TEMPLATE_LABELS.filter(isDatasetGenerationEnabled);
}

export function formatGenerateDataTemplateOptionLabel(templateLabel: string): string {
  const profile = resolveTaskTemplateCapabilities(templateLabel);
  if (profile?.supportsDatasetGeneration === 'planned') {
    return `${templateLabel}（数据生成待接入）`;
  }
  return templateLabel;
}

export function getGenerateDataTemplateOptions(includePlannedTasks = false): string[] {
  const enabled = getStaticGenerateDataTemplateOptions();
  if (!includePlannedTasks) return enabled;
  const plannedLabel = FRANKA_STACK_CUBE_PRODUCT_NAME;
  if (enabled.includes(plannedLabel)) return enabled;
  const profile = resolveTaskTemplateCapabilities(plannedLabel);
  if (profile?.supportsDatasetGeneration === 'planned') {
    return [...enabled, plannedLabel];
  }
  return enabled;
}

export function formatSimulatorBackendLabel(backend: SimulatorBackendId): string {
  return backend === 'isaac_lab' ? 'Isaac Lab' : 'MuJoCo';
}

export function normalizeSimulatorBackendId(raw: string | null | undefined): SimulatorBackendId {
  if (!raw?.trim()) return 'mujoco';
  const value = raw.trim().toLowerCase();
  if (value === 'isaac_lab' || value === 'isaac' || value === 'isaacsim' || value.includes('isaac')) {
    return 'isaac_lab';
  }
  return 'mujoco';
}

export function formatPhysicsBackendLabel(backend: PhysicsBackendId): string {
  return backend === 'physx' ? 'PhysX' : 'MuJoCo';
}

export interface EvaluationUiBinding {
  evalBackendLabel: string;
  simulatorBackend: SimulatorBackendId;
  defaultTaskEnv?: string;
  showCableModel?: boolean;
  showCableDifficulty?: boolean;
  showMuJoCoRobotSelect?: boolean;
  showIsaacTaskEnv?: boolean;
  robotLabel?: string;
  robotReadOnly?: boolean;
}

const EVALUATION_UI_BINDINGS: Record<string, EvaluationUiBinding> = {
  cable_threading_single_arm: {
    evalBackendLabel: 'MuJoCo',
    simulatorBackend: 'mujoco',
    showCableModel: true,
    showCableDifficulty: true,
    showMuJoCoRobotSelect: true,
    robotLabel: 'Panda / UR5e',
  },
  dual_arm_cable_manipulation: {
    evalBackendLabel: 'MuJoCo',
    simulatorBackend: 'mujoco',
    showCableModel: false,
    showCableDifficulty: false,
    showMuJoCoRobotSelect: false,
    robotLabel: 'Dual FR3',
    robotReadOnly: true,
  },
  nut_assembly_single_arm: {
    evalBackendLabel: 'MuJoCo',
    simulatorBackend: 'mujoco',
    showCableModel: false,
    showCableDifficulty: false,
    showMuJoCoRobotSelect: true,
    robotLabel: 'Panda 单臂机械臂',
  },
  [ISAAC_BLOCK_STACKING_TEMPLATE_ID]: {
    evalBackendLabel: 'Isaac Lab',
    simulatorBackend: 'isaac_lab',
    showCableModel: false,
    showCableDifficulty: false,
    showMuJoCoRobotSelect: false,
    showIsaacTaskEnv: true,
    robotLabel: 'Franka Panda',
    robotReadOnly: true,
    defaultTaskEnv: ISAAC_BLOCK_STACKING_DEFAULT_ENV,
  },
};

export function resolveEvaluationUiBinding(
  templateId: string | null | undefined
): EvaluationUiBinding | null {
  if (!templateId) return null;
  const resolvedId = resolveFrankStackCubeEvaluationUiBindingTemplateId(templateId);
  if (EVALUATION_UI_BINDINGS[resolvedId]) return EVALUATION_UI_BINDINGS[resolvedId];
  if (EVALUATION_UI_BINDINGS[templateId]) return EVALUATION_UI_BINDINGS[templateId];
  if (isIsaacBlockStackingTask(templateId) || isFrankStackCubeProductTask(templateId)) {
    return EVALUATION_UI_BINDINGS[ISAAC_BLOCK_STACKING_TEMPLATE_ID];
  }
  if (isNutAssemblyTask(templateId)) {
    return EVALUATION_UI_BINDINGS.nut_assembly_single_arm;
  }
  if (isCableThreadingTask(templateId)) {
    return EVALUATION_UI_BINDINGS.cable_threading_single_arm;
  }
  if (isDualArmCableTask(templateId)) {
    return EVALUATION_UI_BINDINGS.dual_arm_cable_manipulation;
  }
  return null;
}

export function resolveEvalBackendLabel(templateId: string | null | undefined): string | null {
  return resolveEvaluationUiBinding(templateId)?.evalBackendLabel ?? null;
}

export function resolveSupportedGenerationPaths(
  templateLabel: string
): GenerationPath[] {
  const profile = resolveTaskTemplateCapabilities(templateLabel);
  return profile?.supportedGenerationPaths ?? [];
}

export function resolveDefaultGenerationPath(templateLabel: string): GenerationPath | null {
  const profile = resolveTaskTemplateCapabilities(templateLabel);
  return profile?.defaultGenerationPath ?? profile?.supportedGenerationPaths?.[0] ?? null;
}

export function resolveGenerationPathDisabledReason(
  templateLabel: string,
  path: GenerationPath
): string | null {
  const profile = resolveTaskTemplateCapabilities(templateLabel);
  return profile?.generationPathDisabledReasons?.[path] ?? null;
}

export function isGenerationPathEnabled(templateLabel: string, path: GenerationPath): boolean {
  const profile = resolveTaskTemplateCapabilities(templateLabel);
  if (!profile?.supportedGenerationPaths?.includes(path)) return false;
  return !profile.generationPathDisabledReasons?.[path];
}
