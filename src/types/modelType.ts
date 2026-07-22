export type ModelTypeStatus = 'available' | 'draft' | 'disabled' | 'deleted';

export type BaseAlgorithm = 'robomimic_bc' | 'act' | 'diffusion_policy';

export interface ModelTypeDefinition {
  modelTypeId: string;
  name: string;
  baseAlgorithm: BaseAlgorithm | string;
  adapterKey: string;
  simulator?: string | null;
  robotType?: string | null;
  tags: string[];
  description?: string | null;
  structureConfig: Record<string, unknown>;
  trainingDefaults: Record<string, unknown>;
  status: ModelTypeStatus | string;
  trainingReady: boolean;
  trainingReadinessStatus?: 'ready' | 'pending' | 'unavailable' | 'unknown' | 'disabled' | string;
  disabledReason?: string | null;
  isBuiltin: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface CreateModelTypeInput {
  name: string;
  modelTypeId?: string;
  baseAlgorithm: BaseAlgorithm | string;
  simulator?: string;
  robotType?: string;
  tags?: string[];
  description?: string;
  structureConfig?: Record<string, unknown>;
  trainingDefaults?: Record<string, unknown>;
  status?: ModelTypeStatus | string;
}

export interface UpdateModelTypeInput {
  name?: string;
  simulator?: string;
  robotType?: string;
  tags?: string[];
  description?: string;
  structureConfig?: Record<string, unknown>;
  trainingDefaults?: Record<string, unknown>;
  status?: ModelTypeStatus | string;
}

export const BASE_ALGORITHM_OPTIONS = [
  { value: 'robomimic_bc', label: 'Robomimic BC' },
  { value: 'act', label: 'ACT' },
  { value: 'diffusion_policy', label: 'Diffusion Policy' },
  { value: 'pi0', label: 'pi0' },
] as const;

export const SIMULATOR_OPTIONS = [
  { value: 'mujoco', label: 'MuJoCo' },
  { value: 'isaac', label: 'Isaac' },
  { value: 'general', label: '通用' },
] as const;

export const ROBOT_TYPE_OPTIONS = [
  { value: 'panda', label: 'Panda' },
  { value: 'dual_arm', label: 'Dual Arm' },
  { value: 'general', label: '通用' },
] as const;

export const ADAPTER_LABELS: Record<string, string> = {
  robomimic_bc_adapter: 'robomimic_bc_adapter',
  act_adapter: 'act_adapter',
  diffusion_policy_adapter: 'diffusion_policy_adapter',
};

export const MODEL_TYPE_STATUS_LABELS: Record<string, string> = {
  available: '可用',
  draft: '草稿',
  disabled: '已禁用',
  deleted: '已删除',
};
