export type NutAssemblyPhysicsEnhancementMethod = 'pinn_repair';

export interface NutAssemblyPhysicsEnhancementConfig {
  enabled: boolean;
  method?: NutAssemblyPhysicsEnhancementMethod;
  modelId?: string;
  repairStages?: string[];
  candidateSource?: string[];
  maxCandidates?: number;
  maxRepairAttemptsPerCandidate?: number;
  xyErrorThreshold?: number;
  heightErrorThreshold?: number;
  validationMode?: 'mujoco_rollout';
  appendRepairedDemos?: boolean;
}

export const NUT_ASSEMBLY_PINN_DEFAULTS = {
  modelId: 'nut_assembly_pinn_v1',
  modelDisplayName: 'NutAssembly 物理增强 v1',
  maxCandidates: 20,
  maxRepairAttemptsPerCandidate: 2,
  xyErrorThreshold: 0.025,
  heightErrorThreshold: 0.02,
  repairStages: ['align_over_peg', 'descend_insert'] as const,
  candidateSource: ['mimicgen_failed_trials', 'high_error_generated_demos'] as const,
} as const;

/** 用户可见模型名称（不暴露 PINN 品牌词） */
export function formatNutAssemblyEnhancementModelDisplayName(name?: string | null): string {
  const raw = (name ?? NUT_ASSEMBLY_PINN_DEFAULTS.modelDisplayName).trim();
  if (!raw) return NUT_ASSEMBLY_PINN_DEFAULTS.modelDisplayName;
  return raw
    .replace(/NutAssembly-PINN/gi, 'NutAssembly 物理增强')
    .replace(/\bPINN\b/g, '物理增强');
}

export function buildNutAssemblyPhysicsEnhancementPayload(
  enabled: boolean
): NutAssemblyPhysicsEnhancementConfig {
  if (!enabled) {
    return { enabled: false };
  }
  return {
    enabled: true,
    method: 'pinn_repair',
    modelId: NUT_ASSEMBLY_PINN_DEFAULTS.modelId,
    repairStages: [...NUT_ASSEMBLY_PINN_DEFAULTS.repairStages],
    candidateSource: [...NUT_ASSEMBLY_PINN_DEFAULTS.candidateSource],
    maxCandidates: NUT_ASSEMBLY_PINN_DEFAULTS.maxCandidates,
    maxRepairAttemptsPerCandidate: NUT_ASSEMBLY_PINN_DEFAULTS.maxRepairAttemptsPerCandidate,
    xyErrorThreshold: NUT_ASSEMBLY_PINN_DEFAULTS.xyErrorThreshold,
    heightErrorThreshold: NUT_ASSEMBLY_PINN_DEFAULTS.heightErrorThreshold,
    validationMode: 'mujoco_rollout',
    appendRepairedDemos: true,
  };
}

export function formatNutAssemblyDataSourceLabel(input: {
  generationMode?: string | null;
  physicsEnhancementEnabled?: boolean | null;
  enhancementMode?: string | null;
}): string {
  const mimicgen =
    input.generationMode === 'mimicgen_datagen' || input.enhancementMode === 'pinn_repair';
  const pinn = Boolean(input.physicsEnhancementEnabled || input.enhancementMode === 'pinn_repair');
  if (mimicgen && pinn) return 'MimicGen + PINN';
  if (mimicgen) return 'MimicGen 生成';
  if (input.generationMode === 'robosuite_rollout') return '规则生成（调试）';
  return 'MuJoCo 生成';
}
