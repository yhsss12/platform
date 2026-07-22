/** Isaac Lab 物块堆叠 generationMode 用户可见文案（内部调试可保留原始值）。 */

export type IsaacGenerationMode = 'expert_policy' | 'scripted_expert' | 'mimic_auto' | 'teleop_record' | string;

export const ISAAC_GENERATION_MODE_LABELS: Record<string, string> = {
  expert_policy: '专家策略生成',
  scripted_expert: '专家策略生成',
  mimic_auto: 'Mimic 示范扩增',
  teleop_record: '人工遥操作采集',
};

/** Canonical user-facing label; maps legacy scripted_expert to expert_policy display. */
export function formatIsaacGenerationMode(mode: string | null | undefined): string | undefined {
  if (!mode) return undefined;
  if (mode === 'scripted_expert') return ISAAC_GENERATION_MODE_LABELS.expert_policy;
  return ISAAC_GENERATION_MODE_LABELS[mode] ?? mode;
}

export function isExpertPolicyGenerationMode(mode: string | null | undefined): boolean {
  return mode === 'expert_policy' || mode === 'scripted_expert';
}
