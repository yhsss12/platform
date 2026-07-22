export type GenerationPath =
  | 'expert_policy'
  | 'demo_augmentation'
  | 'expert_seed_then_augmentation';

export type AugmentationAlgorithm = 'mimicgen';

export interface GenerationPathOption {
  value: GenerationPath;
  title: string;
  description: string;
}

export const GENERATION_PATH_OPTIONS: GenerationPathOption[] = [
  {
    value: 'expert_policy',
    title: '专家策略生成',
    description: '使用任务模板内置专家策略直接生成训练数据。',
  },
  {
    value: 'demo_augmentation',
    title: '示范数据扩增',
    description: '选择已有优质示范数据，通过 MimicGen 等方法扩增生成新数据。',
  },
  {
    value: 'expert_seed_then_augmentation',
    title: '专家策略种子 + 示范扩增',
    description: '先自动生成少量优质种子轨迹，再基于种子轨迹扩增生成目标数据。',
  },
];

export const AUGMENTATION_ALGORITHM_OPTIONS: Array<{ value: AugmentationAlgorithm; label: string }> = [
  { value: 'mimicgen', label: 'MimicGen' },
];
