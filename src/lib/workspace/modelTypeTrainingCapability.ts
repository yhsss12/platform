import type { ModelTypeDefinition } from '@/types/modelType';

/** 模型类型训练能力短标签（基于 API trainingReady / trainingReadinessStatus）。 */
export function modelTypeTrainingCapabilityLabel(
  item: Pick<ModelTypeDefinition, 'trainingReady' | 'trainingReadinessStatus' | 'disabledReason'>
): string {
  if (item.trainingReadinessStatus === 'pending') return '环境检测中';
  if (item.trainingReady) return '可训练';
  if (item.disabledReason?.trim()) return item.disabledReason.trim();
  return '该训练后端暂未开放';
}

/** 下拉选项展示文案。 */
export function modelTypeSelectOptionLabel(
  item: Pick<ModelTypeDefinition, 'name' | 'trainingReady' | 'trainingReadinessStatus' | 'modelTypeId' | 'baseAlgorithm'>
): string {
  const experimentalPi0 =
    item.modelTypeId === 'pi0' || item.baseAlgorithm === 'pi0' ? 'pi0 / openpi（实验）' : item.name;
  if (item.trainingReadinessStatus === 'pending') return `${experimentalPi0}（环境检测中）`;
  if (item.trainingReady) return experimentalPi0;
  return `${experimentalPi0}（该训练后端暂未开放）`;
}

export function modelTypeHasPendingReadiness(
  items: Pick<ModelTypeDefinition, 'trainingReadinessStatus'>[]
): boolean {
  return items.some((item) => item.trainingReadinessStatus === 'pending');
}
