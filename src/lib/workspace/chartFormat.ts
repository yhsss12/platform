/** 训练 loss 原始数值展示（非百分比、无额外单位） */
export function formatLossValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 10) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

export function formatLossAxisTick(value: number): string {
  return formatLossValue(value);
}

/** 成功率 / 准确率等比例指标 */
export function formatPercentValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(1)}%`;
}
