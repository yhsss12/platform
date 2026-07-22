export function validateHiddenDimsInput(value: string | undefined): string | null {
  const text = (value ?? '').trim();
  if (!text) return null;
  if (!/^\d+(,\d+)*$/.test(text)) {
    return '请输入逗号分隔的正整数，例如 512,512';
  }
  return null;
}

export function validateNonNegativeNumberInput(value: string | undefined): string | null {
  const text = (value ?? '').trim();
  if (!text) return null;
  const num = Number(text);
  if (!Number.isFinite(num) || num < 0) {
    return '请输入非负数字';
  }
  return null;
}
