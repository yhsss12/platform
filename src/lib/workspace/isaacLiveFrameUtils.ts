'use client';

/** 抽样检测 JPEG blob 是否几乎全黑（Isaac warmup 空缓冲）。 */
export async function isImageBlobMostlyBlack(blob: Blob, minMean = 2): Promise<boolean> {
  try {
    const bitmap = await createImageBitmap(blob);
    const sampleW = Math.min(bitmap.width, 64);
    const sampleH = Math.min(bitmap.height, 64);
    const canvas = document.createElement('canvas');
    canvas.width = sampleW;
    canvas.height = sampleH;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      bitmap.close();
      return false;
    }
    ctx.drawImage(bitmap, 0, 0, sampleW, sampleH);
    bitmap.close();
    const { data } = ctx.getImageData(0, 0, sampleW, sampleH);
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) {
      sum += data[i] + data[i + 1] + data[i + 2];
    }
    const mean = sum / ((data.length / 4) * 3);
    return mean < minMean;
  } catch {
    return false;
  }
}
