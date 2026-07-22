import type { Dataset } from '@/types/benchmark';

const JOB_ID_TIMESTAMP = /(?:^|_)(?:ct_gen|dac_gen|isaac_import|isaac_gen)_(\d{8})_(\d{6})/;

function parseCreatedAtMs(value: string | null | undefined): number | null {
  let raw = (value ?? '').trim();
  if (!raw) return null;
  if (/^\d{8}T/.test(raw)) {
    raw = `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}${raw.slice(8)}`;
  }
  const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T');
  const ms = Date.parse(normalized);
  return Number.isNaN(ms) ? null : ms;
}

/** 解析数据集创建时间（毫秒）；缺失时尝试从 sourceJobId 推断。 */
export function resolveDatasetCreatedTimeMs(
  dataset: Pick<Dataset, 'createdAt' | 'sourceJobId'>
): number | null {
  const fromField = parseCreatedAtMs(dataset.createdAt);
  if (fromField !== null) return fromField;

  const jobId = dataset.sourceJobId ?? '';
  const match = jobId.match(JOB_ID_TIMESTAMP);
  if (!match) return null;

  const [, ymd, hms] = match;
  const iso = `${ymd}T${hms.slice(0, 2)}:${hms.slice(2, 4)}:${hms.slice(4, 6)}Z`;
  return parseCreatedAtMs(iso);
}

/** 按创建时间倒序；无创建时间的条目排在末尾。 */
export function sortDatasetsByCreatedAtDesc<T extends Pick<Dataset, 'createdAt' | 'sourceJobId'>>(
  datasets: T[]
): T[] {
  return [...datasets].sort((a, b) => {
    const ta = resolveDatasetCreatedTimeMs(a);
    const tb = resolveDatasetCreatedTimeMs(b);
    if (ta === null && tb === null) return 0;
    if (ta === null) return 1;
    if (tb === null) return -1;
    return tb - ta;
  });
}
