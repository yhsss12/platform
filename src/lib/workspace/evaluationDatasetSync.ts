import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import { resolveDatasetSourceTaskLabel } from '@/lib/workspace/taskTemplateMapping';
import type { Dataset } from '@/types/benchmark';

function isDatasetEvaluationRow(row: EvaluationTaskRow): boolean {
  return (
    row.evaluationMode === '数据过程评测' ||
    Boolean(row.datasetId) ||
    Boolean(row.dataName?.trim()) ||
    /离线数据集评测/i.test(row.name)
  );
}

function findDatasetForEvaluationRow(row: EvaluationTaskRow, datasets: Dataset[]): Dataset | null {
  const datasetId = row.datasetId?.trim();
  if (datasetId) {
    const byId = datasets.find((d) => d.id === datasetId);
    if (byId) return byId;
  }

  const dataName = row.dataName?.trim() || row.name.replace(/^离线数据集评测\s*[·•]\s*/i, '').trim();
  if (dataName) {
    const exact = datasets.find((d) => d.name === dataName);
    if (exact) return exact;

    const normalized = dataName.toLowerCase();
    const byName = datasets.find((d) => {
      const name = d.name.toLowerCase();
      return name === normalized || normalized.includes(name) || name.includes(normalized);
    });
    if (byName) return byName;

    const jobToken = dataName.match(/(ct_gen_[\w]+|dac_gen_[\w]+|isaac_(?:gen|import)_[\w]+)/i)?.[1];
    if (jobToken) {
      const byJob = datasets.find((d) => d.sourceJobId.includes(jobToken));
      if (byJob) return byJob;
    }
  }

  return null;
}

/** 将数据中心数据集的「关联任务」同步到离线数据集评测行的 relatedTask */
export function enrichEvaluationTasksWithDatasetRelatedTask(
  rows: EvaluationTaskRow[],
  datasets: Dataset[]
): EvaluationTaskRow[] {
  if (datasets.length === 0) return rows;

  return rows.map((row) => {
    if (!isDatasetEvaluationRow(row)) return row;

    const dataset = findDatasetForEvaluationRow(row, datasets);
    if (!dataset) return row;

    const relatedTask = resolveDatasetSourceTaskLabel(dataset);
    if (!relatedTask || relatedTask === '—') return row;

    return { ...row, relatedTask };
  });
}
