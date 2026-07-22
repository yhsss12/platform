import { getTaskDisplayName } from '@/lib/workspace/taskDisplayNames';
import {
  normalizeEvaluationTypeLabel,
  type EvaluationTypeLabel,
} from '@/lib/workspace/evaluationType';

export type { EvaluationTypeLabel };

export function resolveEvaluationTypeLabel(
  evaluationMode: string | null | undefined,
  extra?: {
    evaluationType?: string | null;
    evaluationTypeLabel?: string | null;
    evaluationObject?: string | null;
    modelAssetId?: string | null;
    datasetId?: string | null;
    taskType?: string | null;
    taskName?: string | null;
  }
): EvaluationTypeLabel {
  return normalizeEvaluationTypeLabel({
    evaluationMode,
    evaluationType: extra?.evaluationType,
    evaluationTypeLabel: extra?.evaluationTypeLabel,
    evaluationObject: extra?.evaluationObject,
    modelAssetId: extra?.modelAssetId,
    datasetId: extra?.datasetId,
    taskType: extra?.taskType,
    taskName: extra?.taskName,
  });
}

function pad2(n: number): string {
  return String(n).padStart(2, '0');
}

export function formatEvaluationNameTimestamp(date: Date): { ymd: string; hm: string } {
  const ymd = `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}`;
  const hm = `${pad2(date.getHours())}${pad2(date.getMinutes())}`;
  return { ymd, hm };
}

export function parseEvalIdTimestamp(evalJobId: string): { ymd: string; hm: string } | null {
  const ct = evalJobId.match(/ct_eval_(\d{8})_(\d{2})(\d{2})\d{2}/);
  if (ct) return { ymd: ct[1], hm: `${ct[2]}${ct[3]}` };
  const isaac = evalJobId.match(/isaac_eval_(\d{8})_(\d{2})(\d{2})\d{2}/);
  if (isaac) return { ymd: isaac[1], hm: `${isaac[2]}${isaac[3]}` };
  const eval_ = evalJobId.match(/eval_(\d{8})_(\d{2})(\d{2})\d{2}/);
  if (eval_) return { ymd: eval_[1], hm: `${eval_[2]}${eval_[3]}` };
  return null;
}

export function isCanonicalEvaluationDisplayName(name: string): boolean {
  const v = name.trim();
  return /^[^\s_]+(模型评测|专家策略评测|数据集评测)_\d{8}_\d{4}(_\d{2})?$/.test(v);
}

export function buildCanonicalEvaluationDisplayName(params: {
  taskType: string | null | undefined;
  evaluationMode: string | null | undefined;
  evaluationTypeLabel?: EvaluationTypeLabel | null;
  createdAt?: Date | null;
  evalJobId?: string | null;
  seq?: number | null;
}): string {
  const taskDisplay = getTaskDisplayName(params.taskType ?? undefined);
  const typeLabel =
    params.evaluationTypeLabel ??
    resolveEvaluationTypeLabel(params.evaluationMode, { taskType: params.taskType ?? null });

  const ts =
    params.createdAt != null
      ? formatEvaluationNameTimestamp(params.createdAt)
      : params.evalJobId
        ? parseEvalIdTimestamp(params.evalJobId) ?? formatEvaluationNameTimestamp(new Date())
        : formatEvaluationNameTimestamp(new Date());

  const suffix =
    params.seq != null && params.seq > 1 ? `_${String(params.seq).padStart(2, '0')}` : '';

  return `${taskDisplay}${typeLabel}_${ts.ymd}_${ts.hm}${suffix}`;
}

export function normalizeEvaluationDisplayName(params: {
  displayName?: string | null;
  taskType?: string | null;
  evaluationMode?: string | null;
  evaluationTypeLabel?: EvaluationTypeLabel | null;
  createdAtIso?: string | null;
  evalJobId?: string | null;
}): string {
  const displayName = String(params.displayName ?? '').trim();
  if (displayName && isCanonicalEvaluationDisplayName(displayName)) return displayName;

  const createdAt = (() => {
    const raw = String(params.createdAtIso ?? '').trim();
    if (!raw) return null;
    const d = new Date(raw);
    return Number.isFinite(d.getTime()) ? d : null;
  })();

  const derived = buildCanonicalEvaluationDisplayName({
    taskType: params.taskType,
    evaluationMode: params.evaluationMode,
    evaluationTypeLabel: params.evaluationTypeLabel,
    createdAt,
    evalJobId: params.evalJobId,
  });
  if (derived && derived !== '—') return derived;

  const shortId = String(params.evalJobId ?? '').slice(0, 10);
  return shortId ? `评测任务_${shortId}` : '评测任务';
}
