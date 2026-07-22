/** 评测任务真实 job id 校验与解析（删除/回放/报告必须使用）。 */

import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';

const EVAL_JOB_ID_PATTERN = /^eval_\d{8}_\d{6}_[a-f0-9]{4}$/i;
const ISAAC_EVAL_JOB_ID_PATTERN = /^isaac_eval_\d{8}_\d{6}_[a-f0-9]{4}$/i;
const CT_EVAL_JOB_ID_PATTERN = /^ct_eval_\d{8}_\d{6}_[a-f0-9]{4}$/i;
/** 手工注册 / smoke 评测任务（如 ct_eval_20260624_dp_smoke） */
const CT_EVAL_LEGACY_JOB_ID_PATTERN = /^ct_eval_[a-z0-9_]+$/i;
/** 手工导入的 joint-space DP 评测任务 */
const IMPORTED_EVAL_JOB_ID_PATTERN = /^eval_joint_dp_[a-z0-9_]+$/i;
const WORKSPACE_DELETE_KEY_PREFIX = 'ws:';

export function isStrictValidEvaluationJobId(value?: string | null): boolean {
  if (!value || typeof value !== 'string') return false;
  const trimmed = value.trim();
  if (!trimmed || trimmed.includes('评测任务') || /[\u4e00-\u9fff]/.test(trimmed)) {
    return false;
  }
  return (
    EVAL_JOB_ID_PATTERN.test(trimmed) ||
    ISAAC_EVAL_JOB_ID_PATTERN.test(trimmed) ||
    CT_EVAL_JOB_ID_PATTERN.test(trimmed) ||
    CT_EVAL_LEGACY_JOB_ID_PATTERN.test(trimmed) ||
    IMPORTED_EVAL_JOB_ID_PATTERN.test(trimmed)
  );
}

/** @deprecated 使用 isStrictValidEvaluationJobId；保留宽松前缀校验供兼容读取 */
export function isValidEvaluationJobId(value?: string | null): boolean {
  return isStrictValidEvaluationJobId(value);
}

export function isImportedJointDpEvalJobId(value?: string | null): boolean {
  if (!value || typeof value !== 'string') return false;
  return IMPORTED_EVAL_JOB_ID_PATTERN.test(value.trim());
}

export function isPendingEvaluationStatus(status?: string | null): boolean {
  const value = String(status ?? '').trim();
  return (
    value === '待评测' ||
    value === 'pending' ||
    value === 'queued' ||
    value === 'draft' ||
    value === 'unknown'
  );
}

export function canRenderEvaluationRow(
  row: Pick<
    EvaluationTaskRow,
    'evalJobId' | 'jobId' | 'id' | 'runtimePath' | 'workspaceJobId' | 'status'
  >
): boolean {
  const evalJobId = getEvaluationRowJobId(row);
  if (isStrictValidEvaluationJobId(evalJobId)) return true;
  if (row.workspaceJobId != null) return true;
  return false;
}

export function resolveEvaluationJobId(source: {
  evalJobId?: string | null;
  eval_job_id?: string | null;
  jobId?: string | null;
  job_id?: string | null;
  id?: string | null;
  runtimePath?: string | null;
}): string {
  const candidates = [
    source.evalJobId,
    source.eval_job_id,
    source.jobId,
    source.job_id,
  ];
  for (const candidate of candidates) {
    if (isStrictValidEvaluationJobId(candidate)) {
      return String(candidate).trim();
    }
  }

  const rawId = String(source.id ?? '').trim();
  if (isStrictValidEvaluationJobId(rawId)) {
    return rawId;
  }

  const runtimePath = String(source.runtimePath ?? '');
  const fromPath = runtimePath.match(
    /(eval_joint_dp_[a-z0-9_]+|ct_eval_[a-z0-9_]+|eval_\d{8}_\d{6}_[a-f0-9]{4}|isaac_eval_\d{8}_\d{6}_[a-f0-9]{4})/i
  );
  if (fromPath?.[1] && isStrictValidEvaluationJobId(fromPath[1])) {
    return fromPath[1];
  }

  return '';
}

export function getEvaluationRowJobId(row: {
  evalJobId?: string | null;
  jobId?: string | null;
  id?: string | null;
  runtimePath?: string | null;
}): string {
  return resolveEvaluationJobId({
    evalJobId: row.evalJobId,
    jobId: row.jobId,
    id: row.id,
    runtimePath: row.runtimePath,
  });
}

export function toWorkspaceDeleteKey(workspaceJobId?: string | number | null): string {
  if (workspaceJobId == null || workspaceJobId === '') return '';
  return `${WORKSPACE_DELETE_KEY_PREFIX}${workspaceJobId}`;
}

export function parseWorkspaceDeleteKey(key: string): string | null {
  const trimmed = String(key ?? '').trim();
  if (!trimmed.startsWith(WORKSPACE_DELETE_KEY_PREFIX)) return null;
  const id = trimmed.slice(WORKSPACE_DELETE_KEY_PREFIX.length).trim();
  return id || null;
}

export function getEvaluationRowDeleteKey(row: Pick<
  EvaluationTaskRow,
  'evalJobId' | 'jobId' | 'id' | 'runtimePath' | 'workspaceJobId' | 'status'
>): string {
  const evalJobId = getEvaluationRowJobId(row);
  if (isStrictValidEvaluationJobId(evalJobId)) return evalJobId;
  if (row.workspaceJobId != null) {
    return toWorkspaceDeleteKey(row.workspaceJobId);
  }
  return '';
}

export type EvaluationDeleteTarget =
  | { kind: 'evalJob'; evalJobId: string }
  | { kind: 'pendingRecord'; workspaceJobId: string };

export function resolveEvaluationDeleteTarget(row: Pick<
  EvaluationTaskRow,
  'evalJobId' | 'jobId' | 'id' | 'runtimePath' | 'workspaceJobId' | 'status'
>): EvaluationDeleteTarget | null {
  const evalJobId = getEvaluationRowJobId(row);
  if (isStrictValidEvaluationJobId(evalJobId)) {
    return { kind: 'evalJob', evalJobId };
  }
  if (row.workspaceJobId != null) {
    return { kind: 'pendingRecord', workspaceJobId: String(row.workspaceJobId) };
  }
  return null;
}

export function assertValidEvaluationJobId(evalJobId: string): void {
  if (!isStrictValidEvaluationJobId(evalJobId)) {
    throw new Error(`Invalid evaluation job id: ${evalJobId}`);
  }
}

/** 导出报告时从多个来源解析真实 eval job id。 */
export function resolveExportEvaluationJobId(source: {
  evalJobId?: string | null;
  jobId?: string | null;
  id?: string | null;
  runtimePath?: string | null;
  aggregateEvalJobId?: string | null;
  listRowEvalJobId?: string | null;
}): string {
  for (const candidate of [
    source.evalJobId,
    source.jobId,
    source.aggregateEvalJobId,
    source.listRowEvalJobId,
    source.id,
  ]) {
    if (isStrictValidEvaluationJobId(candidate)) {
      return String(candidate).trim();
    }
  }
  return resolveEvaluationJobId({
    evalJobId: source.evalJobId,
    jobId: source.jobId,
    id: source.id,
    runtimePath: source.runtimePath,
  });
}
