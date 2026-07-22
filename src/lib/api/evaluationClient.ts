'use client';

import { apiDelete, apiGet, apiPost } from '@/lib/api/authClient';
import {
  evaluateCableThreadingPolicyAsync,
  type CableThreadingEvaluateAsyncResponse,
} from '@/lib/api/cableThreadingClient';
import { resolveBackendTaskType } from '@/lib/workspace/taskTemplateMapping';
import { assertValidEvaluationJobId } from '@/lib/workspace/evaluationJobId';
import { normalizeEvaluationJobListResponse } from '@/lib/workspace/evaluationJobs';
import type { WorkspaceEvaluationMode } from '@/lib/api/taskTemplatesClient';

export type EvaluationMode =
  | 'policy_evaluation'
  | 'episode_stability'
  | 'expert_policy_evaluation'
  | 'trained_model_evaluation';

export type EvaluationTaskType =
  | 'cable_threading'
  | 'cable_threading_single_arm'
  | 'dual_arm_cable_manipulation';

export interface EvaluateAsyncRequest {
  taskTemplateId?: string;
  taskType?: EvaluationTaskType | string;
  evaluationMode: WorkspaceEvaluationMode | EvaluationMode;
  evaluationObject?: string;
  numEpisodes: number;
  seed?: number;
  seeds?: number[];
  policyType?: string;
  checkpointId?: string | null;
  checkpointPath?: string;
  datasetId?: string;
  modelAssetId?: string;
  record?: boolean;
  headless?: boolean;
  maxCables?: number;
  horizon?: number;
  cableThreading?: Record<string, unknown>;
  dualArmCable?: Record<string, unknown>;
  taskConfigId?: string;
  taskName?: string;
  modelName?: string;
  metrics?: string[];
  config?: Record<string, unknown>;
  taskTemplate?: 'single_task' | 'multi_task';
  selectedTaskIds?: string[];
}

export interface EvaluateAsyncResponse {
  evalJobId: string;
  taskType: string;
  taskTemplateId?: string;
  evaluationMode: string;
  status: 'queued' | 'running';
  runtimePath?: string;
  resultPath?: string;
  createdAt?: string;
  statusUrl?: string;
  logUrl?: string;
  resultUrl?: string;
}

export interface EvaluationReplayUriItem {
  episodeIndex?: number | null;
  uri: string;
  label?: string | null;
  fileName?: string | null;
  recordCamera?: string | null;
  success?: boolean | null;
}

export interface EvaluationJobStatusResponse {
  evalJobId: string;
  taskType: string;
  evaluationMode: string;
  status: string;
  phase?: string | null;
  progress?: number | null;
  currentEpisode?: number | null;
  totalEpisodes?: number | null;
  message: string;
  metrics: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  updatedAt?: string | null;
  requestedEpisodes?: number | null;
  completedEpisodes?: number | null;
  successfulEpisodes?: number | null;
  failedEpisodes?: number | null;
  successRate?: number | null;
  recordedVideoCount?: number | null;
  replayUri?: string | null;
  replayUris?: EvaluationReplayUriItem[];
  videoAvailable?: boolean;
  isRepresentativeVideo?: boolean;
  warning?: string | null;
  workbenchBasicInfo?: Record<string, unknown>;
  taskName?: string | null;
  evaluationTypeLabel?: string | null;
  evaluationObject?: string | null;
  simulationPlatform?: string | null;
  robotType?: string | null;
  modelAssetName?: string | null;
}

export interface EvaluationSuccessStats {
  successEpisodes: number | null;
  totalEpisodes: number | null;
  display: string;
  available: boolean;
  source?: string;
  reason?: string;
}

export interface EvaluationJobListItem {
  workspaceJobId?: number | null;
  evalJobId: string;
  jobId?: string | null;
  taskType?: string | null;
  evaluationMode?: string | null;
  evaluationObject?: string | null;
  evaluationType?: string | null;
  evaluationTypeLabel?: string | null;
  status: string;
  message?: string | null;
  errorMessage?: string | null;
  createdAt?: string | null;
  updatedAt?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  taskName?: string | null;
  templateDisplayName?: string | null;
  runner?: string | null;
  runtimePath?: string | null;
  metrics?: Record<string, unknown>;
  videoAvailable?: boolean;
  requestedEpisodes?: number | null;
  completedEpisodes?: number | null;
  currentEpisode?: number | null;
  totalEpisodes?: number | null;
  progress?: number | null;
  progressPercent?: number | null;
  progressLabel?: string | null;
  successStats?: EvaluationSuccessStats;
}

export interface EvaluationJobDeleteResponse {
  success?: boolean;
  evalJobId?: string;
  deleted: boolean;
  deletedAt?: string | null;
  warning?: string | null;
  workspaceJobId?: number | null;
  jobId?: string | null;
}

export interface EvaluationPendingRecordDeleteResponse {
  success: boolean;
  deleted: boolean;
  workspaceJobId: number;
  jobId?: string | null;
  status: string;
}

export interface EvaluationJobBatchDeleteResponse {
  success: boolean;
  deletedCount: number;
  deleted: string[];
  deletedRecordIds?: number[];
  failed: Array<{
    evalJobId?: string;
    workspaceJobId?: number;
    reason: string;
  }>;
  warnings?: string[];
}

export const EVALUATION_JOB_DELETE_CONFIRM =
  '确认删除该评测任务吗？删除后该任务将从评测中心移除，相关评测摘要也会删除。若任务正在运行，删除后将从列表隐藏，但后台进程可能仍在执行。';

export function evaluationJobBatchDeleteConfirm(count: number): string {
  return `确认删除选中的 ${count} 个评测任务吗？删除后这些任务将从评测中心移除，相关评测摘要也会删除。`;
}

export interface EvaluationCapabilities {
  taskType: string;
  supportedModes: string[];
  supportedPolicyTypes?: string[];
  supportsCheckpoint: boolean;
  supportsPolicyEvaluation: boolean;
  supportsEpisodeStability: boolean;
  supportsTrainModelEvaluation?: boolean;
  supportsVideo?: boolean;
  resultArtifact?: string | null;
  description: string;
}

export async function listEvaluationCapabilities(): Promise<EvaluationCapabilities[]> {
  return apiGet<EvaluationCapabilities[]>('/workspace/evaluation/capabilities');
}

export interface EvaluationJobListResponse {
  jobs: EvaluationJobListItem[];
  total: number;
}

export interface ListEvaluationJobsParams {
  limit?: number;
  offset?: number;
  search?: string;
  status?: string;
  mode?: string;
  backend?: string;
}

export async function listEvaluationJobs(
  params: ListEvaluationJobsParams = {}
): Promise<EvaluationJobListResponse> {
  const qs = new URLSearchParams();
  if (params.limit != null) qs.set('limit', String(params.limit));
  if (params.offset != null) qs.set('offset', String(params.offset));
  if (params.search?.trim()) qs.set('search', params.search.trim());
  if (params.status?.trim()) qs.set('status', params.status.trim());
  if (params.mode?.trim()) qs.set('mode', params.mode.trim());
  if (params.backend?.trim()) qs.set('backend', params.backend.trim());
  const query = qs.toString();
  const raw = await apiGet<EvaluationJobListResponse>(
    query ? `/workspace/evaluation/jobs?${query}` : '/workspace/evaluation/jobs'
  );
  return normalizeEvaluationJobListResponse(raw);
}

export async function getEvaluationCapabilities(
  taskType: EvaluationTaskType
): Promise<EvaluationCapabilities> {
  return apiGet<EvaluationCapabilities>(
    `/workspace/evaluation/capabilities/${encodeURIComponent(taskType)}`
  );
}

export async function getEvaluationCapability(
  taskType: EvaluationTaskType
): Promise<EvaluationCapabilities> {
  return getEvaluationCapabilities(taskType);
}

export async function startEvaluationAsync(
  payload: EvaluateAsyncRequest
): Promise<EvaluateAsyncResponse> {
  return apiPost<EvaluateAsyncResponse>('/workspace/evaluation/evaluate-async', payload);
}

export async function getEvaluationJobStatus(
  evalJobId: string
): Promise<EvaluationJobStatusResponse> {
  return apiGet<EvaluationJobStatusResponse>(
    `/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}/status`
  );
}

export async function getEvaluationJobLog(
  evalJobId: string
): Promise<{ evalJobId: string; tail: string }> {
  return apiGet<{ evalJobId: string; tail: string }>(
    `/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}/log`
  );
}

export async function getEvaluationJobResult(evalJobId: string): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>(
    `/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}/result`
  );
}

export function deleteEvaluationJob(evalJobId: string): Promise<EvaluationJobDeleteResponse> {
  assertValidEvaluationJobId(evalJobId);
  return apiDelete<EvaluationJobDeleteResponse>(
    `/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}`
  );
}

export function deletePendingEvaluationRecord(
  workspaceJobId: string | number
): Promise<EvaluationPendingRecordDeleteResponse> {
  const id = String(workspaceJobId ?? '').trim();
  if (!/^\d+$/.test(id)) {
    return Promise.reject(new Error('workspaceJobId 无效'));
  }
  return apiDelete<EvaluationPendingRecordDeleteResponse>(
    `/workspace/evaluation/records/${encodeURIComponent(id)}`
  );
}

export async function deleteEvaluationJobsBatch(
  evalJobIds: string[],
  workspaceJobIds: Array<string | number> = []
): Promise<EvaluationJobBatchDeleteResponse> {
  const validIds = evalJobIds.filter((id) => {
    try {
      assertValidEvaluationJobId(id);
      return true;
    } catch {
      return false;
    }
  });
  const validRecordIds = workspaceJobIds
    .map((id) => Number(id))
    .filter((id) => Number.isInteger(id) && id > 0);
  if (!validIds.length && !validRecordIds.length) {
    return { success: true, deletedCount: 0, deleted: [], deletedRecordIds: [], failed: [] };
  }
  return apiPost<EvaluationJobBatchDeleteResponse>(
    '/workspace/evaluation/jobs/batch-delete',
    { evalJobIds: validIds, workspaceJobIds: validRecordIds }
  );
}

export function buildEvaluationVideoApiPath(evalJobId: string, episode?: number): string {
  const base = `/api/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}/video`;
  if (episode == null) return base;
  return `${base}?episode=${episode}`;
}

/** 线缆穿杆评测 nested 参数（与 backend cable_threading_eval_params 对齐） */
export interface CableThreadingEvaluationParams {
  robot?: string;
  cableModel?: string;
  difficulty?: string;
  horizon?: number;
  /** 与 numEpisodes 冗余，后端 resolve_cable_eval_episodes 会读取 */
  episodes?: number;
  recordVideo?: boolean;
  device?: string;
  modelName?: string;
  taskName?: string;
  seed?: number;
  policyType?: string;
  checkpointId?: string;
  evalExecutor?: string;
  controllerType?: string;
  actionMode?: string;
  evalDisplayCamera?: string;
  allowCameraFallback?: boolean;
}

/** 双臂线缆评测 nested 参数 */
export interface DualArmCableEvaluationParams {
  stretchMode?: string;
  releaseMode?: string;
  modelName?: string;
  taskName?: string;
  policyType?: string;
  checkpointPath?: string;
  modelAssetId?: string;
}

/** 页面层统一评测创建请求 */
export interface CreateEvaluationJobRequest {
  taskTemplateId: string;
  evaluationMode: WorkspaceEvaluationMode;
  evaluationObject?: string;
  productEvaluationMode?: string;
  evaluationType?: string;
  evaluationTypeLabel?: string;
  numEpisodes: number;
  seed?: number;
  seeds?: number[];
  datasetId?: string;
  modelAssetId?: string;
  checkpointPath?: string;
  taskConfigId?: string;
  taskName?: string;
  modelName?: string;
  metrics?: string[];
  config?: Record<string, unknown>;
  record?: boolean;
  headless?: boolean;
  maxCables?: number;
  horizon?: number;
  taskTemplate?: 'single_task' | 'multi_task';
  selectedTaskIds?: string[];
  cableThreading?: CableThreadingEvaluationParams;
  dualArmCable?: DualArmCableEvaluationParams;
}

export interface CreateEvaluationJobResponse {
  evalJobId: string;
  taskType: string;
  taskTemplateId?: string;
  evaluationMode: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  runtimePath?: string;
  resultPath?: string;
  createdAt?: string;
  statusUrl?: string;
  logUrl?: string;
  resultUrl?: string;
}

export interface DatasetEvaluationConfigPayload {
  datasetId: string;
  datasetName: string;
  metrics: string[];
}

export interface DatasetEvaluationSubmitBody {
  evaluationType: 'dataset';
  config: DatasetEvaluationConfigPayload;
}

export async function startDatasetEvaluation(
  body: DatasetEvaluationSubmitBody
): Promise<CreateEvaluationJobResponse> {
  const response = await apiPost<EvaluateAsyncResponse>(
    '/workspace/evaluation/dataset-evaluate-async',
    body
  );
  return mapEvaluateResponse(response);
}

function mapEvaluateResponse(response: EvaluateAsyncResponse): CreateEvaluationJobResponse {
  return {
    evalJobId: response.evalJobId,
    taskType: response.taskType,
    taskTemplateId: response.taskTemplateId,
    evaluationMode: response.evaluationMode,
    status: response.status,
    runtimePath: response.runtimePath,
    resultPath: response.resultPath,
    createdAt: response.createdAt,
    statusUrl: response.statusUrl,
    logUrl: response.logUrl,
    resultUrl: response.resultUrl,
  };
}

function buildUnifiedEvaluatePayload(payload: CreateEvaluationJobRequest): EvaluateAsyncRequest {
  const backendTaskType = resolveBackendTaskType(payload.taskTemplateId);
  const displayName = payload.modelName?.trim() || payload.taskName?.trim() || undefined;
  const cableThreading = {
    ...(payload.cableThreading ?? {}),
    ...(displayName ? { modelName: displayName, taskName: displayName } : {}),
  };
  const dualArmCable = {
    ...(payload.dualArmCable ?? {}),
    ...(displayName ? { modelName: displayName, taskName: displayName } : {}),
  };
  return {
    taskTemplateId: payload.taskTemplateId,
    taskType: payload.taskTemplateId,
    evaluationMode: payload.evaluationMode,
    evaluationObject: payload.evaluationObject,
    numEpisodes: payload.numEpisodes,
    seed: payload.seed,
    seeds: payload.seeds,
    datasetId: payload.datasetId,
    modelAssetId: payload.modelAssetId,
    checkpointPath: payload.checkpointPath,
    taskConfigId: payload.taskConfigId,
    record: payload.record,
    headless: payload.headless,
    maxCables: payload.maxCables,
    horizon: payload.horizon,
    ...(displayName ? { modelName: displayName, taskName: displayName } : {}),
    cableThreading,
    dualArmCable,
    taskTemplate: payload.taskTemplate,
    selectedTaskIds: payload.selectedTaskIds,
    metrics: payload.metrics ?? (Array.isArray(payload.config?.metrics) ? payload.config.metrics : undefined),
    config: payload.config,
  };
}

async function legacyCableEvaluateFallback(
  payload: CreateEvaluationJobRequest
): Promise<CreateEvaluationJobResponse> {
  const policy =
    payload.evaluationMode === 'trained_model_evaluation' ? 'robomimic' : 'scripted';
  const cableParams = payload.cableThreading ?? {};
  const response: CableThreadingEvaluateAsyncResponse = await evaluateCableThreadingPolicyAsync({
    episodes: payload.numEpisodes,
    robot: cableParams.robot,
    cableModel: cableParams.cableModel,
    difficulty: cableParams.difficulty,
    horizon: cableParams.horizon,
    seed: payload.seed ?? 0,
    policy,
    checkpoint:
      payload.evaluationMode === 'trained_model_evaluation' ? payload.checkpointPath : undefined,
    taskConfigId: payload.taskConfigId,
  });
  return {
    evalJobId: response.evalJobId,
    taskType: response.taskType,
    taskTemplateId: payload.taskTemplateId,
    evaluationMode: payload.evaluationMode,
    status: response.status,
    statusUrl: response.statusUrl,
  };
}

/**
 * 统一评测创建入口。默认调用 POST /workspace/evaluation/evaluate-async。
 * 线缆穿杆 legacy API 仅在统一入口失败时作为 fallback。
 */
export async function createEvaluationJob(
  payload: CreateEvaluationJobRequest
): Promise<CreateEvaluationJobResponse> {
  if (!resolveBackendTaskType(payload.taskTemplateId)) {
    throw new Error(`未知任务模板：${payload.taskTemplateId}`);
  }

  try {
    const response = await startEvaluationAsync(buildUnifiedEvaluatePayload(payload));
    return mapEvaluateResponse(response);
  } catch (err) {
    if (resolveBackendTaskType(payload.taskTemplateId) === 'cable_threading') {
      return legacyCableEvaluateFallback(payload);
    }
    throw err;
  }
}

export type EvaluationReportExportFormat =
  | 'pdf'
  | 'docx'
  | 'json'
  | 'markdown'
  | 'xlsx'
  | 'csv'
  | 'latex'
  | 'zip';

export interface EvaluationReportExportOptions {
  format?: EvaluationReportExportFormat;
  template?: string;
  includeBasicInfo?: boolean;
  includeConfig?: boolean;
  includeMetrics?: boolean;
  includeEpisodes?: boolean;
  includeVideoInfo?: boolean;
  includeDiagnostics?: boolean;
  includeRuntimeIndex?: boolean;
  includeUnavailableMetricReasons?: boolean;
  force?: boolean;
}

function parseFilenameFromDisposition(disposition: string | null): string | undefined {
  if (!disposition) return undefined;
  const match = /filename="?([^";\n]+)"?/.exec(disposition);
  return match ? match[1].trim() : undefined;
}

export interface EvaluationReportExportResult {
  ok: boolean;
  error?: string;
  hint?: string;
  filename?: string;
}

function normalizeExportErrorMessage(
  response: Response,
  payload: unknown,
  rawText: string
): { error: string; hint?: string } {
  const data =
    payload && typeof payload === 'object' && !Array.isArray(payload)
      ? (payload as Record<string, unknown>)
      : null;

  const nestedDetail =
    data?.detail && typeof data.detail === 'object' && !Array.isArray(data.detail)
      ? (data.detail as Record<string, unknown>)
      : null;

  const errorRaw =
    (typeof data?.error === 'string' && data.error) ||
    (typeof nestedDetail?.error === 'string' && nestedDetail.error) ||
    (typeof data?.detail === 'string' && data.detail) ||
    rawText.trim();

  const hintRaw =
    (typeof data?.hint === 'string' && data.hint) ||
    (typeof nestedDetail?.hint === 'string' && nestedDetail.hint) ||
    undefined;

  if (response.status === 404) {
    const isRouteMissing =
      errorRaw === 'Not Found' ||
      errorRaw.toLowerCase() === 'not found' ||
      (!errorRaw && response.url.includes('/report/export'));
    if (isRouteMissing) {
      return {
        error: '导出失败：未找到评测任务或导出接口不存在',
        hint:
          hintRaw ??
          '请确认使用的是真实 job_id（eval_* / ct_eval_* / isaac_eval_*），并确保后端已更新并重启',
      };
    }
    return {
      error: errorRaw === 'Not Found' ? '导出失败：未找到评测任务' : errorRaw || '导出失败：未找到评测任务',
      hint:
        hintRaw ?? '请确认导出使用的是真实 job_id，而不是任务展示名称',
    };
  }

  if (response.status === 503) {
    return {
      error: errorRaw || '导出失败：服务端缺少对应格式依赖',
      hint: hintRaw,
    };
  }

  return {
    error: errorRaw || `导出失败（HTTP ${response.status}）`,
    hint: hintRaw,
  };
}

async function parseExportError(response: Response): Promise<{ error: string; hint?: string }> {
  const text = await response.text();
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    /* ignore */
  }
  return normalizeExportErrorMessage(response, payload, text);
}

export async function downloadEvaluationReport(
  evalJobId: string,
  options: EvaluationReportExportOptions = {}
): Promise<EvaluationReportExportResult> {
  const format = options.format ?? 'json';
  if (!evalJobId?.trim()) {
    return {
      ok: false,
      error: '导出失败：缺少评测任务 ID',
      hint: '请从评测中心任务列表进入报告页，或使用真实 job_id（eval_* / ct_eval_* / isaac_eval_*）',
    };
  }

  try {
    assertValidEvaluationJobId(evalJobId);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error('[downloadEvaluationReport] invalid evalJobId:', evalJobId, message);
    return {
      ok: false,
      error: '导出失败：任务 ID 无效',
      hint: '请确认使用的是真实 job_id，而不是任务展示名称（如「线缆整理评测_20260626_103」）',
    };
  }

  const { getAccessToken } = await import('@/lib/auth/session');
  const token = getAccessToken();
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const requestUrl = `/api/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}/report/export`;
  let response: Response;
  try {
    response = await fetch(requestUrl, {
      method: 'POST',
      headers,
      credentials: 'include',
      body: JSON.stringify({ format, template: 'standard', force: true, ...options }),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error('[downloadEvaluationReport] network error:', requestUrl, message);
    return { ok: false, error: '导出失败：网络请求异常', hint: message };
  }

  if (!response.ok) {
    const parsed = await parseExportError(response);
    console.error('[downloadEvaluationReport] export failed:', {
      evalJobId,
      format,
      status: response.status,
      url: requestUrl,
      error: parsed.error,
      hint: parsed.hint,
    });
    return { ok: false, error: parsed.error, hint: parsed.hint };
  }

  const blob = await response.blob();
  if (blob.size === 0) {
    return { ok: false, error: '导出内容为空' };
  }

  const filename =
    parseFilenameFromDisposition(response.headers.get('Content-Disposition')) ||
    `evaluation_report_${evalJobId}.${format === 'csv' ? 'zip' : format === 'zip' ? 'zip' : format}`;

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  return { ok: true, filename };
}

export function buildEvaluationReportExportUrl(
  evalJobId: string,
  format: EvaluationReportExportFormat = 'json'
): string {
  assertValidEvaluationJobId(evalJobId);
  return `/api/workspace/evaluation/jobs/${encodeURIComponent(evalJobId)}/report/export?format=${encodeURIComponent(format)}`;
}
