import { apiGet, apiPost, apiDelete } from '@/features/data-platform/api/client';
import type { ConversionJob, CreateConversionInput } from './mockConversion';

const BASE_URL = '/api/conversion';

/** 与后端 ConversionBatchSummary / ConversionBatchDetail 对齐（camelCase） */
export type ConversionBatchOverallStatus =
  | 'PENDING'
  | 'RUNNING'
  | 'SUCCESS'
  | 'PARTIAL_SUCCESS'
  | 'FAILED'
  | 'CANCELED';

export interface ConversionBatchSummary {
  batchId: string;
  taskName?: string | null;
  /** 后端若返回则用于任务名称兜底展示（如 MCAP） */
  sourceFormat?: string | null;
  targetFormat: string;
  projectId: string;
  projectName: string;
  /** 创建人 users.id */
  creatorId?: string | null;
  /** 展示名：优先 users.username，否则 account_id；旧单文件任务为 operator_name */
  creatorName?: string | null;
  totalCount: number;
  successCount: number;
  failedCount: number;
  runningCount: number;
  pendingCount: number;
  progressPercent: number;
  overallStatus: ConversionBatchOverallStatus | string;
  createdAt: string;
  updatedAt: string;
  legacySingleFile?: boolean;
}

export interface ConversionBatchChildItem {
  jobId: string;
  sourceFileName: string;
  outputFileName: string;
  itemStatus: string;
  itemStage?: string | null;
  errorMessage?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ConversionBatchDetail {
  batch: ConversionBatchSummary;
  children: ConversionBatchChildItem[];
}

export interface CreateConversionBatchBody {
  taskName?: string | null;
  items: CreateConversionInput[];
}

export async function createJob(input: CreateConversionInput): Promise<ConversionJob> {
  const res = await apiPost<ConversionJob>(`${BASE_URL}/jobs`, input);
  if ((res as any).jobId) {
    return res as any as ConversionJob;
  }
  if ((res as any).ok === false) {
    throw new Error((res as any).error || 'Failed to create job');
  }
  return res as any as ConversionJob;
}

export async function createConversionBatch(body: CreateConversionBatchBody): Promise<ConversionBatchDetail> {
  const res = (await apiPost<ConversionBatchDetail>(`${BASE_URL}/batches`, body)) as unknown as Record<string, unknown>;
  if (res && typeof res === 'object' && (res as any).batch?.batchId) {
    return res as unknown as ConversionBatchDetail;
  }
  if ((res as any)?.ok === false) {
    throw new Error(String((res as any).error || '批量创建失败'));
  }
  throw new Error('批量创建失败');
}

export async function listBatches(): Promise<ConversionBatchSummary[]> {
  const res = (await apiGet<ConversionBatchSummary[]>(`${BASE_URL}/batches`)) as unknown;
  if (Array.isArray(res)) {
    return res as ConversionBatchSummary[];
  }
  if (typeof res === 'object' && res && (res as any).ok === false) {
    throw new Error(String((res as any).error || 'Failed to fetch batches'));
  }
  return [];
}

export async function getBatchDetail(batchId: string): Promise<ConversionBatchDetail> {
  const res = (await apiGet<ConversionBatchDetail>(
    `${BASE_URL}/batches/${encodeURIComponent(batchId)}`
  )) as unknown as Record<string, unknown>;
  if (res && typeof res === 'object' && (res as any).batch?.batchId) {
    return res as unknown as ConversionBatchDetail;
  }
  if ((res as any)?.ok === false) {
    throw new Error(String((res as any).error || 'Failed to fetch batch'));
  }
  throw new Error('加载批量任务失败');
}

export async function deleteConversionBatch(batchId: string): Promise<void> {
  const res = (await apiDelete(`${BASE_URL}/batches/${encodeURIComponent(batchId)}`)) as unknown;
  const r = res as Record<string, unknown>;
  if (r && typeof r === 'object' && r.success === true) {
    return;
  }
  if (r && typeof r === 'object' && r.ok === false) {
    throw new Error(String(r.error || '删除失败'));
  }
}

export async function cancelConversionBatch(batchId: string): Promise<void> {
  const res = (await apiPost<unknown>(
    `${BASE_URL}/batches/${encodeURIComponent(batchId)}/cancel`,
    {}
  )) as unknown as Record<string, unknown>;
  if (!res || typeof res !== 'object') {
    throw new Error('取消失败：响应无效');
  }
  if (res.ok === false) {
    throw new Error(String(res.error || (typeof res.detail === 'string' ? res.detail : '') || '取消失败'));
  }
}

export async function getJobs(): Promise<ConversionJob[]> {
  const res = await apiGet<ConversionJob[]>(`${BASE_URL}/jobs`);
  if (Array.isArray(res)) {
    return res;
  }
  if ((res as any).ok === false) {
    throw new Error((res as any).error || 'Failed to fetch jobs');
  }
  return [];
}

export async function getJob(jobId: string): Promise<ConversionJob> {
  const res = await apiGet<ConversionJob>(`${BASE_URL}/jobs/${jobId}`);
  if ((res as any).jobId) {
    return res as any as ConversionJob;
  }
  if ((res as any).ok === false) {
    throw new Error((res as any).error || 'Failed to fetch job');
  }
  return res as any as ConversionJob;
}

export interface McapAnalysisResult {
  topic: string;
  count: number;
  frequency: number;
  period_ms: number;
  min_delta_ms: number;
  max_delta_ms: number;
}

export async function analyzeDataset(datasetId: string): Promise<McapAnalysisResult[]> {
  const res = await apiGet<McapAnalysisResult[]>(`${BASE_URL}/analyze?datasetId=${datasetId}`);
  if (Array.isArray(res)) {
    return res;
  }
  if ((res as any).ok === false) {
    throw new Error((res as any).error || 'Failed to analyze dataset');
  }
  return [];
}

export async function deleteJob(jobId: string): Promise<void> {
  const res = await apiDelete(`${BASE_URL}/jobs/${jobId}`);
  if ((res as any).success) {
    return;
  }
  if ((res as any).ok === false) {
    throw new Error((res as any).error || 'Failed to delete job');
  }
}

/** 删除转换任务产物（输出文件或目录），并移除任务记录。仅允许删除白名单内路径。 */
export async function deleteConversionResult(jobId: string): Promise<{ ok: boolean; error?: string }> {
  const res = await apiPost<unknown>(`${BASE_URL}/jobs/${jobId}/delete-result`, {});
  if ((res as any).ok === true) return { ok: true };
  return { ok: false, error: (res as any).error || '删除失败' };
}
