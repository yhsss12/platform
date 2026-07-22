'use client';

import { apiDelete, apiGet, apiPost } from '@/lib/api/authClient';
import { getAccessToken, getSessionId, initSessionId } from '@/lib/auth/session';
import type { Dataset } from '@/types/benchmark';
import { sortDatasetsByCreatedAtDesc } from '@/lib/workspace/datasetSort';

export interface DatasetListResponse {
  datasets: Dataset[];
  total: number;
}

export interface ListWorkspaceDatasetsParams {
  limit?: number;
  offset?: number;
  search?: string;
  task?: string;
  source?: string;
  format?: string;
}

export interface DatasetImportUploadResponse {
  dataset: Dataset;
  datasetId: string;
  status: string;
  validationReport: Record<string, unknown>;
}

const API_BASE = typeof window !== 'undefined' ? '/api' : process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

function formatApiErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg?: unknown }).msg ?? '');
        }
        return '';
      })
      .filter(Boolean);
    if (messages.length > 0) return messages.join('；');
  }
  if (detail && typeof detail === 'object' && 'message' in detail) {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === 'string' && message.trim()) return message;
  }
  return fallback;
}

export async function listWorkspaceDatasets(
  params: ListWorkspaceDatasetsParams = {}
): Promise<DatasetListResponse> {
  const qs = new URLSearchParams();
  if (params.limit != null) qs.set('limit', String(params.limit));
  if (params.offset != null) qs.set('offset', String(params.offset));
  if (params.search?.trim()) qs.set('search', params.search.trim());
  if (params.task?.trim()) qs.set('task', params.task.trim());
  if (params.source?.trim()) qs.set('source', params.source.trim());
  if (params.format?.trim()) qs.set('format', params.format.trim());
  const query = qs.toString();
  const path = query ? `/workspace/datasets?${query}` : '/workspace/datasets';
  const response = await apiGet<DatasetListResponse>(path);
  return {
    ...response,
    datasets: sortDatasetsByCreatedAtDesc(response.datasets ?? []),
  };
}

export async function importWorkspaceDataset(form: FormData): Promise<DatasetImportUploadResponse> {
  const token = getAccessToken();
  const sessionId = initSessionId();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (sessionId) headers['X-Session-Id'] = sessionId;

  const response = await fetch(`${API_BASE}/workspace/datasets/import/upload`, {
    method: 'POST',
    headers,
    body: form,
    credentials: 'omit',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = (errorData as { detail?: unknown }).detail;
    throw new Error(formatApiErrorDetail(detail, '导入数据集失败'));
  }

  return response.json() as Promise<DatasetImportUploadResponse>;
}

export async function deleteImportedWorkspaceDataset(datasetId: string): Promise<{ ok: boolean }> {
  return apiDelete<{ ok: boolean }>(`/workspace/datasets/import/${encodeURIComponent(datasetId)}`);
}

export interface Hdf5SchemaField {
  path: string;
  dtype?: string;
  shape?: number[];
}

export interface DatasetSchemaResponse {
  datasetId: string;
  fields: Hdf5SchemaField[];
  recognizedFields?: Record<string, string | null>;
}

export interface DatasetFieldMapping {
  action?: string | null;
  qpos?: string | null;
  image?: string | null;
  qvel?: string | null;
  done?: string | null;
}

export interface BuildDatasetFromImportRequest {
  sourceDatasetId: string;
  outputName: string;
  taskType: string;
  targetFormat: 'standard_hdf5';
  fieldMapping?: DatasetFieldMapping | null;
  auto?: boolean;
  episodeRule?: { type: 'single_episode' };
}

export interface BuildDatasetFromImportResponse {
  builtDatasetId: string;
  status: string;
  trainable: boolean;
  directTrainable: boolean;
  dataset?: Dataset;
}

export async function getDatasetSchema(datasetId: string): Promise<DatasetSchemaResponse> {
  return apiGet<DatasetSchemaResponse>(
    `/workspace/datasets/${encodeURIComponent(datasetId)}/schema`
  );
}

export async function buildDatasetFromImport(
  body: BuildDatasetFromImportRequest
): Promise<BuildDatasetFromImportResponse> {
  return apiPost<BuildDatasetFromImportResponse>('/workspace/datasets/build/from-import', body);
}

export async function deleteBuiltWorkspaceDataset(datasetId: string): Promise<{ ok: boolean }> {
  return apiDelete<{ ok: boolean }>(`/workspace/datasets/built/${encodeURIComponent(datasetId)}`);
}

/** 评测表单「选择数据集」：GET /api/datasets/list */
export async function listEvaluationDatasets(): Promise<DatasetListResponse> {
  const response = await apiGet<DatasetListResponse>('/datasets/list');
  return {
    ...response,
    datasets: sortDatasetsByCreatedAtDesc(response.datasets ?? []),
  };
}

export function formatDatasetOptionLabel(dataset: Dataset): string {
  const version = dataset.sourceJobId?.trim() || dataset.format || 'v1';
  return `${dataset.name} · ${version}`;
}
