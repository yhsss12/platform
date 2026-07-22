'use client';

import { apiDelete, apiGet } from '@/lib/api/authClient';
import { getAccessToken, getSessionId, initSessionId } from '@/lib/auth/session';
import type { ModelAsset } from '@/types/benchmark';
import { resolveModelAssetColumnLabel } from '@/lib/workspace/modelAssetDisplay';

const API_BASE = typeof window !== 'undefined' ? '/api' : process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export type ModelAssetDisplayStatus = 'waiting' | 'ready' | 'generating' | 'superseded';

export interface TrainingJobModelAssetItem extends ModelAsset {
  isPlaceholder?: boolean;
  canEvaluate?: boolean;
  canEvaluateReason?: string | null;
  displayStatus?: ModelAssetDisplayStatus | string;
}

export interface ModelAssetListResponse {
  modelAssets: ModelAsset[];
  total: number;
}

type RawModelAssetListResponse = Partial<ModelAssetListResponse> & {
  assets?: ModelAsset[];
};

/** 兼容后端 `assets` 与历史 `modelAssets` 字段名 */
export function normalizeModelAssetListResponse(
  raw: RawModelAssetListResponse | null | undefined
): ModelAssetListResponse {
  const modelAssets = Array.isArray(raw?.modelAssets)
    ? raw.modelAssets
    : Array.isArray(raw?.assets)
      ? raw.assets
      : [];
  const total =
    typeof raw?.total === 'number' && Number.isFinite(raw.total)
      ? raw.total
      : modelAssets.length;
  return { modelAssets, total };
}

export interface TrainingJobModelAssetListResponse {
  modelAssets: TrainingJobModelAssetItem[];
  total: number;
  listMessage?: string | null;
}

type RawTrainingJobModelAssetListResponse = Partial<TrainingJobModelAssetListResponse> & {
  assets?: TrainingJobModelAssetItem[];
};

export function normalizeTrainingJobModelAssetListResponse(
  raw: RawTrainingJobModelAssetListResponse | null | undefined
): TrainingJobModelAssetListResponse {
  const modelAssets = Array.isArray(raw?.modelAssets)
    ? raw.modelAssets
    : Array.isArray(raw?.assets)
      ? raw.assets
      : [];
  const total =
    typeof raw?.total === 'number' && Number.isFinite(raw.total)
      ? raw.total
      : modelAssets.length;
  return {
    modelAssets,
    total,
    listMessage: raw?.listMessage ?? null,
  };
}

export interface ModelAssetDeleteResponse {
  modelAssetId: string;
  deleted: boolean;
  warnings?: string[];
}

export interface ImportModelAssetResponse {
  modelAssetId: string;
  modelName: string;
  modelType: string;
  taskName?: string | null;
  datasetName?: string | null;
  structureConfig?: Record<string, unknown> | null;
  checkpointPath: string;
  createdAt: string;
  validationResult: Record<string, unknown>;
  assetSource?: string;
}

export async function importPretrainedModelAsset(form: FormData): Promise<ImportModelAssetResponse> {
  const token = getAccessToken();
  const sessionId = initSessionId();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (sessionId) headers['X-Session-Id'] = sessionId;

  const response = await fetch(`${API_BASE}/workspace/model-assets/import`, {
    method: 'POST',
    headers,
    body: form,
    credentials: 'omit',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = (errorData as { detail?: unknown }).detail;
    throw new Error(typeof detail === 'string' ? detail : '导入预训练模型失败');
  }

  return response.json() as Promise<ImportModelAssetResponse>;
}

export const MODEL_ASSET_DELETE_CONFIRM =
  '确认删除该模型资产？将移除平台登记记录，并尝试删除关联的 checkpoint 文件（仅限 runs 安全路径）。';

export function modelAssetBatchDeleteConfirm(count: number): string {
  return `确认删除选中的 ${count} 个模型资产？将移除登记记录，并尝试删除可安全删除的 checkpoint 文件（仅限 runs 路径）。`;
}

export interface ModelAssetFilterOptionsResponse {
  modelTypes: string[];
  datasets: string[];
  sourceTasks: string[];
}

export async function listModelAssetFilterOptions(): Promise<ModelAssetFilterOptionsResponse> {
  return apiGet<ModelAssetFilterOptionsResponse>('/workspace/model-assets/filter-options');
}

export async function listModelAssets(options?: {
  forEvaluation?: boolean;
  taskType?: string;
  search?: string;
  status?: string;
  modelType?: string;
  trainingJobId?: string;
  datasetId?: string;
  source?: string;
  dataset?: string;
  sourceTask?: string;
  limit?: number;
  offset?: number;
}): Promise<ModelAssetListResponse> {
  const params = new URLSearchParams();
  if (options?.forEvaluation) params.set('forEvaluation', 'true');
  if (options?.taskType) params.set('taskType', options.taskType);
  if (options?.search) params.set('search', options.search);
  if (options?.status) params.set('status', options.status);
  if (options?.modelType) params.set('modelType', options.modelType);
  if (options?.trainingJobId) params.set('trainingJobId', options.trainingJobId);
  if (options?.datasetId) params.set('datasetId', options.datasetId);
  if (options?.source) params.set('source', options.source);
  if (options?.dataset) params.set('dataset', options.dataset);
  if (options?.sourceTask) params.set('sourceTask', options.sourceTask);
  if (options?.limit != null) params.set('limit', String(options.limit));
  if (options?.offset != null) params.set('offset', String(options.offset));
  const query = params.toString();
  const raw = await apiGet<RawModelAssetListResponse>(
    query ? `/workspace/model-assets?${query}` : '/workspace/model-assets'
  );
  return normalizeModelAssetListResponse(raw);
}

export async function getModelAsset(modelAssetId: string): Promise<ModelAsset> {
  return apiGet<ModelAsset>(`/workspace/model-assets/${encodeURIComponent(modelAssetId)}`);
}

export async function listModelAssetsByTrainingJob(
  trainJobId: string
): Promise<ModelAssetListResponse> {
  const raw = await apiGet<RawModelAssetListResponse>(
    `/workspace/model-assets/by-training-job/${encodeURIComponent(trainJobId)}`
  );
  return normalizeModelAssetListResponse(raw);
}

export async function listTrainingJobModelAssetsDetail(
  trainJobId: string
): Promise<TrainingJobModelAssetListResponse> {
  const raw = await apiGet<RawTrainingJobModelAssetListResponse>(
    `/workspace/model-assets/by-training-job/${encodeURIComponent(trainJobId)}/detail`
  );
  return normalizeTrainingJobModelAssetListResponse(raw);
}

export async function deleteModelAsset(modelAssetId: string): Promise<ModelAssetDeleteResponse> {
  return apiDelete<ModelAssetDeleteResponse>(
    `/workspace/model-assets/${encodeURIComponent(modelAssetId)}`
  );
}

export async function deleteModelAssetsBatch(modelAssetIds: string[]): Promise<{
  deleted: string[];
  failed: Array<{ modelAssetId: string; error: string }>;
}> {
  const deleted: string[] = [];
  const failed: Array<{ modelAssetId: string; error: string }> = [];
  for (const modelAssetId of modelAssetIds) {
    try {
      await deleteModelAsset(modelAssetId);
      deleted.push(modelAssetId);
    } catch (err) {
      failed.push({
        modelAssetId,
        error: err instanceof Error ? err.message : '删除失败',
      });
    }
  }
  return { deleted, failed };
}

export interface ModelAssetCheckpointOption {
  trainJobId: string;
  modelAssetId: string;
  label: string;
  ready: boolean;
  checkpointPath: string | null;
}

/** 评测弹窗 / 初始化权重：仅真实 ready 资产，排除占位与文件缺失 */
export function modelAssetsToCheckpointOptions(
  assets: Array<
    ModelAsset & { isPlaceholder?: boolean; canEvaluate?: boolean; fileExists?: boolean }
  >
): ModelAssetCheckpointOption[] {
  return assets
    .filter(
      (asset) =>
        !asset.isPlaceholder &&
        asset.canEvaluate !== false &&
        asset.fileExists !== false &&
        Boolean(asset.checkpointPath) &&
        (asset.status === 'available' || asset.status === 'ready')
    )
    .map((asset) => ({
      trainJobId: asset.sourceTrainingJobId,
      modelAssetId: asset.id,
      label: resolveModelAssetColumnLabel(asset),
      ready: Boolean(asset.checkpointPath) && asset.fileExists !== false,
      checkpointPath: asset.checkpointPath || null,
    }));
}

export function isEvaluableModelAsset(
  asset: ModelAsset & { fileExists?: boolean; canEvaluate?: boolean; isPlaceholder?: boolean }
): boolean {
  if (asset.isPlaceholder) return false;
  if (asset.fileExists === false) return false;
  if (asset.canEvaluate === false) return false;
  return (
    Boolean(asset.checkpointPath) &&
    (asset.status === 'available' || asset.status === 'ready')
  );
}
