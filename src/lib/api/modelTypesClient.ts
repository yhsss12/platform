'use client';

import { apiDelete, apiGet, apiPost, apiPut } from '@/lib/api/authClient';
import type {
  CreateModelTypeInput,
  ModelTypeDefinition,
  UpdateModelTypeInput,
} from '@/types/modelType';

export interface ModelTypeListResponse {
  modelTypes: ModelTypeDefinition[];
  total: number;
}

export interface ModelTypeValidateResponse {
  valid: boolean;
  errors: string[];
}

type RawModelTypeListResponse = Partial<ModelTypeListResponse> & {
  items?: ModelTypeDefinition[];
};

export function normalizeModelTypeListResponse(
  raw: RawModelTypeListResponse | null | undefined
): ModelTypeListResponse {
  const modelTypes = Array.isArray(raw?.modelTypes)
    ? raw!.modelTypes
    : Array.isArray(raw?.items)
      ? raw!.items!
      : [];
  const total =
    typeof raw?.total === 'number' && Number.isFinite(raw.total)
      ? raw.total
      : modelTypes.length;
  return { modelTypes, total };
}

export async function listModelTypes(options?: { status?: string }): Promise<ModelTypeListResponse> {
  const params = new URLSearchParams();
  if (options?.status) params.set('status', options.status);
  const query = params.toString();
  const raw = await apiGet<RawModelTypeListResponse>(
    `/workspace/model-types${query ? `?${query}` : ''}`
  );
  return normalizeModelTypeListResponse(raw);
}

export async function listAvailableModelTypes(): Promise<ModelTypeDefinition[]> {
  const response = await listModelTypes({ status: 'available' });
  return response.modelTypes;
}

export async function getModelType(modelTypeId: string): Promise<ModelTypeDefinition> {
  return apiGet<ModelTypeDefinition>(`/workspace/model-types/${encodeURIComponent(modelTypeId)}`);
}

export async function createModelType(input: CreateModelTypeInput): Promise<ModelTypeDefinition> {
  return apiPost<ModelTypeDefinition>('/workspace/model-types', input);
}

export async function updateModelType(
  modelTypeId: string,
  input: UpdateModelTypeInput
): Promise<ModelTypeDefinition> {
  return apiPut<ModelTypeDefinition>(
    `/workspace/model-types/${encodeURIComponent(modelTypeId)}`,
    input
  );
}

export async function deleteModelType(modelTypeId: string): Promise<{ modelTypeId: string; deleted: boolean }> {
  return apiDelete<{ modelTypeId: string; deleted: boolean }>(
    `/workspace/model-types/${encodeURIComponent(modelTypeId)}`
  );
}

export async function validateModelType(modelTypeId: string): Promise<ModelTypeValidateResponse> {
  return apiPost<ModelTypeValidateResponse>(
    `/workspace/model-types/${encodeURIComponent(modelTypeId)}/validate`,
    {}
  );
}

export interface ModelTypeProbeRefreshResponse {
  accepted: boolean;
}

export async function refreshModelTypeTrainingCapabilities(): Promise<ModelTypeProbeRefreshResponse> {
  return apiPost<ModelTypeProbeRefreshResponse>('/workspace/model-types/probe/refresh', {});
}
