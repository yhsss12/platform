/**
 * 大模型厂商/模型 API（标注页「模型选择与管理」弹窗）
 * 数据存 backend/data/assets/assets.db
 */
import { apiGet, apiPatch, apiPost, type ApiResponse } from '@/features/data-platform/api/client';

export interface ProviderItem {
  id: number;
  name: string;
  code: string;
  type: string;
  base_url: string | null;
  api_key_prefix: string | null;
  supports_stream: boolean;
  supports_functions: boolean;
  logo: string | null;
  is_active: boolean;
  sort_order: number;
  is_enabled: boolean;
  has_api_key: boolean;
  is_verified: boolean;
}

export interface ModelItem {
  id: number;
  provider_id: number;
  name: string;
  display_name: string | null;
  context_length: number | null;
  max_output_tokens: number | null;
  is_default: boolean;
  is_active: boolean;
  sort_order: number;
  is_selected: boolean;
}

export interface ProviderDetail extends ProviderItem {
  api_base: string | null;
  api_key_masked?: string | null;
  models: ModelItem[];
}

export async function listProviders(
  search?: string,
  projectId?: string
): Promise<ApiResponse<ProviderItem[]>> {
  const qs = new URLSearchParams();
  if (search) qs.set('search', search);
  if (projectId?.trim()) qs.set('project_id', projectId.trim());
  const q = qs.toString();
  return apiGet<ProviderItem[]>(`/api/label/llm/providers${q ? `?${q}` : ''}`);
}

export async function getProviderDetail(
  providerId: number,
  projectId?: string
): Promise<ApiResponse<ProviderDetail>> {
  const q = projectId?.trim() ? `?project_id=${encodeURIComponent(projectId.trim())}` : '';
  return apiGet<ProviderDetail>(`/api/label/llm/providers/${providerId}${q}`);
}

export async function updateUserProvider(params: {
  provider_id: number;
  api_key?: string;
  api_base?: string;
  is_enabled?: boolean;
  project_id: string;
}): Promise<ApiResponse<{ updated: boolean }>> {
  return apiPatch<{ updated: boolean }>('/api/label/llm/user-providers', params);
}

export async function verifyProvider(params: {
  provider_id: number;
  api_key?: string;
  api_base?: string;
  project_id: string;
}): Promise<ApiResponse<{ success: boolean }>> {
  return apiPost<{ success: boolean }>('/api/label/llm/verify', params);
}

export async function listModels(
  providerId: number,
  search?: string,
  projectId?: string
): Promise<ApiResponse<ModelItem[]>> {
  const q = new URLSearchParams({ provider_id: String(providerId) });
  if (search) q.set('search', search);
  if (projectId?.trim()) q.set('project_id', projectId.trim());
  return apiGet<ModelItem[]>(`/api/label/llm/models?${q}`);
}

export async function updateUserModels(params: {
  provider_id: number;
  model_ids: number[];
  project_id: string;
}): Promise<ApiResponse<{ updated: boolean; selected_count: number }>> {
  return apiPatch<{ updated: boolean; selected_count: number }>('/api/label/llm/user-models', params);
}

export async function createModel(params: {
  project_id: string;
  provider_id: number;
  name: string;
  display_name?: string;
}): Promise<ApiResponse<{ created: boolean }>> {
  return apiPost<{ created: boolean }>('/api/label/llm/models', params);
}

export async function deleteModel(modelId: number, projectId: string): Promise<ApiResponse<{ deleted: boolean }>> {
  const q = `?project_id=${encodeURIComponent(projectId)}`;
  return apiPost<{ deleted: boolean }>(`/api/label/llm/models/${modelId}/delete${q}`, {});
}

export async function updateModel(
  modelId: number,
  params: { name?: string; display_name?: string },
  projectId: string
): Promise<ApiResponse<{ updated: boolean }>> {
  const q = `?project_id=${encodeURIComponent(projectId)}`;
  return apiPatch<{ updated: boolean }>(`/api/label/llm/models/${modelId}${q}`, params);
}
