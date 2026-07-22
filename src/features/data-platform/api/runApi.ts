import type { Run } from '../models';
import { apiGet, apiPost, apiPatch, apiDelete, type ApiResponse } from './client';

/**
 * 获取运行列表
 */
export async function listRuns(): Promise<ApiResponse<Run[]>> {
  return apiGet<Run[]>('/api/runs');
}

/**
 * 根据 ID 获取运行
 */
export async function getRun(id: string): Promise<ApiResponse<Run>> {
  return apiGet<Run>(`/api/runs/${id}`);
}

/**
 * 创建运行
 */
export async function createRun(
  run: Omit<Run, 'id' | 'createdAt' | 'updatedAt'>
): Promise<ApiResponse<Run>> {
  return apiPost<Run>('/api/runs', run);
}

/**
 * 更新运行
 */
export async function updateRun(
  id: string,
  updates: Partial<Omit<Run, 'id' | 'createdAt' | 'updatedAt'>>
): Promise<ApiResponse<Run>> {
  return apiPatch<Run>(`/api/runs/${id}`, updates);
}

/**
 * 删除运行
 */
export async function deleteRun(id: string): Promise<ApiResponse<void>> {
  return apiDelete<void>(`/api/runs/${id}`);
}
