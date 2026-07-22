import type { Dataset } from '../models';
import { apiGet, apiPost, apiPatch, apiDelete, type ApiResponse } from './client';

/**
 * 获取数据集列表
 */
export async function listDatasets(): Promise<ApiResponse<Dataset[]>> {
  return apiGet<Dataset[]>('/api/datasets');
}

/**
 * 创建数据集
 */
export async function createDataset(
  dataset: Omit<Dataset, 'id' | 'createdAt' | 'updatedAt'>
): Promise<ApiResponse<Dataset>> {
  return apiPost<Dataset>('/api/datasets', dataset);
}

/**
 * 更新数据集
 */
export async function updateDataset(
  id: string,
  updates: Partial<Omit<Dataset, 'id' | 'createdAt' | 'updatedAt'>>
): Promise<ApiResponse<Dataset>> {
  return apiPatch<Dataset>(`/api/datasets/${id}`, updates);
}

/**
 * 删除数据集
 */
export async function deleteDataset(id: string): Promise<ApiResponse<void>> {
  return apiDelete<void>(`/api/datasets/${id}`);
}
