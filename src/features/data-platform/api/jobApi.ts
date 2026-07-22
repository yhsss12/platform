import type { Job } from '../models';
import { apiGet, apiPost, apiPatch, apiDelete, type ApiResponse } from './client';
export type { ApiResponse };

/**
 * 获取作业列表
 */
export async function listJobs(taskId?: string, reconcileDisk?: boolean): Promise<ApiResponse<Job[]>> {
  const q = new URLSearchParams();
  if (taskId) q.set('task_id', taskId);
  // 采集作业列表：按采集端磁盘 episode 目录数回写进度（需 Agent 隧道在线）
  if (taskId && reconcileDisk) q.set('reconcile_disk', 'true');
  const query = q.toString();
  return apiGet<Job[]>(`/api/jobs${query ? `?${query}` : ''}`);
}

/**
 * 根据 ID 获取作业
 */
export async function getJob(id: string): Promise<ApiResponse<Job>> {
  return apiGet<Job>(`/api/jobs/${id}`);
}

/**
 * 创建作业
 */
export async function createJob(
  job: Omit<Job, 'id' | 'createdAt' | 'updatedAt'>
): Promise<ApiResponse<Job>> {
  return apiPost<Job>('/api/jobs', job);
}

/**
 * 更新作业
 */
export async function updateJob(
  id: string,
  updates: Partial<Omit<Job, 'id' | 'createdAt' | 'updatedAt'>>
): Promise<ApiResponse<Job>> {
  return apiPatch<Job>(`/api/jobs/${id}`, updates);
}

/**
 * 启动作业
 */
export async function startJob(id: string): Promise<ApiResponse<Job>> {
  return apiPost<Job>(`/api/jobs/${id}/start`, {});
}

/**
 * 取消作业
 */
export async function cancelJob(id: string): Promise<ApiResponse<Job>> {
  return apiPost<Job>(`/api/jobs/${id}/cancel`, {});
}

/**
 * 完成作业
 */
export async function finishJob(
  id: string,
  updates: Partial<Omit<Job, 'id' | 'createdAt' | 'updatedAt'>>
): Promise<ApiResponse<Job>> {
  return apiPost<Job>(`/api/jobs/${id}/finish`, updates);
}

/**
 * 删除作业
 */
export async function deleteJob(id: string): Promise<ApiResponse<void>> {
  return apiDelete<void>(`/api/jobs/${id}`);
}
