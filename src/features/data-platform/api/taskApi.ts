import type { Task } from '../models';
import { apiGet, apiPost, apiPatch, apiDelete, type ApiResponse } from './client';

/**
 * 获取任务列表
 * - 若不传参数：返回所有采集任务（受后端权限控制）
 * - 传入 projectId 时：仅返回该项目下的采集任务
 */
export async function listTasks(params?: { projectId?: string }): Promise<ApiResponse<Task[]>> {
  const query =
    params?.projectId && params.projectId.trim().length > 0
      ? `?project_id=${encodeURIComponent(params.projectId.trim())}`
      : '';
  return apiGet<Task[]>(`/api/tasks${query}`);
}

/**
 * 根据 ID 获取任务
 */
export async function getTask(id: string): Promise<ApiResponse<Task>> {
  return apiGet<Task>(`/api/tasks/${id}`);
}

/**
 * 创建任务
 */
export async function createTask(
  task: Omit<Task, 'id' | 'createdAt' | 'updatedAt'>
): Promise<ApiResponse<Task>> {
  return apiPost<Task>('/api/tasks', task);
}

/**
 * 更新任务
 */
export async function updateTask(
  id: string,
  updates: Partial<Omit<Task, 'id' | 'createdAt' | 'updatedAt'>>
): Promise<ApiResponse<Task>> {
  return apiPatch<Task>(`/api/tasks/${id}`, updates);
}

/**
 * 删除任务
 */
export async function deleteTask(id: string): Promise<ApiResponse<void>> {
  return apiDelete<void>(`/api/tasks/${id}`);
}
