/**
 * 统一 TaskJob 控制（与 backend TaskJob 表、协作式取消配合）
 */
import { apiPost, type ApiResponse } from './client';

export async function cancelTaskJob(taskId: string): Promise<ApiResponse<{ task_id: string; status: string }>> {
  return apiPost<{ task_id: string; status: string }>(
    `/api/tasks/jobs/${encodeURIComponent(taskId)}/cancel`,
    {}
  );
}
