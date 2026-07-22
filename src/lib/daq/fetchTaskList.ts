import { listTasks } from '@/features/data-platform/api/taskApi';
import { listJobs } from '@/features/data-platform/api/jobApi';
import type { Task } from '@/features/data-platform/models/task';
import type { DaqTask } from './types';

function taskToDaqTask(task: Task): DaqTask {
  const creatorUsername = String(
    (task as { creatorUsername?: string; creator_username?: string }).creatorUsername ??
      (task as { creator_username?: string }).creator_username ??
      task.creatorUsername ??
      ''
  ).trim();
  const creatorAccountId = String(
    (task as { creatorAccountId?: string; creator_account_id?: string }).creatorAccountId ??
      (task as { creator_account_id?: string }).creator_account_id ??
      task.creatorAccountId ??
      ''
  ).trim();
  const creatorId = String(
    (task as { creatorId?: string; creator_id?: string }).creatorId ??
      (task as { creator_id?: string }).creator_id ??
      task.creatorId ??
      ''
  ).trim();
  const owner = String(task.owner ?? '').trim();
  const creatorDisplay = creatorUsername || owner || creatorAccountId || creatorId || '';
  return {
    id: task.id,
    taskNumber: task.id.substring(0, 8),
    taskName: task.name,
    taskDescription: task.description,
    owner: task.owner,
    creatorId: creatorId || undefined,
    creatorUsername: creatorUsername || undefined,
    creatorAccountId: creatorAccountId || undefined,
    creatorDisplay: creatorDisplay || undefined,
    deviceName: task.deviceName,
    deviceId: task.deviceId,
    projectId: (task as { projectId?: string; project_id?: string }).projectId ?? (task as { project_id?: string }).project_id,
    episodeCount: task.episodeCount,
    durationSec: task.durationSec,
    storagePath: task.storagePath,
    storageTypes: task.storageTypes,
    createdAt: task.createdAt,
    updatedAt: task.updatedAt,
    remark: task.remark,
    cameraDataFormat: task.cameraDataFormat,
    frequencyConfig: (task as { frequencyConfig?: DaqTask['frequencyConfig'] }).frequencyConfig,
  };
}

/** 采集任务列表（admin/projects 隐藏页数据源） */
export async function fetchTaskList(projectId?: string): Promise<DaqTask[]> {
  const [tasksResponse, jobsResponse] = await Promise.all([
    listTasks(projectId ? { projectId } : undefined),
    listJobs(),
  ]);

  let allJobs: Record<string, unknown>[] = [];
  if (jobsResponse.ok && jobsResponse.data) {
    allJobs = jobsResponse.data as unknown as Record<string, unknown>[];
  }

  const jobStats = new Map<string, { hasJobs: boolean; completedCount: number; jobCount: number }>();
  const jobCollectors = new Map<string, Set<string>>();

  for (const job of allJobs) {
    const target = job.target as { taskId?: string } | undefined;
    const taskId = String(target?.taskId ?? job.taskId ?? job.task_id ?? '').trim();
    if (!taskId) continue;

    if (!jobStats.has(taskId)) {
      jobStats.set(taskId, { hasJobs: false, completedCount: 0, jobCount: 0 });
    }
    const stats = jobStats.get(taskId)!;
    stats.hasJobs = true;
    stats.jobCount += 1;
    const progressObj = job.progress;
    const current =
      progressObj && typeof progressObj === 'object'
        ? Number((progressObj as { current?: number }).current ?? 0)
        : Number(job.completed_count ?? job.completedCount ?? 0);
    stats.completedCount += Number.isFinite(current) ? current : 0;

    const operator = String(job.operatorName ?? job.operator_name ?? job.collector ?? '').trim();
    if (operator) {
      if (!jobCollectors.has(taskId)) jobCollectors.set(taskId, new Set<string>());
      jobCollectors.get(taskId)!.add(operator);
    }
  }

  if (tasksResponse.ok && tasksResponse.data) {
    return tasksResponse.data.map((task) => {
      const daqTask = taskToDaqTask(task);
      const stats = jobStats.get(task.id);
      if (stats) {
        daqTask.hasJobs = stats.hasJobs;
        daqTask.completedCount = stats.completedCount;
        daqTask.jobCount = stats.jobCount;
      } else {
        daqTask.hasJobs = false;
        daqTask.completedCount = 0;
        daqTask.jobCount = 0;
      }
      const collectors = jobCollectors.get(task.id);
      if (collectors && collectors.size > 0) {
        const names = Array.from(collectors);
        daqTask.collectorName = names.join('、');
        daqTask.collector = names[0];
      }
      return daqTask;
    });
  }
  throw new Error(tasksResponse.error || '加载任务列表失败');
}
