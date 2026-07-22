import { normalizeRole } from '@/lib/api/roleLabels';
import type { LabelTask } from '@/features/data-platform/models/labelTask';
import type { Project } from '@/lib/projects/types';

export type LabelTaskActorPayload = {
  project_id?: string | null;
  project_owner_id?: string | null;
  labeler?: string | null;
  reviewer?: string | null;
};

function norm(s: string | null | undefined): string {
  return (s || '').trim();
}

function userNameMatch(username: string | undefined, field: string | null | undefined): boolean {
  const u = norm(username);
  const f = norm(field);
  if (!u || !f) return false;
  return u.toLowerCase() === f.toLowerCase();
}

/** 与后端 label_task_actor_permissions 对齐（含：任务指定的审核员可进入执行页并写标注） */
export function canAnnotateLabelTask(
  authUser: { id?: string; username?: string; role?: string | null } | null | undefined,
  task: LabelTaskActorPayload
): boolean {
  if (!authUser) return false;
  const r = normalizeRole(authUser.role);
  if (r === 'SUPER_ADMIN' || r === 'ADMIN') return true;
  const oid = norm(task.project_owner_id);
  if (oid && oid === norm(authUser.id)) return true;
  return (
    userNameMatch(authUser.username, task.labeler) ||
    userNameMatch(authUser.username, task.reviewer)
  );
}

export function canReviewLabelTask(
  authUser: { id?: string; username?: string; role?: string | null } | null | undefined,
  task: LabelTaskActorPayload
): boolean {
  if (!authUser) return false;
  const r = normalizeRole(authUser.role);
  if (r === 'SUPER_ADMIN' || r === 'ADMIN') return true;
  return userNameMatch(authUser.username, task.reviewer);
}

/** 列表页：用项目列表解析 ownerId */
export function labelTaskToActorPayload(task: LabelTask, projectList: Project[]): LabelTaskActorPayload {
  const proj = projectList.find((p) => p.id === task.projectId);
  return {
    project_id: task.projectId,
    project_owner_id: proj?.ownerId ?? null,
    labeler: task.labeler ?? null,
    reviewer: task.reviewer ?? null,
  };
}
