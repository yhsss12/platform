/**
 * 项目活动事件（用于最近活动流）
 */
export type ProjectActivityType =
  | 'TASK_CREATED'
  | 'TASK_UPDATED'
  | 'TASK_DELETED'
  | 'DATA_IMPORTED'
  | 'DATA_DELETED'
  | 'MEMBER_ADDED'
  | 'MEMBER_REMOVED'
  | 'MEMBER_ROLE_CHANGED'
  | 'PROJECT_UPDATED';

export interface ProjectActivity {
  id: string;
  projectId: string;
  type: ProjectActivityType;
  refId?: string;
  message: string;
  operator: string;
  createdAt: string;
}

const STORAGE_KEY = 'eai.projectActivities.v1';

function loadAll(): ProjectActivity[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveAll(activities: ProjectActivity[]) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(activities));
}

function nextId(): string {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `act_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
}

/** 追加一条活动并写回 */
export function addProjectActivity(activity: Omit<ProjectActivity, 'id' | 'createdAt'>): ProjectActivity {
  const now = new Date().toISOString();
  const full: ProjectActivity = {
    ...activity,
    id: nextId(),
    createdAt: now,
  };
  const all = loadAll();
  all.push(full);
  saveAll(all);
  return full;
}

/** 按项目 ID 取活动，按 createdAt 倒序，取前 limit 条 */
export function getProjectActivities(projectId: string, limit = 10): ProjectActivity[] {
  return loadAll()
    .filter((a) => a.projectId === projectId)
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, limit);
}

/** 全部活动（跨项目），按 createdAt 倒序，取前 limit 条（概览页用） */
export function getAllActivities(limit = 50): ProjectActivity[] {
  return loadAll()
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, limit);
}

