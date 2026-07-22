import type { Project, ProjectMember } from './types';
import type { CreateProjectInput } from './createProject';
import { addProjectActivity } from './projectActivity';
import type { ProjectActivityType } from './projectActivity';
import * as projectApi from './projectApi';

let projectCache: Project[] = [];

function setProjectCache(projects: Project[]): void {
  projectCache = [...projects];
}

function upsertProjectCache(project: Project): void {
  const idx = projectCache.findIndex((p) => p.id === project.id);
  if (idx >= 0) {
    projectCache[idx] = project;
  } else {
    projectCache = [project, ...projectCache];
  }
}

function removeProjectCache(id: string): void {
  projectCache = projectCache.filter((p) => p.id !== id);
}

function mapApiToProject(item: projectApi.ProjectApiItem): Project {
  return {
    id: item.id,
    name: item.name,
    description: item.description ?? undefined,
    tags: Array.isArray(item.tags) ? item.tags : [],
    status: (item.status as Project['status']) || '进行中',
    ownerId: item.owner_id ?? '',
    teamId: item.team_id ?? null,
    viewerInProjectMembers: item.viewer_is_project_member,
    viewerIsProjectOwner: item.viewer_is_project_owner,
    memberCount: typeof item.member_count === 'number' ? item.member_count : undefined,
    members: [],
    tasks: [],
    datasets: [],
    createdAt: item.created_at,
    updatedAt: item.updated_at,
  };
}

/** 列出所有项目（仅后端 API）。withStats=true 时返回统计数据。teamId 与后端 team_id 一致，按团队过滤。 */
export async function listAsync(
  withStats = true,
  teamId?: string | null
): Promise<
  | Project[]
  | { projects: Project[]; stats: Record<string, { label_task_count: number; dataset_count: number }> }
> {
  const res = await projectApi.fetchProjects(undefined, withStats, teamId);
  if (res.ok && res.data?.items) {
    const projects = res.data.items.map(mapApiToProject);
    setProjectCache(projects);
    if (withStats && res.data.stats) {
      return { projects, stats: res.data.stats };
    }
    return projects;
  }
  throw new Error(res.error || '加载项目列表失败');
}

/** 列出缓存中的项目（仅内存缓存，不持久化）。 */
export function list(): Project[] {
  return projectCache;
}

/** 按 id 获取单个项目（仅后端 API）。 */
export async function getAsync(id: string): Promise<Project | null> {
  const res = await projectApi.fetchProject(id);
  if (res.ok && res.data) {
    const project = mapApiToProject(res.data);
    upsertProjectCache(project);
    return project;
  }
  return null;
}

/** 按 id 获取缓存中的项目（仅内存缓存，不持久化）。 */
export function get(id: string): Project | null {
  return projectCache.find((p) => p.id === id) ?? null;
}

/** 创建项目（仅后端 API）。 */
export async function create(input: CreateProjectInput): Promise<Project> {
  const now = new Date().toISOString();
  const ownerMember: ProjectMember = {
    id: input.ownerId,
    name: input.ownerName,
    role: 'Owner',
    addedAt: now,
    lastActiveAt: now,
  };
  const projectId =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `p_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  const tid = (input.teamId ?? '').trim();
  const res = await projectApi.createProjectApi({
    id: projectId,
    name: input.name.trim(),
    description: input.description?.trim() || null,
    tags: Array.isArray(input.tags) ? input.tags.slice(0, 4) : null,
    status: '进行中',
    owner_id: input.ownerId || null,
    team_id: tid ? tid : undefined,
  });
  if (!res.ok || !res.data) {
    throw new Error(res.error || '创建项目失败');
  }
  const project: Project = {
    ...mapApiToProject(res.data),
    members: [ownerMember],
    createdAt: res.data.created_at || now,
    updatedAt: res.data.updated_at || now,
  };
  upsertProjectCache(project);
  return project;
}

/** 局部更新项目（仅后端 API）。 */
export async function update(id: string, patch: Partial<Project>): Promise<Project> {
  const current = get(id);
  if (!current) throw new Error(`Project not found: ${id}`);
  const next = { ...current, ...patch, id: current.id };
  const res = await projectApi.updateProjectApi(id, {
    name: next.name,
    description: next.description ?? null,
    tags: next.tags?.length ? next.tags : null,
    status: next.status,
    owner_id: next.ownerId || null,
  });
  if (!res.ok || !res.data) {
    throw new Error(res.error || '更新项目失败');
  }
  const updated = mapApiToProject(res.data);
  // 后端 PATCH 项目不包含成员列表；mapApiToProject 固定 members: []。
  // 邀请/移除成员时 patch 会带 members，必须写回，否则会覆盖成空表。
  const merged: Project = {
    ...updated,
    members: patch.members !== undefined ? patch.members : current.members,
  };
  upsertProjectCache(merged);
  return merged;
}

/** 归档：将 status 设为 已归档（仅后端 API） */
export async function archive(id: string): Promise<Project> {
  return update(id, { status: '已归档' });
}

/** 删除项目（仅后端 API，fire-and-forget 版本）。 */
export function remove(id: string): void {
  projectApi.deleteProjectApi(id).then((res) => {
    if (res.ok) removeProjectCache(id);
  }).catch(() => {});
}

/** 删除项目（等待后端删除完成）。 */
export async function removeAsync(id: string): Promise<{ ok: boolean; error?: string }> {
  try {
    const res = await projectApi.deleteProjectApi(id);
    if (res.ok) {
      removeProjectCache(id);
      return { ok: true };
    }
    return { ok: false, error: res.error || '删除失败' };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

/** 本地更新缓存时间戳（仅 UI 展示）。 */
export function touchProject(projectId: string): void {
  const idx = projectCache.findIndex((p) => p.id === projectId);
  if (idx < 0) return;
  projectCache[idx] = { ...projectCache[idx], updatedAt: new Date().toISOString() };
}

/** 记录项目活动并更新项目 updatedAt（写操作成功后调用） */
export function recordProjectActivityAndTouch(
  projectId: string,
  type: ProjectActivityType,
  message: string,
  operator: string,
  refId?: string
): void {
  addProjectActivity({ projectId, type, refId, message, operator });
  touchProject(projectId);
}
