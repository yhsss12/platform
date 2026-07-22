/**
 * 项目 API（projects 表存 backend/data/assets/assets.db）
 */
import { apiGet, apiPost, apiPatch, apiDelete, type ApiResponse } from '@/features/data-platform/api/client';

export interface ProjectApiItem {
  id: string;
  name: string;
  description: string | null;
  tags: string[];
  status: string;
  owner_id: string | null;
  team_id?: string | null;
  created_at: string;
  updated_at: string;
  /** 当前用户在 project_members 是否有记录（列表/详情由后端按登录用户填充） */
  viewer_is_project_member?: boolean;
  viewer_is_project_owner?: boolean;
  /** 与 GET .../members 列表条数一致（project_members + 负责人展示行） */
  member_count?: number;
}

export interface ProjectListPayload {
  items: ProjectApiItem[];
  total: number;
  /** 仅当请求 with_stats=true 时存在：project_id -> { label_task_count, dataset_count } */
  stats?: Record<string, { label_task_count: number; dataset_count: number }>;
}

/** GET /api/projects/permissions-context — 团队管理员辖区 team_id 列表；超管为 null */
export interface ProjectPermissionsContextPayload {
  team_admin_team_ids: string[] | null;
}

export async function fetchProjectPermissionsContext(): Promise<
  ApiResponse<ProjectPermissionsContextPayload>
> {
  return apiGet<ProjectPermissionsContextPayload>('/api/projects/permissions-context');
}

export async function fetchProjects(
  status?: string,
  withStats?: boolean,
  teamId?: string | null
): Promise<ApiResponse<ProjectListPayload>> {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  if (withStats) params.set('with_stats', 'true');
  const tid = (teamId ?? '').trim();
  if (tid) params.set('team_id', tid);
  const q = params.toString();
  return apiGet<ProjectListPayload>(`/api/projects${q ? `?${q}` : ''}`);
}

export async function fetchProject(projectId: string): Promise<ApiResponse<ProjectApiItem>> {
  return apiGet<ProjectApiItem>(`/api/projects/${encodeURIComponent(projectId)}`);
}

export interface ProjectCreatePayload {
  name: string;
  description?: string | null;
  tags?: string[] | null;
  status?: string;
  owner_id?: string | null;
  team_id?: string | null;
  id?: string | null;
}

export async function createProjectApi(body: ProjectCreatePayload): Promise<ApiResponse<ProjectApiItem>> {
  return apiPost<ProjectApiItem>('/api/projects', {
    name: body.name,
    description: body.description ?? undefined,
    tags: body.tags ?? undefined,
    status: body.status ?? '进行中',
    owner_id: body.owner_id ?? undefined,
    team_id: body.team_id ?? undefined,
    id: body.id ?? undefined,
  });
}

export interface ProjectUpdatePayload {
  name?: string;
  description?: string | null;
  tags?: string[] | null;
  status?: string;
  owner_id?: string | null;
}

export async function updateProjectApi(
  projectId: string,
  body: ProjectUpdatePayload
): Promise<ApiResponse<ProjectApiItem>> {
  return apiPatch<ProjectApiItem>(`/api/projects/${encodeURIComponent(projectId)}`, body);
}

export async function deleteProjectApi(projectId: string): Promise<ApiResponse<{ deleted: string }>> {
  return apiDelete<{ deleted: string }>(`/api/projects/${encodeURIComponent(projectId)}`);
}

/** 按项目 id 统计：标注任务数、数据资产数（表与表通过 project_id 关联） */
export interface ProjectStatsPayload {
  label_task_count: number;
  dataset_count: number;
}

export async function fetchProjectStats(
  projectId: string
): Promise<ApiResponse<ProjectStatsPayload>> {
  return apiGet<ProjectStatsPayload>(`/api/projects/${encodeURIComponent(projectId)}/stats`);
}

export interface ProjectLabelTaskItem {
  id: number;
  task_id: string;
  name: string;
  dataset_path: string;
  project_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectDatasetsPayload {
  items: Array<{
    id: number;
    dataset_id: string | null;
    filename: string;
    format: string;
    file_path: string;
    project_id: string | null;
    project_name: string | null;
    created_at: string;
  }>;
  total: number;
  page: number;
  page_size: number;
}

export async function fetchProjectLabelTasks(projectId: string): Promise<
  ApiResponse<{ items: ProjectLabelTaskItem[]; total: number }>
> {
  return apiGet<{ items: ProjectLabelTaskItem[]; total: number }>(
    `/api/projects/${encodeURIComponent(projectId)}/label-tasks`
  );
}

export async function fetchProjectDatasets(
  projectId: string,
  page = 1,
  page_size = 20
): Promise<ApiResponse<ProjectDatasetsPayload>> {
  return apiGet<ProjectDatasetsPayload>(
    `/api/projects/${encodeURIComponent(projectId)}/datasets?page=${page}&page_size=${page_size}`
  );
}

export interface ProjectMemberItem {
  user_id: string;
  username: string;
  /** 列表展示用：Owner=项目负责人；Admin/Member 由账号角色映射，非独立存库字段 */
  role: 'Owner' | 'Member' | 'Admin';
}

export async function fetchProjectMembers(projectId: string): Promise<ApiResponse<{ items: ProjectMemberItem[]; total: number }>> {
  return apiGet<{ items: ProjectMemberItem[]; total: number }>(
    `/api/projects/${encodeURIComponent(projectId)}/members`
  );
}

export async function addProjectMember(projectId: string, userId: string): Promise<ApiResponse<{ added: boolean; user_id: string }>> {
  return apiPost<{ added: boolean; user_id: string }>(
    `/api/projects/${encodeURIComponent(projectId)}/members`,
    { user_id: userId }
  );
}

/** POST /api/projects/:id/users — 在项目所属团队下新建 USER，并写入 team_users + project_members */
export interface CreateProjectUserResult {
  user_id: string;
  account_id: string;
  username: string;
}

export async function createProjectUser(
  projectId: string,
  body: { username: string; password: string }
): Promise<ApiResponse<CreateProjectUserResult>> {
  return apiPost<CreateProjectUserResult>(
    `/api/projects/${encodeURIComponent(projectId)}/users`,
    { username: body.username, password: body.password }
  );
}

export async function removeProjectMember(projectId: string, userId: string): Promise<ApiResponse<{ removed: boolean; user_id: string }>> {
  return apiDelete<{ removed: boolean; user_id: string }>(
    `/api/projects/${encodeURIComponent(projectId)}/members/${encodeURIComponent(userId)}`
  );
}
