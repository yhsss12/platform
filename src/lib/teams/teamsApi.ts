/**
 * 团队管理 API（/api/teams）
 */
import {
  apiGet,
  apiPost,
  apiPatch,
  apiDelete,
  type ApiResponse,
} from '@/features/data-platform/api/client';
import type { Team, TeamAdmin, TeamAdminCandidateUser, TeamProject, TeamStatus, TeamUserRow } from './types';

/** 后端列表项（snake_case） */
interface TeamApiRow {
  id: string;
  name: string;
  code: string;
  description: string;
  status: string;
  admin_count: number;
  user_count: number;
  project_count: number;
  created_at: string;
  created_by?: string | null;
}

interface TeamListData {
  items: TeamApiRow[];
  total: number;
}

function mapTeam(row: TeamApiRow): Team {
  const st = (row.status || 'active').toLowerCase();
  return {
    id: row.id,
    name: row.name,
    code: row.code,
    description: row.description ?? '',
    status: st === 'inactive' ? 'inactive' : 'active',
    adminCount: row.admin_count,
    userCount: row.user_count ?? 0,
    projectCount: row.project_count,
    createdAt: row.created_at,
    createdBy: row.created_by ?? '',
  };
}

function pickStr(row: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = row[k];
    if (v != null && String(v).trim() !== '') return String(v);
  }
  return '';
}

/** 兼容 snake_case / camelCase（代理或序列化差异） */
function mapAdmin(row: Record<string, unknown>): TeamAdmin {
  const userId = pickStr(row, 'user_id', 'userId');
  const teamId = pickStr(row, 'team_id', 'teamId');
  const username = pickStr(row, 'username') || userId || '—';
  const displayName = pickStr(row, 'display_name', 'displayName') || username;
  const email = pickStr(row, 'email');
  const statusRaw = pickStr(row, 'status').toLowerCase();
  const id = pickStr(row, 'id') || `row-${userId || Math.random()}`;
  const platformRole = pickStr(row, 'platform_role', 'platformRole');
  return {
    id,
    userId,
    username,
    displayName,
    email,
    status: statusRaw === 'inactive' ? 'inactive' : 'active',
    teamId,
    ...(platformRole ? { platformRole } : {}),
  };
}

function mapProject(row: {
  id: string;
  team_id: string;
  name: string;
  owner: string;
  members: number;
  assets: number;
  updated_at: string;
  status: string;
}): TeamProject {
  return {
    id: row.id,
    teamId: row.team_id,
    name: row.name,
    owner: row.owner,
    members: row.members,
    assets: row.assets,
    updatedAt: row.updated_at,
    status: row.status as TeamProject['status'],
  };
}

export async function fetchTeams(): Promise<Team[]> {
  const res = await apiGet<TeamListData>('/api/teams');
  if (!res.ok || !res.data?.items) {
    throw new Error(res.error || '加载团队列表失败');
  }
  return res.data.items.map(mapTeam);
}

export async function createTeamApi(body: {
  name: string;
  code: string;
  description?: string;
  status: TeamStatus;
}): Promise<Team> {
  const res = await apiPost<TeamApiRow>('/api/teams', {
    name: body.name,
    code: body.code,
    description: body.description || undefined,
    status: body.status,
  });
  if (!res.ok || !res.data) {
    throw new Error(res.error || '创建团队失败');
  }
  return mapTeam(res.data);
}

export async function patchTeamApi(
  teamId: string,
  body: { name?: string; description?: string; status?: TeamStatus },
): Promise<Team> {
  const res = await apiPatch<TeamApiRow>(`/api/teams/${encodeURIComponent(teamId)}`, {
    name: body.name,
    description: body.description,
    status: body.status,
  });
  if (!res.ok || !res.data) {
    throw new Error(res.error || '更新团队失败');
  }
  return mapTeam(res.data);
}

export async function fetchTeamAdmins(teamId: string): Promise<TeamAdmin[]> {
  const res = await apiGet<{ items?: unknown[] }>(
    `/api/teams/${encodeURIComponent(teamId)}/admins`,
  );
  if (!res.ok) {
    throw new Error(res.error || '加载团队管理员失败');
  }
  const raw = res.data?.items;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map((row) => mapAdmin(row as Record<string, unknown>));
}

export async function addTeamAdminApi(teamId: string, userId: string): Promise<void> {
  const res = await apiPost(`/api/teams/${encodeURIComponent(teamId)}/admins`, { user_id: userId });
  if (!res.ok) {
    throw new Error(res.error || '添加管理员失败');
  }
}

export async function removeTeamAdminApi(teamId: string, userId: string): Promise<void> {
  const res = await apiDelete(
    `/api/teams/${encodeURIComponent(teamId)}/admins/${encodeURIComponent(userId)}`,
  );
  if (!res.ok) {
    throw new Error(res.error || '移除管理员失败');
  }
}

export async function fetchTeamUsers(teamId: string): Promise<TeamUserRow[]> {
  const res = await apiGet<{ items?: unknown[] }>(
    `/api/teams/${encodeURIComponent(teamId)}/users`,
  );
  if (!res.ok) {
    throw new Error(res.error || '加载团队成员失败');
  }
  const raw = res.data?.items;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map((row) => mapAdmin(row as Record<string, unknown>));
}

export async function addTeamUserApi(teamId: string, userId: string): Promise<void> {
  const res = await apiPost(`/api/teams/${encodeURIComponent(teamId)}/users`, { user_id: userId });
  if (!res.ok) {
    throw new Error(res.error || '添加团队成员失败');
  }
}

export async function removeTeamUserApi(teamId: string, userId: string): Promise<void> {
  const res = await apiDelete(
    `/api/teams/${encodeURIComponent(teamId)}/users/${encodeURIComponent(userId)}`,
  );
  if (!res.ok) {
    throw new Error(res.error || '移除团队成员失败');
  }
}

export interface DeleteTeamSummary {
  team_id: string;
  team_name: string;
  projects_deleted: number;
  assets_deleted_rows: number;
  team_users_removed: number;
  team_admins_removed: number;
  users_deleted: number;
  storage_warnings: string[];
}

/** POST /api/teams/:id/delete — 物理删除团队（仅超管；须传 confirmation_name） */
export async function deleteTeamHardApi(teamId: string, confirmationName: string): Promise<DeleteTeamSummary> {
  const res = await apiPost<DeleteTeamSummary>(
    `/api/teams/${encodeURIComponent(teamId)}/delete`,
    { confirmation_name: confirmationName },
  );
  if (!res.ok || !res.data) {
    throw new Error(res.error || '删除团队失败');
  }
  return res.data;
}

export async function fetchTeamProjects(teamId: string): Promise<TeamProject[]> {
  const res = await apiGet<{ items: Parameters<typeof mapProject>[0][] }>(
    `/api/teams/${encodeURIComponent(teamId)}/projects`,
  );
  if (!res.ok || !res.data?.items) {
    throw new Error(res.error || '加载团队项目失败');
  }
  return res.data.items.map(mapProject);
}

/**
 * 可添加为团队管理员的候选（平台 role=ADMIN，且可选排除指定团队已有 team_admins）。
 */
export async function fetchTeamAdminCandidateOptions(excludeTeamId?: string): Promise<TeamAdminCandidateUser[]> {
  const q = excludeTeamId?.trim()
    ? `?exclude_team_id=${encodeURIComponent(excludeTeamId.trim())}`
    : '';
  const res = await apiGet<{ items?: { id: string; username: string }[] }>(
    `/api/teams/meta/user-options${q}`,
  );
  if (!res.ok) {
    throw new Error(res.error || '加载用户列表失败');
  }
  const raw = res.data?.items;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map((u) => ({
    id: u.id,
    username: u.username,
    displayName: u.username,
    email: '',
    status: 'active' as const,
  }));
}

export type { ApiResponse };
