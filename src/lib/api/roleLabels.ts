import type { Role } from './types';

/** 团队管理员在 team_admins 中的团队 id；超管接口返回 null（前端用角色判断，不依赖列表） */
export type TeamAdminScope = string[] | null;

/** 与后端 project_permissions 对齐：编辑 = 归档等 PATCH；成员管理与编辑矩阵一致。 */
export function canEditProjectForUser(
  role: Role,
  project: { ownerId: string; teamId?: string | null },
  currentUserId: string,
  teamAdminTeamIds: TeamAdminScope | undefined
): boolean {
  if (role === 'SUPER_ADMIN') return true;
  if (role === 'USER') return false;
  const tid = (project.teamId ?? '').trim();
  const ownerMatch = (project.ownerId ?? '').trim() === (currentUserId ?? '').trim();
  if (role === 'OWNER') {
    return ownerMatch;
  }
  if (role === 'ADMIN') {
    if (teamAdminTeamIds === undefined) return false;
    if (teamAdminTeamIds === null) return false;
    if (!tid) return false;
    return teamAdminTeamIds.includes(tid);
  }
  return false;
}

export function canDeleteProjectForUser(
  role: Role,
  project: { ownerId: string; teamId?: string | null },
  currentUserId: string,
  teamAdminTeamIds: TeamAdminScope | undefined
): boolean {
  if (role === 'SUPER_ADMIN') return true;
  const tid = (project.teamId ?? '').trim();
  const ownerMatch = (project.ownerId ?? '').trim() === (currentUserId ?? '').trim();
  if (role === 'OWNER') {
    return ownerMatch;
  }
  if (role === 'ADMIN') {
    if (teamAdminTeamIds === undefined) return false;
    const scope = teamAdminTeamIds === null ? [] : teamAdminTeamIds;
    if (tid && scope.includes(tid)) return true;
    if (!tid && ownerMatch) return true;
    return false;
  }
  return false;
}

/** 与 canEditProjectForUser 同矩阵（负责人 / 团队辖区 / 超管） */
export function canManageProjectMembersForUser(
  role: Role,
  project: { ownerId: string; teamId?: string | null },
  currentUserId: string,
  teamAdminTeamIds: TeamAdminScope | undefined
): boolean {
  return canEditProjectForUser(role, project, currentUserId, teamAdminTeamIds);
}

/**
 * 将 API / 存库角色统一为四层 Role。
 * 兼容历史：ADMINISTRATOR→SUPER_ADMIN，MEMBER→USER。存库旧 ADMIN 由迁移脚本改为 OWNER。
 */
export function normalizeRole(raw: string | null | undefined): Role {
  const v = (raw || '').trim().toUpperCase();
  if (!v) return 'USER';

  if (v === 'SUPER_ADMIN') return 'SUPER_ADMIN';
  if (v === 'ADMIN') return 'ADMIN';
  if (v === 'OWNER') return 'OWNER';
  if (v === 'USER') return 'USER';

  if (v === 'ADMINISTRATOR') return 'SUPER_ADMIN';
  if (v === 'MEMBER') return 'USER';

  if (v === '成员' || v === '用户') return 'USER';
  if (v === '负责人' || v === '创建者') return 'OWNER';
  if (v === '管理员') return 'ADMIN';

  return 'USER';
}

export function isSuperAdmin(rawRole: string | null | undefined): boolean {
  return normalizeRole(rawRole) === 'SUPER_ADMIN';
}

export function isTeamAdminAccount(rawRole: string | null | undefined): boolean {
  return normalizeRole(rawRole) === 'ADMIN';
}

export function canCreateProject(rawRole: string | null | undefined): boolean {
  return isSuperAdmin(rawRole) || isTeamAdminAccount(rawRole);
}

export function canManageProjectMembers(rawRole: string | null | undefined): boolean {
  const r = normalizeRole(rawRole);
  return r === 'SUPER_ADMIN' || r === 'ADMIN' || r === 'OWNER';
}

/** 设备增删改（与后端 POST/PUT/DELETE /devices 一致）；USER 仅可查看列表与详情 */
export function canMutateDevices(rawRole: string | null | undefined): boolean {
  return canManageProjectMembers(rawRole);
}

/** 用户管理：可创建的角色（不含 SUPER_ADMIN，与 POST /users 一致） */
export type CreatableUserRole = 'ADMIN' | 'OWNER' | 'USER';

export function creatableUserRolesForActor(rawRole: string | null | undefined): CreatableUserRole[] {
  const r = normalizeRole(rawRole);
  if (r === 'SUPER_ADMIN') return ['ADMIN', 'OWNER', 'USER'];
  if (r === 'ADMIN') return ['OWNER', 'USER'];
  return [];
}

/** 修改用户角色时可选项（与 PATCH /users/{id}/role 一致） */
export function assignableUserRolesForEditor(
  editorRole: string | null | undefined,
  targetRole: string | null | undefined
): CreatableUserRole[] {
  const editor = normalizeRole(editorRole);
  const target = normalizeRole(targetRole);
  if (editor === 'SUPER_ADMIN') {
    return ['ADMIN', 'OWNER', 'USER'];
  }
  if (editor === 'ADMIN') {
    if (target === 'SUPER_ADMIN' || target === 'ADMIN') {
      return [];
    }
    return ['OWNER', 'USER'];
  }
  return [];
}

export function canEditOtherUserRole(
  editorRole: string | null | undefined,
  editorUserId: string,
  target: { id: string; role: string }
): boolean {
  if (!editorUserId || target.id === editorUserId) return false;
  return assignableUserRolesForEditor(editorRole, target.role).length > 0;
}

/**
 * 历史：负责人曾可对辖区内用户做账号级操作。已收紧为仅 SUPER_ADMIN / 团队 ADMIN 可走 /users 账号接口。
 * 负责人协作请使用项目成员管理；此处恒为 false，避免用户页误显账号操作按钮。
 */
export function canOwnerScopedUserMutations(
  _editorRole: string | null | undefined,
  _editorUserId: string,
  _target: { id: string; role: string }
): boolean {
  return false;
}

/** 团队管理员：可与后端 PATCH/DELETE 一致，对辖区内非超管/非团队管理员用户执行重置密码、禁用/启用、删除 */
export function canTeamAdminScopedUserMutations(
  editorRole: string | null | undefined,
  editorUserId: string,
  target: { id: string; role: string }
): boolean {
  if (normalizeRole(editorRole) !== 'ADMIN') return false;
  if (!editorUserId || target.id === editorUserId) return false;
  const tr = normalizeRole(target.role);
  if (tr === 'SUPER_ADMIN' || tr === 'ADMIN') return false;
  return true;
}

export function getRoleLabelKey(
  rawRole: string | null | undefined,
):
  | 'adminUsersPage.roleSuperAdmin'
  | 'adminUsersPage.roleAdmin'
  | 'adminUsersPage.roleOwner'
  | 'adminUsersPage.roleMember' {
  const role = normalizeRole(rawRole);
  if (role === 'SUPER_ADMIN') return 'adminUsersPage.roleSuperAdmin';
  if (role === 'ADMIN') return 'adminUsersPage.roleAdmin';
  if (role === 'OWNER') return 'adminUsersPage.roleOwner';
  return 'adminUsersPage.roleMember';
}
