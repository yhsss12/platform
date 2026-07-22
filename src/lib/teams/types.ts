/**
 * 团队管理前端类型（原型阶段，后续可对接 API）
 */

export type TeamStatus = 'active' | 'inactive';

export interface Team {
  id: string;
  name: string;
  code: string;
  description: string;
  status: TeamStatus;
  adminCount: number;
  /** team_users 行数（普通成员，不含管理员） */
  userCount: number;
  projectCount: number;
  createdAt: string;
  createdBy: string;
}

export type TeamAdminMemberStatus = 'active' | 'inactive';

export interface TeamAdmin {
  id: string;
  /** 平台用户 ID，用于调用移除接口 */
  userId: string;
  username: string;
  displayName: string;
  email: string;
  status: TeamAdminMemberStatus;
  teamId: string;
  /** 主库账号角色（如 OWNER/USER），团队接口返回；与项目内成员角色列无关 */
  platformRole?: string;
}

/** 团队普通成员（与 TeamAdmin 字段一致，便于复用表格） */
export type TeamUserRow = TeamAdmin;

export type TeamProjectStatus = '进行中' | '已暂停' | '已归档';

export interface TeamProject {
  id: string;
  teamId: string;
  name: string;
  /** 项目负责人（与团队管理员区分） */
  owner: string;
  members: number;
  assets: number;
  updatedAt: string;
  status: TeamProjectStatus;
}

/** 供「添加管理员」下拉的 mock 用户候选项（非接口用户） */
export interface TeamAdminCandidateUser {
  id: string;
  username: string;
  displayName: string;
  email: string;
  status: TeamAdminMemberStatus;
}
