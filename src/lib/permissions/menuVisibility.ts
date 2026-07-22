/**
 * 左侧管理域菜单显隐（权限体系第一阶段）。
 * 与后端四层角色一致：SUPER_ADMIN / ADMIN / OWNER / USER
 */
import type { Role } from '@/lib/api/types';
import { normalizeRole } from '@/lib/api/roleLabels';

export function canSeeProjectMenu(role: string | null | undefined): boolean {
  const r = normalizeRole(role);
  return r === 'ADMIN' || r === 'OWNER' || r === 'USER' || r === 'SUPER_ADMIN';
}

export function canSeeDeviceMenu(role: string | null | undefined): boolean {
  return canSeeProjectMenu(role);
}

/** 用户管理页入口（与 GET /users 一致：仅 SUPER_ADMIN / 团队 ADMIN；OWNER 用项目成员管理） */
export function canSeeUserMenu(role: string | null | undefined): boolean {
  const r = normalizeRole(role);
  return r === 'SUPER_ADMIN' || r === 'ADMIN';
}

/** 团队管理页入口 */
export function canSeeTeamMenu(role: string | null | undefined): boolean {
  return normalizeRole(role) === 'SUPER_ADMIN';
}

/** 审计日志页入口 */
export function canSeeLogMenu(role: string | null | undefined): boolean {
  const r = normalizeRole(role);
  return r === 'SUPER_ADMIN' || r === 'ADMIN';
}

/** 管理域中是否出现「项目」「设备」成组菜单（含 SUPER_ADMIN，与 canSeeProjectMenu 一致） */
export function canSeeProjectDeviceAdminSection(role: string | null | undefined): boolean {
  return canSeeProjectMenu(role);
}

export function normalizedPlatformRole(role: string | null | undefined): Role {
  return normalizeRole(role);
}
