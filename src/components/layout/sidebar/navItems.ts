/**
 * Sidebar 菜单配置
 */

import type { LucideIcon } from 'lucide-react';
import {
  Home,
  LayoutDashboard,
  Settings,
  FolderOpen,
  Cpu,
  User,
  UsersRound,
  Sparkles,
  Play,
  Database,
  LineChart,
  FlaskConical,
  Library,
  BookOpen,
  GraduationCap,
} from 'lucide-react';

export interface NavItem {
  path: string;
  label: string;
  /** i18n path like "sidebar.data" */
  labelKey?: string;
  icon: LucideIcon;
  children?: string[];
  section: 'main' | 'system';
  /** 禁用态：不跳转、不选中、灰显、cursor not-allowed */
  disabled?: boolean;
  /** 禁用原因，悬停时展示（如「暂未开放」） */
  disabledReason?: string;
}

// IconRail 菜单项（最左侧），labelKey 用于 i18n tooltip
export const iconRailItems: Array<{ path: string; label: string; labelKey?: string; icon: LucideIcon }> = [
  { path: '/workspace', label: '概览', labelKey: 'sidebar.overview', icon: Home },
  { path: '/workspace', label: '工作台', labelKey: 'sidebar.workspace', icon: LayoutDashboard },
  { path: '/admin', label: '管理', labelKey: 'sidebar.admin', icon: Settings },
];

/**
 * Phase 1 前数据平台工作台菜单（保留代码，菜单中已隐藏）
 */
export const legacyDataPlatformSideMenuItems: NavItem[] = [
  { path: '/data', label: '数据', labelKey: 'sidebar.data', icon: Library, section: 'main' },
  {
    path: '/collect',
    label: '采集',
    labelKey: 'sidebar.collect',
    icon: Play,
    children: ['/collect/tasks', '/collect/jobs', '/collect/realtime', '/collect/quality'],
    section: 'main',
  },
  { path: '/label', label: '标注', labelKey: 'sidebar.annotate', icon: Sparkles, section: 'main' },
  { path: '/convert', label: '转换', labelKey: 'sidebar.transform', icon: LineChart, section: 'main' },
  {
    path: '/clean',
    label: '清洗',
    labelKey: 'sidebar.clean',
    icon: FlaskConical,
    section: 'main',
    disabled: true,
    disabledReason: '暂未开放',
  },
];

/**
 * 工作台 SideMenu（数据 / 训练 / 评测 / 资源）
 * 仿真相关路由保留但不展示：/workspace/simulation、/workspace/simulation/console 等
 */
export const workspaceSideMenuItems: NavItem[] = [
  {
    path: '/workspace/data',
    label: '数据',
    labelKey: 'sidebarWorkspace.data',
    icon: Database,
    section: 'main',
  },
  {
    path: '/workspace/training',
    label: '训练',
    labelKey: 'sidebarWorkspace.training',
    icon: GraduationCap,
    section: 'main',
  },
  {
    path: '/workspace/evaluation',
    label: '评测',
    labelKey: 'sidebarWorkspace.evaluation',
    icon: LineChart,
    section: 'main',
  },
  {
    path: '/workspace/resources',
    label: '资源',
    labelKey: 'sidebarWorkspace.resources',
    icon: Library,
    children: [
      '/workspace/resources/task-templates',
      '/workspace/resources/datasets',
      '/workspace/resources/model-assets',
      '/workspace/resources/metrics',
      '/workspace/resources/scenes',
      '/workspace/resources/assets',
      '/workspace/resources/robots',
      '/workspace/resources/policies',
      '/workspace/resources/model-types',
      '/workspace/resources/physics-proxies',
      '/workspace/resources/craft-config',
    ],
    section: 'main',
  },
];

/** 仿真菜单（菜单中已隐藏，路由保留） */
export const legacySimulationSideMenuItem: NavItem = {
  path: '/workspace/simulation',
  label: '仿真',
  labelKey: 'sidebarWorkspace.simulation',
  icon: Play,
  children: [
    '/workspace/simulation/console',
    '/workspace/task-build',
    '/workspace/replay',
    '/workspace/experiments',
  ],
  section: 'main',
};

/** @deprecated 使用 workspaceSideMenuItems；保留别名避免旧引用断裂 */
export const dataSideMenuItems = workspaceSideMenuItems;

// 管理域 SideMenu 菜单项（Phase 1 菜单：用户 / 运行节点 / 日志；项目、系统配置从菜单隐藏，代码保留）
/** 项目管理（菜单隐藏，路由 /admin/projects 保留） */
export const adminProjectsNavItem: NavItem = {
  path: '/admin/projects',
  label: '项目',
  labelKey: 'sidebarAdmin.projects',
  icon: FolderOpen,
  section: 'main',
};

/** 运行节点（Phase 1 从管理菜单隐藏，路由 /devices 保留） */
export const adminRuntimeNodesNavItem: NavItem = {
  path: '/devices',
  label: '运行节点',
  labelKey: 'sidebarAdmin.devices',
  icon: Cpu,
  section: 'main',
};

/** @deprecated 菜单不再使用；保留导出避免旧引用断裂 */
export const adminSideMenuItems: NavItem[] = [adminProjectsNavItem, adminRuntimeNodesNavItem];

/** 用户管理（仅平台管理员可见，与 layout 组合顺序一致） */
export const adminUsersNavItem: NavItem = {
  path: '/admin/users',
  label: '用户',
  labelKey: 'sidebarAdmin.users',
  icon: User,
  section: 'main',
};

/** 团队（Phase 1 从菜单隐藏，代码保留） */
export const adminTeamsNavItem: NavItem = {
  path: '/admin/teams',
  label: '团队',
  labelKey: 'sidebarAdmin.teams',
  icon: UsersRound,
  section: 'main',
};

/** 审计日志（仅平台管理员可见） */
export const adminAuditNavItem: NavItem = {
  path: '/admin/audit',
  label: '日志',
  labelKey: 'sidebarAdmin.audit',
  icon: BookOpen,
  section: 'main',
};

/** 系统配置占位 */
export const adminSettingsNavItem: NavItem = {
  path: '/settings',
  label: '系统配置',
  labelKey: 'sidebarAdmin.settings',
  icon: Settings,
  section: 'main',
};

/** @deprecated 请使用 layout 内按顺序组合 adminUsersNavItem / adminAuditNavItem */
export const superAdminMenuItems: NavItem[] = [adminUsersNavItem, adminAuditNavItem];
