/**
 * Workspace 演示数据展示策略（Phase 1）。
 * 默认仅展示真实 job / manifest；演示数据需显式开启环境变量。
 */

/** 是否允许在 UI 中展示演示数据（需显式 NEXT_PUBLIC_WORKSPACE_SHOW_DEMO=true） */
export function isWorkspaceDemoEnabled(): boolean {
  return process.env.NEXT_PUBLIC_WORKSPACE_SHOW_DEMO === 'true';
}

/** 当前是否应合并演示 mock 数据到列表 */
export function shouldShowWorkspaceDemo(): boolean {
  return isWorkspaceDemoEnabled();
}

/** @deprecated 使用 shouldShowWorkspaceDemo */
export function isWorkspaceDemoRealOnly(): boolean {
  return !isWorkspaceDemoEnabled();
}

/** @deprecated 使用 shouldShowWorkspaceDemo */
export function shouldShowWorkspaceMockByDefault(): boolean {
  return shouldShowWorkspaceDemo();
}
