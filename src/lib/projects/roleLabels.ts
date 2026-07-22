/** 角色 UI 显示映射（仅展示用，不改变存储）。兼容已有中文/其它值：未命中则原样显示。 */
export const roleLabelMap: Record<string, string> = {
  Owner: '创建者',
  Admin: '负责人',
  Member: '成员',
  Viewer: '访客',
  // 兼容已存中文
  创建者: '创建者',
  管理员: '管理员',
  负责人: '负责人',
  成员: '成员',
  用户: '成员',
  访客: '访客',
};

export function getRoleLabel(role: string): string {
  return roleLabelMap[role] ?? role;
}
