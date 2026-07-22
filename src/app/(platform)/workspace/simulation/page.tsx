import { redirect } from 'next/navigation';

/** 仿真中心入口已合并至数据中心；保留路由供内部跳转与旧链接兼容 */
export default function WorkspaceSimulationRedirectPage() {
  redirect('/workspace/data');
}
