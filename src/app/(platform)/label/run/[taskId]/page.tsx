import { redirect } from 'next/navigation';

/** 遗留标注执行页，已迁移至任务模板 */
export default function LegacyRouteRedirectPage() {
  redirect('/workspace/resources/task-templates');
}
