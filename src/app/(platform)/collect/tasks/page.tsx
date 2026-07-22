import { redirect } from 'next/navigation';

/** 遗留采集任务列表，已迁移至工作台 */
export default function LegacyRouteRedirectPage() {
  redirect('/workspace/data');
}
