import { redirect } from 'next/navigation';

/** 遗留采集执行页，已迁移至工作台 */
export default function LegacyRouteRedirectPage() {
  redirect('/workspace/data');
}
