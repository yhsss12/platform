import { redirect } from 'next/navigation';

/** 遗留数据转换页，功能已收敛至工作台数据中心 */
export default function LegacyRouteRedirectPage() {
  redirect('/workspace/data');
}
