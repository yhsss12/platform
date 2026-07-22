import { redirect } from 'next/navigation';

/** 旧版 data-platform 概览已废弃，统一进入 workspace 数据中心。 */
export default function OverviewPage() {
  redirect('/workspace/data');
}
