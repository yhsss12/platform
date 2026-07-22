import { redirect } from 'next/navigation';

/** 旧 mock 向导入口 → 双路径任务构建 hub */
export default function LegacyTaskGenerationRedirect() {
  redirect('/workspace/task-build');
}
