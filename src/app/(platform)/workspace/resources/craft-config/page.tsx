'use client';

import ResourceLibraryPage from '@/components/workspace/ResourceLibraryPage';

export default function CraftConfigPage() {
  return (
    <ResourceLibraryPage
      title="工艺配置"
      subtitle="任务工艺参数、运行入口与任务级配置资源。"
      assetTypes="task"
      createLabel="新建工艺配置"
      importLabel="导入工艺配置"
    />
  );
}
