'use client';

import ResourceLibraryPage from '@/components/workspace/ResourceLibraryPage';

export default function ScenesPage() {
  return (
    <ResourceLibraryPage
      title="仿真场景"
      subtitle="仿真场景、工位布局与物理环境配置。"
      assetTypes="scene"
      createLabel="新建场景"
      importLabel="导入场景"
    />
  );
}
