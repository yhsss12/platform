'use client';

import ResourceLibraryPage from '@/components/workspace/ResourceLibraryPage';

export default function RobotsPage() {
  return (
    <ResourceLibraryPage
      title="机器人库"
      subtitle="机器人模型、运动学与驱动配置，用于仿真运行与策略部署。"
      assetTypes="robot"
      createLabel="添加机器人"
      importLabel="导入模型"
    />
  );
}
