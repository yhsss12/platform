'use client';

import ResourceLibraryPage from '@/components/workspace/ResourceLibraryPage';

export default function PoliciesPage() {
  return (
    <ResourceLibraryPage
      title="策略资产"
      subtitle="策略模型、推理入口与 checkpoint 配置。"
      assetTypes="policy"
      createLabel="注册策略"
      importLabel="导入 checkpoint"
    />
  );
}
