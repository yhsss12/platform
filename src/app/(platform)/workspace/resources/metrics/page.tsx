'use client';

import ResourceLibraryPage from '@/components/workspace/ResourceLibraryPage';

export default function MetricsPage() {
  return (
    <ResourceLibraryPage
      title="评测指标库"
      subtitle="管理平台评测指标定义，维护各任务模板下的字段映射与计算方式。"
      assetTypes="metric"
      createLabel="新建指标"
      importLabel="导入指标"
    />
  );
}
