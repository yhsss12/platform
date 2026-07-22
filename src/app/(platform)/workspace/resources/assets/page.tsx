'use client';

import { useI18n } from '@/components/common/I18nProvider';
import ResourceLibraryPage from '@/components/workspace/ResourceLibraryPage';

export default function AssetsPage() {
  const { t } = useI18n();
  return (
    <ResourceLibraryPage
      title={t('workspacePages.manipulationObjectLibraryTitle')}
      subtitle="管理跨场景可交互操作对象，覆盖精密制造、线缆操作、生化实验与通用操作任务。"
      assetTypes={['object', 'end_effector']}
    />
  );
}
