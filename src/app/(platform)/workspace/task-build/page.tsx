'use client';

import { useI18n } from '@/components/common/I18nProvider';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { TaskBuildPathSelector } from '@/components/workspace/taskBuild/TaskBuildUi';

export default function TaskBuildHubPage() {
  const { t } = useI18n();

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={t('workspacePages.taskBuildTitle')}
        subtitle={t('workspacePages.taskBuildSubtitle')}
      />
      <TaskBuildPathSelector />
    </ModulePageContainer>
  );
}
