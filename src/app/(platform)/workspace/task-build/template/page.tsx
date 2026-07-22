'use client';

import { Suspense } from 'react';
import { StandardTemplateBuildFlow } from '@/components/workspace/taskBuild/StandardTemplateBuildFlow';

export default function TaskBuildTemplatePage() {
  return (
    <Suspense fallback={null}>
      <StandardTemplateBuildFlow />
    </Suspense>
  );
}
