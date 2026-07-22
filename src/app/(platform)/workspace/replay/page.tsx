'use client';

import { Suspense, useCallback, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useI18n } from '@/components/common/I18nProvider';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { ReplayWorkbench } from '@/components/workspace/replay/ReplayWorkbench';
import { CableEvalReplayPanel } from '@/components/workspace/replay/CableEvalReplayPanel';
import { DualArmEvalReplayPanel } from '@/components/workspace/replay/DualArmEvalReplayPanel';
import { IsaacEvalReplayPanel } from '@/components/workspace/replay/IsaacEvalReplayPanel';
import { DUAL_ARM_CABLE_TASK_TYPE } from '@/lib/workspace/dualArmCable';
import {
  buildDualArmEvalReportHref,
} from '@/lib/workspace/dualArmEvaluation';
import { buildIsaacEvalReportHref } from '@/lib/workspace/isaacBlockStacking';
import {
  hasReplayUrlTarget,
  resolveReplayPageKind,
} from '@/lib/workspace/replayPageKind';
import { isUnifiedReplayWorkbenchMode } from '@/lib/workspace/datasetReplayHref';
import { buildReplayViewModel } from '@/lib/workspace/replayViewModel';
import type { ReplayAdapterResult } from '@/lib/workspace/replayAdapters';

function ReplayPageContent() {
  const { t } = useI18n();
  const searchParams = useSearchParams();
  const taskType = searchParams.get('taskType') ?? undefined;
  const evalId = searchParams.get('evalId') ?? undefined;
  const evalJobId = searchParams.get('evalJobId') ?? undefined;
  const jobId = searchParams.get('jobId') ?? undefined;
  const datasetId = searchParams.get('datasetId') ?? undefined;
  const replayJobId = searchParams.get('replayJobId') ?? undefined;
  const replayType = searchParams.get('replayType') ?? undefined;
  const episode = Number(searchParams.get('episode') ?? '0') || 0;

  const sourceParams = {
    replayType,
    jobId,
    evalId,
    evalJobId,
    datasetId,
  };

  const viewModel = buildReplayViewModel(sourceParams, {
    replayDataTitle: t('workspacePages.replayDataTitle'),
    replayDataSubtitle: t('workspacePages.replayDataSubtitle'),
    replayEvalTitle: t('workspacePages.replayEvalTitle'),
    replayEvalSubtitle: t('workspacePages.replayEvalSubtitle'),
  });

  const replayKind = resolveReplayPageKind(sourceParams);
  const hasUrlTarget = hasReplayUrlTarget({
    jobId,
    evalId,
    evalJobId,
    datasetId,
    taskType,
    replayJobId,
  });
  const unifiedReplay = isUnifiedReplayWorkbenchMode({
    replayType,
    taskType,
    jobId,
    datasetId,
    replayJobId,
    evalId,
    evalJobId,
    hasUrlTarget,
  });

  const dualArmEvalJobId =
    evalJobId?.startsWith('eval_')
      ? evalJobId
      : evalId?.startsWith('eval_')
        ? evalId
        : undefined;

  const dualArmEvalReplay =
    viewModel.sourceKind === 'evaluation' &&
    taskType === DUAL_ARM_CABLE_TASK_TYPE &&
    Boolean(dualArmEvalJobId);

  const isaacEvalReplay =
    viewModel.sourceKind === 'evaluation' && Boolean(evalId?.startsWith('isaac_eval_'));

  const cableEvalJobId =
    evalId?.startsWith('ct_eval_') || evalId?.startsWith('eval_joint_dp_')
      ? evalId
      : evalJobId?.startsWith('ct_eval_') || evalJobId?.startsWith('eval_joint_dp_')
        ? evalJobId
        : undefined;

  const cableEvalReplay =
    viewModel.sourceKind === 'evaluation' &&
    Boolean(cableEvalJobId ?? (taskType === 'cable_threading' && evalId));

  const isDatasetReplay = viewModel.sourceKind === 'dataset';
  const [resolvedHeader, setResolvedHeader] = useState<{
    title: string;
    subtitle: string;
  } | null>(null);

  const handleContentResolved = useCallback((adapter: ReplayAdapterResult) => {
    if (!adapter.pageTitle) return;
    setResolvedHeader({
      title: adapter.pageTitle,
      subtitle: adapter.pageSubtitle || viewModel.subtitle,
    });
  }, [viewModel.subtitle]);

  const pageTitle = resolvedHeader?.title ?? viewModel.title;
  const pageSubtitle = resolvedHeader?.subtitle ?? viewModel.subtitle;

  const headerActions =
    isaacEvalReplay && evalId ? (
      <>
        <Link href={buildIsaacEvalReportHref({ evalJobId: evalId })}>
          <SecondaryButton>查看报告</SecondaryButton>
        </Link>
        <Link href={viewModel.returnPath}>
          <SecondaryButton>{viewModel.returnLabel}</SecondaryButton>
        </Link>
      </>
    ) : dualArmEvalReplay && dualArmEvalJobId ? (
      <>
        <Link href={buildDualArmEvalReportHref({ evalJobId: dualArmEvalJobId })}>
          <SecondaryButton>查看报告</SecondaryButton>
        </Link>
        <Link href={viewModel.returnPath}>
          <SecondaryButton>{viewModel.returnLabel}</SecondaryButton>
        </Link>
      </>
    ) : (
      <Link href={viewModel.returnPath}>
        <SecondaryButton>{isDatasetReplay ? '返回' : viewModel.returnLabel}</SecondaryButton>
      </Link>
    );

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={pageTitle}
        subtitle={pageSubtitle}
        actions={headerActions}
      />
      {isaacEvalReplay && evalId ? (
        <IsaacEvalReplayPanel evalJobId={evalId} episode={episode} replayKind={replayKind} />
      ) : cableEvalReplay && (cableEvalJobId ?? evalId) ? (
        <CableEvalReplayPanel
          evalJobId={cableEvalJobId ?? evalId!}
          replayKind={replayKind}
          initialEpisode={episode > 0 ? episode : undefined}
        />
      ) : dualArmEvalReplay && dualArmEvalJobId ? (
        <DualArmEvalReplayPanel evalJobId={dualArmEvalJobId} episode={episode} replayKind={replayKind} />
      ) : (
        <ReplayWorkbench
          initialTaskType={taskType}
          initialEvalId={evalId}
          initialEvalJobId={evalJobId}
          initialJobId={jobId}
          initialDatasetId={datasetId}
          initialReplayJobId={replayJobId}
          initialReplayType={replayType}
          replayKind={replayKind}
          replaySourceKind={viewModel.sourceKind}
          hasUrlTarget={hasUrlTarget}
          unifiedReplay={unifiedReplay}
          returnPath={viewModel.returnPath}
          returnLabel={viewModel.returnLabel}
          onContentResolved={unifiedReplay ? handleContentResolved : undefined}
        />
      )}
    </ModulePageContainer>
  );
}

export default function ReplayPage() {
  return (
    <Suspense fallback={null}>
      <ReplayPageContent />
    </Suspense>
  );
}
