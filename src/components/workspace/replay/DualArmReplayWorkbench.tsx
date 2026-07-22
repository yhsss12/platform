'use client';

import { ReplayWorkbench } from '@/components/workspace/replay/ReplayWorkbench';

/** @deprecated 请使用统一 ReplayWorkbench */
export function DualArmReplayWorkbench({ initialJobId }: { initialJobId?: string }) {
  return (
    <ReplayWorkbench
      initialTaskType="dual_arm_cable_manipulation"
      initialJobId={initialJobId}
      initialReplayType="dataset"
      replayKind="data_generation"
      replaySourceKind="dataset"
      hasUrlTarget={Boolean(initialJobId)}
    />
  );
}
