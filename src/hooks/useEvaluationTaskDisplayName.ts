'use client';

import { useEffect, useState } from 'react';
import { getWorkspaceJob } from '@/lib/api/workspaceJobClient';
import { resolveEvaluationTaskDisplayName } from '@/lib/workspace/evaluationReport';
import { workspaceEvaluationJobToRow } from '@/lib/workspace/workspaceJobMapper';

export function useEvaluationTaskDisplayName(
  evalJobId: string | undefined,
  fallback = '评测回放'
): string {
  const [displayName, setDisplayName] = useState(fallback);

  useEffect(() => {
    if (!evalJobId?.trim()) {
      setDisplayName(fallback);
      return;
    }

    let cancelled = false;
    void getWorkspaceJob(evalJobId)
      .then((job) => {
        if (cancelled) return;
        const row = workspaceEvaluationJobToRow(job);
        setDisplayName(
          resolveEvaluationTaskDisplayName({
            jobName: row.name,
            sourceJobName: job.taskName,
            taskName: job.taskName,
            metadata: job.metadata as Record<string, unknown> | undefined,
            fallback,
          })
        );
      })
      .catch(() => {
        if (!cancelled) setDisplayName(fallback);
      });

    return () => {
      cancelled = true;
    };
  }, [evalJobId, fallback]);

  return displayName;
}
