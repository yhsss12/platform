'use client';

import { useEffect, useMemo, useState } from 'react';
import { useI18n } from '@/components/common/I18nProvider';
import { EvaluationReplayStatusHeader } from '@/components/workspace/replay/EvaluationReplayStatusHeader';
import { EvalReplaySidePanel } from '@/components/workspace/replay/EvalReplaySidePanel';
import { EvaluationReplayVideoSection } from '@/components/workspace/replay/EvaluationReplayVideoSection';
import { REPLAY_PAGE_STYLES } from '@/components/workspace/replay/ReplayWorkbench';
import {
  getEvaluationJobResult,
  getEvaluationJobStatus,
} from '@/lib/api/evaluationClient';
import { normalizeEvaluationJobResultPayload } from '@/lib/workspace/evaluationMetricRegistry';
import { EvaluationWorkbenchBasicInfoRows } from '@/components/workspace/replay/EvaluationWorkbenchBasicInfoRows';
import { ISAAC_BLOCK_STACKING_DISPLAY_NAME } from '@/lib/workspace/isaacBlockStacking';
import { mergeEvaluationReplayInfo } from '@/lib/workspace/evaluationReplayInfo';
import { resolveEvaluationWorkbenchBasicInfo } from '@/lib/workspace/evaluationWorkbenchBasicInfo';
import { mapEvaluationJobStatusLabel } from '@/lib/workspace/evaluationWorkbenchCopy';
import { replayVideoSourceUserLabel } from '@/lib/workspace/replayAdapters';
import type { ReplayPageKind } from '@/lib/workspace/replayPageKind';

const cardStyle: React.CSSProperties = {
  backgroundColor: '#fff',
  border: '1px solid #e5e7eb',
  borderRadius: 16,
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
  padding: 20,
};

const POLL_MS = 2000;

export function IsaacEvalReplayPanel({
  evalJobId,
  episode = 0,
  replayKind,
}: {
  evalJobId: string;
  episode?: number;
  replayKind: ReplayPageKind;
}) {
  const { t } = useI18n();
  const [aggregate, setAggregate] = useState<Record<string, unknown> | null>(null);
  const [statusPayload, setStatusPayload] = useState<Record<string, unknown> | null>(null);
  const [statusMetrics, setStatusMetrics] = useState<Record<string, unknown> | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('加载中…');
  const [jobStatus, setJobStatus] = useState<string>('loading');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | undefined;

    const load = async () => {
      try {
        const status = await getEvaluationJobStatus(evalJobId);
        if (cancelled) return;
        setJobStatus(status.status);
        setStatusMessage(status.message || status.status);
        setStatusMetrics(status.metrics ?? null);
        setStatusPayload(status as unknown as Record<string, unknown>);

        if (status.status === 'completed' || status.status === 'failed') {
          const result = await getEvaluationJobResult(evalJobId);
          if (!cancelled) {
            const normalized = normalizeEvaluationJobResultPayload(result);
            setAggregate(normalized.aggregate);
            setStatusPayload((prev) => ({
              ...mergeEvaluationReplayInfo(prev ?? undefined, result as Record<string, unknown>),
            }));
          }
          if (timer) {
            clearInterval(timer);
            timer = undefined;
          }
        }
      } catch (err) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message.toLowerCase() : '';
          if (msg.includes('not found') || msg.includes('404')) {
            setError(t('workspacePages.replayEmptyDeleted'));
          } else {
            setError(err instanceof Error ? err.message : '加载评测回放失败');
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();
    timer = setInterval(() => {
      void load();
    }, POLL_MS);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [evalJobId, t]);

  const replayInfo = useMemo(
    () => mergeEvaluationReplayInfo(statusPayload ?? undefined, aggregate ?? undefined),
    [statusPayload, aggregate]
  );

  const videoTitle =
    replayKind === 'evaluation'
      ? null
      : t('workspacePages.replayVideoGenericTitle');

  const workbenchBasicInfo = useMemo(
    () =>
      resolveEvaluationWorkbenchBasicInfo({
        evalJobId,
        status: statusPayload ?? undefined,
        aggregate: aggregate ?? undefined,
        fallbackTaskName: ISAAC_BLOCK_STACKING_DISPLAY_NAME,
      }),
    [evalJobId, statusPayload, aggregate]
  );

  const statusBarLabel = mapEvaluationJobStatusLabel(loading ? 'loading' : jobStatus);
  const displayTaskName = workbenchBasicInfo.taskName;
  const evaluationMode =
    replayInfo.evaluationMode ??
    (typeof statusPayload?.evaluationMode === 'string' ? statusPayload.evaluationMode : null);
  const rolloutFooterLabel = replayVideoSourceUserLabel('evaluation', 'evaluation', evaluationMode);

  if (error) {
    return (
      <section style={cardStyle}>
        <p style={{ margin: 0, color: '#b45309', fontSize: 13, lineHeight: 1.6 }}>{error}</p>
      </section>
    );
  }

  return (
    <>
      <style>{REPLAY_PAGE_STYLES}</style>
      <section className="replay-workspace-card">
        <div className="replay-main-area">
          <div className="replay-header">
            {replayKind === 'evaluation' ? (
              <EvaluationReplayStatusHeader
                taskName={displayTaskName}
                statusLabel={statusBarLabel}
              />
            ) : (
              <span className="replay-header-title">{videoTitle}</span>
            )}
          </div>
          <div className="replay-content-row">
            <div className="replay-player-column">
              <EvaluationReplayVideoSection
                evalJobId={evalJobId}
                replayInfo={replayInfo}
                initialEpisode={episode > 0 ? episode : undefined}
                footerLabel={rolloutFooterLabel}
              />
            </div>
            <EvalReplaySidePanel
              evalJobId={evalJobId}
              taskType="isaac_block_stacking"
              jobStatus={jobStatus}
              loading={loading}
              aggregate={aggregate}
              metrics={statusMetrics}
              basicInfo={<EvaluationWorkbenchBasicInfoRows info={workbenchBasicInfo} />}
            />
          </div>
        </div>
      </section>
    </>
  );
}
