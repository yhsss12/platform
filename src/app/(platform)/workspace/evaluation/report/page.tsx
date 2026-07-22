'use client';

import { Suspense, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { EvaluationReportCoreMetricsPanel } from '@/components/workspace/evaluation/EvaluationReportCoreMetricsPanel';
import { EvaluationReportBasicInfoSection } from '@/components/workspace/evaluation/EvaluationReportBasicInfoSection';
import { EvaluationReportExportButton } from '@/components/workspace/evaluation/EvaluationReportExportModal';
import { CableThreadingEvaluationReport } from '@/components/workspace/evaluation/CableThreadingEvaluationReport';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import { listWorkspaceEvaluationTasksForUi } from '@/lib/workspace/workspaceDataSources';
import {
  buildEvaluationReport,
  findEvaluationTaskById,
} from '@/lib/mock/workspaceEvaluationRecordsMock';
import { buildReportBasicInfo } from '@/lib/workspace/evaluationReportBasicInfo';
import {
  buildCableThreadingReplayHref,
} from '@/lib/workspace/cableThreading';
import {
  buildDualArmEvalReplayHref,
} from '@/lib/workspace/dualArmEvaluation';
import { DUAL_ARM_CABLE_TASK_TYPE } from '@/lib/workspace/dualArmCable';
import { getEvaluationJobResult } from '@/lib/api/evaluationClient';
import { getCableThreadingEvalResult } from '@/lib/api/cableThreadingClient';
import { getWorkspaceJob } from '@/lib/api/workspaceJobClient';
import { workspaceEvaluationJobToRow } from '@/lib/workspace/workspaceJobMapper';
import {
  parseEvaluationReportPayload,
  resolveEvaluationReportCardTitle,
  type ParsedEvaluationReport,
} from '@/lib/workspace/evaluationReport';
import { resolveReportPerEpisode } from '@/lib/workspace/evaluationReportCoreMetrics';
import {
  normalizeEvaluationJobResultPayload,
} from '@/lib/workspace/evaluationMetricRegistry';
import {
  buildIsaacEvalReplayHref,
  ISAAC_BLOCK_STACKING_TASK_TYPE,
} from '@/lib/workspace/isaacBlockStacking';
import {
  getEvaluationRowJobId,
  resolveExportEvaluationJobId,
} from '@/lib/workspace/evaluationJobId';

const REPORT_PAGE_TITLE = '评测报告';

function EvaluationReportContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const evalId = searchParams.get('evalId');
  const taskTypeParam = searchParams.get('taskType');

  const [dualArmAggregate, setDualArmAggregate] = useState<Record<string, unknown> | null>(null);
  const [dualArmLoadError, setDualArmLoadError] = useState<string | null>(null);
  const [workspaceEvalJob, setWorkspaceEvalJob] = useState<ReturnType<typeof workspaceEvaluationJobToRow> | null>(
    null
  );
  const [workspaceEvalJobMetadata, setWorkspaceEvalJobMetadata] = useState<Record<string, unknown> | null>(
    null
  );
  const [workspaceEvalDeleted, setWorkspaceEvalDeleted] = useState(false);
  const [cableReport, setCableReport] = useState<ParsedEvaluationReport | null>(null);
  const [cableReportLoading, setCableReportLoading] = useState(false);
  const [cableReportError, setCableReportError] = useState<string | null>(null);
  const [isaacAggregate, setIsaacAggregate] = useState<Record<string, unknown> | null>(null);
  const [isaacPerEpisode, setIsaacPerEpisode] = useState<Record<string, unknown> | null>(null);
  const [isaacLoadError, setIsaacLoadError] = useState<string | null>(null);
  const [isaacLoading, setIsaacLoading] = useState(false);

  const isDualArmEvalReport =
    taskTypeParam === DUAL_ARM_CABLE_TASK_TYPE && Boolean(evalId?.startsWith('eval_'));

  const isIsaacEvalReport = Boolean(evalId?.startsWith('isaac_eval_'));

  const isCableThreadingEval =
    taskTypeParam === 'cable_threading' ||
    Boolean(evalId?.startsWith('ct_eval_')) ||
    workspaceEvalJob?.taskType === 'cable_threading';

  const hasCableReport = isCableThreadingEval && Boolean(evalId);

  useEffect(() => {
    if (!evalId) return;
    let cancelled = false;
    setWorkspaceEvalDeleted(false);
    void getWorkspaceJob(evalId)
      .then((job) => {
        if (!cancelled) {
          setWorkspaceEvalJob(workspaceEvaluationJobToRow(job));
          setWorkspaceEvalJobMetadata(
            job.metadata && typeof job.metadata === 'object' && !Array.isArray(job.metadata)
              ? (job.metadata as Record<string, unknown>)
              : null
          );
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWorkspaceEvalJob(null);
          setWorkspaceEvalJobMetadata(null);
          if (evalId.startsWith('eval_')) {
            setWorkspaceEvalDeleted(true);
          }
        }
      });
    return () => {
      cancelled = true;
    };
  }, [evalId]);

  useEffect(() => {
    if (!isDualArmEvalReport || !evalId) return;
    let cancelled = false;
    void getEvaluationJobResult(evalId)
      .then((data) => {
        if (!cancelled) setDualArmAggregate(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setDualArmLoadError(err instanceof Error ? err.message : '加载评测报告失败');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [isDualArmEvalReport, evalId]);

  useEffect(() => {
    if (!isIsaacEvalReport || !evalId) return;
    let cancelled = false;
    setIsaacLoading(true);
    setIsaacLoadError(null);
    void getEvaluationJobResult(evalId)
      .then((data) => {
        if (cancelled) return;
        const normalized = normalizeEvaluationJobResultPayload(data);
        setIsaacAggregate(normalized.aggregate);
        setIsaacPerEpisode(normalized.perEpisode);
      })
      .catch((err) => {
        if (!cancelled) {
          setIsaacLoadError(err instanceof Error ? err.message : '加载评测报告失败');
        }
      })
      .finally(() => {
        if (!cancelled) setIsaacLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isIsaacEvalReport, evalId]);

  useEffect(() => {
    if (!evalId || !isCableThreadingEval) return;
    let cancelled = false;
    setCableReportLoading(true);
    setCableReportError(null);

    const load = async () => {
      try {
        const payload = evalId.startsWith('ct_eval_')
          ? await getCableThreadingEvalResult(evalId)
          : evalId.startsWith('eval_')
            ? await getEvaluationJobResult(evalId)
            : null;
        if (cancelled) return;
        if (!payload) {
          setCableReport(null);
          setCableReportError('未找到评测结果');
          return;
        }
        setCableReport(parseEvaluationReportPayload(evalId, payload));
      } catch (err) {
        if (!cancelled) {
          setCableReport(null);
          setCableReportError(err instanceof Error ? err.message : '加载评测结果失败');
        }
      } finally {
        if (!cancelled) setCableReportLoading(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [evalId, isCableThreadingEval, isIsaacEvalReport]);

  const dualArmPerEpisode = useMemo(() => {
    if (!dualArmAggregate) return [];
    return resolveReportPerEpisode(dualArmAggregate);
  }, [dualArmAggregate]);

  const isaacPerEpisodeRows = useMemo(() => {
    if (!isaacPerEpisode) return [];
    if (Array.isArray(isaacPerEpisode.episodes)) return isaacPerEpisode.episodes as unknown[];
    return resolveReportPerEpisode(isaacPerEpisode);
  }, [isaacPerEpisode]);

  const listRow = useMemo(() => {
    if (!evalId) return null;
    return findEvaluationTaskById(evalId, listWorkspaceEvaluationTasksForUi());
  }, [evalId]);

  const report = useMemo(() => {
    if (!listRow || isCableThreadingEval || isIsaacEvalReport) return null;
    return buildEvaluationReport(listRow);
  }, [listRow, isCableThreadingEval, isIsaacEvalReport]);

  const reportCardTitle = useMemo(() => {
    return resolveEvaluationReportCardTitle({
      jobName: listRow?.taskName ?? listRow?.rawName ?? listRow?.name ?? workspaceEvalJob?.name,
      recordName: listRow?.taskName ?? listRow?.name ?? workspaceEvalJob?.name,
      taskName: listRow?.taskName ?? listRow?.rawName ?? workspaceEvalJob?.name,
      metadata: workspaceEvalJobMetadata,
      evaluationName: listRow?.name,
      reportJobName: report?.title,
      fallback: report?.title,
    });
  }, [workspaceEvalJob, workspaceEvalJobMetadata, listRow, report]);

  const resolvedTaskType = useMemo(() => {
    if (isIsaacEvalReport || evalId?.startsWith('isaac_eval_')) {
      return ISAAC_BLOCK_STACKING_TASK_TYPE;
    }
    if (isDualArmEvalReport || listRow?.taskType === DUAL_ARM_CABLE_TASK_TYPE) {
      return DUAL_ARM_CABLE_TASK_TYPE;
    }
    if (isCableThreadingEval || evalId?.startsWith('ct_eval_')) {
      return 'cable_threading';
    }
    return taskTypeParam ?? workspaceEvalJob?.taskType ?? listRow?.taskType ?? undefined;
  }, [
    evalId,
    taskTypeParam,
    isIsaacEvalReport,
    isDualArmEvalReport,
    isCableThreadingEval,
    workspaceEvalJob?.taskType,
    listRow?.taskType,
  ]);

  const reportAggregate = useMemo(() => {
    if (isIsaacEvalReport) return isaacAggregate;
    if (isDualArmEvalReport) return dualArmAggregate;
    if (hasCableReport) return cableReport?.rawAggregate ?? null;
    return null;
  }, [
    isIsaacEvalReport,
    isDualArmEvalReport,
    hasCableReport,
    isaacAggregate,
    dualArmAggregate,
    cableReport,
  ]);

  const reportPerEpisode = useMemo(() => {
    if (isIsaacEvalReport) return isaacPerEpisodeRows;
    if (isDualArmEvalReport) return dualArmPerEpisode;
    if (hasCableReport) return cableReport?.episodes ?? [];
    return [];
  }, [
    isIsaacEvalReport,
    isDualArmEvalReport,
    hasCableReport,
    isaacPerEpisodeRows,
    dualArmPerEpisode,
    cableReport,
  ]);

  const basicInfoFields = useMemo(() => {
    const metadata = workspaceEvalJobMetadata ?? {};
    const metrics =
      metadata.metrics && typeof metadata.metrics === 'object' && !Array.isArray(metadata.metrics)
        ? (metadata.metrics as Record<string, unknown>)
        : {};

    return buildReportBasicInfo({
      taskName: reportCardTitle,
      relatedTask: workspaceEvalJob?.relatedTask ?? listRow?.relatedTask ?? report?.relatedTask,
      taskDisplayName:
        typeof metadata.taskDisplayName === 'string' ? metadata.taskDisplayName : undefined,
      taskTemplateId:
        typeof metadata.taskTemplateId === 'string' ? metadata.taskTemplateId : undefined,
      taskType: resolvedTaskType,
      metadata,
      metrics,
      aggregate: reportAggregate ?? undefined,
      evaluationMode:
        workspaceEvalJob?.evaluationMode ??
        (typeof reportAggregate?.evaluationMode === 'string'
          ? reportAggregate.evaluationMode
          : undefined),
      modelType: workspaceEvalJob?.modelType ?? listRow?.modelType ?? report?.modelType,
      episodeCount:
        cableReport?.totalEpisodes ??
        (typeof reportAggregate?.episodeCount === 'number'
          ? reportAggregate.episodeCount
          : typeof reportAggregate?.totalEpisodes === 'number'
            ? reportAggregate.totalEpisodes
            : workspaceEvalJob?.evalRounds ?? report?.evalRounds),
    });
  }, [
    reportCardTitle,
    workspaceEvalJob,
    workspaceEvalJobMetadata,
    listRow,
    report,
    resolvedTaskType,
    reportAggregate,
    cableReport?.totalEpisodes,
  ]);

  const coreMetricsLoading =
    (hasCableReport && cableReportLoading) ||
    (isIsaacEvalReport && isaacLoading) ||
    (isDualArmEvalReport && !dualArmAggregate && !dualArmLoadError);

  const coreMetricsError =
    (hasCableReport ? cableReportError : null) ??
    (isIsaacEvalReport ? isaacLoadError : null) ??
    (isDualArmEvalReport ? dualArmLoadError : null);

  if (workspaceEvalDeleted) {
    return (
      <ModulePageContainer>
        <ModulePageHeader
          title={REPORT_PAGE_TITLE}
          actions={
            <SecondaryButton onClick={() => router.push('/workspace/evaluation')}>
              返回
            </SecondaryButton>
          }
        />
        <div
          style={{
            padding: 40,
            textAlign: 'center',
            color: '#6b7280',
            fontSize: 14,
            backgroundColor: '#fff',
            borderRadius: 12,
            border: '1px solid #e5e7eb',
          }}
        >
          该评测产物已被删除，无法生成报告。
        </div>
      </ModulePageContainer>
    );
  }

  const hasContent =
    report ||
    hasCableReport ||
    isDualArmEvalReport ||
    dualArmLoadError ||
    isIsaacEvalReport ||
    isaacLoadError ||
    Boolean(workspaceEvalJob);

  if (!evalId || !hasContent) {
    return (
      <ModulePageContainer>
        <ModulePageHeader
          title={REPORT_PAGE_TITLE}
          actions={
            <SecondaryButton onClick={() => router.push('/workspace/evaluation')}>
              返回
            </SecondaryButton>
          }
        />
        <div
          style={{
            padding: 40,
            textAlign: 'center',
            color: '#6b7280',
            fontSize: 14,
            backgroundColor: '#fff',
            borderRadius: 12,
            border: '1px solid #e5e7eb',
          }}
        >
          未找到对应评测报告，请从评测中心任务列表进入。
        </div>
      </ModulePageContainer>
    );
  }

  const exportEvalJobId = useMemo(() => {
    const aggregateEvalJobId =
      reportAggregate && typeof reportAggregate.evalJobId === 'string'
        ? reportAggregate.evalJobId
        : undefined;
    return (
      resolveExportEvaluationJobId({
        evalJobId: evalId,
        jobId: workspaceEvalJob?.evalJobId ?? workspaceEvalJob?.id,
        runtimePath: workspaceEvalJob?.runtimePath,
        aggregateEvalJobId,
        listRowEvalJobId: listRow ? getEvaluationRowJobId(listRow) : undefined,
      }) || evalId || ''
    );
  }, [evalId, workspaceEvalJob, reportAggregate, listRow]);

  const reportEvalId = exportEvalJobId || evalId || '';

  const replayHref = hasCableReport
    ? buildCableThreadingReplayHref({ evalId: reportEvalId })
    : isIsaacEvalReport
      ? buildIsaacEvalReplayHref({ evalJobId: reportEvalId })
      : isDualArmEvalReport
      ? buildDualArmEvalReplayHref({ evalJobId: reportEvalId })
      : `/workspace/replay?replayType=evaluation&evalId=${encodeURIComponent(reportEvalId)}`;

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={REPORT_PAGE_TITLE}
        actions={
          <div style={{ display: 'flex', gap: 8 }}>
            <EvaluationReportExportButton evalJobId={reportEvalId} label="导出" />
            <Link href={replayHref}>
              <SecondaryButton>查看回放</SecondaryButton>
            </Link>
            <SecondaryButton onClick={() => router.push('/workspace/evaluation')}>
              返回
            </SecondaryButton>
          </div>
        }
      />

      <div
        style={{
          backgroundColor: '#fff',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: '24px 28px',
        }}
      >
        <h2 style={{ margin: '0 0 24px', fontSize: 20, fontWeight: 600, color: '#111827' }}>
          {reportCardTitle}
        </h2>

        <Section title="基本信息">
          <EvaluationReportBasicInfoSection items={basicInfoFields} />
        </Section>

        <Section title="核心指标">
          {hasCableReport ? (
            <CableThreadingEvaluationReport
              report={cableReport}
              loading={cableReportLoading}
              error={cableReportError}
            />
          ) : (
            <EvaluationReportCoreMetricsPanel
              aggregate={reportAggregate}
              perEpisode={reportPerEpisode}
              loading={coreMetricsLoading}
              error={coreMetricsError}
            />
          )}
        </Section>
      </div>
    </ModulePageContainer>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: 28 }}>
      <h3
        style={{
          margin: '0 0 12px',
          fontSize: 14,
          fontWeight: 600,
          color: '#374151',
          borderBottom: '1px solid #f3f4f6',
          paddingBottom: 8,
        }}
      >
        {title}
      </h3>
      {children}
    </section>
  );
}

export default function EvaluationReportPage() {
  return (
    <Suspense fallback={null}>
      <EvaluationReportContent />
    </Suspense>
  );
}
