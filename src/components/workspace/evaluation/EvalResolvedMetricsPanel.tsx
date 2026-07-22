'use client';

import { useEffect, useMemo, useState } from 'react';
import { listRegistryResources } from '@/lib/api/resourceRegistryClient';
import {
  filterMetricDefinitions,
  listImplementedMetricDefinitions,
  normalizeEvaluationMode,
  normalizeEvaluationTaskType,
  resolveEvalMetrics,
  type MetricDefinition,
  type ResolvedEvalMetric,
} from '@/lib/workspace/evaluationMetricRegistry';

function MetricRow({ metric }: { metric: ResolvedEvalMetric }) {
  const hint =
    metric.unavailableReason ||
    (metric.status !== 'computed' ? metric.sourceHint : undefined);
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '120px 1fr',
        gap: 12,
        fontSize: 13,
        lineHeight: 1.55,
        padding: '8px 0',
        borderBottom: '1px solid #f1f5f9',
      }}
      title={hint}
    >
      <span style={{ color: '#64748b' }}>{metric.displayName}</span>
      <span
        style={{
          color: metric.status === 'computed' ? '#0f172a' : '#94a3b8',
          fontWeight: 500,
        }}
      >
        {metric.valueText}
      </span>
    </div>
  );
}

export function EvalResolvedMetricsPanel({
  taskType,
  evaluationMode,
  aggregate,
  perEpisode,
  metricIds,
  title = '评测指标',
  loading: externalLoading,
}: {
  taskType: string;
  evaluationMode: string;
  aggregate: Record<string, unknown> | null | undefined;
  perEpisode?: Record<string, unknown> | null;
  metricIds?: string[];
  title?: string;
  loading?: boolean;
}) {
  const [definitions, setDefinitions] = useState<MetricDefinition[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void listRegistryResources({ assetType: 'metric' })
      .then((response) => {
        if (cancelled) return;
        setDefinitions(listImplementedMetricDefinitions(response.resources));
      })
      .catch(() => {
        if (!cancelled) setDefinitions([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const resolved = useMemo(() => {
    const normalizedTaskType = normalizeEvaluationTaskType(taskType);
    const normalizedMode = normalizeEvaluationMode(evaluationMode);
    const applicable = filterMetricDefinitions(definitions, normalizedTaskType, normalizedMode);
    return resolveEvalMetrics({
      definitions: applicable,
      aggregate,
      perEpisode,
      metricIds,
    });
  }, [aggregate, definitions, evaluationMode, metricIds, perEpisode, taskType]);

  if (externalLoading || loading) {
    return <p style={{ margin: 0, fontSize: 13, color: '#94a3b8' }}>加载评测指标…</p>;
  }

  if (resolved.length === 0) {
    return null;
  }

  return (
    <div>
      {title ? (
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 8 }}>{title}</div>
      ) : null}
      <div
        style={{
          padding: '4px 12px',
          borderRadius: 10,
          border: '1px solid #e5eaf2',
          background: '#f8fafc',
        }}
      >
        {resolved.map((metric) => (
          <MetricRow key={metric.metricId} metric={metric} />
        ))}
      </div>
    </div>
  );
}

export function EvalResolvedMetricsReadonlyList({
  metricIds,
  definitions,
}: {
  metricIds: string[];
  definitions: MetricDefinition[];
}) {
  const labels = metricIds
    .map((id) => definitions.find((item) => item.metricId === id)?.displayName ?? id)
    .filter(Boolean);

  if (labels.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {labels.map((label) => (
        <span
          key={label}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            padding: '4px 10px',
            borderRadius: 999,
            fontSize: 12,
            color: '#1e40af',
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
          }}
        >
          {label}
        </span>
      ))}
    </div>
  );
}
