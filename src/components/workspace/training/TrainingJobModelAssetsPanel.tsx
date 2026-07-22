'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  listTrainingJobModelAssetsDetail,
  type TrainingJobModelAssetItem,
} from '@/lib/api/modelAssetsClient';
import { buildModelEvaluationCreateFromAssetUrl } from '@/lib/workspace/evaluationCreateNavigation';
import { resolveModelAssetColumnLabel } from '@/lib/workspace/modelAssetDisplay';
import {
  canEvaluateModelAsset,
  getModelAssetEvalDisabledReason,
  isTrainingJobInProgressForAssets,
  modelAssetDisplayStatusLabel,
  sortModelAssetsForDisplay,
} from '@/lib/workspace/modelAssetRules';
import { trainingSectionTitleStyle } from '@/components/workspace/training/trainingDetailShared';

const TABLE_COLUMNS = ['模型资产名称', '类型', '状态', '操作'] as const;

const headerHintStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 500,
  color: '#9ca3af',
};

function checkpointKindLabel(asset: TrainingJobModelAssetItem): string {
  const kind = (asset.checkpointKind ?? '').toLowerCase();
  if (kind === 'final') return 'Final';
  if (kind === 'best') {
    const metric = asset.checkpointMetricName?.trim() || 'Loss';
    return `Best ${metric}`;
  }
  if (kind === 'epoch' && asset.checkpointEpoch != null) {
    return `Epoch ${asset.checkpointEpoch}`;
  }
  return '—';
}

function evalDisabledTitle(asset: TrainingJobModelAssetItem, jobInProgress: boolean): string {
  return getModelAssetEvalDisabledReason(asset, { jobInProgress }) ?? '模型未就绪或缺少 checkpoint，暂无法发起评测';
}

function AssetRow({
  asset,
  jobInProgress,
  onStartEvaluation,
}: {
  asset: TrainingJobModelAssetItem;
  jobInProgress: boolean;
  onStartEvaluation?: () => void;
}) {
  const router = useRouter();
  const name = resolveModelAssetColumnLabel(asset);
  const evalEnabled = canEvaluateModelAsset(asset, { jobInProgress });
  const statusLabel = modelAssetDisplayStatusLabel(asset.displayStatus);
  const statusColor =
    asset.displayStatus === 'ready'
      ? '#047857'
      : asset.displayStatus === 'generating'
        ? '#2563eb'
        : asset.displayStatus === 'missing'
          ? '#b45309'
          : asset.displayStatus === 'superseded'
            ? '#9ca3af'
            : '#b45309';

  const handleStartEvaluation = () => {
    onStartEvaluation?.();
    router.push(buildModelEvaluationCreateFromAssetUrl(asset));
  };

  return (
    <tr style={{ borderTop: '1px solid #f3f4f6' }}>
      <td
        style={{
          padding: '8px 10px',
          maxWidth: 200,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={name}
      >
        {name}
      </td>
      <td style={{ padding: '8px 10px' }}>{checkpointKindLabel(asset)}</td>
      <td style={{ padding: '8px 10px', color: statusColor }}>{statusLabel}</td>
      <td style={{ padding: '8px 10px' }}>
        {evalEnabled ? (
          <button
            type="button"
            onClick={handleStartEvaluation}
            style={{
              color: '#2563eb',
              fontSize: 12,
              background: 'none',
              border: 'none',
              padding: 0,
              cursor: 'pointer',
              textDecoration: 'underline',
            }}
          >
            发起评测
          </button>
        ) : (
          <span
            title={evalDisabledTitle(asset, jobInProgress)}
            style={{ fontSize: 12, color: '#9ca3af', cursor: 'not-allowed' }}
          >
            发起评测
          </span>
        )}
      </td>
    </tr>
  );
}

function TablePlaceholderRow({ message, tone = 'muted' }: { message: string; tone?: 'muted' | 'warn' }) {
  return (
    <tr>
      <td
        colSpan={TABLE_COLUMNS.length}
        style={{
          padding: '14px 10px',
          textAlign: 'center',
          fontSize: 12,
          color: tone === 'warn' ? '#b45309' : '#9ca3af',
        }}
      >
        {message}
      </td>
    </tr>
  );
}

export function TrainingJobModelAssetsPanel({
  trainJobId,
  jobStatus,
  jobBackendStatus,
  jobCurrentEpoch,
  jobTotalEpochs,
  jobProgressPercent,
  onNavigateToEvaluation,
}: {
  trainJobId: string;
  jobStatus?: string | null;
  jobBackendStatus?: string | null;
  jobCurrentEpoch?: number;
  jobTotalEpochs?: number;
  jobProgressPercent?: number | null;
  onNavigateToEvaluation?: () => void;
}) {
  const [assets, setAssets] = useState<TrainingJobModelAssetItem[]>([]);
  const [listMessage, setListMessage] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const hasLoadedOnceRef = useRef(false);

  const jobInProgress = useMemo(() => {
    const signals = {
      currentEpoch: jobCurrentEpoch,
      totalEpochs: jobTotalEpochs,
      progressPercent: jobProgressPercent,
    };
    return (
      isTrainingJobInProgressForAssets(jobBackendStatus, signals) ||
      isTrainingJobInProgressForAssets(jobStatus, signals)
    );
  }, [
    jobBackendStatus,
    jobStatus,
    jobCurrentEpoch,
    jobTotalEpochs,
    jobProgressPercent,
  ]);

  useEffect(() => {
    hasLoadedOnceRef.current = false;
    setHasLoadedOnce(false);
    setAssets([]);
    setListMessage(null);
    setInitialLoading(true);
    setRefreshing(false);
    setRefreshError(null);
  }, [trainJobId]);

  useEffect(() => {
    let cancelled = false;

    const load = async (background: boolean) => {
      if (background) {
        setRefreshing(true);
      } else if (!hasLoadedOnceRef.current) {
        setInitialLoading(true);
      }

      try {
        const res = await listTrainingJobModelAssetsDetail(trainJobId);
        if (cancelled) return;
        const nextAssets = sortModelAssetsForDisplay(res.modelAssets);
        setAssets((prev) => {
          if (background && nextAssets.length === 0 && prev.length > 0) {
            return prev;
          }
          return nextAssets;
        });
        if (!(background && (res.modelAssets?.length ?? 0) === 0 && hasLoadedOnceRef.current)) {
          setListMessage(res.listMessage ?? null);
        }
        setRefreshError(null);
        hasLoadedOnceRef.current = true;
        setHasLoadedOnce(true);
      } catch {
        if (cancelled) return;
        if (!background || !hasLoadedOnceRef.current) {
          setRefreshError('刷新失败，可稍后重试');
        }
      } finally {
        if (!cancelled) {
          setInitialLoading(false);
          setRefreshing(false);
        }
      }
    };

    void load(false);

    if (!jobInProgress) {
      return () => {
        cancelled = true;
      };
    }

    const timer = window.setInterval(() => {
      void load(true);
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [trainJobId, jobInProgress]);

  const emptyTableMessage = useMemo(() => {
    if (assets.some((asset) => asset.displayStatus === 'missing')) {
      return null;
    }
    if (jobInProgress && assets.length === 0) {
      return listMessage ?? '模型资产将在当前训练任务完成后生成。';
    }
    return listMessage ?? '暂无模型资产';
  }, [assets, listMessage, jobInProgress]);

  const showInitialPlaceholder = initialLoading && assets.length === 0;
  const showEmptyState = !initialLoading && !refreshing && assets.length === 0 && hasLoadedOnce;
  const showFirstLoadError =
    Boolean(refreshError) && !hasLoadedOnce && assets.length === 0 && !initialLoading;

  return (
    <div style={{ marginTop: 20 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          marginBottom: 12,
        }}
      >
        <div style={{ ...trainingSectionTitleStyle, marginBottom: 0 }}>生成的模型资产</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          {refreshing ? <span style={headerHintStyle}>刷新中</span> : null}
          {refreshError && !refreshing ? (
            <span style={{ ...headerHintStyle, color: '#b45309' }}>{refreshError}</span>
          ) : null}
        </div>
      </div>

      <div style={{ overflowX: 'auto', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ backgroundColor: '#f9fafb', color: '#6b7280', textAlign: 'left' }}>
              {TABLE_COLUMNS.map((label) => (
                <th key={label} style={{ padding: '8px 10px', fontWeight: 600 }}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {assets.map((asset) => (
              <AssetRow
                key={asset.id}
                asset={asset}
                jobInProgress={jobInProgress}
                onStartEvaluation={onNavigateToEvaluation}
              />
            ))}
            {showInitialPlaceholder ? <TablePlaceholderRow message="加载中…" /> : null}
            {showEmptyState ? (
              <TablePlaceholderRow
                message={emptyTableMessage ?? '暂无模型资产'}
                tone={listMessage ? 'warn' : 'muted'}
              />
            ) : null}
            {showFirstLoadError ? (
              <TablePlaceholderRow message={refreshError ?? '刷新失败，可稍后重试'} tone="warn" />
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
