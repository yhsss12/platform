'use client';

import { useEffect, useState } from 'react';
import { fetchAssetPipelineFileBlob } from '@/lib/api/sam3dAssetPipelineClient';

export interface CutoutManifestItem {
  cutoutIndex: number;
  label?: string | null;
  score?: number | null;
  bbox?: number[] | null;
  cutoutPath?: string | null;
  previewPath?: string | null;
  originalCutoutPath?: string | null;
  selectable?: boolean;
}

export function cutoutFileName(item: CutoutManifestItem): string {
  const path = item.cutoutPath || item.previewPath || '';
  const parts = path.split('/');
  return parts[parts.length - 1] || `${item.cutoutIndex}.png`;
}

export function cutoutRelPath(item: CutoutManifestItem): string | null {
  return item.cutoutPath || item.previewPath || null;
}

const CHECKERBOARD_BG =
  'linear-gradient(45deg, #eceff3 25%, transparent 25%), linear-gradient(-45deg, #eceff3 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #eceff3 75%), linear-gradient(-45deg, transparent 75%, #eceff3 75%)';

function useCutoutBlobUrl(jobId: string, relPath: string | null) {
  const [thumbUrl, setThumbUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!relPath) {
      setThumbUrl(null);
      setLoadError('缺少 cutout 路径');
      setLoading(false);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;
    setLoadError(null);
    setLoading(true);

    void fetchAssetPipelineFileBlob(jobId, relPath)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setThumbUrl(objectUrl);
      })
      .catch((err) => {
        if (cancelled) return;
        setThumbUrl(null);
        setLoadError(err instanceof Error ? err.message : 'cutout 加载失败');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [jobId, relPath]);

  return { thumbUrl, loadError, loading };
}

interface CutoutLargePreviewProps {
  jobId: string;
  item: CutoutManifestItem | null;
  emptyLabel?: string;
}

export function CutoutLargePreview({
  jobId,
  item,
  emptyLabel = '请选择一个 cutout 作为三维重建输入',
}: CutoutLargePreviewProps) {
  const relPath = item ? cutoutRelPath(item) : null;
  const { thumbUrl, loadError, loading } = useCutoutBlobUrl(jobId, relPath);

  return (
    <div
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        height: 'min(560px, 58vh)',
        minHeight: 320,
        overflow: 'hidden',
        borderRadius: 12,
        border: '1px solid #e2e8f0',
        backgroundColor: '#f8fafc',
        backgroundImage: CHECKERBOARD_BG,
        backgroundSize: '16px 16px',
        backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0',
      }}
    >
      {!item ? (
        <div style={{ fontSize: 14, color: '#6b7280', padding: 24, textAlign: 'center' }}>{emptyLabel}</div>
      ) : thumbUrl ? (
        <img
          src={thumbUrl}
          alt={`cutout ${item.cutoutIndex}`}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            userSelect: 'none',
          }}
        />
      ) : (
        <div style={{ fontSize: 13, color: loadError ? '#b91c1c' : '#6b7280', padding: 16 }}>
          {loadError || (loading ? '加载 cutout…' : '无预览')}
        </div>
      )}
    </div>
  );
}

interface CutoutCandidateCardProps {
  jobId: string;
  item: CutoutManifestItem;
  selected: boolean;
  onSelect: (cutoutIndex: number) => void;
  compact?: boolean;
}

export function CutoutCandidateCard({
  jobId,
  item,
  selected,
  onSelect,
  compact = false,
}: CutoutCandidateCardProps) {
  const relPath = cutoutRelPath(item);
  const { thumbUrl, loadError, loading } = useCutoutBlobUrl(jobId, relPath);
  const thumbHeight = compact ? 72 : 120;

  return (
    <button
      type="button"
      onClick={() => onSelect(item.cutoutIndex)}
      style={{
        border: selected ? '2px solid #2563eb' : '1px solid #e5e7eb',
        borderRadius: 8,
        padding: compact ? 6 : 8,
        background: selected ? '#eff6ff' : '#fff',
        cursor: 'pointer',
        textAlign: 'left',
      }}
    >
      {thumbUrl ? (
        <img
          src={thumbUrl}
          alt={`cutout ${item.cutoutIndex}`}
          style={{
            width: '100%',
            height: thumbHeight,
            objectFit: 'contain',
            background: '#f3f4f6',
            borderRadius: 4,
          }}
        />
      ) : (
        <div
          style={{
            width: '100%',
            height: thumbHeight,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#f3f4f6',
            color: loadError ? '#b91c1c' : '#6b7280',
            fontSize: compact ? 10 : 12,
            borderRadius: 4,
            padding: 6,
            textAlign: 'center',
          }}
        >
          {loadError || (loading ? '…' : '无图')}
        </div>
      )}
      <div style={{ fontSize: compact ? 11 : 12, marginTop: compact ? 4 : 6, fontWeight: 600 }}>
        #{item.cutoutIndex}
      </div>
      {!compact ? (
        <>
          <div style={{ fontSize: 11, color: '#6b7280' }}>文件名：{cutoutFileName(item)}</div>
          {item.score != null ? (
            <div style={{ fontSize: 11, color: '#6b7280' }}>score={item.score.toFixed(3)}</div>
          ) : null}
          {item.bbox && item.bbox.length === 4 ? (
            <div style={{ fontSize: 10, color: '#9ca3af', marginTop: 2 }}>
              bbox: {item.bbox.map((v) => Math.round(v)).join(', ')}
            </div>
          ) : null}
        </>
      ) : (
        <>
          <div
            style={{
              fontSize: 10,
              color: '#9ca3af',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {cutoutFileName(item)}
          </div>
          {item.score != null ? (
            <div style={{ fontSize: 10, color: '#6b7280' }}>{item.score.toFixed(2)}</div>
          ) : null}
        </>
      )}
    </button>
  );
}

interface CutoutCandidateGridProps {
  jobId: string;
  items: CutoutManifestItem[];
  selectedCutoutIndex: number | null | undefined;
  onSelect: (cutoutIndex: number) => void;
  compact?: boolean;
}

export function CutoutCandidateGrid({
  jobId,
  items,
  selectedCutoutIndex,
  onSelect,
  compact = false,
}: CutoutCandidateGridProps) {
  if (!items.length) {
    return null;
  }
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: compact
          ? 'repeat(auto-fill, minmax(96px, 1fr))'
          : 'repeat(auto-fill,minmax(160px,1fr))',
        gap: compact ? 8 : 12,
      }}
    >
      {items.map((item) => (
        <CutoutCandidateCard
          key={item.cutoutIndex}
          jobId={jobId}
          item={item}
          selected={item.cutoutIndex === selectedCutoutIndex}
          onSelect={onSelect}
          compact={compact}
        />
      ))}
    </div>
  );
}
