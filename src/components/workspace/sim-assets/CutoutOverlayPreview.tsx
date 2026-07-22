'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchAssetPipelineFileBlob } from '@/lib/api/sam3dAssetPipelineClient';
import {
  CutoutLargePreview,
  cutoutRelPath,
  type CutoutManifestItem,
} from '@/components/workspace/sim-assets/CutoutCandidateGrid';

const PREVIEW_HEIGHT = 'min(560px, 58vh)';
const PREVIEW_MIN_HEIGHT = 320;

export interface CutoutOverlayPreviewProps {
  jobId: string;
  originalImagePath: string | null;
  selectedItem: CutoutManifestItem | null;
  emptyLabel?: string;
}

type PlacementMode = 'full-size' | 'bbox-size' | 'fallback-bbox';

type Placement = {
  x: number;
  y: number;
  width: number;
  height: number;
  mode: PlacementMode;
};

type PlacementDebugInfo = {
  original: [number, number];
  cutout: [number, number];
  bbox: [number, number, number, number] | null;
  mode: PlacementMode;
  placement: Placement;
};

function parseBbox(bbox: number[] | null | undefined): [number, number, number, number] | null {
  if (!bbox || bbox.length !== 4) return null;
  const x0 = Math.round(bbox[0]);
  const y0 = Math.round(bbox[1]);
  const x1 = Math.round(bbox[2]);
  const y1 = Math.round(bbox[3]);
  if (x1 <= x0 || y1 <= y0) return null;
  return [x0, y0, x1, y1];
}

function nearlyEqual(a: number, b: number, tolerance = 3): boolean {
  return Math.abs(a - b) <= tolerance;
}

function resolvePlacement(
  originalWidth: number,
  originalHeight: number,
  cutoutWidth: number,
  cutoutHeight: number,
  bbox: [number, number, number, number]
): Placement {
  const [x0, y0, x1, y1] = bbox;
  const bboxWidth = x1 - x0;
  const bboxHeight = y1 - y0;

  const isFullSizeCutout =
    nearlyEqual(cutoutWidth, originalWidth) && nearlyEqual(cutoutHeight, originalHeight);

  const isBboxSizeCutout =
    nearlyEqual(cutoutWidth, bboxWidth) && nearlyEqual(cutoutHeight, bboxHeight);

  if (isFullSizeCutout) {
    return {
      x: 0,
      y: 0,
      width: originalWidth,
      height: originalHeight,
      mode: 'full-size',
    };
  }

  if (isBboxSizeCutout) {
    return {
      x: x0,
      y: y0,
      width: bboxWidth,
      height: bboxHeight,
      mode: 'bbox-size',
    };
  }

  console.warn(
    '[CutoutOverlayPreview] cutout size does not match original or bbox; fallback to bbox placement',
    {
      original: [originalWidth, originalHeight],
      cutout: [cutoutWidth, cutoutHeight],
      bbox,
    }
  );

  return {
    x: x0,
    y: y0,
    width: bboxWidth,
    height: bboxHeight,
    mode: 'fallback-bbox',
  };
}

async function loadImageFromBlob(blob: Blob): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('图片解码失败'));
      img.src = url;
    });
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function drawOverlayPreview(
  canvas: HTMLCanvasElement,
  originalImg: HTMLImageElement,
  cutoutImg: HTMLImageElement,
  bbox: [number, number, number, number]
): PlacementDebugInfo {
  const [x0, y0, x1, y1] = bbox;
  const bboxWidth = x1 - x0;
  const bboxHeight = y1 - y0;
  const originalWidth = originalImg.naturalWidth;
  const originalHeight = originalImg.naturalHeight;
  const cutoutWidth = cutoutImg.naturalWidth;
  const cutoutHeight = cutoutImg.naturalHeight;

  const placement = resolvePlacement(
    originalWidth,
    originalHeight,
    cutoutWidth,
    cutoutHeight,
    bbox
  );

  const debugInfo: PlacementDebugInfo = {
    original: [originalWidth, originalHeight],
    cutout: [cutoutWidth, cutoutHeight],
    bbox,
    mode: placement.mode,
    placement,
  };

  console.debug('[CutoutOverlayPreview] placement', debugInfo);

  canvas.width = originalWidth;
  canvas.height = originalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return debugInfo;

  ctx.clearRect(0, 0, originalWidth, originalHeight);
  ctx.drawImage(originalImg, 0, 0, originalWidth, originalHeight);

  ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
  ctx.fillRect(0, 0, originalWidth, originalHeight);

  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = placement.width;
  overlayCanvas.height = placement.height;
  const overlayCtx = overlayCanvas.getContext('2d');
  if (!overlayCtx) return debugInfo;

  overlayCtx.clearRect(0, 0, placement.width, placement.height);
  overlayCtx.drawImage(cutoutImg, 0, 0, placement.width, placement.height);
  overlayCtx.globalCompositeOperation = 'source-atop';
  overlayCtx.fillStyle = 'rgba(34, 197, 94, 0.42)';
  overlayCtx.fillRect(0, 0, placement.width, placement.height);
  overlayCtx.globalCompositeOperation = 'source-over';

  ctx.drawImage(overlayCanvas, placement.x, placement.y, placement.width, placement.height);

  ctx.strokeStyle = 'rgba(34, 197, 94, 0.95)';
  ctx.lineWidth = Math.max(3, Math.round(originalWidth / 400));
  ctx.strokeRect(
    x0 + ctx.lineWidth / 2,
    y0 + ctx.lineWidth / 2,
    bboxWidth - ctx.lineWidth,
    bboxHeight - ctx.lineWidth
  );

  return debugInfo;
}

export function CutoutOverlayPreview({
  jobId,
  originalImagePath,
  selectedItem,
  emptyLabel = '请选择一个 cutout 作为三维重建输入',
}: CutoutOverlayPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [placementWarning, setPlacementWarning] = useState<string | null>(null);

  const bbox = useMemo(() => parseBbox(selectedItem?.bbox ?? null), [selectedItem?.bbox]);
  const cutoutPath = selectedItem ? cutoutRelPath(selectedItem) : null;

  const fallbackHint = useMemo(() => {
    if (!selectedItem) return null;
    if (!originalImagePath) return '缺少原图路径，已回退为 cutout 预览';
    if (!cutoutPath) return '缺少 cutout 路径，已回退为 cutout 预览';
    if (!bbox) return '当前 cutout 缺少 bbox，已回退为原始 cutout 预览';
    return null;
  }, [selectedItem, originalImagePath, cutoutPath, bbox]);

  useEffect(() => {
    setError(null);
    setPlacementWarning(null);
    if (!selectedItem || fallbackHint || !originalImagePath || !cutoutPath || !bbox) {
      return;
    }

    let cancelled = false;
    setLoading(true);

    void (async () => {
      try {
        const [originalBlob, cutoutBlob] = await Promise.all([
          fetchAssetPipelineFileBlob(jobId, originalImagePath),
          fetchAssetPipelineFileBlob(jobId, cutoutPath),
        ]);
        const [originalImg, cutoutImg] = await Promise.all([
          loadImageFromBlob(originalBlob),
          loadImageFromBlob(cutoutBlob),
        ]);
        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;

        const debugInfo = drawOverlayPreview(canvas, originalImg, cutoutImg, bbox);
        if (debugInfo.mode === 'fallback-bbox') {
          setPlacementWarning('cutout 尺寸与 bbox 不一致，预览可能存在偏差');
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : '预览合成失败');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [jobId, originalImagePath, cutoutPath, bbox, selectedItem, fallbackHint]);

  const panelStyle = {
    position: 'relative' as const,
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: PREVIEW_HEIGHT,
    minHeight: PREVIEW_MIN_HEIGHT,
    overflow: 'hidden',
    borderRadius: 12,
    border: '1px solid #e2e8f0',
    background: '#f8fafc',
  };

  if (!selectedItem) {
    return (
      <div style={panelStyle}>
        <div style={{ fontSize: 14, color: '#6b7280', padding: 24, textAlign: 'center' }}>{emptyLabel}</div>
      </div>
    );
  }

  if (fallbackHint) {
    return (
      <div style={{ width: '100%' }}>
        <div
          style={{
            marginBottom: 8,
            fontSize: 12,
            color: '#b45309',
            padding: '8px 10px',
            background: '#fffbeb',
            borderRadius: 8,
            border: '1px solid #fde68a',
          }}
        >
          {fallbackHint}
        </div>
        <CutoutLargePreview jobId={jobId} item={selectedItem} emptyLabel={emptyLabel} />
      </div>
    );
  }

  return (
    <div style={panelStyle}>
      {loading ? (
        <div style={{ position: 'absolute', zIndex: 2, fontSize: 13, color: '#6b7280' }}>合成预览中…</div>
      ) : null}
      {error ? (
        <div style={{ position: 'absolute', zIndex: 2, fontSize: 13, color: '#b91c1c', padding: 16, textAlign: 'center' }}>
          {error}
        </div>
      ) : null}
      {placementWarning ? (
        <div
          style={{
            position: 'absolute',
            zIndex: 2,
            bottom: 8,
            right: 8,
            maxWidth: '70%',
            fontSize: 11,
            color: '#92400e',
            padding: '4px 8px',
            background: 'rgba(255, 251, 235, 0.92)',
            borderRadius: 6,
            border: '1px solid #fde68a',
            pointerEvents: 'none',
          }}
        >
          {placementWarning}
        </div>
      ) : null}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%',
          padding: 8,
          boxSizing: 'border-box',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            display: 'block',
            maxWidth: '100%',
            maxHeight: '100%',
            userSelect: 'none',
          }}
        />
      </div>
    </div>
  );
}
