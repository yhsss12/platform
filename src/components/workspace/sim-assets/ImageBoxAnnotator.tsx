'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type BoxMode = 'positive' | 'negative';

export type AnnotationBox = {
  id: string;
  type: BoxMode;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

/** @deprecated use AnnotationBox */
export type SelectionBox = AnnotationBox;

/** @deprecated use BoxMode */
export type BoxType = BoxMode;

interface ImageBoxAnnotatorProps {
  imageUrl: string;
  boxes: AnnotationBox[];
  mode: BoxMode;
  disabled?: boolean;
  onBoxesChange: (boxes: AnnotationBox[]) => void;
  onImageSizeChange?: (size: { width: number; height: number }) => void;
  onImageError?: (message: string) => void;
  onHoverPointChange?: (point: { x: number; y: number } | null) => void;
  maxPanelHeight?: number;
}

const DEFAULT_MAX_PANEL_HEIGHT = 560;
const MIN_PANEL_HEIGHT = 360;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function normalizeBox(x0: number, y0: number, x1: number, y1: number) {
  return {
    x0: Math.round(Math.min(x0, x1)),
    y0: Math.round(Math.min(y0, y1)),
    x1: Math.round(Math.max(x0, x1)),
    y1: Math.round(Math.max(y0, y1)),
  };
}

function boxStroke(type: BoxMode): string {
  return type === 'positive' ? '#16a34a' : '#dc2626';
}

function boxLabel(type: BoxMode, index: number): string {
  return type === 'positive' ? `+${index + 1}` : `-${index + 1}`;
}

function computeDisplaySize(
  natural: { width: number; height: number },
  panel: { width: number; height: number }
): { width: number; height: number } {
  if (natural.width <= 0 || natural.height <= 0 || panel.width <= 0 || panel.height <= 0) {
    return { width: 0, height: 0 };
  }
  const scale = Math.min(panel.width / natural.width, panel.height / natural.height);
  return {
    width: Math.max(1, Math.round(natural.width * scale)),
    height: Math.max(1, Math.round(natural.height * scale)),
  };
}

export function ImageBoxAnnotator({
  imageUrl,
  boxes,
  mode,
  disabled,
  onBoxesChange,
  onImageSizeChange,
  onImageError,
  onHoverPointChange,
  maxPanelHeight = DEFAULT_MAX_PANEL_HEIGHT,
}: ImageBoxAnnotatorProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const boxesRef = useRef(boxes);
  const modeRef = useRef(mode);
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [panelSize, setPanelSize] = useState({ width: 0, height: 0 });
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);

  boxesRef.current = boxes;
  modeRef.current = mode;

  const displaySize = computeDisplaySize(naturalSize, panelSize);

  useEffect(() => {
    setNaturalSize({ width: 0, height: 0 });
    setDragStart(null);
    setDragCurrent(null);
    setSelectedBoxId(null);
  }, [imageUrl]);

  useEffect(() => {
    if (!selectedBoxId) return;
    if (!boxes.some((box) => box.id === selectedBoxId)) {
      setSelectedBoxId(null);
    }
  }, [boxes, selectedBoxId]);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;

    const updatePanelSize = () => {
      const rect = panel.getBoundingClientRect();
      setPanelSize({
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      });
    };

    updatePanelSize();
    const observer = new ResizeObserver(updatePanelSize);
    observer.observe(panel);
    return () => observer.disconnect();
  }, []);

  const getSvgPoint = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg || naturalSize.width <= 0 || naturalSize.height <= 0) {
        return null;
      }
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const svgPoint = point.matrixTransform(ctm.inverse());
      return {
        x: clamp(svgPoint.x, 0, naturalSize.width),
        y: clamp(svgPoint.y, 0, naturalSize.height),
      };
    },
    [naturalSize.height, naturalSize.width]
  );

  const finishDrag = useCallback(
    (start: { x: number; y: number }, end: { x: number; y: number }) => {
      const normalized = normalizeBox(start.x, start.y, end.x, end.y);
      if (normalized.x1 - normalized.x0 < 5 || normalized.y1 - normalized.y0 < 5) {
        return;
      }
      const next: AnnotationBox = {
        id: `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
        type: modeRef.current,
        ...normalized,
      };
      onBoxesChange([...boxesRef.current, next]);
    },
    [onBoxesChange]
  );

  const handleDeleteSelected = useCallback(() => {
    if (!selectedBoxId) return;
    onBoxesChange(boxesRef.current.filter((box) => box.id !== selectedBoxId));
    setSelectedBoxId(null);
  }, [onBoxesChange, selectedBoxId]);

  const positiveIndexMap = getBoxIndexMap(boxes, 'positive');
  const negativeIndexMap = getBoxIndexMap(boxes, 'negative');

  const strokeWidth = naturalSize.width > 0 ? Math.max(2, naturalSize.width / 400) : 2;
  const selectedStrokeWidth = strokeWidth * 2;
  const hitStrokeWidth = naturalSize.width > 0 ? Math.max(14, naturalSize.width / 80) : 14;
  const labelFontSize = naturalSize.width > 0 ? Math.max(12, naturalSize.width / 120) : 14;

  const selectedBox = selectedBoxId ? boxes.find((box) => box.id === selectedBoxId) : null;
  const selectedLabel = selectedBox
    ? `${selectedBox.type === 'positive' ? '正框' : '负框'} #${
        (selectedBox.type === 'positive'
          ? (positiveIndexMap.get(selectedBox.id) ?? 0)
          : (negativeIndexMap.get(selectedBox.id) ?? 0)) + 1
      }`
    : null;

  const panelHeightCss = `min(${Math.round(maxPanelHeight)}px, 58vh)`;

  return (
    <div
      ref={panelRef}
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        height: panelHeightCss,
        minHeight: MIN_PANEL_HEIGHT,
        overflow: 'hidden',
        borderRadius: 12,
        border: '1px solid #e2e8f0',
        background: '#f8fafc',
      }}
    >
      {displaySize.width > 0 && displaySize.height > 0 ? (
        <div
          style={{
            position: 'relative',
            display: 'inline-block',
            width: displaySize.width,
            height: displaySize.height,
            maxWidth: '100%',
            maxHeight: '100%',
            flexShrink: 0,
          }}
        >
          <img
            src={imageUrl}
            alt="框选原图"
            draggable={false}
            onLoad={(event) => {
              const img = event.currentTarget;
              const width = img.naturalWidth;
              const height = img.naturalHeight;
              if (width <= 0 || height <= 0) {
                onImageError?.('图片已加载但无法读取原始尺寸');
                return;
              }
              setNaturalSize({ width, height });
              onImageSizeChange?.({ width, height });
            }}
            onError={() => onImageError?.('原图加载失败')}
            style={{
              display: 'block',
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              userSelect: 'none',
              pointerEvents: 'none',
            }}
          />
          <svg
            ref={svgRef}
            viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
            preserveAspectRatio="none"
            style={{
              position: 'absolute',
              inset: 0,
              zIndex: 10,
              width: '100%',
              height: '100%',
              cursor: disabled ? 'not-allowed' : 'crosshair',
              pointerEvents: disabled ? 'none' : 'auto',
              touchAction: 'none',
            }}
            onPointerDown={(event) => {
              if (disabled) return;
              setSelectedBoxId(null);
              event.preventDefault();
              event.currentTarget.setPointerCapture(event.pointerId);
              const point = getSvgPoint(event);
              if (!point) return;
              setDragStart(point);
              setDragCurrent(point);
            }}
            onPointerMove={(event) => {
              const point = getSvgPoint(event);
              if (point) onHoverPointChange?.({ x: Math.round(point.x), y: Math.round(point.y) });
              if (!dragStart) return;
              if (point) setDragCurrent(point);
            }}
            onPointerUp={(event) => {
              if (!dragStart) return;
              const point = getSvgPoint(event) || dragCurrent;
              if (point) finishDrag(dragStart, point);
              setDragStart(null);
              setDragCurrent(null);
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerCancel={(event) => {
              setDragStart(null);
              setDragCurrent(null);
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerLeave={() => {
              if (!dragStart) onHoverPointChange?.(null);
            }}
          >
            {boxes.map((box) => {
              const isSelected = box.id === selectedBoxId;
              const labelIndex =
                box.type === 'positive'
                  ? positiveIndexMap.get(box.id) ?? 0
                  : negativeIndexMap.get(box.id) ?? 0;
              const label = boxLabel(box.type, labelIndex);
              const stroke = boxStroke(box.type);
              return (
                <g key={box.id}>
                  <rect
                    x={box.x0}
                    y={box.y0}
                    width={box.x1 - box.x0}
                    height={box.y1 - box.y0}
                    fill={box.type === 'positive' ? 'rgba(22,163,74,0.15)' : 'rgba(220,38,38,0.15)'}
                    stroke={stroke}
                    strokeWidth={isSelected ? selectedStrokeWidth : strokeWidth}
                    pointerEvents="none"
                  />
                  <rect
                    x={box.x0}
                    y={box.y0}
                    width={box.x1 - box.x0}
                    height={box.y1 - box.y0}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={hitStrokeWidth}
                    pointerEvents="stroke"
                    onPointerDown={(event) => {
                      if (disabled) return;
                      event.stopPropagation();
                      setSelectedBoxId(box.id);
                      setDragStart(null);
                      setDragCurrent(null);
                    }}
                  />
                  <rect
                    x={box.x0}
                    y={Math.max(0, box.y0 - labelFontSize * 1.2)}
                    width={labelFontSize * 1.6}
                    height={labelFontSize * 1.1}
                    fill={stroke}
                    pointerEvents="none"
                  />
                  <text
                    x={box.x0 + labelFontSize * 0.3}
                    y={Math.max(labelFontSize, box.y0 - labelFontSize * 0.25)}
                    fill="#fff"
                    fontSize={labelFontSize}
                    fontWeight={600}
                    pointerEvents="none"
                  >
                    {label}
                  </text>
                </g>
              );
            })}
            {dragStart && dragCurrent ? (
              <rect
                x={Math.min(dragStart.x, dragCurrent.x)}
                y={Math.min(dragStart.y, dragCurrent.y)}
                width={Math.abs(dragCurrent.x - dragStart.x)}
                height={Math.abs(dragCurrent.y - dragStart.y)}
                fill={mode === 'positive' ? 'rgba(22,163,74,0.12)' : 'rgba(220,38,38,0.12)'}
                stroke={boxStroke(mode)}
                strokeDasharray="6 4"
                strokeWidth={strokeWidth}
                pointerEvents="none"
              />
            ) : null}
          </svg>
        </div>
      ) : (
        <img
          src={imageUrl}
          alt="框选原图"
          draggable={false}
          onLoad={(event) => {
            const img = event.currentTarget;
            const width = img.naturalWidth;
            const height = img.naturalHeight;
            if (width <= 0 || height <= 0) {
              onImageError?.('图片已加载但无法读取原始尺寸');
              return;
            }
            setNaturalSize({ width, height });
            onImageSizeChange?.({ width, height });
          }}
          onError={() => onImageError?.('原图加载失败')}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            opacity: 0,
            pointerEvents: 'none',
            position: 'absolute',
          }}
        />
      )}

      {selectedBox && selectedLabel ? (
        <div
          style={{
            position: 'absolute',
            right: 12,
            top: 12,
            zIndex: 20,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 12px',
            borderRadius: 8,
            border: '1px solid #e2e8f0',
            background: 'rgba(255,255,255,0.95)',
            boxShadow: '0 4px 12px rgba(15,23,42,0.12)',
            fontSize: 13,
            color: '#334155',
          }}
        >
          <span style={{ fontWeight: 600 }}>已选中：{selectedLabel}</span>
          <button
            type="button"
            onClick={handleDeleteSelected}
            disabled={disabled}
            style={{
              padding: '4px 10px',
              borderRadius: 6,
              border: '1px solid #fecaca',
              background: '#fef2f2',
              color: '#b91c1c',
              cursor: disabled ? 'not-allowed' : 'pointer',
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            删除该框
          </button>
          <button
            type="button"
            onClick={() => setSelectedBoxId(null)}
            style={{
              padding: '4px 10px',
              borderRadius: 6,
              border: '1px solid #e2e8f0',
              background: '#fff',
              color: '#475569',
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            取消
          </button>
        </div>
      ) : null}
    </div>
  );
}

function getBoxIndexMap(boxes: AnnotationBox[], type: BoxMode): Map<string, number> {
  const map = new Map<string, number>();
  boxes
    .filter((box) => box.type === type)
    .forEach((box, index) => map.set(box.id, index));
  return map;
}

export function boxesToPayload(boxes: AnnotationBox[]) {
  const positiveBoxes = boxes
    .filter((box) => box.type === 'positive')
    .map((box) => [box.x0, box.y0, box.x1, box.y1]);
  const negativeBoxes = boxes
    .filter((box) => box.type === 'negative')
    .map((box) => [box.x0, box.y0, box.x1, box.y1]);
  return { positiveBoxes, negativeBoxes };
}

export function validateBoxCoords(
  box: Pick<AnnotationBox, 'x0' | 'y0' | 'x1' | 'y1'>,
  naturalSize: { width: number; height: number } | null
): string | null {
  const { x0, y0, x1, y1 } = box;
  if (x0 >= x1 || y0 >= y1) return '必须满足 x0 < x1 且 y0 < y1';
  if (x0 < 0 || y0 < 0) return '坐标不能为负数';
  if (naturalSize) {
    if (x1 > naturalSize.width || y1 > naturalSize.height) {
      return `坐标超出原图范围（${naturalSize.width}×${naturalSize.height}）`;
    }
  }
  if (x1 - x0 < 5 || y1 - y0 < 5) return '框宽高至少 5 像素';
  return null;
}
