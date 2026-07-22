'use client';

import { useMemo, useState } from 'react';
import { GhostButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  type AnnotationBox,
  type BoxMode,
  validateBoxCoords,
} from '@/components/workspace/sim-assets/ImageBoxAnnotator';
import { formControlStyle } from '@/components/workspace/sim-assets/simAssetUi';

interface SegmentParamsPanelProps {
  prompt: string;
  boxes: AnnotationBox[];
  confidenceThreshold: number;
  naturalSize: { width: number; height: number } | null;
  onBoxesChange: (boxes: AnnotationBox[]) => void;
}

function boxCommand(box: AnnotationBox): string {
  const coords = `${box.x0},${box.y0},${box.x1},${box.y1}`;
  return box.type === 'positive' ? `--pos-box ${coords}` : `--neg-box ${coords}`;
}

function buildCommandPreview(prompt: string, boxes: AnnotationBox[], confidenceThreshold: number): string {
  const lines = ['python scripts/run_sam3_box_select.py \\'];
  if (prompt.trim()) {
    lines.push(`  --prompt "${prompt.trim()}" \\`);
  }
  boxes
    .filter((box) => box.type === 'positive')
    .forEach((box) => lines.push(`  ${boxCommand(box)} \\`));
  boxes
    .filter((box) => box.type === 'negative')
    .forEach((box) => lines.push(`  ${boxCommand(box)} \\`));
  lines.push(`  --confidence-threshold ${confidenceThreshold} \\`);
  lines.push('  --no-neg-interactive');
  return lines.join('\n');
}

export function SegmentParamsPanel({
  prompt,
  boxes,
  confidenceThreshold,
  naturalSize,
  onBoxesChange,
}: SegmentParamsPanelProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState({ x0: 0, y0: 0, x1: 0, y1: 0 });
  const [editError, setEditError] = useState<string | null>(null);

  const positiveBoxes = useMemo(() => boxes.filter((box) => box.type === 'positive'), [boxes]);
  const negativeBoxes = useMemo(() => boxes.filter((box) => box.type === 'negative'), [boxes]);

  const payloadPreview = useMemo(() => {
    const positive = positiveBoxes.map((box) => [box.x0, box.y0, box.x1, box.y1]);
    const negative = negativeBoxes.map((box) => [box.x0, box.y0, box.x1, box.y1]);
    return JSON.stringify(
      {
        prompt: prompt.trim() || null,
        positiveBoxes: positive,
        negativeBoxes: negative,
        confidenceThreshold,
        textOnly: positive.length === 0,
      },
      null,
      2
    );
  }, [confidenceThreshold, negativeBoxes, positiveBoxes, prompt]);

  const commandPreview = useMemo(
    () => buildCommandPreview(prompt, boxes, confidenceThreshold),
    [boxes, confidenceThreshold, prompt]
  );

  const startEdit = (box: AnnotationBox) => {
    setEditingId(box.id);
    setEditDraft({ x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1 });
    setEditError(null);
  };

  const saveEdit = (boxId: string) => {
    const err = validateBoxCoords(editDraft, naturalSize);
    if (err) {
      setEditError(err);
      return;
    }
    onBoxesChange(
      boxes.map((box) =>
        box.id === boxId
          ? {
              ...box,
              x0: Math.round(editDraft.x0),
              y0: Math.round(editDraft.y0),
              x1: Math.round(editDraft.x1),
              y1: Math.round(editDraft.y1),
            }
          : box
      )
    );
    setEditingId(null);
    setEditError(null);
  };

  const renderBoxList = (items: AnnotationBox[], emptyHint: string, typeLabel: string) => {
    if (!items.length) {
      return <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 8 }}>{emptyHint}</div>;
    }
    return items.map((box, index) => (
      <div
        key={box.id}
        style={{
          border: '1px solid #e5e7eb',
          borderRadius: 8,
          padding: 10,
          marginBottom: 8,
          background: '#fafafa',
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6, color: box.type === 'positive' ? '#16a34a' : '#dc2626' }}>
          {typeLabel} #{index + 1}
        </div>
        {editingId === box.id ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 8 }}>
            {(['x0', 'y0', 'x1', 'y1'] as const).map((key) => (
              <input
                key={key}
                type="number"
                value={editDraft[key]}
                onChange={(e) => setEditDraft((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                style={formControlStyle}
              />
            ))}
          </div>
        ) : (
          <div style={{ fontSize: 13, marginBottom: 6 }}>
            x0={box.x0}, y0={box.y0}, x1={box.x1}, y1={box.y1}
          </div>
        )}
        <code style={{ display: 'block', fontSize: 12, color: '#374151', marginBottom: 8 }}>{boxCommand(box)}</code>
        {editError && editingId === box.id ? (
          <div style={{ color: '#b91c1c', fontSize: 12, marginBottom: 6 }}>{editError}</div>
        ) : null}
        <div style={{ display: 'flex', gap: 8 }}>
          {editingId === box.id ? (
            <>
              <SecondaryButton onClick={() => saveEdit(box.id)}>保存</SecondaryButton>
              <GhostButton onClick={() => setEditingId(null)}>取消</GhostButton>
            </>
          ) : (
            <>
              <GhostButton onClick={() => startEdit(box)}>编辑</GhostButton>
              <GhostButton onClick={() => onBoxesChange(boxes.filter((item) => item.id !== box.id))}>删除</GhostButton>
            </>
          )}
        </div>
      </div>
    ));
  };

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 10 }}>分割参数</div>

      <div style={{ fontSize: 13, marginBottom: 12 }}>
        <div style={{ color: '#6b7280', marginBottom: 4 }}>Prompt:</div>
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
          {prompt.trim() || '（空）'}
        </pre>
      </div>

      <div style={{ fontSize: 13, marginBottom: 8, fontWeight: 500 }}>正框 positiveBoxes:</div>
      {renderBoxList(
        positiveBoxes,
        '暂无正框，请在图片上使用正框模式拖拽选择目标区域。',
        '正框'
      )}

      <div style={{ fontSize: 13, margin: '12px 0 8px', fontWeight: 500 }}>负框 negativeBoxes:</div>
      {renderBoxList(negativeBoxes, '暂无负框，可选。', '负框')}

      <div style={{ fontSize: 13, marginTop: 12 }}>
        <div style={{ color: '#6b7280', marginBottom: 4 }}>命令行预览（仅展示，不会在此执行）:</div>
        <pre
          style={{
            margin: 0,
            padding: 12,
            borderRadius: 8,
            background: '#111827',
            color: '#e5e7eb',
            fontSize: 12,
            overflowX: 'auto',
          }}
        >
          {commandPreview}
        </pre>
      </div>

      <div style={{ fontSize: 13, marginTop: 12 }}>
        <div style={{ color: '#6b7280', marginBottom: 4 }}>请求 payload 预览:</div>
        <pre
          style={{
            margin: 0,
            padding: 12,
            borderRadius: 8,
            background: '#f3f4f6',
            fontSize: 12,
            overflowX: 'auto',
          }}
        >
          {payloadPreview}
        </pre>
      </div>
    </div>
  );
}

export function clearBoxesByType(boxes: AnnotationBox[], type: BoxMode | 'all'): AnnotationBox[] {
  if (type === 'all') return [];
  return boxes.filter((box) => box.type !== type);
}
