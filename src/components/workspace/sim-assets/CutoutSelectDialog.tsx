'use client';

import { useEffect, useState } from 'react';
import { GhostButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  CutoutCandidateGrid,
  type CutoutManifestItem,
} from '@/components/workspace/sim-assets/CutoutCandidateGrid';

interface CutoutSelectDialogProps {
  open: boolean;
  jobId: string;
  items: CutoutManifestItem[];
  selectedCutoutIndex: number | null | undefined;
  onConfirm: (cutoutIndex: number) => void;
  onCancel: () => void;
}

export function CutoutSelectDialog({
  open,
  jobId,
  items,
  selectedCutoutIndex,
  onConfirm,
  onCancel,
}: CutoutSelectDialogProps) {
  const [pendingCutoutIndex, setPendingCutoutIndex] = useState<number | null>(
    selectedCutoutIndex ?? null
  );

  useEffect(() => {
    if (!open) return;
    if (
      selectedCutoutIndex != null &&
      items.some((item) => item.cutoutIndex === selectedCutoutIndex)
    ) {
      setPendingCutoutIndex(selectedCutoutIndex);
      return;
    }
    const first = items.find((item) => item.selectable !== false) ?? items[0];
    setPendingCutoutIndex(first?.cutoutIndex ?? null);
  }, [open, items, selectedCutoutIndex]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="cutout-select-dialog-title"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
        background: 'rgba(15, 23, 42, 0.45)',
      }}
      onClick={onCancel}
    >
      <div
        style={{
          width: 'min(720px, 90vw)',
          maxHeight: '70vh',
          display: 'flex',
          flexDirection: 'column',
          background: '#fff',
          borderRadius: 12,
          border: '1px solid #e5e7eb',
          boxShadow: '0 20px 40px rgba(15, 23, 42, 0.18)',
          overflow: 'hidden',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div
          id="cutout-select-dialog-title"
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            fontSize: 16,
            fontWeight: 600,
            color: '#111827',
          }}
        >
          选择 cutout
        </div>
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: 16,
            minHeight: 120,
          }}
        >
          {items.length ? (
            <CutoutCandidateGrid
              compact
              jobId={jobId}
              items={items}
              selectedCutoutIndex={pendingCutoutIndex}
              onSelect={setPendingCutoutIndex}
            />
          ) : (
            <div style={{ fontSize: 13, color: '#6b7280', padding: 12 }}>暂无可用 cutout</div>
          )}
        </div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
            padding: '12px 16px',
            borderTop: '1px solid #e5e7eb',
            background: '#fafafa',
          }}
        >
          <GhostButton onClick={onCancel}>取消</GhostButton>
          <SecondaryButton
            onClick={() => {
              if (pendingCutoutIndex == null) return;
              onConfirm(pendingCutoutIndex);
            }}
            disabled={pendingCutoutIndex == null}
          >
            确认选择
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
}
