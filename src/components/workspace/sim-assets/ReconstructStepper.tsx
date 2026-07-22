'use client';

import type { AssetPipelineJobStatus } from '@/lib/api/sam3dAssetPipelineClient';

export type ReconstructStep = 1 | 2 | 3 | 4 | 5;

export const RECONSTRUCT_STEPS: { step: ReconstructStep; title: string }[] = [
  { step: 1, title: '上传图片' },
  { step: 2, title: '目标选择' },
  { step: 3, title: '分割确认' },
  { step: 4, title: '三维重建' },
  { step: 5, title: '转成资产' },
];

export function resolveReconstructStep(
  jobId: string | null,
  jobStatus: AssetPipelineJobStatus | null,
  manualStep?: ReconstructStep | null
): ReconstructStep {
  if (manualStep != null) return manualStep;
  if (!jobId) return 1;
  const status = jobStatus?.status || 'created';
  if (status === 'created') return 2;
  if (status === 'segmenting') return 2;
  if (status === 'segmented') return 3;
  if (status === 'reconstructing') return 4;
  if (status === 'reconstructed') return 4;
  if (status === 'failed') {
    const phase = jobStatus?.phase || '';
    if (phase.startsWith('sam3')) return 2;
    if (phase.startsWith('sam3d')) return 4;
    return 2;
  }
  return 2;
}

export function stepCompleted(step: ReconstructStep, current: ReconstructStep, status?: string): boolean {
  if (step < current) return true;
  if (step === 1 && status && status !== 'created') return true;
  if (step === 2 && ['segmented', 'reconstructing', 'reconstructed'].includes(status || '')) return true;
  if (step === 3 && ['reconstructing', 'reconstructed'].includes(status || '')) return true;
  if (step === 4 && status === 'reconstructed') return true;
  return false;
}

interface ReconstructStepperProps {
  currentStep: ReconstructStep;
  status?: string;
  onStepClick?: (step: ReconstructStep) => void;
}

export function ReconstructStepper({ currentStep, status, onStepClick }: ReconstructStepperProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
        gap: 8,
        marginBottom: 20,
      }}
    >
      {RECONSTRUCT_STEPS.map(({ step, title }) => {
        const active = step === currentStep;
        const done = stepCompleted(step, currentStep, status);
        const clickable = done || step <= currentStep;
        return (
          <button
            key={step}
            type="button"
            disabled={!clickable}
            onClick={() => clickable && onStepClick?.(step)}
            style={{
              border: `1px solid ${active ? '#2563eb' : done ? '#86efac' : '#d1d5db'}`,
              background: active ? '#eff6ff' : done ? '#f0fdf4' : '#fff',
              color: active ? '#1d4ed8' : done ? '#166534' : '#374151',
              borderRadius: 10,
              padding: '10px 8px',
              fontSize: 12,
              fontWeight: active ? 600 : 500,
              cursor: clickable ? 'pointer' : 'default',
              opacity: clickable ? 1 : 0.65,
            }}
          >
            <div>{step}. {title}</div>
            {done && !active ? <div style={{ fontSize: 11, marginTop: 4 }}>已完成</div> : null}
          </button>
        );
      })}
    </div>
  );
}
