'use client';

import type { TrainingBackendRequest } from '@/lib/api/trainingClient';

export type TrainingSaveCapabilities = {
  final: boolean;
  best: boolean;
  interval: boolean;
};

export const TRAINING_SAVE_CAPABILITIES: Record<string, TrainingSaveCapabilities> = {
  robomimic_bc: { final: true, best: true, interval: true },
  isaac_robomimic_bc: { final: true, best: false, interval: true },
  diffusion_policy: { final: true, best: false, interval: false },
  torch_bc: { final: true, best: false, interval: false },
};

export function trainingSaveCapabilities(
  backend: TrainingBackendRequest | string
): TrainingSaveCapabilities {
  return TRAINING_SAVE_CAPABILITIES[backend] ?? { final: true, best: false, interval: false };
}

const labelStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  fontSize: 13,
  color: '#374151',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: '#6b7280',
  marginBottom: 10,
};

const hintStyle: React.CSSProperties = {
  marginTop: 8,
  fontSize: 12,
  color: '#b45309',
};

export function TrainingCheckpointSaveSection({
  backend,
  totalEpochs,
  saveFinal,
  saveBest,
  intervalEnabled,
  checkpointIntervalEpochs,
  onSaveFinalChange,
  onSaveBestChange,
  onIntervalEnabledChange,
  onCheckpointIntervalChange,
}: {
  backend: TrainingBackendRequest | string;
  totalEpochs: number;
  saveFinal: boolean;
  saveBest: boolean;
  intervalEnabled: boolean;
  checkpointIntervalEpochs: number;
  onSaveFinalChange: (value: boolean) => void;
  onSaveBestChange: (value: boolean) => void;
  onIntervalEnabledChange: (value: boolean) => void;
  onCheckpointIntervalChange: (value: number) => void;
}) {
  const caps = trainingSaveCapabilities(backend);
  const intervalInvalid =
    intervalEnabled &&
    caps.interval &&
    (checkpointIntervalEpochs <= 0 || checkpointIntervalEpochs > totalEpochs);

  return (
    <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #f3f4f6' }}>
      <div style={sectionTitleStyle}>模型保存</div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '12px 20px',
        }}
      >
        <label style={{ ...labelStyle, opacity: caps.final ? 1 : 0.5 }}>
          <input
            type="checkbox"
            checked={saveFinal}
            disabled={!caps.final}
            onChange={(e) => onSaveFinalChange(e.target.checked)}
          />
          保存最终模型
        </label>
        {caps.best ? (
          <label style={labelStyle}>
            <input
              type="checkbox"
              checked={saveBest}
              onChange={(e) => onSaveBestChange(e.target.checked)}
            />
            保存最佳模型
          </label>
        ) : null}
        <label style={{ ...labelStyle, opacity: caps.interval ? 1 : 0.5 }}>
          <input
            type="checkbox"
            checked={intervalEnabled}
            disabled={!caps.interval}
            onChange={(e) => onIntervalEnabledChange(e.target.checked)}
          />
          按 Epoch 间隔保存
        </label>
        {intervalEnabled && caps.interval ? (
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 13, color: '#6b7280' }}>间隔</span>
            <input
              type="number"
              min={1}
              max={totalEpochs}
              value={checkpointIntervalEpochs}
              onChange={(e) => onCheckpointIntervalChange(Number(e.target.value) || 1)}
              style={{
                width: 64,
                padding: '5px 8px',
                borderRadius: 6,
                border: '1px solid #d1d5db',
                fontSize: 13,
              }}
            />
            <span style={{ fontSize: 13, color: '#6b7280' }}>epochs</span>
          </div>
        ) : null}
      </div>
      {!caps.interval ? (
        <div style={{ marginTop: 8, fontSize: 12, color: '#9ca3af' }}>
          当前算法暂不支持按 Epoch 间隔保存中间 checkpoint。
        </div>
      ) : null}
      {intervalInvalid ? (
        <div style={hintStyle}>
          保存间隔需为 1–{totalEpochs} 的正整数；总 Epoch 为 {totalEpochs} 时不会保存中间 checkpoint。
        </div>
      ) : null}
    </div>
  );
}
