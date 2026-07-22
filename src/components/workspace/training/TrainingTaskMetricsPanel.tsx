'use client';

import {
  CartesianGrid,
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { TrainingMetricPoint } from '@/lib/workspace/trainingLogParser';
import { formatLossAxisTick, formatLossValue } from '@/lib/workspace/chartFormat';
import { resolveTrainingLossFieldLabel } from '@/lib/workspace/trainingLossDisplay';
import type { TrainingDisplayState } from '@/lib/workspace/trainingDisplayState';

const GRID_STROKE = 'rgba(15, 23, 42, 0.06)';
const TICK_FILL = '#9ca3af';

function chartData(points: TrainingMetricPoint[]) {
  return points.map((p) => ({
    epoch: p.epoch,
    trainLoss: p.trainLoss,
    validLoss: p.validLoss,
  }));
}

function lossYDomain(values: number[]): [number, number] | ['auto', 'auto'] {
  if (values.length === 0) return ['auto', 'auto'];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    const pad = Math.max(Math.abs(min) * 0.05, 0.01);
    return [min - pad, max + pad];
  }
  const span = max - min;
  const pad = Math.max(span * 0.08, 0.01);
  return [min - pad, max + pad];
}

function MetricsChart({
  title,
  points,
  totalEpochs,
}: {
  title: string;
  points: TrainingMetricPoint[];
  totalEpochs?: number;
}) {
  const data = chartData(points);
  const hasTrainLoss = data.some((p) => p.trainLoss != null);
  const hasValid = data.some((p) => p.validLoss != null);
  const showValidLossNote = hasTrainLoss && !hasValid;
  const maxEpoch = data.length > 0 ? Math.max(...data.map((p) => p.epoch)) : 1;
  const axisMax = totalEpochs && totalEpochs > 0 ? totalEpochs : maxEpoch;
  const shellData = data.length > 0 ? data : [{ epoch: 0 }, { epoch: Math.max(axisMax, 1) }];
  const trainValues = data.map((p) => p.trainLoss).filter((v): v is number => v != null && Number.isFinite(v));
  const validValues = data.map((p) => p.validLoss).filter((v): v is number => v != null && Number.isFinite(v));
  const allLossValues = [...trainValues, ...validValues];
  const yDomain = lossYDomain(allLossValues);
  const xDomain: [number, number] | ['auto', 'auto'] =
    data.length === 1
      ? [Math.max(0, data[0].epoch - 0.5), Math.max(data[0].epoch + 0.5, axisMax)]
      : axisMax > 0
        ? [0.5, axisMax + 0.5]
        : ['auto', 'auto'];

  return (
    <div
      style={{
        padding: '12px 14px',
        borderRadius: 8,
        border: '1px solid #e5e7eb',
        backgroundColor: '#fff',
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 10 }}>{title}</div>
      <div style={{ height: 168 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={shellData} margin={{ top: 8, right: 16, left: 2, bottom: 4 }}>
            <CartesianGrid stroke={GRID_STROKE} vertical={false} />
            <XAxis
              dataKey="epoch"
              type="number"
              domain={xDomain}
              allowDecimals={false}
              tickCount={axisMax > 20 ? 8 : undefined}
              tick={{ fontSize: 11, fill: TICK_FILL }}
              axisLine={{ stroke: GRID_STROKE }}
              tickLine={false}
              label={{
                value: 'Epoch',
                position: 'insideBottomRight',
                offset: -2,
                fontSize: 10,
                fill: TICK_FILL,
              }}
            />
            <YAxis
              domain={yDomain}
              allowDecimals
              tickFormatter={formatLossAxisTick}
              tick={{ fontSize: 11, fill: TICK_FILL }}
              width={52}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{ fontSize: 12, border: '1px solid rgba(15,23,42,0.06)', borderRadius: 8 }}
              labelFormatter={(epoch) => `Epoch ${epoch}`}
              formatter={(value: number, name: string) => [
                formatLossValue(value),
                name === 'trainLoss' ? 'Train Loss' : 'Valid Loss',
              ]}
            />
            {hasTrainLoss ? (
              <Line
                type="monotone"
                dataKey="trainLoss"
                name="trainLoss"
                stroke="#2563eb"
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 4 }}
                connectNulls
                isAnimationActive={false}
              />
            ) : null}
            {hasValid ? (
              <Line
                type="monotone"
                dataKey="validLoss"
                name="validLoss"
                stroke="#059669"
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 4 }}
                connectNulls
                isAnimationActive={false}
              />
            ) : null}
            {hasTrainLoss && hasValid ? (
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(v) => (v === 'trainLoss' ? 'Train Loss' : 'Valid Loss')}
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {showValidLossNote ? (
        <p style={{ margin: '8px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.45 }}>
          该训练后端仅记录 Train Loss（日志格式为 Epoch N Loss），未单独输出 Validation Loss，因此不显示 Valid Loss 曲线。
        </p>
      ) : null}
    </div>
  );
}

export function TrainingTaskMetricsPanel({
  metrics,
  status,
  displayState,
}: {
  metrics: {
    currentEpoch: number;
    totalEpochs: number;
    loss: number | null;
    lossSeries: TrainingMetricPoint[];
    progressPercent: number | null;
    bestLoss?: number | null;
    finalLoss?: number | null;
  };
  status?: string | null;
  displayState?: TrainingDisplayState;
}) {
  const progressPercent = displayState?.showProgressBar ? displayState.progressPercent : metrics.progressPercent;
  const lossLabel = resolveTrainingLossFieldLabel(status);
  const showSummaryLoss =
    displayState?.showFinalLoss !== false &&
    (metrics.bestLoss != null || metrics.finalLoss != null || metrics.loss != null);
  const isLaunching =
    displayState?.phase === 'created' || displayState?.phase === 'launching';
  const progressTitle = displayState?.progressLabel ?? `Epoch ${metrics.currentEpoch}/${metrics.totalEpochs || '—'}`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {displayState?.progressIndeterminate ? (
        <style>{`
          @keyframes trainingMetricsIndeterminate {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(250%); }
          }
        `}</style>
      ) : null}
      <div
        style={{
          padding: '12px 14px',
          borderRadius: 8,
          border: '1px solid #e5e7eb',
          backgroundColor: '#f9fafb',
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 10 }}>训练进度</div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 12,
            color: '#6b7280',
            marginBottom: 6,
          }}
        >
          <span>{progressTitle}</span>
          {displayState?.showProgressBar && progressPercent != null ? <span>{progressPercent}%</span> : null}
        </div>
        {displayState?.showProgressBar !== false ? (
          <div
            style={{
              height: 6,
              borderRadius: 3,
              backgroundColor: '#dbeafe',
              overflow: 'hidden',
            }}
          >
            {displayState?.progressIndeterminate ? (
              <div
                style={{
                  width: '40%',
                  height: '100%',
                  borderRadius: 3,
                  backgroundColor: '#2563eb',
                  animation: 'trainingMetricsIndeterminate 1.2s ease-in-out infinite',
                }}
              />
            ) : (
              <div
                style={{
                  width: `${progressPercent ?? 0}%`,
                  height: '100%',
                  borderRadius: 3,
                  backgroundColor: '#2563eb',
                  transition: 'width 0.25s ease',
                }}
              />
            )}
          </div>
        ) : null}
        {isLaunching && displayState?.progressHint ? (
          <p style={{ margin: '8px 0 0', fontSize: 12, color: '#92400e', lineHeight: 1.45 }}>
            {displayState.progressHint}
          </p>
        ) : null}
        {showSummaryLoss && !isLaunching ? (
          <div style={{ marginTop: 8, fontSize: 12, color: '#475569', display: 'flex', flexWrap: 'wrap', gap: '8px 16px' }}>
            {metrics.loss != null ? (
              <span>
                {lossLabel}：
                <span style={{ fontFamily: 'ui-monospace, monospace' }}>{formatLossValue(metrics.loss)}</span>
              </span>
            ) : null}
            {metrics.bestLoss != null ? (
              <span>
                Best Loss：
                <span style={{ fontFamily: 'ui-monospace, monospace' }}>{formatLossValue(metrics.bestLoss)}</span>
              </span>
            ) : null}
            {metrics.finalLoss != null && lossLabel === '最终 Loss' ? (
              <span>
                Final Loss：
                <span style={{ fontFamily: 'ui-monospace, monospace' }}>{formatLossValue(metrics.finalLoss)}</span>
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      <MetricsChart
        title="训练 Loss"
        points={displayState?.showLossChart === false ? [] : metrics.lossSeries}
        totalEpochs={metrics.totalEpochs}
      />
    </div>
  );
}
