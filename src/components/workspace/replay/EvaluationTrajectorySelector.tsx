'use client';

import {
  getTrajectoryLabel,
  type EvaluationReplayUriItem,
} from '@/lib/workspace/evaluationReplayInfo';

export interface EvaluationTrajectorySelectorProps {
  replayItems: EvaluationReplayUriItem[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  className?: string;
}

export function EvaluationTrajectorySelector({
  replayItems,
  selectedIndex,
  onSelect,
  className = '',
}: EvaluationTrajectorySelectorProps) {
  if (replayItems.length <= 1) return null;

  const isRepresentativeOnly =
    replayItems.length === 1 && replayItems[0]?.label === '代表性回放';
  if (isRepresentativeOnly) return null;

  return (
    <select
      value={selectedIndex}
      onChange={(event) => onSelect(Number(event.target.value))}
      aria-label="选择轨迹轮次"
      className={
        className ||
        'h-8 min-w-[140px] rounded-md border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200'
      }
      style={{ cursor: 'pointer' }}
    >
      {replayItems.map((item, index) => (
        <option key={`${item.episodeIndex ?? index}-${item.uri}`} value={index}>
          {getTrajectoryLabel(item, index)}
        </option>
      ))}
    </select>
  );
}
