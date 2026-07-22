'use client';

import { InfoRow } from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import {
  buildIncompleteEpisodesWarning,
  buildRepresentativeVideoHint,
  formatEpisodeProgressPercent,
  type EvaluationReplayInfo,
} from '@/lib/workspace/evaluationReplayInfo';

function displayValue(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—';
  return String(value);
}

export function EvaluationReplayEpisodeStats({
  replay,
  horizon,
  seed,
  statusLabel,
}: {
  replay: EvaluationReplayInfo;
  horizon?: number | null;
  seed?: number | null;
  statusLabel: string;
}) {
  const progressPercent = formatEpisodeProgressPercent(
    replay.completedEpisodes,
    replay.requestedEpisodes
  );
  const representativeHint = buildRepresentativeVideoHint(replay);
  const incompleteWarning = buildIncompleteEpisodesWarning(replay);

  return (
    <>
      <InfoRow label="计划轮数" value={displayValue(replay.requestedEpisodes)} />
      <InfoRow label="实际完成" value={displayValue(replay.completedEpisodes)} />
      <InfoRow label="成功轮数" value={displayValue(replay.successfulEpisodes)} />
      <InfoRow label="失败轮数" value={displayValue(replay.failedEpisodes)} />
      <InfoRow
        label="真实成功率"
        value={
          replay.successRate != null
            ? `${(replay.successRate * 100).toFixed(1)}%`
            : '—'
        }
      />
      <InfoRow
        label="回放视频"
        value={
          replay.recordedVideoCount != null && replay.recordedVideoCount > 0
            ? `${replay.recordedVideoCount} 段${
                replay.isRepresentativeVideo ? '（代表性）' : ''
              }`
            : '暂无'
        }
      />
      {horizon != null ? <InfoRow label="Horizon" value={String(horizon)} /> : null}
      {seed != null ? <InfoRow label="Seed" value={String(seed)} /> : null}
      <InfoRow
        label="进度"
        value={
          progressPercent ??
          (statusLabel === '已完成'
            ? '100%'
            : statusLabel === '失败'
              ? '评测失败'
              : statusLabel)
        }
      />
      {representativeHint ? (
        <p style={{ margin: '8px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          {representativeHint}
        </p>
      ) : null}
      {incompleteWarning ? (
        <p style={{ margin: '8px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          {incompleteWarning}
        </p>
      ) : null}
      {replay.warning ? (
        <p style={{ margin: '8px 0 0', fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          {replay.warning}
        </p>
      ) : null}
    </>
  );
}
