'use client';

import {
  ReplayPanelSectionTitle,
  ReplaySidePanelLayout,
} from '@/components/workspace/replay/ReplaySidePanelLayout';
import { EvaluationReplayMetricsBlock } from '@/components/workspace/replay/EvaluationReplayMetricsBlock';
import {
  formatGenerationMode,
  type ReplayAdapterResult,
} from '@/lib/workspace/replayAdapters';
import { resolveDatasetSimulatorBackendLabel } from '@/lib/workspace/datasetDisplay';
import { metricsInputFromReplayAdapter } from '@/lib/workspace/replayMetricsInput';
import type { ReplaySourceKind } from '@/lib/workspace/replayViewModel';
import type { ReplayContentKind } from '@/lib/workspace/replayContentKind';

function InfoRow({ label, value }: { label: string; value?: string | number | boolean | null }) {
  if (value == null || value === '') return null;
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 12, lineHeight: 1.6 }}>
      <span style={{ color: '#9ca3af', minWidth: 72, flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#374151', wordBreak: 'break-all' }}>{String(value)}</span>
    </div>
  );
}

interface ReplayInfoPanelProps {
  adapter: ReplayAdapterResult;
  sourceKind: ReplaySourceKind;
  activeReplayTab?: ReplayContentKind | null;
  selectedTrajectory?: string | null;
  generating?: boolean;
  returnPath?: string;
  returnLabel?: string;
  onGenerateReplay?: () => void;
}

export function ReplayInfoPanel({
  adapter,
  sourceKind,
  activeReplayTab,
  selectedTrajectory,
}: ReplayInfoPanelProps) {
  const isDataset = sourceKind === 'dataset';
  const generationMode = formatGenerationMode(adapter.generationMode);
  const showTrajectoryStats =
    adapter.replayContentKind === 'dataset_trajectory_replay' ||
    activeReplayTab === 'dataset_trajectory_replay';
  const failureRecords = adapter.failureRecords ?? [];

  return (
    <ReplaySidePanelLayout fitContent={isDataset}>
      {isDataset ? (
        <div>
          {showTrajectoryStats ? (
            <>
              <ReplayPanelSectionTitle>统计信息</ReplayPanelSectionTitle>
              <InfoRow
                label="有效轨迹"
                value={
                  adapter.trajectoryCount != null ? `${adapter.trajectoryCount} 条` : adapter.episodeCount
                }
              />
              <InfoRow
                label="本次生成"
                value={adapter.totalEpisodes != null ? `${adapter.totalEpisodes} 轮` : undefined}
              />
              <InfoRow
                label="失败轮次"
                value={
                  adapter.failedEpisodesCount != null ? `${adapter.failedEpisodesCount} 轮` : undefined
                }
              />
              <InfoRow label="数据来源" value={adapter.primarySource ?? 'dataset.hdf5'} />
              {selectedTrajectory ? (
                <InfoRow label="当前轨迹" value={selectedTrajectory} />
              ) : null}
              {failureRecords.length > 0 ? (
                <div style={{ marginTop: 16 }}>
                  <ReplayPanelSectionTitle>失败记录</ReplayPanelSectionTitle>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {failureRecords.map((record, index) => (
                      <div
                        key={`${record.episodeIndex ?? 'na'}-${record.seed ?? index}`}
                        style={{
                          padding: '10px 12px',
                          borderRadius: 10,
                          border: '1px solid #e5e7eb',
                          background: '#f9fafb',
                        }}
                      >
                        <InfoRow
                          label="失败轮次"
                          value={
                            record.episodeIndex != null ? `第 ${record.episodeIndex} 轮` : undefined
                          }
                        />
                        <InfoRow label="seed" value={record.seed ?? undefined} />
                        <InfoRow label="失败原因" value={record.failureReason ?? undefined} />
                        <InfoRow
                          label="写入数据集"
                          value={record.writtenToDataset ? 'true' : 'false'}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          ) : null}

          <div style={{ marginTop: showTrajectoryStats ? 16 : 0 }}>
            <ReplayPanelSectionTitle>数据集信息</ReplayPanelSectionTitle>
            <InfoRow label="数据集名称" value={adapter.datasetName} />
            <InfoRow label="来源任务" value={adapter.taskName} />
            <InfoRow label="数据来源" value={adapter.datasetSourceLabel} />
            <InfoRow
              label="仿真后端"
              value={resolveDatasetSimulatorBackendLabel(adapter.simulatorBackend) ?? adapter.simulatorBackend}
            />
            <InfoRow label="数据格式" value={adapter.datasetFormat} />
            {!showTrajectoryStats ? (
              <InfoRow label="有效轨迹" value={adapter.episodeCount} />
            ) : null}
            <InfoRow label="状态" value={adapter.status} />
            {generationMode ? <InfoRow label="生成方式" value={generationMode} /> : null}
            {adapter.trajectoryQualityLabel ? (
              <InfoRow label="轨迹质量" value={adapter.trajectoryQualityLabel} />
            ) : null}
            {adapter.sourceJobId ? <InfoRow label="sourceJobId" value={adapter.sourceJobId} /> : null}
            {adapter.createdAt ? <InfoRow label="生成时间" value={adapter.createdAt} /> : null}
            {adapter.metricsSummary?.success != null ? (
              <InfoRow label="Episode 成功" value={String(adapter.metricsSummary.success)} />
            ) : null}
            {adapter.metricsSummary?.duration_sec != null ? (
              <InfoRow label="时长 (s)" value={String(adapter.metricsSummary.duration_sec)} />
            ) : null}
            {adapter.metricsSummary?.episode_length != null ? (
              <InfoRow label="Episode 步数" value={String(adapter.metricsSummary.episode_length)} />
            ) : null}
            {adapter.metricsSummary?.pick_success != null ? (
              <InfoRow label="抓取成功" value={String(adapter.metricsSummary.pick_success)} />
            ) : null}
            {adapter.metricsSummary?.place_success != null ? (
              <InfoRow label="放置成功" value={String(adapter.metricsSummary.place_success)} />
            ) : null}
            {adapter.metricsSummary?.controller_done != null ? (
              <InfoRow label="控制器完成" value={String(adapter.metricsSummary.controller_done)} />
            ) : null}
            {adapter.metadata?.dataset_hdf5_path ? (
              <InfoRow label="dataset.hdf5" value={String(adapter.metadata.dataset_hdf5_path)} />
            ) : null}
            {adapter.metadata?.episode_count != null ? (
              <InfoRow label="episode_count" value={String(adapter.metadata.episode_count)} />
            ) : null}
            {adapter.metadata?.successfulEpisodes != null ? (
              <InfoRow
                label="successfulEpisodes"
                value={String(adapter.metadata.successfulEpisodes)}
              />
            ) : null}
            {adapter.metadata?.manifestPath ? (
              <InfoRow label="Manifest" value={String(adapter.metadata.manifestPath)} />
            ) : null}
            {adapter.metadata?.trajectory_path ? (
              <InfoRow label="Trajectory" value={String(adapter.metadata.trajectory_path)} />
            ) : null}
            {adapter.metadata?.metrics_path ? (
              <InfoRow label="Metrics" value={String(adapter.metadata.metrics_path)} />
            ) : null}
          </div>
        </div>
      ) : (
        <>
          <div style={{ marginBottom: 16 }}>
            <ReplayPanelSectionTitle>评测信息</ReplayPanelSectionTitle>
            <InfoRow label="来源任务" value={adapter.taskName} />
            <InfoRow label="仿真后端" value={adapter.simulatorBackend} />
            <InfoRow label="状态" value={adapter.sourceJobStatus ?? adapter.status} />
            {adapter.metadata.evalJobId != null ? (
              <InfoRow label="评测任务 ID" value={String(adapter.metadata.evalJobId)} />
            ) : null}
          </div>
          <div>
            <ReplayPanelSectionTitle>评测指标</ReplayPanelSectionTitle>
            <EvaluationReplayMetricsBlock {...metricsInputFromReplayAdapter(adapter)} />
          </div>
        </>
      )}
    </ReplaySidePanelLayout>
  );
}
