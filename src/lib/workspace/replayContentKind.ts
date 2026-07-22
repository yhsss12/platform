export type ReplayContentKind =
  | 'dataset_trajectory_replay'
  | 'generation_process_preview'
  | 'evaluation_replay';

export interface ReplayContentTab {
  id: ReplayContentKind | string;
  label: string;
}

export interface ReplayFailureRecord {
  sourceEpisodeIndex?: number | null;
  displayEpisodeNumber?: number | null;
  episodeIndex?: number | null;
  seed?: number | null;
  failureReason?: string | null;
  failureCode?: string | null;
  writtenToDataset?: boolean;
}

export interface ReplayTrajectoryRecord {
  demoName: string;
  sourceEpisodeIndex?: number | null;
  displayEpisodeNumber?: number | null;
  successfulTrajectoryIndex?: number | null;
  seed?: number | null;
  writtenToDataset?: boolean;
}

export interface ReplayContentDetection {
  replayContentKind: ReplayContentKind;
  hasHdf5Trajectories: boolean;
  trajectoryCount?: number | null;
  totalEpisodes?: number | null;
  failedEpisodes?: number | null;
  hasGenerationPreview: boolean;
  hasFailures: boolean;
  hasEvaluationResult?: boolean;
  primarySource?: string | null;
  tabs: ReplayContentTab[];
  trajectories?: string[];
  trajectoryRecords?: ReplayTrajectoryRecord[];
  failureRecords?: ReplayFailureRecord[];
  debug?: Record<string, unknown>;
  hasRgbObservation?: boolean;
  rgbCameras?: string[];
  trajectoryDisplayMode?: 'rgb_frame_replay' | 'state_trajectory';
}

export const REPLAY_CONTENT_COPY: Record<
  ReplayContentKind,
  { title: string; subtitle: string; tag?: string | null }
> = {
  dataset_trajectory_replay: {
    title: '数据集轨迹回放',
    subtitle: '当前展示 HDF5 数据集中的有效轨迹。',
  },
  generation_process_preview: {
    title: '生成过程预览',
    subtitle: '用于查看本次 expert 数据生成过程。',
    tag: 'Expert Rollout / Demonstration Generation',
  },
  evaluation_replay: {
    title: '评测回放',
    subtitle: '查看评测任务回放与指标结果。',
  },
};

export function resolveDefaultReplayTab(content: ReplayContentDetection): ReplayContentKind {
  if (content.replayContentKind === 'evaluation_replay') {
    return 'evaluation_replay';
  }
  if (content.hasHdf5Trajectories) {
    return 'dataset_trajectory_replay';
  }
  if (content.hasGenerationPreview) {
    return 'generation_process_preview';
  }
  return content.replayContentKind;
}

export function resolveReplayTabVideoPlayable(
  activeTab: ReplayContentKind,
  content: Pick<ReplayContentDetection, 'hasGenerationPreview'>
): boolean {
  return activeTab === 'generation_process_preview' && content.hasGenerationPreview;
}
