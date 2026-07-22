import type { SimRunDisplayStatus } from '@/components/workspace/simulation/SimulationRunConsoleLayout';

export type RunConsoleSimulatorBackend = 'MuJoCo' | 'Isaac Lab';

export type RunConsoleDisplayStatus = SimRunDisplayStatus;

export interface RunConsoleInfoRow {
  label: string;
  value: string;
}

export type RunConsoleLiveFrameKind = 'cable_threading' | 'dual_arm_cable' | 'isaac_lab';

export interface RunConsoleLiveFrameConfig {
  kind: RunConsoleLiveFrameKind;
  jobId: string;
  pollEnabled: boolean;
  status: RunConsoleDisplayStatus;
  frameCount?: number;
  phase?: string | null;
}

export type RunConsoleViewportMode =
  | 'failed'
  | 'init'
  | 'live'
  | 'cameras_disabled'
  | 'video'
  | 'hdf5_replay';

export interface RunConsoleSceneViewModel {
  title: string;
  backendLabel: string;
  initializingText: string;
  frameStatusLine: string;
  frameStatusAccent: string;
  viewportMode: RunConsoleViewportMode;
  failedMessage?: string;
  liveFrame?: RunConsoleLiveFrameConfig;
  previewVideoApiPath?: string | null;
  hdf5ReplayHref?: string | null;
}

export interface RunConsoleSectionsViewModel {
  basicInfo: RunConsoleInfoRow[];
  assetConfig: RunConsoleInfoRow[];
  results: RunConsoleInfoRow[];
  debug?: RunConsoleInfoRow[];
}

export interface RunConsoleActionsViewModel {
  backToDataCenterHref: string;
  canViewReplay: boolean;
  showViewDataRecord: boolean;
  openReplay?: () => void;
}

export interface RunConsoleViewModel {
  jobId: string;
  taskName: string;
  taskTypeLabel: string;
  taskKindLabel: string;
  simulatorBackend: RunConsoleSimulatorBackend;
  status: RunConsoleDisplayStatus;
  progress: number;
  scene: RunConsoleSceneViewModel;
  sections: RunConsoleSectionsViewModel;
  actions: RunConsoleActionsViewModel;
}

export interface MimicGenProgressPanelViewModel {
  isFailed: boolean;
  message: string;
  stageLabel: string;
  generationMode: string;
  policyMode: string;
  sourceDemoPath: string;
  elapsedSeconds: number | null;
  lastHeartbeatAt?: string | null;
  episodesGenerated: number | string;
  episodesRequested: number | string;
  datagenFailedTrials: number | string;
  errorMessage?: string | null;
  logTail: string;
  logStaleHint?: string | null;
  traceback?: string | null;
  completionHint: string;
  replayHref?: string | null;
}

export function runConsoleResultStatusLabel(status: RunConsoleDisplayStatus): string {
  switch (status) {
    case 'completed':
      return '已完成';
    case 'failed':
      return '失败';
    case 'queued':
      return '等待中';
    default:
      return '运行中';
  }
}

export function runConsoleFileReadyLabel(
  exists: boolean | undefined,
  running: boolean,
  disabledLabel?: string
): string {
  if (disabledLabel) return disabledLabel;
  if (exists) return '已生成';
  return running ? '等待生成' : '未生成';
}
