import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import { CABLE_OBJECT_MODEL_OPTIONS } from '@/lib/mock/generateDataTaskParams';
import {
  getCableThreadingGenerateRun,
} from '@/lib/mock/workspaceMockFlowStore';
import {
  listWorkspaceDataItemsForUi,
  listWorkspaceEvaluationTasksForUi,
} from '@/lib/workspace/workspaceDataSources';
import { CABLE_THREADING_TASK_NAME, resolveCableThreadingBackendJobId } from '@/lib/workspace/cableThreading';
import { resolveEvaluationTaskDisplayName } from '@/lib/workspace/evaluationReport';

export type CableReplayRecordType = 'data_generation' | 'policy_eval';

export type CableReplayStatus = 'completed' | 'failed' | 'generating' | 'running' | 'pending';

export interface CableReplayRecord {
  id: string;
  recordType: CableReplayRecordType;
  title: string;
  runNumber: string;
  status: CableReplayStatus;
  successRate: number | null;
  hasVideo: boolean;
  videoJobId?: string;
  frameJobId?: string;
  backendJobId?: string;
  createdAt: string;
  dataItem?: WorkspaceDataItem;
  evalRow?: EvaluationTaskRow;
}

const STATUS_LABEL: Record<CableReplayStatus, string> = {
  completed: '已完成',
  failed: '失败',
  generating: '生成中',
  running: '运行中',
  pending: '待生成',
};

const CABLE_PHASE_LABELS: Record<string, string> = {
  approach_above_end: '接近线缆末端',
  attach: '建立线缆连接',
  pull_through: '牵引穿过杆间间隙',
  release: '释放线缆',
  settle_wait: '等待稳定',
  backoff_clearance: '后退清障',
  align_to_gap_entry: '对准杆间入口',
  enter_gap: '进入杆间间隙',
  lower_after_gap: '穿过后下降',
};

export const CABLE_DEFAULT_TIMELINE = [
  '环境初始化',
  '接近线缆末端',
  '建立线缆连接 / 抓取末端',
  '牵引线缆穿过杆间间隙',
  '释放线缆并等待稳定',
  '成功条件判定',
  '任务完成',
] as const;

const EVAL_METRIC_KEYS = [
  'success_rate',
  'ever_success_rate',
  'mean_endpoint_goal_error_final',
  'mean_straightness_error_final',
  'mean_anchor_error_final',
  'mean_tabletop_spread_final',
  'mean_thread_completion_max',
] as const;

export function cableReplayStatusLabel(status: CableReplayStatus): string {
  return STATUS_LABEL[status];
}

export function mapCablePhaseLabel(phase: string): string {
  return CABLE_PHASE_LABELS[phase] ?? phase;
}

export function cableObjectModelLabel(internalValue?: string): string {
  if (!internalValue) return '—';
  const option = CABLE_OBJECT_MODEL_OPTIONS.find((o) => o.value === internalValue);
  return option ? `${option.label} / ${internalValue}` : internalValue;
}

export function mapDataItemStatus(status: WorkspaceDataItem['status']): CableReplayStatus {
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'generating') return 'generating';
  return 'pending';
}

export function mapEvalStatus(status: EvaluationTaskRow['status']): CableReplayStatus {
  if (status === '已完成') return 'completed';
  if (status === '失败') return 'failed';
  if (status === '评测中') return 'running';
  return 'pending';
}

export function resolveBackendJobIdForDataItem(item: WorkspaceDataItem): string | undefined {
  const fromFields = resolveCableThreadingBackendJobId(item);
  if (fromFields) return fromFields;

  const pendingMatch = item.id.match(/^ct-pending-(.+)$/);
  if (pendingMatch) {
    return getCableThreadingGenerateRun(pendingMatch[1])?.backendJobId;
  }
  if (item.simulationId?.startsWith('ct-run_')) {
    return getCableThreadingGenerateRun(item.simulationId)?.backendJobId;
  }
  return undefined;
}

export function resolveCableReplayJobId(record: CableReplayRecord): string | undefined {
  return record.backendJobId ?? record.videoJobId ?? record.frameJobId;
}

function deriveJobLogPath(artifactPath?: string): string | undefined {
  if (!artifactPath) return undefined;
  const match = artifactPath.match(/^(.*\/jobs\/[^/]+)\//);
  if (!match) return undefined;
  return `${match[1]}/logs/run.log`;
}

function parseEpisodesFromDataVolume(dataVolume?: string): string {
  if (!dataVolume) return '—';
  const digits = dataVolume.replace(/[^\d]/g, '');
  return digits || '—';
}

function dataFormatsForItem(item: WorkspaceDataItem): string {
  const formats: string[] = [];
  if (item.npzPath || item.contents?.includes('NPZ')) formats.push('NPZ');
  if (item.hdf5Path || item.contents?.includes('HDF5')) formats.push('HDF5');
  if (item.collectCsvPath || item.contents?.includes('CSV')) formats.push('CSV');
  if (item.failuresPath || item.contents?.includes('JSON')) formats.push('JSON');
  if (item.generateVideoExists || item.contents?.includes('过程视频')) formats.push('MP4');
  return formats.length ? formats.join(' / ') : '—';
}

function dataRecordFromItem(item: WorkspaceDataItem): CableReplayRecord {
  const frameJobId = resolveBackendJobIdForDataItem(item);
  const videoJobId =
    item.videoJobId ??
    (item.generateVideoExists && frameJobId ? frameJobId : undefined);
  const backendJobId = frameJobId ?? videoJobId;
  return {
    id: `ct-data-${item.id}`,
    recordType: 'data_generation',
    title: item.name || '数据生成',
    runNumber: item.simulationId || item.id,
    status: mapDataItemStatus(item.status),
    successRate: item.successRate ?? null,
    hasVideo: Boolean(item.generateVideoExists && videoJobId),
    videoJobId,
    frameJobId,
    backendJobId,
    createdAt: item.generatedAt,
    dataItem: item,
  };
}

function evalRecordFromRow(row: EvaluationTaskRow): CableReplayRecord {
  const backendJobId = row.videoJobId ?? row.id;
  const hasEvalVideo = Boolean(
    (row.evalVideoExists || row.videoExists) && (row.videoJobId ?? row.id)
  );
  return {
    id: `ct-eval-${row.id}`,
    recordType: 'policy_eval',
    title: resolveEvaluationTaskDisplayName({
      recordName: row.name,
      taskName: row.name,
      evaluationName: row.name,
    }),
    runNumber: row.id,
    status: mapEvalStatus(row.status),
    successRate: row.successRate,
    hasVideo: hasEvalVideo,
    videoJobId: row.videoJobId ?? row.id,
    frameJobId: row.videoJobId ?? row.id,
    backendJobId,
    createdAt: row.createdAt,
    evalRow: row,
  };
}

export function buildCableReplayRecords(): CableReplayRecord[] {
  const dataItems = listWorkspaceDataItemsForUi().filter(
    (item) => item.taskType === 'cable_threading'
  );
  const evalRows = listWorkspaceEvaluationTasksForUi().filter(
    (row) => row.taskType === 'cable_threading'
  );

  const records: CableReplayRecord[] = [
    ...dataItems.map(dataRecordFromItem),
    ...evalRows.map(evalRecordFromRow),
  ];

  return records.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function resolveCableReplayRecordId(
  records: CableReplayRecord[],
  params: { evalId?: string; jobId?: string }
): string | null {
  const { evalId, jobId } = params;
  if (evalId) {
    const match = records.find((r) => r.evalRow?.id === evalId);
    if (match) return match.id;
  }
  if (jobId) {
    if (jobId.startsWith('ct_gen_') || jobId.startsWith('ct_eval_')) {
      const direct = records.find((r) => r.id === jobId || r.backendJobId === jobId);
      if (direct) return direct.id;
      return jobId;
    }
    const match = records.find(
      (r) =>
        r.dataItem?.id === jobId ||
        r.dataItem?.simulationId === jobId ||
        r.evalRow?.id === jobId ||
        r.frameJobId === jobId ||
        r.backendJobId === jobId ||
        r.dataItem?.id === `ct-pending-${jobId}` ||
        (jobId.startsWith('ct-run_') && r.dataItem?.simulationId === jobId)
    );
    if (match) return match.id;
  }
  return records[0]?.id ?? null;
}

export interface CableTimelineEvent {
  episode: number;
  frameIndex: number;
  step: number;
  timeSec: number;
  phase: string;
  label: string;
}

export interface CableReplayPhasePoint {
  label: string;
  active?: boolean;
  timestamp?: number;
}

export interface CableReplayPhaseView {
  points: CableReplayPhasePoint[];
  hasVideoSync: boolean;
  syncFootnote?: string;
}

export function parseCableReplayTimeline(data: unknown): CableTimelineEvent[] | null {
  if (!data || typeof data !== 'object') return null;
  const raw = (data as { events?: unknown }).events;
  if (!Array.isArray(raw) || raw.length === 0) return null;

  const events: CableTimelineEvent[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') return null;
    const item = entry as Record<string, unknown>;
    const timeSec = item.timeSec ?? item.timestamp;
    const label = item.label;
    const phase = item.phase;
    if (typeof timeSec !== 'number' || Number.isNaN(timeSec)) return null;
    if (typeof label !== 'string' || !label.trim()) return null;
    if (typeof phase !== 'string' || !phase.trim()) return null;
    events.push({
      episode: typeof item.episode === 'number' ? item.episode : 0,
      frameIndex: typeof item.frameIndex === 'number' ? item.frameIndex : 0,
      step: typeof item.step === 'number' ? item.step : 0,
      timeSec,
      phase: phase.trim(),
      label: label.trim(),
    });
  }
  return events.length > 0 ? events : null;
}

export function cableReplayStaticPhases(): CableReplayPhaseView {
  return {
    hasVideoSync: false,
    syncFootnote: '展示该任务的标准执行阶段，暂未与视频帧级同步。',
    points: CABLE_DEFAULT_TIMELINE.map((label) => ({
      label,
      active: false,
    })),
  };
}

export function cableReplaySyncedPhases(
  events: CableTimelineEvent[],
  currentTimeSec: number | null
): CableReplayPhaseView {
  let activeIndex = -1;
  if (currentTimeSec != null && Number.isFinite(currentTimeSec)) {
    for (let i = 0; i < events.length; i++) {
      if (events[i].timeSec <= currentTimeSec) activeIndex = i;
      else break;
    }
  }

  return {
    hasVideoSync: true,
    points: events.map((event, index) => ({
      label: event.label,
      timestamp: event.timeSec,
      active: index === activeIndex,
    })),
  };
}

export function cableReplayMetrics(record: CableReplayRecord): { label: string; value: string }[] {
  if (record.recordType === 'data_generation' && record.dataItem) {
    const item = record.dataItem;
    const episodes = parseEpisodesFromDataVolume(item.dataVolume);
    return [
      {
        label: '成功轨迹数',
        value: item.successfulEpisodes != null ? String(item.successfulEpisodes) : '—',
      },
      { label: '采集轮次', value: episodes !== '—' ? episodes : '—' },
      {
        label: '成功率',
        value: record.successRate != null ? `${record.successRate}%` : '—',
      },
      { label: '数据格式', value: dataFormatsForItem(item) },
      { label: '对象模型', value: cableObjectModelLabel(item.cableModel) },
      { label: '最大步数', value: item.horizon != null ? String(item.horizon) : '—' },
    ];
  }

  const aggregate = (record.evalRow?.aggregate ?? {}) as Record<string, unknown>;
  return EVAL_METRIC_KEYS.map((key) => {
    let value: unknown = aggregate[key];
    if (key === 'success_rate' && value == null && record.successRate != null) {
      value = record.successRate / 100;
    }
    if (key === 'ever_success_rate' && value == null && record.evalRow?.everSuccessRate != null) {
      value = record.evalRow.everSuccessRate / 100;
    }
    return {
      label: key,
      value: value != null && value !== '' ? String(value) : '—',
    };
  });
}

export function cableReplayLogPaths(record: CableReplayRecord): { label: string; value: string }[] {
  if (record.evalRow) {
    const row = record.evalRow;
    const paths = [
      { label: 'eval.csv', value: row.evalCsvPath ?? '—' },
      { label: 'eval.results.json', value: row.resultPath ?? '—' },
      { label: 'eval.failures.json', value: row.failuresPath ?? '—' },
      { label: 'eval.mp4', value: row.evalVideoPath ?? row.videoPath ?? '—' },
      { label: 'run.log', value: deriveJobLogPath(row.evalCsvPath) ?? '—' },
    ];
    return paths;
  }

  if (record.dataItem) {
    const item = record.dataItem;
    const logPath =
      deriveJobLogPath(item.npzPath) ??
      deriveJobLogPath(item.collectCsvPath) ??
      deriveJobLogPath(item.manifestPath);
    const paths = [
      { label: 'run.log', value: logPath ?? '—' },
      { label: 'collect.csv', value: item.collectCsvPath ?? '—' },
      { label: 'failures.json', value: item.failuresPath ?? '—' },
      { label: 'manifest.json', value: item.manifestPath ?? '—' },
      { label: 'dataset.npz', value: item.npzPath ?? '—' },
    ];
    if (item.hdf5Path) {
      paths.push({ label: 'dataset.hdf5', value: item.hdf5Path });
    }
    if (item.generateVideoPath) {
      paths.push({ label: 'generate.mp4', value: item.generateVideoPath });
    }
    const timelinePath = deriveJobLogPath(item.npzPath)?.replace(
      '/logs/run.log',
      '/live/generate_timeline.json'
    );
    if (timelinePath) {
      paths.push({ label: 'generate_timeline.json', value: timelinePath });
    }
    return paths;
  }

  return [];
}
