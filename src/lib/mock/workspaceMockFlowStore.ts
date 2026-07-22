/**
 * 工作台前端闭环 mock 状态（sessionStorage，不接后端）
 * 串联：仿真任务 → 控制台 → 数据中心 → 评测任务 → 过程评测
 */

import type { BuildDatasetPayload } from '@/lib/workspace/buildDatasetLegacyTypes';
import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import type { CreateSimulationTaskPayload } from '@/components/workspace/simulation/CreateSimulationTaskModal';
import { hasPersistedGenerateJobId, type WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import {
  buildDatasetManifest,
  formatQualityStatusLabel,
  purposeToUsageKey,
  resolveSourceArtifacts,
  type DatasetManifest,
} from '@/lib/workspace/datasetManifest';
import type { CreateEvaluationPayload } from '@/components/workspace/evaluation/CreateEvaluationModal';
import type { EvaluationTaskRow } from '@/lib/mock/workspaceEvaluationRecordsMock';
import type { CurrentSimulation, SimulationRunStatus } from '@/lib/mock/workspaceSimulationMock';
import { isStalePendingDataItem } from '@/lib/workspace/backendJobIds';
import { resolveCableThreadingBackendJobId } from '@/lib/workspace/cableThreading';
import { resolveDualArmBackendJobId } from '@/lib/workspace/dualArmCable';

export type SimulationRunListStatus = '待运行' | '运行中' | '已完成' | '失败';

export type CableThreadingGenerateRunStatus = 'running' | 'completed' | 'failed';

export interface CableThreadingGenerateRunResult {
  successRate?: number;
  successfulEpisodes?: number;
  npzPath?: string;
  hdf5Path?: string;
  manifestPath?: string;
  collectCsvPath?: string;
  failuresPath?: string;
  logPath?: string;
  backendCommand?: string;
  generateVideoPath?: string;
  generateVideoExists?: boolean;
  generateVideoSizeBytes?: number;
  generateVideoStatus?: string;
}

export interface CableThreadingGenerateRun {
  localRunId: string;
  dataItemId: string;
  status: CableThreadingGenerateRunStatus;
  payload: GenerateDataPayload;
  startedAt: string;
  apiStarted?: boolean;
  backendJobId?: string;
  errorMessage?: string;
  result?: CableThreadingGenerateRunResult;
}

export type CableThreadingEvaluateRunStatus = 'running' | 'completed' | 'failed';

export interface CableThreadingEvaluateRunResult {
  successRate?: number;
  everSuccessRate?: number;
  evalCsvPath?: string;
  resultPath?: string;
  failuresPath?: string;
  logPath?: string;
  backendCommand?: string;
}

export interface CableThreadingEvaluateRun {
  evalJobId: string;
  status: CableThreadingEvaluateRunStatus;
  payload: CreateEvaluationPayload;
  startedAt: string;
  recordWritten?: boolean;
  errorMessage?: string;
  result?: CableThreadingEvaluateRunResult;
}

export type DualArmCableGenerateRunStatus = 'running' | 'completed' | 'failed';

export interface DualArmCableGenerateRun {
  jobId: string;
  dataItemId: string;
  status: DualArmCableGenerateRunStatus;
  payload: GenerateDataPayload;
  startedAt: string;
  errorMessage?: string;
}

export interface SimulationTaskRun {
  id: string;
  template: string;
  scene: string;
  robot: string;
  policy: string;
  rounds: number;
  seed: number;
  generateData: boolean;
  saveVideo: boolean;
  autoEvaluate: boolean;
  status: SimulationRunListStatus;
  progressPercent: number;
  createdAt: string;
  creator: string;
}

interface MockFlowState {
  simulationRuns: SimulationTaskRun[];
  activeSimulationRunId: string | null;
  extraDataItems: WorkspaceDataItem[];
  extraEvaluationTasks: EvaluationTaskRow[];
  lastProcessEvalId: string | null;
  activeDataGenerationItemId: string | null;
  activeDataGenerationContext: {
    itemId: string;
    name: string;
    episodes: number;
    seed: number;
    template: string;
  } | null;
  cableThreadingGenerateRuns: CableThreadingGenerateRun[];
  activeCableThreadingRunId: string | null;
  dualArmCableGenerateRuns: DualArmCableGenerateRun[];
  cableThreadingEvaluateRuns: CableThreadingEvaluateRun[];
  datasetManifests: Record<string, DatasetManifest>;
}

const STORAGE_KEY = 'epi_workspace_mock_flow_v1';
const STALE_LOCAL_PENDING_MS = 30 * 60 * 1000;

function nowLabel() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function defaultState(): MockFlowState {
  return {
    simulationRuns: seedSimulationRuns(),
    activeSimulationRunId: 'sim-run-2041',
    extraDataItems: [],
    extraEvaluationTasks: [],
    lastProcessEvalId: null,
    activeDataGenerationItemId: null,
    activeDataGenerationContext: null,
    cableThreadingGenerateRuns: [],
    activeCableThreadingRunId: null,
    dualArmCableGenerateRuns: [],
    cableThreadingEvaluateRuns: [],
    datasetManifests: {},
  };
}

function seedSimulationRuns(): SimulationTaskRun[] {
  return [
    {
      id: 'sim-run-cable-001',
      template: '线缆穿杆',
      scene: '桌面双杆穿线工位',
      robot: 'Panda',
      policy: 'scripted',
      rounds: 10,
      seed: 42,
      generateData: true,
      saveVideo: true,
      autoEvaluate: false,
      status: '运行中',
      progressPercent: 68,
      createdAt: '2026-06-03 10:20',
      creator: '平台',
    },
    {
      id: 'sim-run-dual-arm-001',
      template: '线缆整理',
      scene: '双臂桌面线缆整理工位',
      robot: 'Dual FR3',
      policy: '感知驱动操控',
      rounds: 8,
      seed: 7,
      generateData: true,
      saveVideo: true,
      autoEvaluate: false,
      status: '待运行',
      progressPercent: 0,
      createdAt: '2026-06-02 16:30',
      creator: '平台',
    },
  ];
}

function readState(): MockFlowState {
  if (typeof window === 'undefined') return defaultState();
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState();
    const parsed = JSON.parse(raw) as MockFlowState;
    if (!parsed || !Array.isArray(parsed.simulationRuns)) return defaultState();
    return {
      ...defaultState(),
      ...parsed,
      simulationRuns: parsed.simulationRuns.length ? parsed.simulationRuns : seedSimulationRuns(),
      cableThreadingGenerateRuns: Array.isArray(parsed.cableThreadingGenerateRuns)
        ? parsed.cableThreadingGenerateRuns
        : [],
    };
  } catch {
    return defaultState();
  }
}

function writeState(state: MockFlowState) {
  if (typeof window === 'undefined') return;
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function getMockFlowState(): MockFlowState {
  return readState();
}

export function listSimulationRuns(): SimulationTaskRun[] {
  return readState().simulationRuns;
}

export function addSimulationRun(
  payload: CreateSimulationTaskPayload,
  status: SimulationRunListStatus
): SimulationTaskRun {
  const state = readState();
  const id = `sim-run-${Date.now()}`;
  const run: SimulationTaskRun = {
    id,
    template: payload.template,
    scene: payload.scene,
    robot: payload.robot,
    policy: payload.policy,
    rounds: payload.rounds,
    seed: payload.seed ?? 0,
    generateData: payload.generateData,
    saveVideo: payload.saveVideo,
    autoEvaluate: payload.autoEvaluate,
    status,
    progressPercent: status === '运行中' ? 12 : 0,
    createdAt: nowLabel(),
    creator: '当前用户',
  };
  state.simulationRuns = [run, ...state.simulationRuns];
  if (status === '运行中') {
    state.activeSimulationRunId = id;
  }
  writeState(state);
  return run;
}

export function setActiveSimulationRun(id: string) {
  const state = readState();
  state.activeSimulationRunId = id;
  writeState(state);
}

export function getActiveSimulationRun(): SimulationTaskRun | null {
  const state = readState();
  const id = state.activeSimulationRunId;
  if (!id) return state.simulationRuns[0] ?? null;
  return state.simulationRuns.find((r) => r.id === id) ?? state.simulationRuns[0] ?? null;
}

export function runToCurrentSimulation(run: SimulationTaskRun): CurrentSimulation {
  const statusMap: Record<SimulationRunListStatus, SimulationRunStatus> = {
    待运行: 'idle',
    运行中: 'running',
    已完成: 'completed',
    失败: 'failed',
  };
  return {
    id: run.id,
    taskName: run.template,
    scene: run.scene,
    robot: run.robot,
    policy: run.policy,
    status: statusMap[run.status],
    runDuration: run.status === '运行中' ? '00:04:38' : '00:00:00',
    progressPercent: run.progressPercent,
    currentStepLabel: run.status === '运行中' ? '拧紧第一颗螺丝' : '等待启动',
    engine: 'MuJoCo',
    simTime: '00:02:14.6',
    frame: 4024,
    objectsInScene: ['螺丝', '工件', '电批', '夹具'],
  };
}

function resolveCableThreadingBackendJobIdFromStore(item: WorkspaceDataItem): string | undefined {
  const fromFields = resolveCableThreadingBackendJobId(item);
  if (fromFields) return fromFields;

  const pendingMatch = item.id.match(/^ct-pending-(.+)$/);
  if (pendingMatch) {
    const backend = getCableThreadingGenerateRun(pendingMatch[1])?.backendJobId;
    if (backend) return backend;
  }
  if (item.simulationId?.startsWith('ct-run_')) {
    const backend = getCableThreadingGenerateRun(item.simulationId)?.backendJobId;
    if (backend) return backend;
  }
  return undefined;
}

function sanitizeStalePendingDataItem(item: WorkspaceDataItem): WorkspaceDataItem {
  if (item.taskType === 'cable_threading') {
    const backendJobId = resolveCableThreadingBackendJobIdFromStore(item);
    if (backendJobId) {
      return {
        ...item,
        jobId: backendJobId,
        backendJobId,
        sourceJobId: backendJobId,
        simulationId: backendJobId,
        staleLocalPending: false,
      };
    }
  }

  if (item.taskType === 'dual_arm_cable_manipulation') {
    const backendJobId = resolveDualArmBackendJobId(item);
    if (backendJobId) {
      return {
        ...item,
        jobId: backendJobId,
        backendJobId,
        sourceJobId: backendJobId,
        simulationId: backendJobId,
        staleLocalPending: false,
      };
    }
  }

  if (isStalePendingDataItem(item)) {
    return {
      ...item,
      staleLocalPending: true,
      backendJobStatus: item.backendJobStatus ?? 'stale_local_pending',
    };
  }

  return item;
}

export function listExtraDataItems(): WorkspaceDataItem[] {
  const state = readState();
  const purged = purgeOrphanStaleLocalPendingItems(state.extraDataItems);
  if (purged.length !== state.extraDataItems.length) {
    state.extraDataItems = purged;
    writeState(state);
  }
  return purged.map(sanitizeStalePendingDataItem);
}

export function saveDatasetManifest(manifest: DatasetManifest) {
  const state = readState();
  state.datasetManifests[manifest.datasetId] = manifest;
  writeState(state);
}

export function getDatasetManifest(datasetId: string): DatasetManifest | undefined {
  return readState().datasetManifests[datasetId];
}

export function listDatasetManifests(): DatasetManifest[] {
  return Object.values(readState().datasetManifests);
}

export function upsertMockDataItem(item: WorkspaceDataItem) {
  const state = readState();
  const sourceJobId = item.sourceJobId ?? item.jobId ?? item.backendJobId;
  state.extraDataItems = state.extraDataItems.filter(
    (existing) =>
      !(
        existing.isDatasetAsset &&
        sourceJobId &&
        existing.sourceJobId === sourceJobId &&
        existing.id !== item.id
      )
  );
  const index = state.extraDataItems.findIndex(
    (existing) =>
      existing.id === item.id ||
      (item.jobId != null && (existing.jobId === item.jobId || existing.id === item.jobId)) ||
      (sourceJobId != null && (existing.id === sourceJobId || existing.jobId === sourceJobId))
  );
  if (index >= 0) {
    state.extraDataItems[index] = item;
  } else {
    state.extraDataItems.unshift(item);
  }
  writeState(state);
}

export function createDatasetFromBuild(
  payload: BuildDatasetPayload,
  sourceItem: WorkspaceDataItem
): WorkspaceDataItem {
  const source = resolveSourceArtifacts(sourceItem);
  const manifest = buildDatasetManifest(sourceItem, payload);
  saveDatasetManifest(manifest);

  const usageEpisodes =
    payload.usageScope === '全部成功轨迹'
      ? source.successfulEpisodes
      : payload.usageScope === '全部轨迹'
        ? source.episodes
        : Math.min(payload.customEpisodeCount ?? source.successfulEpisodes, source.episodes);

  const splitLabel =
    payload.splitMode === 'train_val_80_20'
      ? '训练 80% / 验证 20%'
      : payload.splitMode === 'custom'
        ? `训练 ${Math.round((payload.customTrainRatio ?? 0.8) * 100)}% / 验证 ${Math.round((1 - (payload.customTrainRatio ?? 0.8)) * 100)}%`
        : '不划分';

  const contents: string[] = [];
  if (payload.includeTrajectory) contents.push('轨迹');
  if (payload.includeImageObservation) contents.push('图像');
  if (payload.includeStateAction) contents.push('状态', '动作');
  if (payload.includeProcessVideo) contents.push('过程视频');
  if (payload.includeRunLog) contents.push('日志');
  if (payload.includeFailures) contents.push('失败记录');
  if (payload.includeTimeline) contents.push('阶段同步');

  const manifestPath =
    manifest.artifacts.manifest ??
    `${source.sourceJobId}/datasets/dataset_manifest.json`;

  const generationStatus =
    sourceItem.status === 'generating' || sourceItem.status === 'pending'
      ? sourceItem.status
      : 'completed';

  return {
    ...sourceItem,
    id: sourceItem.id,
    name: sourceItem.taskName?.trim() || sourceItem.name,
    taskId: sourceItem.taskId,
    taskName: sourceItem.taskName,
    simulationId: source.sourceJobId,
    dataCategory: sourceItem.dataCategory,
    source: sourceItem.source,
    targetModelFormat: payload.downstreamModelType as WorkspaceDataItem['targetModelFormat'],
    dataVolume: `${usageEpisodes} 条`,
    size: sourceItem.size || '—',
    status: generationStatus,
    generatedAt: sourceItem.generatedAt || nowLabel(),
    creator: sourceItem.creator || '当前用户',
    simBackend: source.backend,
    robot: source.robot,
    cableModel: source.objectModel,
    difficulty: source.difficulty,
    successRate: Math.round(source.successRate * 1000) / 10,
    successfulEpisodes: source.successfulEpisodes,
    taskType: sourceItem.taskType,
    jobId: sourceItem.jobId ?? sourceItem.backendJobId ?? source.sourceJobId,
    backendJobId: sourceItem.backendJobId ?? sourceItem.jobId ?? source.sourceJobId,
    npzPath: manifest.artifacts.npz,
    hdf5Path: manifest.artifacts.hdf5,
    manifestPath,
    collectCsvPath: manifest.artifacts.collectCsv,
    failuresPath: manifest.artifacts.failures,
    generateVideoPath: manifest.artifacts.generateVideo,
    generateVideoExists: Boolean(manifest.artifacts.generateVideo),
    contents,
    sampleRate: sourceItem.sampleRate ?? '混合',
    frameOrTrajectoryCount: `${usageEpisodes} 条 · ${splitLabel} · ${payload.dataOrganizationFormat}`,
    isDatasetAsset: false,
    datasetBuildStatus: 'built',
    datasetId: manifest.datasetId,
    sourceJobId: manifest.sourceJobId,
    sourceRecordName: manifest.sourceRecordName,
    downstreamModelType: payload.downstreamModelType,
    dataOrganizationFormat: payload.dataOrganizationFormat,
    trainingView: manifest.trainingView,
    datasetUsage: purposeToUsageKey(payload.purpose),
    qualityStatus: formatQualityStatusLabel(manifest.quality.status),
    mainFormats: manifest.mainFormats,
    datasetManifestPath: manifestPath,
    datasetBuildSupported: sourceItem.datasetBuildSupported,
  };
}

export function createDataFromGeneration(payload: GenerateDataPayload): WorkspaceDataItem {
  const id = `demo-${Date.now().toString(36)}`;
  const name = payload.outputName.trim() || `${payload.template}-${payload.episodes}`;
  const status = payload.launch === 'save' ? 'pending' : 'generating';
  const volume = `${payload.episodes} 条`;
  return {
    id,
    name,
    taskId: `task-${Date.now().toString(36).slice(-4)}`,
    taskName: payload.template,
    simulationId: `collect-${Date.now().toString(36).slice(-6)}`,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: volume,
    size: payload.launch === 'save' ? '—' : '1.1 GB',
    status,
    generatedAt: nowLabel(),
    creator: '当前用户',
    taskConfig: payload.taskConfig,
    simBackend: 'MuJoCo',
    saveVideo: payload.saveVideo,
    saveTrajectory: payload.saveTrajectory,
    saveStateLog: payload.saveStateLog,
    contents: [
      ...(payload.saveTrajectory ? ['轨迹'] : []),
      ...(payload.saveStructuredData ? ['结构化数据'] : []),
      ...(payload.saveVideo ? ['视频'] : []),
      ...(payload.saveStateLog ? ['状态', '日志'] : []),
    ],
    sampleRate: '30 Hz',
    frameOrTrajectoryCount:
      payload.launch === 'save' ? '待生成' : `${volume} · seed ${payload.seed ?? 0}`,
    physicsProxyMode: payload.physicsProxyMode,
    physicsProxyModel: payload.physicsProxyModel,
    physicsProxyErrorThreshold: payload.physicsProxyErrorThreshold,
    physicsProxyReviewRatio: payload.physicsProxyReviewRatio,
  };
}

export function setActiveDataGeneration(
  item: WorkspaceDataItem,
  context: { episodes: number; seed: number; template: string }
) {
  const state = readState();
  state.activeDataGenerationItemId = item.id;
  state.activeDataGenerationContext = {
    itemId: item.id,
    name: item.name,
    episodes: context.episodes,
    seed: context.seed,
    template: context.template,
  };
  writeState(state);
}

export function getActiveDataGenerationItemId(): string | null {
  return readState().activeDataGenerationItemId;
}

export function getActiveDataGenerationContext() {
  return readState().activeDataGenerationContext;
}

export function clearActiveDataGeneration() {
  const state = readState();
  state.activeDataGenerationItemId = null;
  state.activeDataGenerationContext = null;
  writeState(state);
}

export function updateMockDataItem(
  id: string,
  updates: Partial<WorkspaceDataItem>
): WorkspaceDataItem | null {
  const state = readState();
  let updated: WorkspaceDataItem | null = null;
  state.extraDataItems = state.extraDataItems.map((item) => {
    if (item.id !== id) return item;
    updated = { ...item, ...updates };
    return updated;
  });
  if (updated) {
    if (updates.status === 'completed') {
      state.activeDataGenerationItemId = null;
      state.activeDataGenerationContext = null;
    }
    writeState(state);
  }
  return updated;
}

export function completeDataGenerationItem(
  itemId: string,
  context?: { episodes?: number; seed?: number }
): WorkspaceDataItem | null {
  const stored = getActiveDataGenerationContext();
  const episodes = context?.episodes ?? stored?.episodes ?? 50;
  const seed = context?.seed ?? stored?.seed;
  const volume = `${episodes} 条`;
  const frameOrTrajectoryCount =
    seed != null ? `${volume} · seed ${seed}` : volume;
  return updateMockDataItem(itemId, {
    status: 'completed',
    dataVolume: volume,
    size: '1.1 GB',
    frameOrTrajectoryCount,
  });
}

export function appendMockDataItem(item: WorkspaceDataItem) {
  const state = readState();
  state.extraDataItems = [item, ...state.extraDataItems];
  writeState(state);
}

function collectLocalDataItemMatchKeys(input: string | WorkspaceDataItem): Set<string> {
  const keys = new Set<string>();
  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (trimmed) keys.add(trimmed);
    return keys;
  }
  for (const key of [
    input.id,
    input.jobId,
    input.backendJobId,
    input.sourceJobId,
    input.simulationId,
  ]) {
    if (key) keys.add(key);
  }
  const dacPending = input.id.match(/^dac-pending-(dac_gen_\d{8}_\d{6}_[a-f0-9]{4})$/);
  if (dacPending) keys.add(dacPending[1]);
  const ctPending = input.id.match(/^ct-pending-(.+)$/);
  if (ctPending) keys.add(ctPending[1]);
  return keys;
}

function extraDataItemMatchesKeys(item: WorkspaceDataItem, keys: Set<string>): boolean {
  if (keys.has(item.id)) return true;
  for (const key of [item.jobId, item.backendJobId, item.sourceJobId, item.simulationId]) {
    if (key && keys.has(key)) return true;
  }
  return false;
}

function parseGeneratedAtMs(label: string | undefined): number | null {
  if (!label) return null;
  const match = label.match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/);
  if (!match) return null;
  const [, y, mo, d, h, mi] = match;
  return new Date(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi)).getTime();
}

function isOrphanStaleLocalPending(item: WorkspaceDataItem, nowMs: number): boolean {
  if (hasPersistedGenerateJobId(item)) return false;
  if (item.status !== 'pending' && item.status !== 'generating') return false;
  if (
    !item.id.startsWith('dac-save-') &&
    !item.id.startsWith('dac-pending-') &&
    !item.id.startsWith('ct-pending-') &&
    !isStalePendingDataItem(item)
  ) {
    return false;
  }
  const createdAtMs = parseGeneratedAtMs(item.generatedAt);
  if (createdAtMs == null) return false;
  return nowMs - createdAtMs >= STALE_LOCAL_PENDING_MS;
}

function purgeOrphanStaleLocalPendingItems(items: WorkspaceDataItem[]): WorkspaceDataItem[] {
  const nowMs = Date.now();
  return items.filter((item) => !isOrphanStaleLocalPending(item, nowMs));
}

/**
 * 从 sessionStorage 删除数据中心本地 pending / save 记录。
 * 可按 id、jobId、backendJobId 或整条 WorkspaceDataItem 匹配。
 */
export function removeLocalDataItem(input: string | WorkspaceDataItem): number {
  const keys = collectLocalDataItemMatchKeys(input);
  if (keys.size === 0) return 0;

  const state = readState();
  const beforeCount = state.extraDataItems.length;
  const beforeCableRuns = state.cableThreadingGenerateRuns.length;
  const beforeDualRuns = state.dualArmCableGenerateRuns.length;
  const hadActiveDataGen =
    state.activeDataGenerationItemId != null && keys.has(state.activeDataGenerationItemId);
  const hadActiveCableRun =
    state.activeCableThreadingRunId != null && keys.has(state.activeCableThreadingRunId);

  state.extraDataItems = state.extraDataItems.filter((item) => !extraDataItemMatchesKeys(item, keys));

  state.cableThreadingGenerateRuns = state.cableThreadingGenerateRuns.filter(
    (run) =>
      !keys.has(run.localRunId) &&
      !keys.has(run.dataItemId) &&
      !(run.backendJobId && keys.has(run.backendJobId))
  );

  state.dualArmCableGenerateRuns = state.dualArmCableGenerateRuns.filter(
    (run) => !keys.has(run.jobId) && !keys.has(run.dataItemId)
  );

  if (hadActiveDataGen) {
    state.activeDataGenerationItemId = null;
    state.activeDataGenerationContext = null;
  }

  if (hadActiveCableRun) {
    state.activeCableThreadingRunId = null;
  }

  const removedCount = beforeCount - state.extraDataItems.length;
  const changed =
    removedCount > 0 ||
    state.cableThreadingGenerateRuns.length !== beforeCableRuns ||
    state.dualArmCableGenerateRuns.length !== beforeDualRuns ||
    hadActiveDataGen ||
    hadActiveCableRun;

  if (changed) {
    writeState(state);
  }

  return removedCount;
}

/** @deprecated 使用 removeLocalDataItem */
export const removeMockDataItem = removeLocalDataItem;

export function createDataItemFromSimulation(run: SimulationTaskRun): WorkspaceDataItem {
  const slug = run.template.replace(/\s+/g, '-').slice(0, 24).toLowerCase();
  const id = `${slug}-traj-${Date.now().toString(36)}`;
  return {
    id,
    name: id,
    taskId: run.id,
    taskName: run.template,
    simulationId: run.id,
    dataCategory: '示范数据',
    source: 'MuJoCo 生成',
    targetModelFormat: '—',
    dataVolume: `${run.rounds} 条`,
    size: '1.1 GB',
    status: 'completed',
    generatedAt: nowLabel(),
    creator: run.creator,
    scene: run.scene,
    robot: run.robot,
    policy: run.policy,
    contents: ['轨迹', '状态', '动作', ...(run.saveVideo ? ['视频'] : [])],
    sampleRate: '30 Hz',
    frameOrTrajectoryCount: `${run.rounds} 条`,
  };
}

export function appendEvaluationTask(row: EvaluationTaskRow) {
  const state = readState();
  state.extraEvaluationTasks = [row, ...state.extraEvaluationTasks];
  if (row.evaluationMode === '数据过程评测') {
    state.lastProcessEvalId = row.id;
  }
  writeState(state);
}

export function updateExtraEvaluationTask(
  id: string,
  updates: Partial<EvaluationTaskRow>
): EvaluationTaskRow | null {
  const state = readState();
  let updated: EvaluationTaskRow | null = null;
  state.extraEvaluationTasks = state.extraEvaluationTasks.map((row) => {
    if (row.id !== id) return row;
    updated = { ...row, ...updates };
    return updated;
  });
  if (updated) writeState(state);
  return updated;
}

function isPersistedEvaluationSessionRow(row: EvaluationTaskRow): boolean {
  const evalJobId = String(row.evalJobId ?? row.jobId ?? row.id ?? '').trim();
  const hasStrictEvalId =
    /^eval_\d{8}_\d{6}_[a-f0-9]{4}$/i.test(evalJobId) ||
    /^isaac_eval_\d{8}_\d{6}_[a-f0-9]{4}$/i.test(evalJobId) ||
    /^ct_eval_\d{8}_\d{6}_[a-f0-9]{4}$/i.test(evalJobId) ||
    /^ct_eval_[a-z0-9_]+$/i.test(evalJobId);
  return hasStrictEvalId || row.workspaceJobId != null;
}

export function listExtraEvaluationTasks(): EvaluationTaskRow[] {
  return readState().extraEvaluationTasks.filter(isPersistedEvaluationSessionRow);
}

/** 清理 sessionStorage 中缺少真实 evalJobId / workspaceJobId 的评测脏数据。 */
export function purgeInvalidEvaluationSessionRows(): number {
  if (typeof window === 'undefined') return 0;
  const state = readState();
  const before = state.extraEvaluationTasks.length;
  state.extraEvaluationTasks = state.extraEvaluationTasks.filter(isPersistedEvaluationSessionRow);
  const removed = before - state.extraEvaluationTasks.length;
  if (removed > 0) {
    writeState(state);
  }
  return removed;
}

export function getLastProcessEvalId(): string | null {
  return readState().lastProcessEvalId;
}

export function updateSimulationRunProgress(id: string, progressPercent: number) {
  const state = readState();
  state.simulationRuns = state.simulationRuns.map((r) =>
    r.id === id ? { ...r, progressPercent, status: progressPercent >= 100 ? '已完成' : r.status } : r
  );
  writeState(state);
}

export function removeSimulationRun(id: string) {
  const state = readState();
  state.simulationRuns = state.simulationRuns.filter((r) => r.id !== id);
  if (state.activeSimulationRunId === id) {
    state.activeSimulationRunId = state.simulationRuns[0]?.id ?? null;
  }
  writeState(state);
}

export function createCableThreadingGenerateRun(
  dataItem: WorkspaceDataItem,
  payload: GenerateDataPayload,
  backendJobId: string
): CableThreadingGenerateRun {
  const run: CableThreadingGenerateRun = {
    localRunId: backendJobId,
    backendJobId,
    dataItemId: dataItem.id,
    status: 'running',
    payload,
    startedAt: nowLabel(),
    apiStarted: true,
  };
  const state = readState();
  state.cableThreadingGenerateRuns = [run, ...state.cableThreadingGenerateRuns];
  state.activeCableThreadingRunId = backendJobId;
  writeState(state);
  return run;
}

export function getCableThreadingGenerateRun(
  runId: string
): CableThreadingGenerateRun | null {
  return (
    readState().cableThreadingGenerateRuns.find(
      (r) => r.localRunId === runId || r.backendJobId === runId
    ) ?? null
  );
}

export function updateCableThreadingGenerateRun(
  runId: string,
  updates: Partial<CableThreadingGenerateRun>
): CableThreadingGenerateRun | null {
  const state = readState();
  let updated: CableThreadingGenerateRun | null = null;
  state.cableThreadingGenerateRuns = state.cableThreadingGenerateRuns.map((run) => {
    if (run.localRunId !== runId && run.backendJobId !== runId) return run;
    updated = { ...run, ...updates };
    return updated;
  });
  if (updated) writeState(state);
  return updated;
}

export function bindCableThreadingBackendJobToDataItem(
  dataItemId: string,
  backendJobId: string
): WorkspaceDataItem | null {
  return updateMockDataItem(dataItemId, {
    jobId: backendJobId,
    backendJobId,
    sourceJobId: backendJobId,
    simulationId: backendJobId,
    staleLocalPending: false,
    backendJobStatus: 'running',
  });
}

export function createDualArmCableGenerateRun(
  dataItem: WorkspaceDataItem,
  payload: GenerateDataPayload,
  jobId: string
): DualArmCableGenerateRun {
  const run: DualArmCableGenerateRun = {
    jobId,
    dataItemId: dataItem.id,
    status: 'running',
    payload,
    startedAt: nowLabel(),
  };
  const state = readState();
  state.dualArmCableGenerateRuns = [run, ...state.dualArmCableGenerateRuns];
  writeState(state);
  return run;
}

export function getDualArmCableGenerateRun(jobId: string): DualArmCableGenerateRun | null {
  return readState().dualArmCableGenerateRuns.find((r) => r.jobId === jobId) ?? null;
}

export function updateDualArmCableGenerateRun(
  jobId: string,
  updates: Partial<DualArmCableGenerateRun>
): DualArmCableGenerateRun | null {
  const state = readState();
  let updated: DualArmCableGenerateRun | null = null;
  state.dualArmCableGenerateRuns = state.dualArmCableGenerateRuns.map((run) => {
    if (run.jobId !== jobId) return run;
    updated = { ...run, ...updates };
    return updated;
  });
  if (updated) writeState(state);
  return updated;
}

export function createCableThreadingEvaluateRun(
  evalJobId: string,
  payload: CreateEvaluationPayload
): CableThreadingEvaluateRun {
  const run: CableThreadingEvaluateRun = {
    evalJobId,
    status: 'running',
    payload,
    startedAt: nowLabel(),
    recordWritten: false,
  };
  const state = readState();
  state.cableThreadingEvaluateRuns = [run, ...state.cableThreadingEvaluateRuns];
  writeState(state);
  return run;
}

export function getCableThreadingEvaluateRun(
  evalJobId: string
): CableThreadingEvaluateRun | null {
  return (
    readState().cableThreadingEvaluateRuns.find((r) => r.evalJobId === evalJobId) ?? null
  );
}

export function updateCableThreadingEvaluateRun(
  evalJobId: string,
  updates: Partial<CableThreadingEvaluateRun>
): CableThreadingEvaluateRun | null {
  const state = readState();
  let updated: CableThreadingEvaluateRun | null = null;
  state.cableThreadingEvaluateRuns = state.cableThreadingEvaluateRuns.map((run) => {
    if (run.evalJobId !== evalJobId) return run;
    updated = { ...run, ...updates };
    return updated;
  });
  if (updated) writeState(state);
  return updated;
}

export function replaceMockDataItem(oldId: string, newItem: WorkspaceDataItem) {
  const state = readState();
  const idx = state.extraDataItems.findIndex((item) => item.id === oldId);
  if (idx >= 0) {
    state.extraDataItems[idx] = newItem;
  } else {
    state.extraDataItems = [newItem, ...state.extraDataItems];
  }
  if (state.activeDataGenerationItemId === oldId) {
    state.activeDataGenerationItemId = newItem.id;
    if (state.activeDataGenerationContext) {
      state.activeDataGenerationContext = {
        ...state.activeDataGenerationContext,
        itemId: newItem.id,
        name: newItem.name,
      };
    }
  }
  writeState(state);
}
