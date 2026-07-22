/**
 * 数据转换页 mock：资产、作业、runner（阶段进度 + 日志）
 * 结构与后续真实 API 对齐，前端先画出来
 */

export type ConversionJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';

export type ConversionStage = 'Parse' | 'Align' | 'Write' | 'Validate';

/** 数据资产中的数据集（UI 用 datasetId/datasetName，内部兼容 assetId/assetName） */
export interface MockAsset {
  /** 数据集 ID，与 assetId 一致；仅保留类型定义，实际列表来自后端 API */
  datasetId: string;
  datasetName: string;
  assetId: string;
  assetName: string;
  projectId: string;
  projectName?: string;
  deviceName: string;
  sizeMB: number;
  durationSec: number;
  createdAt: string;
  format?: string;
}

export interface ConversionJobStageProgress {
  stage: ConversionStage;
  status: 'pending' | 'running' | 'done' | 'error';
  progressPercent: number;
  durationMs?: number;
}

export interface ConversionJobArtifact {
  type: 'hdf5' | 'lerobot_zip' | 'report';
  name: string;
  path: string;
  sizeBytes?: number;
}

export interface ConversionJobMetrics {
  dropRate?: number;
  timeSkewMs?: number;
  steps?: number;
  outputSize?: number;
}

export interface ConversionJob {
  jobId: string;
  shortCode: string;
  /** 任务编号，四位数字 0001、0002…（后端生成或前端 sequence fallback） */
  taskNo: string;
  /** 任务名称（创建时填写，展示在任务编号后面） */
  taskName?: string | null;
  /** 输出文件名，列表「文件名」列展示 */
  outputFileName: string;
  /** 兼容旧字段，列表优先用 outputFileName */
  fileName: string;
  /** 输入数据集 id（创建时必填） */
  assetId: string;
  assetName: string;
  projectId: string;
  projectName: string;
  deviceName: string;
  outputFormat: 'HDF5' | 'LeRobot';
  /** 源数据资产格式（如 MCAP），与 outputFormat（转换目标）不同 */
  fileFormat: string;
  outputLocation: 'local' | 'cloud';
  outputPath: string;
  status: ConversionJobStatus;
  progressPercent: number;
  currentStage: ConversionStage | null;
  stages: ConversionJobStageProgress[];
  logs: string[];
  createdAt: string;
  updatedAt: string;
  artifactReady: boolean;
  /** 前端 fallback 用，后端生成时可不传 */
  sequence?: number;
  metrics?: ConversionJobMetrics;
  artifacts?: ConversionJobArtifact[];
  errorMessage?: string;
}

export interface CreateConversionInput {
  projectId: string;
  inputDatasetId: string;
  outputFormat: 'HDF5' | 'LeRobot';
  outputLocation: 'local' | 'cloud';
  taskName?: string;
  outputFileName: string;
  outputPath: string;
  /** @deprecated 用 inputDatasetId */
  assetId?: string;
  /** 采样频率 (Hz)，可选 */
  frequency?: number;
  /** 选中的话题列表，可选 */
  topics?: string[];
}

const STAGES: ConversionStage[] = ['Parse', 'Align', 'Write', 'Validate'];
const STAGE_RANGES: [number, number][] = [[0, 25], [25, 55], [55, 90], [90, 100]];

/** 全局序号，用于 taskNo 四位数字（后端未接时 fallback）；不再依赖任何虚拟资产列表 */
let jobSequence = 0;
function nextTaskNo(): string {
  jobSequence += 1;
  return String(jobSequence).padStart(4, '0');
}

let jobs: ConversionJob[] = [];
const listeners: Array<() => void> = [];

function emit() {
  listeners.forEach((f) => f());
}

function defaultStages(): ConversionJobStageProgress[] {
  return STAGES.map((s) => ({ stage: s, status: 'pending', progressPercent: 0 }));
}

export function getMockJobs(): ConversionJob[] {
  return [...jobs];
}

export function getMockJob(jobId: string): ConversionJob | undefined {
  return jobs.find((j) => j.jobId === jobId);
}

export function subscribeMockJobs(cb: () => void): () => void {
  listeners.push(cb);
  return () => {
    const i = listeners.indexOf(cb);
    if (i >= 0) listeners.splice(i, 1);
  };
}

export function createJob(input: CreateConversionInput): ConversionJob {
  const inputId = input.inputDatasetId ?? (input as { assetId?: string }).assetId ?? '';
  const jobId = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  const now = new Date().toISOString();
  const taskNo = nextTaskNo();
  const job: ConversionJob = {
    jobId,
    shortCode: taskNo,
    taskNo,
    outputFileName: input.outputFileName,
    fileName: input.outputFileName,
    assetId: inputId,
    assetName: inputId,
    projectId: input.projectId,
    projectName: input.projectId,
    deviceName: '—',
    outputFormat: input.outputFormat,
    fileFormat: input.outputFormat,
    outputLocation: input.outputLocation,
    outputPath: input.outputPath,
    status: 'queued',
    progressPercent: 0,
    currentStage: null,
    stages: defaultStages(),
    logs: [],
    createdAt: now,
    updatedAt: now,
    artifactReady: false,
    sequence: jobSequence,
  };
  jobs = [job, ...jobs];
  emit();
  return job;
}

let runnerTimer: ReturnType<typeof setInterval> | null = null;

export function startMockRunner(jobId: string): void {
  const idx = jobs.findIndex((j) => j.jobId === jobId);
  if (idx < 0) return;
  const job = jobs[idx];
  if (job.status !== 'queued' && job.status !== 'running') return;

  jobs[idx] = { ...job, status: 'running', updatedAt: new Date().toISOString() };
  emit();

  const willFail = Math.random() < 0.1;
  let progress = 0;
  let stageIndex = 0;
  const stageLogs: Record<ConversionStage, string[]> = {
    Parse: ['[Parse] 打开 MCAP 流', '[Parse] 解析元数据', '[Parse] 读取消息索引'],
    Align: ['[Align] 时间轴对齐', '[Align] 插值采样 targetHz=30', '[Align] 帧序列生成'],
    Write: ['[Write] 写入 HDF5 组', '[Write] 压缩 chunk', '[Write] 刷新缓冲区'],
    Validate: ['[Validate] 校验 checksum', '[Validate] 写入 report.json', '[Validate] 完成'],
  };

  const tick = () => {
    const i = jobs.findIndex((j) => j.jobId === jobId);
    if (i < 0) {
      if (runnerTimer) clearInterval(runnerTimer);
      runnerTimer = null;
      return;
    }
    const j = jobs[i];
    if (j.status === 'canceled' || j.status === 'succeeded' || j.status === 'failed') {
      if (runnerTimer) clearInterval(runnerTimer);
      runnerTimer = null;
      return;
    }

    progress += 4 + Math.floor(Math.random() * 6);
    if (progress >= 100) progress = 100;

    const [lo, hi] = STAGE_RANGES[stageIndex];
    if (progress >= hi && stageIndex < STAGES.length - 1) {
      stageIndex += 1;
      const stage = STAGES[stageIndex];
      const lines = stageLogs[stage];
      const nextStages = j.stages.map((s, si) => {
        if (si < stageIndex) return { ...s, status: 'done' as const, progressPercent: 100, durationMs: 800 + si * 200 };
        if (si === stageIndex) return { ...s, status: 'running' as const, progressPercent: (progress - lo) / (hi - lo) * 100, durationMs: 200 };
        return s;
      });
      jobs[i] = {
        ...j,
        progressPercent: progress,
        currentStage: stage,
        stages: nextStages,
        logs: [...j.logs, ...lines],
        updatedAt: new Date().toISOString(),
      };
    } else {
      const nextStages = j.stages.map((s, si) => {
        if (si < stageIndex) return { ...s, status: 'done' as const, durationMs: s.durationMs ?? 600 };
        if (si === stageIndex) return { ...s, status: 'running' as const, progressPercent: (progress - STAGE_RANGES[si][0]) / (STAGE_RANGES[si][1] - STAGE_RANGES[si][0]) * 100, durationMs: 400 };
        return s;
      });
      jobs[i] = {
        ...j,
        progressPercent: progress,
        currentStage: STAGES[stageIndex],
        stages: nextStages,
        logs: progress % 20 < 5 ? [...j.logs, `[${STAGES[stageIndex]}] 进度 ${progress}%`] : j.logs,
        updatedAt: new Date().toISOString(),
      };
    }

    if (progress >= 100) {
      if (willFail) {
        jobs[i] = {
          ...jobs[i],
          status: 'failed',
          logs: [...jobs[i].logs, '[ERROR] 校验失败: 帧序不一致'],
          errorMessage: '校验失败: 帧序不一致',
          updatedAt: new Date().toISOString(),
        };
      } else {
        jobs[i] = {
          ...jobs[i],
          status: 'succeeded',
          currentStage: 'Validate',
          stages: jobs[i].stages.map((s) => ({ ...s, status: 'done' as const, progressPercent: 100, durationMs: 500 })),
          logs: [...jobs[i].logs, '[Validate] 完成'],
          metrics: { dropRate: 0.02, timeSkewMs: 12, steps: 3600, outputSize: 48 },
          artifacts: [
            { type: 'hdf5', name: 'output.hdf5', path: jobs[i].outputPath + '/output.hdf5', sizeBytes: 52 * 1024 * 1024 },
            { type: 'report', name: 'report.json', path: jobs[i].outputPath + '/report.json' },
          ],
          artifactReady: true,
          updatedAt: new Date().toISOString(),
        };
      }
      if (runnerTimer) clearInterval(runnerTimer);
      runnerTimer = null;
    }
    emit();
  };

  if (runnerTimer) clearInterval(runnerTimer);
  runnerTimer = setInterval(tick, 350 + Math.floor(Math.random() * 250));
}

export function deleteJob(jobId: string): void {
  jobs = jobs.filter((j) => j.jobId !== jobId);
  emit();
}

export function cancelJob(jobId: string): void {
  const idx = jobs.findIndex((j) => j.jobId === jobId);
  if (idx < 0) return;
  if (jobs[idx].status === 'queued' || jobs[idx].status === 'running') {
    jobs[idx] = { ...jobs[idx], status: 'canceled', updatedAt: new Date().toISOString() };
    if (runnerTimer) {
      clearInterval(runnerTimer);
      runnerTimer = null;
    }
    emit();
  }
}

export function retryJob(jobId: string): ConversionJob | null {
  const idx = jobs.findIndex((j) => j.jobId === jobId);
  if (idx < 0) return null;
  const j = jobs[idx];
  if (j.status !== 'failed') return null;
  const input: CreateConversionInput = {
    projectId: j.projectId,
    inputDatasetId: j.assetId,
    outputFormat: j.outputFormat,
    outputLocation: j.outputLocation,
    outputFileName: j.outputFileName ?? j.fileName,
    outputPath: j.outputPath,
  };
  jobs = jobs.filter((x) => x.jobId !== jobId);
  const newJob = createJob(input);
  emit();
  startMockRunner(newJob.jobId);
  return newJob;
}

// 不再预置任何虚拟任务数据；列表默认为空，由用户基于真实数据资产新建任务
