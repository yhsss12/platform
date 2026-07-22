import type { NutAssemblyJobStatusResponse } from '@/lib/api/nutAssemblyClient';
import { isTerminalSimJobStatus } from '@/lib/workspace/simulationPolling';

export type NutAssemblyTimelineStageState = 'done' | 'active' | 'pending' | 'failed';

export interface NutAssemblyTimelineStage {
  id: string;
  label: string;
  state: NutAssemblyTimelineStageState;
}

export interface NutAssemblyProgressViewModel {
  status: string;
  terminal: boolean;
  stage: string;
  stageLabel: string;
  progressPercent: number;
  progressMode: 'stage_based' | 'count_based';
  progressCaption: string;
  statusBadge: { label: string; bg: string; color: string };
  subtitle: string;
  generationModeLabel: string;
  sourceDemoLabel: string;
  episodesRequested: number | null;
  episodesGenerated: number | null;
  datagenFailedTrials: number | null;
  datagenSuccessRatePct: string;
  statsResolved: boolean;
  metricCards: {
    requested: string;
    mimicgenWritten: string;
    pinnRepaired: string;
    finalCount: string;
  };
  summaryRow: {
    datagenWriteRate: string;
    pinnEnhancementGain: string;
    taskEvalLabel: string;
  };
  physicsEnhancementEnabled: boolean;
  pinnBackend: string | null;
  modelLoaded: boolean | null;
  pinnCandidateCount: number | null;
  pinnRepairAttempted: number | null;
  enhancementStatus: string | null;
  pinnStatusMessage: string | null;
  timeline: NutAssemblyTimelineStage[];
  artifacts: {
    hdf5: string;
    video: string;
    registry: string;
  };
  taskEvalLabel: string;
  hdf5Exists: boolean;
  videoExists: boolean;
  canViewReplay: boolean;
  advancedPaths: {
    sourceDemoPath: string;
    hdf5Path: string;
    videoPath: string;
    manifestPath: string;
    summaryPath: string;
    sourceDemoHash: string;
    envName: string;
    objectPoseKeys: string;
  };
}

const STAGE_PROGRESS: Record<string, number> = {
  queued: 5,
  env_check: 10,
  prepare_source: 25,
  mimicgen_generate: 60,
  pinn_repair: 72,
  pinn_validation: 78,
  robosuite_rollout: 60,
  write_summary: 80,
  write_manifest: 82,
  write_dataset: 80,
  render_video: 90,
  completed: 100,
};

const STAGE_LABELS: Record<string, string> = {
  env_check: '环境检查',
  prepare_source: '准备示教数据',
  mimicgen_generate: 'MimicGen 生成',
  pinn_repair: 'PINN 轨迹修复',
  pinn_validation: 'MuJoCo 复核',
  robosuite_rollout: '规则采集',
  write_summary: '写入 HDF5',
  write_manifest: '写入 HDF5',
  write_dataset: '写入 HDF5',
  render_video: '生成回放',
  completed: '完成',
  failed: '失败',
  stalled: '可能卡住',
  queued: '排队中',
};

const STATUS_BADGE: Record<string, { label: string; bg: string; color: string }> = {
  running: { label: '运行中', bg: '#dbeafe', color: '#1d4ed8' },
  queued: { label: '运行中', bg: '#dbeafe', color: '#1d4ed8' },
  completed: { label: '已完成', bg: '#d1fae5', color: '#047857' },
  failed: { label: '生成失败', bg: '#fee2e2', color: '#b91c1c' },
  partial_success: { label: '部分完成', bg: '#fef3c7', color: '#b45309' },
  stalled: { label: '已卡住', bg: '#ffedd5', color: '#c2410c' },
  cancelled: { label: '已取消', bg: '#f3f4f6', color: '#6b7280' },
  canceled: { label: '已取消', bg: '#f3f4f6', color: '#6b7280' },
};

function normalizeStage(stage: string | null | undefined): string {
  const raw = (stage ?? '').trim();
  if (!raw || raw === '—') return 'env_check';
  if (raw === 'write_summary' || raw === 'write_manifest') return 'write_dataset';
  return raw;
}

export function resolveNutAssemblyNumericField(
  job: NutAssemblyJobStatusResponse | null,
  live: Record<string, unknown>,
  key: 'episodesRequested' | 'episodesGenerated' | 'datagenFailedTrials'
): number | null {
  const fromJob = job?.[key];
  if (typeof fromJob === 'number') return fromJob;
  const fromLive = live[key];
  if (typeof fromLive === 'number') return fromLive;
  if (key === 'episodesRequested') {
    const episodes = live.episodes;
    if (typeof episodes === 'number') return episodes;
  }
  if (key === 'episodesGenerated') {
    const episode = live.episode;
    if (typeof episode === 'number' && episode > 0) return episode;
  }
  return null;
}

export function hasNutAssemblyDatagenStatsResolved(
  job: NutAssemblyJobStatusResponse | null,
  live: Record<string, unknown>,
  terminal: boolean
): boolean {
  if (terminal) return true;
  if (job?.hasDatagenInfo === true || live.hasDatagenInfo === true) return true;
  const generated = resolveNutAssemblyNumericField(job, live, 'episodesGenerated');
  const failed = resolveNutAssemblyNumericField(job, live, 'datagenFailedTrials');
  if (generated != null && generated > 0) return true;
  if (failed != null && failed > 0) return true;
  return false;
}

export function formatNutAssemblySourceDemoLabel(
  origin: string | null | undefined,
  sourceDemoPath?: string | null
): string {
  if (origin === 'official_mimicgen_source') return '系统内置示教数据';
  if (origin === 'local_source') return '本地示教数据';
  if (origin === 'custom_source') return '自定义示教数据';
  if (sourceDemoPath) return '示教数据已配置';
  return '—';
}

export function formatNutAssemblyProgressSubtitle(
  generationMode: string | null | undefined,
  sourceDemoOrigin: string | null | undefined,
  episodesRequested: number | null
): string {
  const mode =
    generationMode === 'robosuite_rollout'
      ? '规则生成（调试）'
      : generationMode === 'mimicgen_datagen'
        ? 'MimicGen 生成'
        : '数据生成';
  const source =
    sourceDemoOrigin === 'official_mimicgen_source'
      ? '官方示教数据'
      : formatNutAssemblySourceDemoLabel(sourceDemoOrigin);
  const attempts = episodesRequested != null ? `${episodesRequested} 次尝试` : '—';
  return `${mode} · ${source} · ${attempts}`;
}

function stageProgressPercent(stage: string, status: string, jobProgress?: number | null): number {
  if (status === 'completed' || stage === 'completed') return 100;
  if (status === 'failed') {
    return typeof jobProgress === 'number' ? Math.max(0, Math.min(jobProgress, 99)) : STAGE_PROGRESS[stage] ?? 10;
  }
  if (typeof jobProgress === 'number' && jobProgress > 0) {
    return Math.max(STAGE_PROGRESS[stage] ?? 10, Math.min(jobProgress, 99));
  }
  return STAGE_PROGRESS[stage] ?? 10;
}

function buildTimeline(
  generationMode: string | null | undefined,
  currentStage: string,
  status: string,
  physicsEnhancementEnabled: boolean,
): NutAssemblyTimelineStage[] {
  const isRollout = generationMode === 'robosuite_rollout';
  const defs = isRollout
    ? [
        { id: 'env_check', label: '环境检查' },
        { id: 'robosuite_rollout', label: '规则采集' },
        { id: 'write_dataset', label: '写入 HDF5' },
        { id: 'render_video', label: '生成回放' },
        { id: 'completed', label: '完成' },
      ]
    : physicsEnhancementEnabled
      ? [
          { id: 'env_check', label: '环境检查' },
          { id: 'prepare_source', label: '准备示教数据' },
          { id: 'mimicgen_generate', label: 'MimicGen 生成' },
          { id: 'pinn_repair', label: 'PINN 轨迹修复' },
          { id: 'pinn_validation', label: 'MuJoCo 复核' },
          { id: 'write_dataset', label: '写入 HDF5' },
          { id: 'render_video', label: '生成回放' },
          { id: 'completed', label: '完成' },
        ]
      : [
          { id: 'env_check', label: '环境检查' },
          { id: 'prepare_source', label: '准备示教数据' },
          { id: 'mimicgen_generate', label: 'MimicGen 生成' },
          { id: 'write_dataset', label: '写入 HDF5' },
          { id: 'render_video', label: '生成回放' },
          { id: 'completed', label: '完成' },
        ];

  const normalizedCurrent = normalizeStage(currentStage);
  const currentIndex = defs.findIndex((d) => d.id === normalizedCurrent);
  const failed = status === 'failed';
  const done = status === 'completed' || status === 'partial_success';

  return defs.map((def, index) => {
    let state: NutAssemblyTimelineStageState = 'pending';
    if (failed && def.id === normalizedCurrent) state = 'failed';
    else if (done) state = 'done';
    else if (currentIndex < 0) state = index === 0 ? 'active' : 'pending';
    else if (index < currentIndex) state = 'done';
    else if (index === currentIndex) state = 'active';
    return { ...def, state };
  });
}

function formatMetricGenerated(
  status: string,
  value: number | null,
  statsResolved: boolean
): string {
  if (!isTerminalSimJobStatus(status)) {
    if (!statsResolved || value == null || value <= 0) return '生成中';
    return `${value} 条`;
  }
  if (value == null) return '—';
  return `${value} 条`;
}

function formatMetricFailedTrials(
  status: string,
  value: number | null,
  statsResolved: boolean
): string {
  if (!isTerminalSimJobStatus(status)) {
    if (!statsResolved || value == null || value <= 0) return '待统计';
    return `${value} 次`;
  }
  if (value == null) return '—';
  return `${value} 次`;
}

function formatWriteRatePct(
  status: string,
  rate: number | null | undefined,
  generated: number | null,
  requested: number | null,
  statsResolved: boolean
): string {
  if (!isTerminalSimJobStatus(status)) {
    if (!statsResolved) return '待统计';
  }
  let resolved = rate;
  if (resolved == null && generated != null && requested != null && requested > 0) {
    resolved = generated / requested;
  }
  if (resolved == null) return isTerminalSimJobStatus(status) ? '—' : '待统计';
  return `${Math.round(resolved * 1000) / 10}%`;
}

function formatTaskEvalLabel(successRate: number | null | undefined, generationMode: string | null): string {
  if (generationMode === 'mimicgen_datagen') return '未评测';
  if (successRate == null || successRate === 0) return '未评测';
  return `${Math.round(Number(successRate) * 1000) / 10}%`;
}

function formatPinnStatusMessage(input: {
  physicsEnhancementEnabled: boolean;
  pinnBackend: string | null;
  modelLoaded: boolean | null;
  pinnCandidateCount: number | null;
  pinnRepairAttempted: number | null;
  pinnValidationSucceeded: number | null;
  enhancementStatus: string | null;
  terminal: boolean;
  stage: string;
}): string | null {
  if (!input.physicsEnhancementEnabled) return null;

  const backendLabel =
    input.pinnBackend === 'torch_model' && input.modelLoaded
      ? '已加载 NutAssembly-PINN v1（torch_model，assets/models/pinn/nut_assembly_pinn_v1/model.pt）。'
      : '当前使用启发式修复流程，尚未加载 PINN 权重。';

  if (!input.terminal && ['pinn_repair', 'pinn_validation'].includes(input.stage)) {
    return `PINN 修复阶段执行中。${backendLabel}`;
  }

  if (input.enhancementStatus === 'completed_no_candidates' && input.terminal) {
    return `PINN 修复阶段已执行，但本批次未筛选到需要修复的候选轨迹。${backendLabel}`;
  }

  if (input.pinnCandidateCount != null && input.pinnCandidateCount > 0) {
    const attempted = input.pinnRepairAttempted ?? 0;
    const validated = input.pinnValidationSucceeded ?? 0;
    if (input.terminal) {
      return `PINN 已执行：候选 ${input.pinnCandidateCount} 条，修复尝试 ${attempted} 次，复核通过 ${validated} 条。${backendLabel}`;
    }
    return `PINN 修复进行中：候选 ${input.pinnCandidateCount} 条。${backendLabel}`;
  }

  if (input.terminal) {
    return `PINN 修复阶段已执行。${backendLabel}`;
  }
  return backendLabel;
}

export function buildNutAssemblyProgressViewModel(
  job: NutAssemblyJobStatusResponse | null,
  jobId: string
): NutAssemblyProgressViewModel {
  const live = (job?.live ?? {}) as Record<string, unknown>;
  const status = job?.status ?? 'running';
  const terminal = isTerminalSimJobStatus(status);
  const stage = normalizeStage(String(job?.stage ?? live.stage ?? 'env_check'));
  const generationMode =
    job?.generationMode ??
    (typeof live.generationMode === 'string' ? live.generationMode : null) ??
    (typeof live.generationModePreference === 'string' ? live.generationModePreference : null);
  const sourceDemoOrigin =
    job?.sourceDemoOrigin ?? (typeof live.sourceDemoOrigin === 'string' ? live.sourceDemoOrigin : null);

  const episodesRequested = resolveNutAssemblyNumericField(job, live, 'episodesRequested');
  const episodesGenerated = resolveNutAssemblyNumericField(job, live, 'episodesGenerated');
  const datagenFailedTrials = resolveNutAssemblyNumericField(job, live, 'datagenFailedTrials');
  const statsResolved = hasNutAssemblyDatagenStatsResolved(job, live, terminal);

  const countBased =
    statsResolved &&
    episodesRequested != null &&
    episodesRequested > 0 &&
    episodesGenerated != null &&
    episodesGenerated >= 0;

  const stagePercent = stageProgressPercent(stage, status, job?.progress);
  const countPercent =
    countBased && episodesRequested
      ? Math.min(99, Math.round(((episodesGenerated ?? 0) / episodesRequested) * 100))
      : stagePercent;
  const progressPercent = terminal ? 100 : countBased ? Math.max(stagePercent, countPercent) : stagePercent;
  const progressMode: 'stage_based' | 'count_based' = countBased && !terminal ? 'count_based' : 'stage_based';

  let progressCaption: string;
  if (terminal && status === 'completed') {
    progressCaption = '数据生成已完成';
  } else if (terminal && status === 'failed') {
    progressCaption = '生成失败，请查看日志';
  } else if (countBased && episodesRequested != null && episodesGenerated != null && episodesGenerated > 0) {
    progressCaption = `已写入 ${episodesGenerated} / ${episodesRequested} 条`;
  } else if (!terminal) {
    progressCaption = '生成中，正在等待 MimicGen 写入统计';
  } else {
    progressCaption = STAGE_LABELS[stage] ?? stage;
  }

  const hdf5Exists = Boolean(
    job?.paths?.hdf5?.exists ||
      job?.hdf5Path ||
      live.savedHdf5 ||
      (terminal && status === 'completed')
  );
  const videoExists = Boolean(
    job?.generateVideoExists ||
      job?.paths?.video?.exists ||
      live.generateVideoExists ||
      job?.videoUrl
  );

  const hdf5Path =
    job?.hdf5Path ??
    (job?.paths?.hdf5?.exists ? job.paths.hdf5.path : null) ??
    'datasets/nut_assembly_generated.hdf5';
  const videoPath =
    job?.videoPath ??
    (videoExists ? 'videos/generate.mp4' : null) ??
    'videos/generate.mp4';
  const manifestPath = job?.paths?.manifest?.path ?? 'manifest.json';
  const summaryPath = job?.paths?.summary?.path ?? 'results/generation_summary.json';

  const sourceDemoHash =
    (typeof live.sourceDemoMd5 === 'string' && live.sourceDemoMd5) ||
    (typeof live.sourceDemoHash === 'string' && live.sourceDemoHash) ||
    '—';
  const envName =
    job?.runtimeEnvName ??
    job?.sourceEnvName ??
    (typeof live.runtimeEnvName === 'string' ? live.runtimeEnvName : null) ??
    (typeof live.sourceEnvName === 'string' ? live.sourceEnvName : null) ??
    '—';
  const objectPoseKeys = (job?.objectPoseKeys ?? live.objectPoseKeys ?? []) as string[];

  const canViewReplay =
    (status === 'completed' || status === 'partial_success') && (videoExists || hdf5Exists);

  const physicsEnhancementEnabled = Boolean(
    live.physicsEnhancementEnabled ?? job?.metrics?.physicsEnhancementEnabled
  );
  const rawDemoCount =
    typeof live.rawDemoCount === 'number'
      ? live.rawDemoCount
      : typeof live.mimicgenGeneratedDemos === 'number'
        ? live.mimicgenGeneratedDemos
        : null;
  const pinnValidationSucceeded =
    typeof live.pinnValidationSucceeded === 'number' ? live.pinnValidationSucceeded : null;
  const pinnCandidateCount =
    typeof live.pinnCandidateCount === 'number' ? live.pinnCandidateCount : null;
  const pinnRepairAttempted =
    typeof live.pinnRepairAttempted === 'number' ? live.pinnRepairAttempted : null;
  const pinnBackend =
    (typeof live.pinnBackend === 'string' ? live.pinnBackend : null) ??
    (typeof job?.metrics?.pinnBackend === 'string' ? job.metrics.pinnBackend : null);
  const modelLoaded =
    typeof live.modelLoaded === 'boolean'
      ? live.modelLoaded
      : typeof job?.metrics?.modelLoaded === 'boolean'
        ? job.metrics.modelLoaded
        : null;
  const enhancementStatus =
    (typeof live.enhancementStatus === 'string' ? live.enhancementStatus : null) ??
    (typeof job?.metrics?.enhancementStatus === 'string' ? job.metrics.enhancementStatus : null);
  const finalDemoCount =
    typeof live.finalDemoCount === 'number'
      ? live.finalDemoCount
      : terminal
        ? episodesGenerated
        : null;
  const enhancementGain =
    typeof live.enhancementGain === 'number'
      ? live.enhancementGain
      : rawDemoCount != null && finalDemoCount != null
        ? Math.max(finalDemoCount - rawDemoCount, 0)
        : null;

  const mimicgenWrittenDisplay = (() => {
    if (!terminal && (rawDemoCount == null || rawDemoCount <= 0) && !statsResolved) return '生成中';
    const value = rawDemoCount ?? (statsResolved ? episodesGenerated : null);
    if (value == null) return terminal ? '—' : '生成中';
    return `${value} 条`;
  })();

  const pinnRepairedDisplay = (() => {
    if (!physicsEnhancementEnabled) return '未启用';
    if (
      !terminal &&
      pinnValidationSucceeded == null &&
      stage !== 'pinn_validation' &&
      stage !== 'pinn_repair'
    ) {
      return '待执行';
    }
    if (!terminal && pinnValidationSucceeded == null) return '修复中';
    if (pinnValidationSucceeded == null) return terminal ? '0 条' : '待统计';
    return `${pinnValidationSucceeded} 条`;
  })();

  const finalCountDisplay = (() => {
    if (!terminal && finalDemoCount == null) return '统计中';
    if (finalDemoCount == null) return '—';
    return `${finalDemoCount} 条`;
  })();

  const pinnStatusMessage = formatPinnStatusMessage({
    physicsEnhancementEnabled,
    pinnBackend,
    modelLoaded,
    pinnCandidateCount,
    pinnRepairAttempted,
    pinnValidationSucceeded,
    enhancementStatus,
    terminal,
    stage,
  });

  return {
    status,
    terminal,
    stage,
    stageLabel: STAGE_LABELS[stage] ?? stage,
    progressPercent,
    progressMode,
    progressCaption,
    statusBadge: STATUS_BADGE[status] ?? STATUS_BADGE.running,
    subtitle: formatNutAssemblyProgressSubtitle(generationMode, sourceDemoOrigin, episodesRequested),
    generationModeLabel:
      generationMode === 'mimicgen_datagen'
        ? 'MimicGen 生成'
        : generationMode === 'robosuite_rollout'
          ? '规则生成（调试）'
          : generationMode ?? '—',
    sourceDemoLabel: formatNutAssemblySourceDemoLabel(sourceDemoOrigin, job?.sourceDemoPath),
    episodesRequested,
    episodesGenerated,
    datagenFailedTrials,
    datagenSuccessRatePct: formatWriteRatePct(
      status,
      job?.datagenSuccessRate,
      episodesGenerated,
      episodesRequested,
      statsResolved
    ),
    statsResolved,
    physicsEnhancementEnabled,
    pinnBackend,
    modelLoaded,
    pinnCandidateCount,
    pinnRepairAttempted,
    enhancementStatus,
    pinnStatusMessage,
    metricCards: {
      requested: episodesRequested != null ? `${episodesRequested} 次` : '—',
      mimicgenWritten: mimicgenWrittenDisplay,
      pinnRepaired: pinnRepairedDisplay,
      finalCount: finalCountDisplay,
    },
    summaryRow: {
      datagenWriteRate: formatWriteRatePct(
        status,
        job?.datagenSuccessRate,
        rawDemoCount ?? episodesGenerated,
        episodesRequested,
        statsResolved
      ),
      pinnEnhancementGain:
        enhancementGain != null ? `+${enhancementGain}` : physicsEnhancementEnabled ? '待统计' : '—',
      taskEvalLabel: formatTaskEvalLabel(job?.successRate, generationMode),
    },
    timeline: buildTimeline(generationMode, stage, status, physicsEnhancementEnabled),
    artifacts: {
      hdf5: terminal ? (hdf5Exists ? '已生成' : '未生成') : hdf5Exists ? '已生成' : '待生成',
      video: terminal
        ? videoExists
          ? '已生成'
          : '未生成'
        : videoExists
          ? '已生成'
          : '待生成',
      registry: terminal ? '已完成' : '待完成',
    },
    taskEvalLabel: formatTaskEvalLabel(job?.successRate, generationMode),
    hdf5Exists,
    videoExists,
    canViewReplay,
    advancedPaths: {
      sourceDemoPath: job?.sourceDemoPath ?? '—',
      hdf5Path: String(hdf5Path),
      videoPath: String(videoPath),
      manifestPath: String(manifestPath),
      summaryPath: String(summaryPath),
      sourceDemoHash: String(sourceDemoHash),
      envName: String(envName),
      objectPoseKeys: objectPoseKeys.length > 0 ? objectPoseKeys.join(', ') : '—',
    },
  };
}

export function nutAssemblyStageStatusLabelShort(stage: string | null | undefined): string {
  const normalized = normalizeStage(stage);
  return STAGE_LABELS[normalized] ?? normalized ?? '—';
}
