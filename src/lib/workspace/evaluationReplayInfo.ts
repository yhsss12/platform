export interface EvaluationReplayUriItem {
  episodeIndex?: number | null;
  uri: string;
  label?: string | null;
  fileName?: string | null;
  recordCamera?: string | null;
  sourceKind?: string | null;
  evaluationMode?: string | null;
}

export interface EvaluationReplayInfo {
  requestedEpisodes?: number | null;
  completedEpisodes?: number | null;
  successfulEpisodes?: number | null;
  failedEpisodes?: number | null;
  recordedVideoCount?: number | null;
  replayUri?: string | null;
  replayUris?: EvaluationReplayUriItem[];
  videoAvailable?: boolean;
  videoSourceKind?: string | null;
  evaluationMode?: string | null;
  isRepresentativeVideo?: boolean;
  currentEpisodeIndex?: number | null;
  successRate?: number | null;
  warning?: string | null;
}

function pickNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

export function parseEvaluationReplayInfo(
  source: Record<string, unknown> | null | undefined
): EvaluationReplayInfo {
  if (!source) return {};
  const replayUrisRaw = source.replayUris;
  const replayUris = Array.isArray(replayUrisRaw)
    ? replayUrisRaw
        .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
        .map((item) => ({
          episodeIndex: pickNumber(item.episodeIndex),
          uri: String(item.uri ?? ''),
          label: typeof item.label === 'string' ? item.label : null,
          fileName: typeof item.fileName === 'string' ? item.fileName : null,
          sourceKind: typeof item.sourceKind === 'string' ? item.sourceKind : null,
          evaluationMode: typeof item.evaluationMode === 'string' ? item.evaluationMode : null,
        }))
        .filter((item) => item.uri)
    : [];

  return {
    requestedEpisodes: pickNumber(source.requestedEpisodes),
    completedEpisodes: pickNumber(source.completedEpisodes),
    successfulEpisodes: pickNumber(source.successfulEpisodes),
    failedEpisodes: pickNumber(source.failedEpisodes),
    recordedVideoCount: pickNumber(source.recordedVideoCount),
    replayUri: typeof source.replayUri === 'string' ? source.replayUri : null,
    replayUris,
    videoAvailable: source.videoAvailable === true,
    videoSourceKind: typeof source.videoSourceKind === 'string' ? source.videoSourceKind : null,
    evaluationMode: typeof source.evaluationMode === 'string' ? source.evaluationMode : null,
    isRepresentativeVideo: source.isRepresentativeVideo === true,
    currentEpisodeIndex: pickNumber(source.currentEpisodeIndex),
    successRate: typeof source.successRate === 'number' ? source.successRate : null,
    warning: typeof source.warning === 'string' ? source.warning : null,
  };
}

export function mergeEvaluationReplayInfo(
  ...sources: Array<Record<string, unknown> | null | undefined>
): EvaluationReplayInfo {
  const merged: EvaluationReplayInfo = {};
  for (const source of sources) {
    const parsed = parseEvaluationReplayInfo(source ?? undefined);
    const { replayUris: _uris, ...rest } = parsed;
    Object.assign(merged, {
      ...merged,
      ...Object.fromEntries(
        Object.entries(rest).filter(([, value]) => value !== null && value !== undefined)
      ),
    });
    if (parsed.replayUris?.length) {
      merged.replayUris = parsed.replayUris;
    }
  }
  return merged;
}

/** 从 replayUri 条目解析真实轮次（1-based），用于排序与中文标签。 */
export function getEpisodeIndex(item: EvaluationReplayUriItem, fallbackIndex: number): number {
  if (typeof item.episodeIndex === 'number' && item.episodeIndex > 0) {
    return item.episodeIndex;
  }
  const source = item.fileName || item.uri || '';
  const match = source.match(/episode[_-](\d+)/i);
  if (match) return Number(match[1]);
  return fallbackIndex + 1;
}

/** 生成中文轨迹按钮文案。 */
export function getTrajectoryLabel(item: EvaluationReplayUriItem, index: number): string {
  if (item.label === '代表性回放') return '代表性回放';
  const round = getEpisodeIndex(item, index);
  return `第 ${round} 轮轨迹`;
}

/** 归一化、排序 replayUris，并覆盖为中文标签。 */
export function normalizeReplayTrajectoryItems(
  replay: EvaluationReplayInfo
): EvaluationReplayUriItem[] {
  let items: EvaluationReplayUriItem[] = [];
  if (replay.replayUris?.length) {
    items = replay.replayUris.map((item) => ({ ...item }));
  } else if (replay.replayUri) {
    items = [
      {
        uri: replay.replayUri,
        episodeIndex: replay.isRepresentativeVideo ? null : 1,
        fileName: replay.isRepresentativeVideo ? 'eval.mp4' : undefined,
      },
    ];
  }

  if (!items.length) return [];

  const representativeOnly =
    replay.isRepresentativeVideo || (items.length === 1 && !items[0]?.fileName?.match(/episode_/i));

  const normalized = items.map((item, index) => {
    if (representativeOnly && items.length === 1) {
      return {
        ...item,
        label: '代表性回放',
        episodeIndex: null,
      };
    }
    const round = getEpisodeIndex(item, index);
    return {
      ...item,
      episodeIndex: round,
      label: getTrajectoryLabel({ ...item, episodeIndex: round }, index),
    };
  });

  if (representativeOnly && normalized.length === 1) {
    return normalized;
  }

  return normalized.sort((a, b) => {
    const ai = typeof a.episodeIndex === 'number' ? a.episodeIndex : 0;
    const bi = typeof b.episodeIndex === 'number' ? b.episodeIndex : 0;
    return ai - bi;
  });
}

/** 按 URL ?episode=N（1-based 轮次）查找选中下标。 */
export function findTrajectoryIndexByRound(
  items: EvaluationReplayUriItem[],
  round: number | null | undefined
): number {
  if (!round || round < 1 || !items.length) return 0;
  const idx = items.findIndex((item, index) => getEpisodeIndex(item, index) === round);
  return idx >= 0 ? idx : 0;
}

export function formatEpisodeProgressPercent(
  completed: number | null | undefined,
  requested: number | null | undefined
): string | null {
  if (!requested || requested <= 0 || completed == null) return null;
  return `${Math.min(100, Math.round((completed / requested) * 100))}%`;
}

export function buildRepresentativeVideoHint(replay: EvaluationReplayInfo): string | null {
  const requested = replay.requestedEpisodes;
  const recorded = replay.recordedVideoCount ?? 0;
  if (!requested || recorded <= 0) return null;
  if (recorded >= requested) return null;
  if (replay.isRepresentativeVideo || recorded === 1) {
    const executed = replay.completedEpisodes ?? requested;
    return `当前旧任务仅保留 1 段代表性回放，评测实际执行 ${executed} 轮。`;
  }
  return `当前回放视频 ${recorded} 段，少于计划 ${requested} 轮。`;
}

export function buildIncompleteEpisodesWarning(replay: EvaluationReplayInfo): string | null {
  const requested = replay.requestedEpisodes;
  const completed = replay.completedEpisodes;
  if (!requested || completed == null) return null;
  if (completed < requested) {
    return `实际完成轮数（${completed}）少于计划轮数（${requested}），请检查执行日志。`;
  }
  const recorded = replay.recordedVideoCount ?? 0;
  if (completed > recorded && recorded > 0 && !replay.isRepresentativeVideo) {
    return `视频段数（${recorded}）少于实际完成轮数（${completed}），请检查录制配置或 runner 输出。`;
  }
  return null;
}

export function resolveReplayVideoApiPath(uri: string): string {
  if (uri.startsWith('http://') || uri.startsWith('https://')) return uri;
  if (uri.startsWith('/api/')) return uri;
  if (uri.startsWith('/')) return `/api${uri}`;
  return uri;
}
