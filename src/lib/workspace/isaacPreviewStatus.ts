/** Isaac 物块堆叠 preview 状态（与 job status.json / generation_manifest 对齐）。 */

export type PreviewStatus = 'pending' | 'generating' | 'completed' | 'failed' | string;

export function resolvePreviewStatusDisplay(input: {
  previewStatus?: string | null;
  videoAvailable?: boolean | null;
  videoNote?: string | null;
  jobStatus?: string | null;
  phase?: string | null;
}): {
  label: string;
  canOpenReplay: boolean;
  hint: string | null;
} {
  const status = (input.previewStatus ?? '').trim();
  const phase = (input.phase ?? '').trim();
  const jobRunning = input.jobStatus === 'running' || input.jobStatus === 'queued';

  if (status === 'generating' || (phase === 'replay_preview' && jobRunning)) {
    return { label: '生成中', canOpenReplay: false, hint: null };
  }
  if (status === 'completed' || (status === '' && input.videoAvailable)) {
    return { label: '已生成', canOpenReplay: true, hint: null };
  }
  if (status === 'failed') {
    return {
      label: '生成失败',
      canOpenReplay: false,
      hint: input.videoNote ?? 'preview 生成失败，可重新生成回放',
    };
  }
  if (status === 'pending' && jobRunning) {
    return { label: '等待生成', canOpenReplay: false, hint: null };
  }
  if (input.jobStatus === 'completed' && !input.videoAvailable) {
    return {
      label: '未生成',
      canOpenReplay: false,
      hint: input.videoNote ?? '暂无回放视频',
    };
  }
  return {
    label: input.videoAvailable ? '已生成' : '未生成',
    canOpenReplay: Boolean(input.videoAvailable),
    hint: input.videoNote ?? null,
  };
}
