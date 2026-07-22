/** 真实后端 jobId 与 pending 临时 ID 识别（前端 UI 占位，不可作为控制台 jobId） */

export const CT_GEN_JOB_ID_PATTERN = /^ct_gen_\d{8}_\d{6}_[a-f0-9]{4}$/;
export const DAC_GEN_JOB_ID_PATTERN = /^dac_gen_\d{8}_\d{6}_[a-f0-9]{4}$/;
export const ISAAC_GEN_JOB_ID_PATTERN = /^isaac_gen_\d{8}_\d{6}_[a-f0-9]{4}$/;
export const ISAAC_REPLAY_JOB_ID_PATTERN = /^isaac_replay_\d{8}_\d{6}_[a-f0-9]{4}$/;
export const DATA_GEN_JOB_ID_PATTERN = /^data_gen_\d{8}_\d{6}_[a-f0-9]{4}$/;
export const NA_GEN_JOB_ID_PATTERN = /^na_gen_\d{8}_\d{6}_[a-f0-9]{4}$/;

export function isPendingLocalJobId(id: string | null | undefined): boolean {
  if (!id) return false;
  return (
    id.startsWith('ct-pending-') ||
    id.startsWith('dac-pending-') ||
    id.startsWith('pending-') ||
    id.startsWith('ct-run_')
  );
}

export function isValidCableThreadingGenerateJobId(id: string | null | undefined): boolean {
  return Boolean(id && CT_GEN_JOB_ID_PATTERN.test(id));
}

export function isValidDualArmGenerateJobId(id: string | null | undefined): boolean {
  return Boolean(id && DAC_GEN_JOB_ID_PATTERN.test(id));
}

export function isValidIsaacGenerateJobId(id: string | null | undefined): boolean {
  return Boolean(id && ISAAC_GEN_JOB_ID_PATTERN.test(id));
}

export function isValidIsaacReplayJobId(id: string | null | undefined): boolean {
  return Boolean(id && ISAAC_REPLAY_JOB_ID_PATTERN.test(id));
}

export function isValidDataGenJobId(id: string | null | undefined): boolean {
  return Boolean(id && DATA_GEN_JOB_ID_PATTERN.test(id));
}

export function isValidNutAssemblyGenerateJobId(id: string | null | undefined): boolean {
  return Boolean(id && NA_GEN_JOB_ID_PATTERN.test(id));
}

export function isStalePendingDataItem(item: {
  id: string;
  status: string;
  backendJobId?: string | null;
  jobId?: string | null;
}): boolean {
  const active =
    item.status === 'generating' ||
    item.status === 'pending' ||
    item.status === 'running' ||
    item.status === 'processing';
  if (!active) return false;
  if (item.backendJobId && !isPendingLocalJobId(item.backendJobId)) return false;
  if (item.jobId && !isPendingLocalJobId(item.jobId)) return false;
  return (
    item.id.startsWith('ct-pending-') ||
    item.id.startsWith('dac-pending-') ||
    isPendingLocalJobId(item.id)
  );
}
