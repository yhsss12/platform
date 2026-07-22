import type { CableThreadingGenerateRequest } from '@/lib/api/cableThreadingClient';
import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import { CABLE_THREADING_DEFAULTS } from '@/lib/workspace/cableThreading';

export const LEROBOT_DEFAULTS = {
  taskInstruction: 'thread the cable through the pole',
  robot: 'Panda',
  fps: 20,
} as const;

export type CableThreadingOutputFormat = 'hdf5' | 'npz' | 'lerobot';

const UNSUPPORTED_FORMAT_MESSAGE =
  '所选数据格式尚未支持，请选择 HDF5 或 LeRobot';

/** Map UI dataFormat to backend outputFormat without silent NPZ fallback. */
export function resolveCableThreadingOutputFormat(
  dataFormat: GenerateDataPayload['dataFormat']
): CableThreadingOutputFormat {
  if (dataFormat === 'hdf5') return 'hdf5';
  if (dataFormat === 'lerobot') return 'lerobot';
  throw new Error(UNSUPPORTED_FORMAT_MESSAGE);
}

export function buildCableThreadingGenerateRequest(
  payload: GenerateDataPayload,
  taskConfigId?: string | null
): CableThreadingGenerateRequest {
  const outputFormat = resolveCableThreadingOutputFormat(payload.dataFormat);
  const base = {
    episodes: payload.episodes,
    robot: payload.cableThreadingRobot ?? CABLE_THREADING_DEFAULTS.robot,
    cableModel: payload.cableThreadingCableModel ?? CABLE_THREADING_DEFAULTS.cableModel,
    difficulty: payload.cableThreadingDifficulty ?? CABLE_THREADING_DEFAULTS.difficulty,
    horizon: payload.cableThreadingHorizon ?? CABLE_THREADING_DEFAULTS.horizon,
    seed: payload.seed ?? CABLE_THREADING_DEFAULTS.seed,
    saveProcessVideo: payload.cableThreadingSaveProcessVideo ?? true,
    taskConfigId: taskConfigId ?? undefined,
  };

  if (outputFormat === 'lerobot') {
    return {
      ...base,
      outputFormat: 'lerobot',
      saveHdf5: false,
      lerobotTaskInstruction: LEROBOT_DEFAULTS.taskInstruction,
      lerobotRobot: payload.cableThreadingRobot ?? LEROBOT_DEFAULTS.robot,
      lerobotFps: LEROBOT_DEFAULTS.fps,
    };
  }

  if (outputFormat === 'hdf5') {
    return {
      ...base,
      outputFormat: 'hdf5',
      saveHdf5: true,
    };
  }

  return {
    ...base,
    outputFormat: 'npz',
    saveHdf5: false,
  };
}

export function isCableThreadingDataFormatDisabled(dataFormat: GenerateDataPayload['dataFormat']): boolean {
  return dataFormat === 'mcap' || dataFormat === 'ROS Bag';
}
