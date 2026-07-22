import { describe, expect, it } from 'vitest';
import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import {
  buildCableThreadingGenerateRequest,
  isCableThreadingDataFormatDisabled,
  resolveCableThreadingOutputFormat,
} from '@/lib/workspace/cableThreadingGeneratePayload';

const basePayload: GenerateDataPayload = {
  template: 'cable_threading_single_arm',
  simBackend: 'MuJoCo',
  taskConfig: 'default',
  episodes: 1,
  saveVideo: false,
  saveTrajectory: true,
  saveStateLog: true,
  saveStructuredData: true,
  saveProcessVideo: true,
  dataFormat: 'hdf5',
  outputName: 'test',
  launch: 'start',
  physicsProxyMode: 'off',
  physicsProxyModel: null,
  physicsProxyErrorThreshold: 0.1,
  physicsProxyReviewRatio: 0.1,
  cableThreadingRobot: 'Panda',
};

describe('cableThreadingGeneratePayload', () => {
  it('maps LeRobot to outputFormat=lerobot without npz fallback', () => {
    const req = buildCableThreadingGenerateRequest({
      ...basePayload,
      dataFormat: 'lerobot',
    });
    expect(req.outputFormat).toBe('lerobot');
    expect(req.saveHdf5).toBe(false);
    expect(req.lerobotTaskInstruction).toBe('thread the cable through the pole');
    expect(req.lerobotRobot).toBe('Panda');
    expect(req.lerobotFps).toBe(20);
  });

  it('keeps HDF5 mapping unchanged', () => {
    const req = buildCableThreadingGenerateRequest(basePayload);
    expect(req.outputFormat).toBe('hdf5');
    expect(req.saveHdf5).toBe(true);
  });

  it('rejects unsupported mcap format', () => {
    expect(() =>
      resolveCableThreadingOutputFormat('mcap')
    ).toThrow(/尚未支持/);
  });

  it('flags disabled formats', () => {
    expect(isCableThreadingDataFormatDisabled('mcap')).toBe(true);
    expect(isCableThreadingDataFormatDisabled('ROS Bag')).toBe(true);
    expect(isCableThreadingDataFormatDisabled('lerobot')).toBe(false);
  });
});
