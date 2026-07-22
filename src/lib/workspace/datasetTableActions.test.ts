import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import type { Dataset } from '@/types/benchmark';
import {
  datasetHasReplayResources,
  isLegacyIsaacLabRegistryDataset,
  resolveUnifiedDatasetReplayHref,
} from './datasetTableActions';
import { isIsaacSimFrankaPickPlaceDataset } from './isaacsimFrankaPickPlace';
import { isIsaacLabFrankaStackCubeDataset } from './isaaclabFrankaStackCube';

function baseDataset(overrides: Partial<Dataset> = {}): Dataset {
  return {
    id: 'dataset_test',
    name: 'test',
    sourceJobId: 'data_gen_20260622_104548_a331',
    manifestPath: '/tmp/manifest.json',
    storagePath: '/tmp',
    format: 'hdf5',
    status: 'available',
    createdAt: '2026-06-22T10:45:48+00:00',
    updatedAt: '2026-06-22T10:45:48+00:00',
    ...overrides,
  };
}

describe('Franka Stack Cube dataset table actions', () => {
  it('does not classify stack cube as Isaac Sim pick place', () => {
    const dataset = baseDataset({
      taskType: 'isaaclab_franka_stack_cube',
      simulatorBackend: 'isaac_lab',
    });
    assert.equal(isIsaacSimFrankaPickPlaceDataset(dataset), false);
    assert.equal(isIsaacLabFrankaStackCubeDataset(dataset), true);
  });

  it('does not classify stack cube as legacy Isaac Lab registry dataset', () => {
    const dataset = baseDataset({
      taskType: 'isaaclab_franka_stack_cube',
      simulatorBackend: 'isaac_lab',
    });
    assert.equal(isLegacyIsaacLabRegistryDataset(dataset), false);
  });

  it('enables replay href only when replay resources exist', () => {
    const withVideo = baseDataset({
      taskType: 'isaaclab_franka_stack_cube',
      replayAvailable: true,
    });
    const withoutVideo = baseDataset({
      taskType: 'isaaclab_franka_stack_cube',
      replayAvailable: false,
    });

    assert.match(resolveUnifiedDatasetReplayHref(withVideo) ?? '', /isaaclab_franka_stack_cube/);
    assert.equal(resolveUnifiedDatasetReplayHref(withoutVideo), null);
    assert.equal(datasetHasReplayResources(withVideo), true);
    assert.equal(datasetHasReplayResources(withoutVideo), false);
  });

  it('still classifies Isaac Sim pick place by task type', () => {
    const dataset = baseDataset({
      taskType: 'isaacsim_franka_pick_place',
      sourceJobId: 'data_gen_20260622_104548_pick',
      simulatorBackend: 'isaacsim',
      replayAvailable: true,
    });
    assert.equal(isIsaacSimFrankaPickPlaceDataset(dataset), true);
    assert.equal(isIsaacLabFrankaStackCubeDataset(dataset), false);
  });
});
