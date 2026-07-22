import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  DATASET_SOURCE_DATA_BUILD,
  DATASET_SOURCE_EXTERNAL_IMPORT,
  DATASET_SOURCE_SIMULATION,
  normalizeDatasetSource,
  resolveDatasetCountText,
  resolveDatasetSizeText,
  resolveDatasetSourceLabel,
  resolveDatasetValidTrajectoryText,
} from './datasetDisplay';

describe('normalizeDatasetSource', () => {
  it('maps legacy Chinese labels to platform categories', () => {
    assert.equal(normalizeDatasetSource('MuJoCo 生成'), DATASET_SOURCE_SIMULATION);
    assert.equal(normalizeDatasetSource('Isaac Lab 生成'), DATASET_SOURCE_SIMULATION);
    assert.equal(normalizeDatasetSource('Isaac Sim 生成'), DATASET_SOURCE_SIMULATION);
    assert.equal(normalizeDatasetSource('真实导入'), DATASET_SOURCE_EXTERNAL_IMPORT);
    assert.equal(normalizeDatasetSource('真实数据构建'), DATASET_SOURCE_DATA_BUILD);
  });

  it('classifies by sourceType and job id hints', () => {
    assert.equal(
      normalizeDatasetSource(null, {
        sourceType: 'simulation_generated',
        simulatorBackend: 'mujoco',
        sourceJobId: 'ct_gen_20260706_151708',
      }),
      DATASET_SOURCE_SIMULATION
    );
    assert.equal(
      normalizeDatasetSource(null, {
        sourceType: 'real_robot_imported',
        dataSourceLabel: '真实导入',
      }),
      DATASET_SOURCE_EXTERNAL_IMPORT
    );
    assert.equal(
      normalizeDatasetSource(null, {
        sourceType: 'real_robot_built',
        dataSourceLabel: '真实数据构建',
      }),
      DATASET_SOURCE_DATA_BUILD
    );
    assert.equal(
      normalizeDatasetSource(null, {
        sourceJobId: 'isaac_import_demo_001',
        sourceType: 'imported_demo',
      }),
      DATASET_SOURCE_EXTERNAL_IMPORT
    );
  });
});

describe('resolveDatasetSourceLabel', () => {
  it('never exposes simulator backend names in data source label', () => {
    assert.equal(
      resolveDatasetSourceLabel({
        sourceType: 'simulation_generated',
        simulatorBackend: 'mujoco',
        sourceJobId: 'ct_gen_123',
        dataSourceLabel: 'MuJoCo 生成',
      }),
      DATASET_SOURCE_SIMULATION
    );
    assert.equal(
      resolveDatasetSourceLabel({
        sourceType: 'simulation_generated',
        simulatorBackend: 'isaac_lab',
        sourceJobId: 'isaac_gen_123',
      }),
      DATASET_SOURCE_SIMULATION
    );
  });
});

describe('resolveDatasetValidTrajectoryText', () => {
  it('caps successful episodes at total episodes', () => {
    assert.equal(
      resolveDatasetValidTrajectoryText({
        successfulEpisodes: 4,
        totalEpisodes: 2,
      }),
      '2/2'
    );
  });

  it('caps successful episodes at episode count fallback', () => {
    assert.equal(
      resolveDatasetValidTrajectoryText({
        successfulEpisodes: 4,
        episodeCount: 2,
      }),
      '2/2'
    );
  });
});

describe('resolveDatasetCountText', () => {
  it('reads backend dataCount', () => {
    assert.equal(resolveDatasetCountText({ dataCount: 10 }), '10');
    assert.equal(resolveDatasetCountText({ dataCount: 1 }), '1');
  });

  it('shows dash when dataCount missing or zero', () => {
    assert.equal(resolveDatasetCountText({}), '—');
    assert.equal(resolveDatasetCountText({ dataCount: 0 }), '—');
    assert.equal(resolveDatasetCountText({ dataCount: null }), '—');
  });
});

describe('resolveDatasetSizeText', () => {
  it('formats fileSizeBytes from backend', () => {
    const label = resolveDatasetSizeText({ fileSizeBytes: 172_912_640 });
    assert.match(label, /^[\d.]+ MB$/);
  });

  it('returns dash when size missing or zero', () => {
    assert.equal(resolveDatasetSizeText({}), '—');
    assert.equal(resolveDatasetSizeText({ fileSizeBytes: 0 }), '—');
  });
});
