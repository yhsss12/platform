import { describe, expect, it } from 'vitest';
import type { Dataset } from '@/types/benchmark';
import {
  filterBuildSourceDatasets,
  filterBuildSourceDatasetsByKeyword,
  isBuildSourceImportedHdf5Dataset,
} from '@/lib/workspace/buildSourceDatasetPicker';

function makeDataset(overrides: Partial<Dataset> = {}): Dataset {
  return {
    id: 'ds_import_test',
    name: '测试导入',
    sourceJobId: 'import_ds_import_test',
    sourceTaskTemplateId: null,
    sourceType: 'real_robot_imported',
    manifestPath: '',
    episodeCount: 1,
    storagePath: '',
    format: 'hdf5',
    status: 'needs_mapping',
    createdAt: '2026-06-26T00:00:00+00:00',
    updatedAt: '2026-06-26T00:00:00+00:00',
    dataSourceLabel: '真实导入',
    ...overrides,
  };
}

describe('buildSourceDatasetPicker', () => {
  it('includes real imported hdf5 even when not directly trainable', () => {
    expect(isBuildSourceImportedHdf5Dataset(makeDataset({ status: 'needs_build' }))).toBe(true);
    expect(isBuildSourceImportedHdf5Dataset(makeDataset({ status: 'needs_mapping' }))).toBe(true);
  });

  it('excludes simulation, built, failed and non-hdf5 datasets', () => {
    expect(isBuildSourceImportedHdf5Dataset(makeDataset({ status: 'failed' }))).toBe(false);
    expect(
      isBuildSourceImportedHdf5Dataset(
        makeDataset({ id: 'ds_built_x', sourceJobId: 'built_ds_built_x', sourceType: 'real_robot_built' })
      )
    ).toBe(false);
    expect(
      isBuildSourceImportedHdf5Dataset(
        makeDataset({ sourceJobId: 'ct_gen_123', sourceType: 'simulation_generated', simulatorBackend: 'mujoco' })
      )
    ).toBe(false);
    expect(isBuildSourceImportedHdf5Dataset(makeDataset({ format: 'manifest' }))).toBe(false);
    expect(isBuildSourceImportedHdf5Dataset(makeDataset({ format: 'hdf5', datasetFormat: 'lerobot' }))).toBe(false);
  });

  it('filters datasets by keyword', () => {
    const rows = filterBuildSourceDatasets([
      makeDataset({ id: 'ds_import_a', name: 'Alpha 导入' }),
      makeDataset({ id: 'ds_import_b', name: 'Beta 导入', taskType: 'custom', taskDisplayName: '自定义' }),
    ]);
    const filtered = filterBuildSourceDatasetsByKeyword(rows, 'alpha');
    expect(filtered).toHaveLength(1);
    expect(filtered[0]?.id).toBe('ds_import_a');
  });
});
