import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import type { TrainingDatasetOption } from '@/lib/mock/workspaceTrainingMock';
import {
  datasetPickerFilterWithColumn,
  filterTrainingDatasetOptions,
  filterTrainingDatasets,
  formatDatasetDate,
  formatDatasetDemoCount,
  formatDatasetFormatLabel,
  formatDatasetRobotLabel,
  formatDatasetSchemaLabel,
  formatDatasetStatusLabel,
  formatSelectedTrainingDatasetsTriggerLabel,
  getDatasetFilterValues,
  getFilteredDatasetSummary,
  matchesTrajectoryRange,
  paginateTrainingDatasetOptions,
  resetDatasetPickerFilter,
  toggleTrainingDatasetDraftSelection,
  TRAINING_DATASET_PICKER_EMPTY_TEXT,
  TRAINING_DATASET_PICKER_FILTERED_EMPTY_TEXT,
  validateTrainingDatasetSelection,
  type TrainingDatasetPickerMeta,
} from './trainingDatasetPicker';

const metaById: Record<string, TrainingDatasetPickerMeta> = {
  ds_joint_1: {
    actionSchema: 'Joint-Space',
    robotType: 'Panda',
    status: 'available',
    createdAt: '2026-06-26T10:00:00Z',
  },
  ds_eef_1: {
    actionSchema: 'EEF-OSC',
    robotType: 'Panda',
    status: 'ready',
    createdAt: '2026-06-24T10:00:00Z',
  },
  ds_mid_1: {
    actionSchema: 'Joint-Space',
    robotType: 'Panda',
    status: 'available',
    createdAt: '2026-06-23T10:00:00Z',
  },
};

const sampleOptions: TrainingDatasetOption[] = [
  {
    id: 'ds_joint_1',
    datasetName: '线缆穿杆数据_20260624_joint_space_replay_full',
    taskName: '线缆穿杆',
    taskType: 'cable_threading',
    modelFormat: 'HDF5',
    dataFormat: 'HDF5',
    sampleCount: 81,
    sourceJobId: 'ct_gen_001',
  },
  {
    id: 'ds_eef_1',
    datasetName: 'eef_osc_dataset',
    taskName: '线缆穿杆',
    taskType: 'cable_threading',
    modelFormat: 'HDF5',
    dataFormat: 'HDF5',
    sampleCount: 4,
    sourceJobId: 'ct_gen_002',
  },
  {
    id: 'ds_mid_1',
    datasetName: 'joint_mid_dataset',
    taskName: '线缆穿杆',
    taskType: 'cable_threading',
    modelFormat: 'HDF5',
    dataFormat: 'HDF5',
    sampleCount: 15,
    sourceJobId: 'ct_gen_003',
  },
];

describe('trainingDatasetPicker', () => {
  it('uses unified empty state copy', () => {
    assert.equal(TRAINING_DATASET_PICKER_EMPTY_TEXT, '暂无可用的训练数据集');
    assert.equal(TRAINING_DATASET_PICKER_FILTERED_EMPTY_TEXT, '暂无符合条件的训练数据集');
  });

  it('formats demo count labels', () => {
    assert.equal(formatDatasetDemoCount(1), '1 demo');
    assert.equal(formatDatasetDemoCount(4), '4 demos');
    assert.equal(formatDatasetDemoCount(81), '81 demos');
  });

  it('does not filter when column is none', () => {
    const filtered = filterTrainingDatasets(sampleOptions, '', resetDatasetPickerFilter(), metaById);
    assert.equal(filtered.length, 3);
  });

  it('does not filter when value is all', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'schema', value: 'all' },
      metaById
    );
    assert.equal(filtered.length, 3);
  });

  it('filters by schema column and value', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'schema', value: 'Joint-Space' },
      metaById
    );
    assert.deepEqual(
      filtered.map((item) => item.id),
      ['ds_joint_1', 'ds_mid_1']
    );
  });

  it('filters by robot column and value', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'robot', value: 'Panda' },
      metaById
    );
    assert.equal(filtered.length, 3);
  });

  it('filters by format column and value', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'format', value: 'HDF5' },
      metaById
    );
    assert.equal(filtered.length, 3);
  });

  it('filters by status column and value', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'status', value: 'available' },
      metaById
    );
    assert.deepEqual(
      filtered.map((item) => item.id),
      ['ds_joint_1', 'ds_mid_1']
    );
  });

  it('filters by trajectory count column and value', () => {
    assert.equal(matchesTrajectoryRange(4, '1-5'), true);
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'trajectoryCount', value: '1-5' },
      metaById
    );
    assert.deepEqual(filtered.map((item) => item.id), ['ds_eef_1']);
  });

  it('combines search and single-column filter', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      'joint',
      { column: 'schema', value: 'Joint-Space' },
      metaById
    );
    assert.deepEqual(
      filtered.map((item) => item.id),
      ['ds_joint_1', 'ds_mid_1']
    );
  });

  it('resets filter value when column changes', () => {
    const next = datasetPickerFilterWithColumn('robot');
    assert.deepEqual(next, { column: 'robot', value: 'all' });
  });

  it('resets filter to show all datasets', () => {
    const filtered = filterTrainingDatasets(sampleOptions, '', resetDatasetPickerFilter(), metaById);
    assert.equal(filtered.length, 3);
  });

  it('builds dynamic filter values for selected column', () => {
    const schemaValues = getDatasetFilterValues(sampleOptions, 'schema', metaById);
    assert.deepEqual(
      schemaValues.map((item) => item.value),
      ['all', 'EEF-OSC', 'Joint-Space']
    );
    const trajectoryValues = getDatasetFilterValues(sampleOptions, 'trajectoryCount', metaById);
    assert.deepEqual(trajectoryValues.map((item) => item.value), ['all', '1-5', '6-20', '21-100', '100+']);
    const noneValues = getDatasetFilterValues(sampleOptions, 'none', metaById);
    assert.deepEqual(noneValues, [{ value: 'all', label: '全部' }]);
  });

  it('updates pagination totals after filtering', () => {
    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'schema', value: 'Joint-Space' },
      metaById
    );
    const page1 = paginateTrainingDatasetOptions(filtered, 1, 1);
    assert.equal(page1.totalPages, 2);
    assert.equal(filtered.length, 2);
  });

  it('shows filtered summary when filters reduce results', () => {
    assert.equal(getFilteredDatasetSummary(3, 2), '已筛选 2 / 共 3 个数据集');
  });

  it('does not clear draft selection when filtering hidden datasets', () => {
    const parentIds = ['ds_joint_1'];
    const draftResult = toggleTrainingDatasetDraftSelection([], 'ds_eef_1', sampleOptions, false);
    assert.deepEqual(draftResult.nextIds, ['ds_eef_1']);
    assert.deepEqual(parentIds, ['ds_joint_1']);

    const filtered = filterTrainingDatasets(
      sampleOptions,
      '',
      { column: 'schema', value: 'EEF-OSC' },
      metaById
    );
    assert.equal(filtered.some((item) => item.id === 'ds_joint_1'), false);
    assert.deepEqual(parentIds, ['ds_joint_1']);
  });

  it('filters datasets by keyword only through legacy helper', () => {
    const filtered = filterTrainingDatasetOptions(sampleOptions, 'joint_space');
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0]?.id, 'ds_joint_1');
  });

  it('formats labels and trigger text', () => {
    assert.equal(formatDatasetSchemaLabel(sampleOptions[0], metaById.ds_joint_1), 'Joint-Space');
    assert.equal(formatDatasetRobotLabel(sampleOptions[0], metaById.ds_joint_1), 'Panda');
    assert.equal(formatDatasetFormatLabel(sampleOptions[0]), 'HDF5');
    assert.equal(formatDatasetStatusLabel('available'), 'available');
    assert.equal(formatDatasetDate('2026-06-26T10:00:00Z'), '2026/06/26');
    assert.equal(
      formatSelectedTrainingDatasetsTriggerLabel(['ds_eef_1'], sampleOptions),
      '线缆穿杆 · 4 demo'
    );
  });

  it('validates compatible joint-space selection for ACT/DP shared datasets', () => {
    const validation = validateTrainingDatasetSelection(['ds_joint_1'], [sampleOptions[0]]);
    assert.equal(validation.ok, true);
  });
});
