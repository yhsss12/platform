import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  buildStackCubeHdf5PreviewNotice,
  buildStackCubeReplayModeNotice,
  resolveStackCubeReplayOutcome,
} from './isaaclabFrankaStackCubeReplayLogic';

const COMPLETED_MANIFEST = {
  jobStatus: 'completed',
  taskIdValidated: true,
  episodeTaskId: 'isaaclab_franka_stack_cube',
  datasetTaskId: 'isaaclab_franka_stack_cube',
  datasetHdf5Path: 'datasets/dataset.hdf5',
  episodeCount: 2,
  successfulEpisodes: 2,
};

describe('resolveStackCubeReplayOutcome', () => {
  it('running job does not surface asset validation failure', () => {
    const outcome = resolveStackCubeReplayOutcome({
      jobStatus: 'running',
      progress: 5,
      taskIdValidated: true,
      episodeTaskId: undefined,
      datasetTaskId: undefined,
    });
    assert.equal(outcome.kind, 'in_progress');
    assert.notEqual(outcome.message, '任务资产校验失败，无法播放视频');
    assert.match(outcome.message, /数据生成中|回放资产尚未就绪/);
  });

  it('running job with high progress suggests preview generation', () => {
    const outcome = resolveStackCubeReplayOutcome({
      jobStatus: 'running',
      progress: 50,
      taskIdValidated: true,
    });
    assert.equal(outcome.kind, 'in_progress');
    assert.equal(outcome.message, '正在生成回放预览，请稍后刷新。');
  });

  it('completed job without video reports HDF5 ready state', () => {
    const outcome = resolveStackCubeReplayOutcome({
      ...COMPLETED_MANIFEST,
      videoExists: false,
      videoStatus: 'pending',
    });
    assert.equal(outcome.kind, 'hdf5_ready_no_video');
    assert.equal(
      outcome.message,
      '该数据集已生成 HDF5 数据，但当前未生成视频回放资产。'
    );
    if (outcome.kind === 'hdf5_ready_no_video') {
      assert.equal(outcome.episodeCount, 2);
      assert.equal(outcome.successfulEpisodes, 2);
      assert.equal(outcome.datasetHdf5Path, 'datasets/dataset.hdf5');
    }
  });

  it('completed job with available video stays playable', () => {
    const outcome = resolveStackCubeReplayOutcome({
      ...COMPLETED_MANIFEST,
      videoExists: true,
      videoStatus: 'available',
    });
    assert.equal(outcome.kind, 'video_available');
  });

  it('completed job with partial open-loop preview stays playable', () => {
    const outcome = resolveStackCubeReplayOutcome({
      ...COMPLETED_MANIFEST,
      videoExists: true,
      videoStatus: 'partial',
    });
    assert.equal(outcome.kind, 'video_available');
  });

  it('buildStackCubeReplayModeNotice distinguishes replay modes', () => {
    assert.equal(
      buildStackCubeReplayModeNotice('state_based'),
      '当前视频为基于 HDF5 状态轨迹生成的严格回放。'
    );
    assert.equal(
      buildStackCubeReplayModeNotice('open_loop_preview'),
      '当前视频为 open-loop 预览，可能与 HDF5 状态轨迹存在偏差。'
    );
    assert.equal(buildStackCubeReplayModeNotice('unknown'), null);
  });

  it('buildStackCubeHdf5PreviewNotice includes dataset summary fields', () => {
    const outcome = resolveStackCubeReplayOutcome({
      ...COMPLETED_MANIFEST,
      videoExists: false,
      videoStatus: 'pending',
    });
    assert.equal(outcome.kind, 'hdf5_ready_no_video');
    if (outcome.kind !== 'hdf5_ready_no_video') return;
    const notice = buildStackCubeHdf5PreviewNotice(outcome, 'data_gen_20260622_111105_5a69');
    assert.match(notice, /dataset\.hdf5: datasets\/dataset\.hdf5/);
    assert.match(notice, /episode_count: 2/);
    assert.match(notice, /successfulEpisodes: 2/);
    assert.match(notice, /sourceJobId: data_gen_20260622_111105_5a69/);
  });
});
