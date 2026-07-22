import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluationListEmptyMessage,
  normalizeEvaluationJobListResponse,
  resolveEvaluationListLoadState,
} from './evaluationJobs';

describe('normalizeEvaluationJobListResponse', () => {
  it('reads jobs and total', () => {
    const result = normalizeEvaluationJobListResponse({
      jobs: [{ evalJobId: 'ct_eval_20260630_133008_3c5f', status: 'completed' }],
      total: 1,
    });
    assert.equal(result.total, 1);
    assert.equal(result.jobs[0]?.evalJobId, 'ct_eval_20260630_133008_3c5f');
  });

  it('falls back to items/evaluations/evaluationJobs', () => {
    assert.equal(
      normalizeEvaluationJobListResponse({ items: [{ evalJobId: 'eval_a', status: 'completed' }], total: 1 })
        .jobs.length,
      1
    );
    assert.equal(
      normalizeEvaluationJobListResponse({
        evaluations: [{ evalJobId: 'eval_b', status: 'completed' }],
      }).total,
      1
    );
    assert.equal(
      normalizeEvaluationJobListResponse({
        evaluationJobs: [{ evalJobId: 'eval_c', status: 'completed' }],
        total: 3,
      }).total,
      3
    );
  });

  it('handles null/undefined safely', () => {
    assert.deepEqual(normalizeEvaluationJobListResponse(null), { jobs: [], total: 0 });
    assert.deepEqual(normalizeEvaluationJobListResponse(undefined), { jobs: [], total: 0 });
  });
});

describe('resolveEvaluationListLoadState', () => {
  it('distinguishes loading, error, empty, success', () => {
    assert.equal(
      resolveEvaluationListLoadState({ isPending: true, isError: false, hasResponse: false, total: 0 }),
      'loading'
    );
    assert.equal(
      resolveEvaluationListLoadState({ isPending: false, isError: true, hasResponse: false, total: 0 }),
      'error'
    );
    assert.equal(
      resolveEvaluationListLoadState({ isPending: false, isError: false, hasResponse: true, total: 0 }),
      'empty'
    );
    assert.equal(
      resolveEvaluationListLoadState({ isPending: false, isError: false, hasResponse: true, total: 109 }),
      'success'
    );
  });
});

describe('evaluationListEmptyMessage', () => {
  it('shows error text instead of empty list copy', () => {
    const msg = evaluationListEmptyMessage('error', 'HTTP 500');
    assert.match(msg, /评测任务加载失败/);
    assert.match(msg, /HTTP 500/);
  });

  it('shows loading and empty copy', () => {
    assert.equal(evaluationListEmptyMessage('loading'), '加载评测任务…');
    assert.match(evaluationListEmptyMessage('empty'), /暂无评测任务/);
  });
});

describe('evaluationListItemToRow pi0 mapping', () => {
  it('keeps successRate=0.0 and modelType pi0', async () => {
    const { evaluationListItemToRow } = await import('./workspaceJobMapper');
    const row = evaluationListItemToRow({
      evalJobId: 'ct_eval_20260630_133008_3c5f',
      status: 'completed',
      taskName: 'pi0 Platform Eval Smoke',
      evaluationMode: 'trained_model_evaluation',
      taskType: 'cable_threading',
      metrics: {
        modelType: 'pi0',
        evalExecutor: 'joint_position',
        successRate: 0.0,
        modelAssetId: 'model__123947_ebd2_final',
      },
      successStats: {
        successEpisodes: 0,
        totalEpisodes: 1,
        display: '0/1',
        available: true,
      },
    });
    assert.equal(row.successRate, 0);
    assert.equal(row.modelType, 'pi0');
    assert.equal(row.successStats?.display, '0/1');
    assert.equal(row.status, '已完成');
  });
});
