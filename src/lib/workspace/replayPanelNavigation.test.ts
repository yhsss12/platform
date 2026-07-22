import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  REPLAY_DATA_CENTER_HREF,
  buildReplayPanelHref,
  resolveReplayPanelNavigationTarget,
} from '@/lib/workspace/replayPanelNavigation';

describe('replayPanelNavigation', () => {
  it('module exports are importable', () => {
    assert.equal(REPLAY_DATA_CENTER_HREF, '/workspace/data');
    assert.equal(typeof resolveReplayPanelNavigationTarget, 'function');
    assert.equal(typeof buildReplayPanelHref, 'function');
  });

  it('returns safe fallback for empty id', () => {
    const target = resolveReplayPanelNavigationTarget(null);
    assert.equal(target.href, REPLAY_DATA_CENTER_HREF);
    assert.equal(target.disabled, true);
    assert.ok(target.reason);
    assert.doesNotThrow(() => buildReplayPanelHref(undefined));
  });

  it('routes na_gen_ ids to nut assembly replay href', () => {
    const jobId = 'na_gen_20260703_141635_6967';
    const target = resolveReplayPanelNavigationTarget(jobId);
    assert.match(target.href, /\/workspace\/replay\?/);
    assert.match(target.href, /taskType=nut_assembly/);
    assert.match(target.href, new RegExp(`jobId=${encodeURIComponent(jobId)}`));
    assert.notEqual(target.disabled, true);
  });

  it('does not throw for unknown id', () => {
    assert.doesNotThrow(() => resolveReplayPanelNavigationTarget('unknown_job_xyz'));
    const target = resolveReplayPanelNavigationTarget('unknown_job_xyz');
    assert.equal(target.href, REPLAY_DATA_CENTER_HREF);
    assert.equal(target.disabled, true);
  });
});
