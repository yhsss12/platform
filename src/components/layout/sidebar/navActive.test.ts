import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  isExactActive,
  isSectionActive,
  isWorkspaceHomeActive,
} from '@/components/layout/sidebar/navActive';

describe('navActive', () => {
  describe('isWorkspaceHomeActive', () => {
    it('matches only workspace home paths', () => {
      assert.equal(isWorkspaceHomeActive('/workspace'), true);
      assert.equal(isWorkspaceHomeActive('/workspace/'), true);
      assert.equal(isWorkspaceHomeActive('/workspace/dashboard'), true);
      assert.equal(isWorkspaceHomeActive('/workspace/overview'), true);
      assert.equal(isWorkspaceHomeActive('/workspace/data'), false);
      assert.equal(isWorkspaceHomeActive('/workspace/datasets'), false);
      assert.equal(isWorkspaceHomeActive('/workspace/training'), false);
      assert.equal(isWorkspaceHomeActive('/workspace/evaluation'), false);
      assert.equal(isWorkspaceHomeActive('/workspace/resources'), false);
    });
  });

  describe('isSectionActive', () => {
    it('matches section root and nested paths', () => {
      assert.equal(isSectionActive('/workspace/data', '/workspace/data'), true);
      assert.equal(isSectionActive('/workspace/data/import', '/workspace/data'), true);
      assert.equal(isSectionActive('/workspace/training/jobs', '/workspace/training'), true);
      assert.equal(isSectionActive('/workspace', '/workspace/data'), false);
    });
  });

  describe('isExactActive', () => {
    it('normalizes trailing slashes', () => {
      assert.equal(isExactActive('/workspace/', ['/workspace']), true);
      assert.equal(isExactActive('/workspace/data/', ['/workspace/data']), true);
    });
  });
});
