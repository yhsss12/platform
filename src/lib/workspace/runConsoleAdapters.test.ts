import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  getRunConsoleKindDisplayName,
  nutAssemblyArtifactStageLabel,
  resolveRunConsoleKind,
} from '@/lib/workspace/runConsoleAdapters';

describe('runConsoleAdapters', () => {
  describe('nutAssemblyArtifactStageLabel', () => {
    it('returns Chinese label for known stages', () => {
      assert.equal(nutAssemblyArtifactStageLabel('completed'), '已完成');
      assert.equal(nutAssemblyArtifactStageLabel('mimicgen_generate'), 'MimicGen 生成中');
    });

    it('returns original stage for unknown stages', () => {
      assert.equal(nutAssemblyArtifactStageLabel('custom_stage'), 'custom_stage');
    });

    it('returns em dash for empty stage', () => {
      assert.equal(nutAssemblyArtifactStageLabel(''), '—');
    });

    it('module imports without compile errors', () => {
      assert.equal(typeof nutAssemblyArtifactStageLabel, 'function');
      assert.equal(typeof resolveRunConsoleKind, 'function');
      assert.equal(typeof getRunConsoleKindDisplayName, 'function');
    });
  });
});
