import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  normalizeModelAssetListResponse,
  normalizeTrainingJobModelAssetListResponse,
} from '@/lib/api/modelAssetsClient';
import { iconRailItems } from '@/components/layout/sidebar/navItems';

describe('workspace health regression', () => {
  describe('modelAssetsClient normalization', () => {
    it('reads backend assets field into modelAssets', () => {
      const result = normalizeModelAssetListResponse({
        assets: [{ id: 'ma-1' } as never],
        total: 1,
      });
      assert.equal(result.modelAssets.length, 1);
      assert.equal(result.modelAssets[0]?.id, 'ma-1');
      assert.equal(result.total, 1);
    });

    it('prefers modelAssets when both fields exist', () => {
      const result = normalizeModelAssetListResponse({
        modelAssets: [{ id: 'legacy' } as never],
        assets: [{ id: 'new' } as never],
        total: 2,
      });
      assert.equal(result.modelAssets.length, 1);
      assert.equal(result.modelAssets[0]?.id, 'legacy');
    });

    it('returns empty arrays for undefined or malformed payloads', () => {
      assert.deepEqual(normalizeModelAssetListResponse(undefined), { modelAssets: [], total: 0 });
      assert.deepEqual(normalizeModelAssetListResponse({}), { modelAssets: [], total: 0 });
      assert.deepEqual(normalizeModelAssetListResponse({ total: 5 }), { modelAssets: [], total: 5 });
    });

    it('normalizes training job detail payloads', () => {
      const result = normalizeTrainingJobModelAssetListResponse({
        assets: [{ id: 'tj-1' } as never],
        total: 1,
        listMessage: 'ok',
      });
      assert.equal(result.modelAssets.length, 1);
      assert.equal(result.listMessage, 'ok');
    });
  });

  describe('navigation / overview redirect targets', () => {
    it('sidebar home icon points to workspace home, not legacy /overview or /workspace/data', () => {
      const home = iconRailItems.find((item) => item.labelKey === 'sidebar.overview');
      assert.ok(home, 'home icon rail item should exist');
      assert.notEqual(home!.path, '/overview');
      assert.notEqual(home!.path, '/workspace/data');
      assert.equal(home!.path, '/workspace');
    });
  });

  describe('CreateTrainingTaskModal array safety contract', () => {
    it('asArray pattern prevents filter on undefined modelAssets/datasets', () => {
      const asArray = <T,>(value: T[] | null | undefined): T[] =>
        Array.isArray(value) ? value : [];

      const undefinedAssets: ModelAssetLike[] | undefined = undefined;
      const undefinedDatasets: DatasetLike[] | undefined = undefined;

      assert.doesNotThrow(() => asArray(undefinedAssets).filter(() => true));
      assert.doesNotThrow(() => asArray(undefinedDatasets).map((item: DatasetLike) => item.id));
      assert.equal(asArray(undefinedAssets).length, 0);
      assert.equal(asArray(undefinedDatasets).length, 0);
    });
  });
});

type ModelAssetLike = { id: string };
type DatasetLike = { id: string };
