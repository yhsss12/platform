import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

function normalizeModelTypeListResponse(raw) {
  const modelTypes = Array.isArray(raw?.modelTypes)
    ? raw.modelTypes
    : Array.isArray(raw?.items)
      ? raw.items
      : [];
  const total =
    typeof raw?.total === 'number' && Number.isFinite(raw.total)
      ? raw.total
      : modelTypes.length;
  return { modelTypes, total };
}

describe('normalizeModelTypeListResponse', () => {
  it('prefers modelTypes field', () => {
    const result = normalizeModelTypeListResponse({
      modelTypes: [{ modelTypeId: 'act' }],
      total: 1,
    });
    assert.equal(result.total, 1);
    assert.equal(result.modelTypes[0]?.modelTypeId, 'act');
  });

  it('falls back to legacy items field', () => {
    const result = normalizeModelTypeListResponse({
      items: [{ modelTypeId: 'pi0' }],
    });
    assert.equal(result.total, 1);
    assert.equal(result.modelTypes[0]?.modelTypeId, 'pi0');
  });

  it('returns empty list for null response', () => {
    assert.deepEqual(normalizeModelTypeListResponse(null), { modelTypes: [], total: 0 });
  });
});
