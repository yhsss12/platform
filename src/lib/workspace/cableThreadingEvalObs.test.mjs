import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

function modelAssetSupportsCableThreadingEvalObs(asset) {
  const framework = String(
    asset.framework ?? asset.trainingBackend ?? asset.backendType ?? asset.modelType ?? ''
  ).toLowerCase();
  const modelTypeId = String(asset.modelTypeId ?? '').toLowerCase();
  const baseAlgorithm = String(asset.baseAlgorithm ?? '').toLowerCase();
  const modelType = String(asset.modelType ?? '').toLowerCase();

  const supportsCableThreadingPolicy =
    framework.includes('robomimic') ||
    framework.includes('diffusion') ||
    framework.includes('diffusion_policy') ||
    framework === 'act' ||
    modelType === 'act' ||
    baseAlgorithm === 'act' ||
    modelTypeId === 'act' ||
    framework === 'pi0' ||
    modelType === 'pi0' ||
    baseAlgorithm === 'pi0' ||
    modelTypeId === 'pi0';
  if (!supportsCableThreadingPolicy) return false;
  const schema = asset.observationSchema?.trim();
  if (schema && schema !== 'cable_threading_robomimic_v1') return false;
  return true;
}

describe('modelAssetSupportsCableThreadingEvalObs', () => {
  it('accepts diffusion policy cable threading assets', () => {
    assert.equal(
      modelAssetSupportsCableThreadingEvalObs({
        framework: 'Diffusion Policy',
        modelType: 'diffusion_policy',
        trainingBackend: 'diffusion_policy',
      }),
      true
    );
  });

  it('accepts act assets', () => {
    assert.equal(
      modelAssetSupportsCableThreadingEvalObs({
        framework: 'ACT',
        modelType: 'act',
        trainingBackend: 'act',
      }),
      true
    );
  });

  it('accepts pi0 assets', () => {
    assert.equal(
      modelAssetSupportsCableThreadingEvalObs({
        framework: 'pi0',
        modelType: 'pi0',
        modelTypeId: 'pi0',
      }),
      true
    );
  });
});
