import type { SimAsset, SimAssetTabFilter } from '@/types/simAsset';

/** Phase 0.5：仿真资产列表 mock（空列表，后续接 API） */
export const MOCK_SIM_ASSETS: SimAsset[] = [];

export function filterSimAssetsByTab(
  assets: SimAsset[],
  tab: SimAssetTabFilter
): SimAsset[] {
  if (tab === 'all') return assets;
  if (tab === 'reconstruction') {
    return assets.filter((item) => item.source === 'reconstructed');
  }
  if (tab === 'scene') {
    return assets.filter((item) => item.assetType === 'scene');
  }
  if (tab === 'object') {
    return assets.filter(
      (item) =>
        item.assetType === 'object' ||
        item.assetType === 'robot' ||
        item.assetType === 'fixture'
    );
  }
  return assets;
}
