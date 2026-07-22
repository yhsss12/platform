import type {
  SimAssetSource,
  SimAssetStatus,
  SimAssetTargetEngine,
  SimAssetType,
} from '@/types/simAsset';

export const SIM_ASSET_TYPE_LABELS: Record<SimAssetType, string> = {
  scene: '场景资产',
  object: '操作对象',
  robot: '机器人资产',
  fixture: '工装/夹具',
};

export const SIM_ASSET_SOURCE_LABELS: Record<SimAssetSource, string> = {
  imported: '导入',
  reconstructed: '图像重建',
  generated: '生成',
};

export const SIM_ASSET_TARGET_LABELS: Record<SimAssetTargetEngine, string> = {
  mujoco: 'MuJoCo',
  isaac: 'Isaac Sim',
  generic: '通用',
};

export const SIM_ASSET_STATUS_LABELS: Record<SimAssetStatus, string> = {
  draft: '草稿',
  processing: '处理中',
  ready: '可用',
  failed: '失败',
};

export function simAssetTypeLabel(type: SimAssetType): string {
  return SIM_ASSET_TYPE_LABELS[type];
}

export function simAssetSourceLabel(source: SimAssetSource): string {
  return SIM_ASSET_SOURCE_LABELS[source];
}

export function simAssetTargetLabel(target: SimAssetTargetEngine): string {
  return SIM_ASSET_TARGET_LABELS[target];
}

export function simAssetStatusLabel(status: SimAssetStatus): string {
  return SIM_ASSET_STATUS_LABELS[status];
}
