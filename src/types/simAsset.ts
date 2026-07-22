export type SimAssetType = 'scene' | 'object' | 'robot' | 'fixture';

export type SimAssetSource = 'imported' | 'reconstructed' | 'generated';

export type SimAssetTargetEngine = 'mujoco' | 'isaac' | 'generic';

export type SimAssetStatus = 'draft' | 'processing' | 'ready' | 'failed';

export interface SimAsset {
  id: string;
  name: string;
  assetType: SimAssetType;
  source: SimAssetSource;
  targetEngine: SimAssetTargetEngine;
  primaryFormat: string;
  status: SimAssetStatus;
  updatedAt: string;
  thumbnailUrl?: string;
  description?: string;
}

export interface ReconstructionJobDraft {
  name: string;
  inputImage?: File | null;
  prompt: string;
  positiveBoxes: number[][];
  negativeBoxes: number[][];
  selectedMaskIndex?: number;
  targetEngine: SimAssetTargetEngine;
  assetType: SimAssetType;
}

export type SimAssetTabFilter = 'all' | 'scene' | 'object' | 'reconstruction';

export type SimAssetExportFormat =
  | 'ply'
  | 'glb'
  | 'obj'
  | 'mjcf'
  | 'usd'
  | 'urdf';
