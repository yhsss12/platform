import type { PhysicsProxyMode } from '@/lib/mock/physicsProxiesMock';
import type { NutAssemblyGenerationMode } from '@/lib/api/nutAssemblyClient';
import type { AugmentationAlgorithm, GenerationPath } from '@/lib/workspace/generateDataTypes';

export type GenerateDataPurpose = '训练数据' | '评测数据' | '训练与评测';

export type IsaacStackingGenerationMode =
  | 'mimic_auto'
  | 'expert'
  | 'mimicgen'
  | 'expert_policy'
  | 'scripted_expert';

export interface GenerateDataPayload {
  template: string;
  simBackend: string;
  taskConfig: string;
  episodes: number;
  seed: number;
  dataPurpose: GenerateDataPurpose;
  saveVideo: boolean;
  saveTrajectory: boolean;
  saveStateLog: boolean;
  saveStructuredData: boolean;
  saveImageData: boolean;
  saveProcessVideo: boolean;
  outputName: string;
  /** 保存任务时为 pending，启动时为 generating/completed */
  launch: 'save' | 'start';
  physicsProxyMode: PhysicsProxyMode;
  physicsProxyModel: string | null;
  physicsProxyErrorThreshold: number;
  physicsProxyReviewRatio: number;
  cableThreadingRobot?: string;
  cableThreadingCableModel?: string;
  cableThreadingDifficulty?: string;
  cableThreadingHorizon?: number;
  cableThreadingSaveHdf5?: boolean;
  cableThreadingLerobot?: boolean;
  cableThreadingSaveProcessVideo?: boolean;
  dualArmMaxCables?: number;
  dualArmStretchMode?: string;
  dualArmReleaseMode?: string;
  dualArmRecord?: boolean;
  dualArmHeadless?: boolean;
  isaacGenerationMode?: IsaacStackingGenerationMode;
  isaacSeedDatasetId?: string;
  isaacSeedDatasetFile?: string;
  isaacNumDemos?: number;
  isaacHeadless?: boolean;
  isaaclabHeadless?: boolean;
  isaacEnableCameras?: boolean;
  isaacParallelNumEnvs?: number;
  isaacsimHeadless?: boolean;
  /** 平台级生成路径 */
  generationPath?: GenerationPath;
  generationCount?: number;
  maxSteps?: number;
  expertPolicy?: string;
  successFilter?: boolean;
  keepFailedTrajectories?: boolean;
  sourceDemoDatasetId?: string;
  augmentationAlgorithm?: AugmentationAlgorithm;
  targetCount?: number;
  seedGenerationCount?: number;
  seedKeepCount?: number;
  autoSelectBestSeeds?: boolean;
  replayValidation?: boolean;
  useExistingSeedDataset?: boolean;
  enablePinnRepair?: boolean;
  outputFormat?: string;
  nutAssemblyEnvName?: string;
  nutAssemblyHorizon?: number;
  nutAssemblyRenderVideo?: boolean;
  /** @deprecated 使用 generationPath 映射 */
  nutAssemblyGenerationMode?: NutAssemblyGenerationMode;
  nutAssemblyPinnRepairEnabled?: boolean;
  nutAssemblySourceDemoSelection?: string;
  nutAssemblySourceDemoPath?: string;
  nutAssemblyRobot?: string;
  dataFormat?: string;
}
