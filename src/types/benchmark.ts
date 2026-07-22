/**
 * Workspace Benchmark 平台统一领域对象类型（Phase 1）。
 * 与现有 mock row / workspace job 类型并存，逐步迁移。
 */

export type TaskTemplateSourceType = 'standard_template' | 'real_data_reconstruction';

export type SimulatorType = 'mujoco' | 'isaac' | 'gazebo' | 'unknown';

export type ExpertPolicyType = 'scripted' | 'teleop' | 'rl' | 'bc' | 'external';

export type DatasetSourceType =
  | 'simulation_generated'
  | 'real_robot_imported'
  | 'real_data_constructed'
  | 'imported_demo'
  | 'imported'
  | 'converted'
  | 'mixed'
  | string;

export type DatasetFormat =
  | 'hdf5'
  | 'npz'
  | 'zarr'
  | 'manifest'
  | 'lerobot'
  | 'robomimic_hdf5'
  | 'unknown'
  | string;

export type BenchmarkEntityStatus =
  | 'draft'
  | 'available'
  | 'running'
  | 'completed'
  | 'failed'
  | 'deprecated'
  | 'unknown';

export interface TaskTemplate {
  id: string;
  name: string;
  description: string;
  sourceType: TaskTemplateSourceType;
  taskFamily: string;
  taskType: string;
  simulatorType: SimulatorType;
  supportedRobotTypes: string[];
  supportedPolicyTypes: string[];
  defaultSceneId: string | null;
  defaultMetricProfileId: string | null;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
  updatedAt: string;
}

export interface TaskBuildConfig {
  id: string;
  name: string;
  taskTemplateId: string;
  taskFamily: string;
  simulatorType: SimulatorType | string;
  registryTaskConfigId?: string | null;
  linkedDatasetId?: string | null;
  linkedModelAssetId?: string | null;
  supportedEvaluationModes?: string[];
  createdAt: string;
}

export interface SimulationScene {
  id: string;
  name: string;
  taskTemplateId: string;
  simulatorType: SimulatorType;
  sceneFile: string;
  assetIds: string[];
  cameraConfig: Record<string, unknown>;
  physicsConfig: Record<string, unknown>;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
  updatedAt: string;
}

export interface ExpertPolicy {
  id: string;
  name: string;
  taskTemplateId: string;
  policyType: ExpertPolicyType;
  entrypoint: string;
  configPath: string | null;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
  updatedAt: string;
}

export interface Dataset {
  id: string;
  name: string;
  sourceJobId: string;
  sourceTaskTemplateId: string | null;
  sourceType: DatasetSourceType;
  manifestPath: string;
  episodeCount: number;
  storagePath: string;
  format: DatasetFormat;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
  updatedAt: string;
  /** Workspace 扩展字段（API / 列表展示） */
  displayName?: string | null;
  taskDisplayName?: string | null;
  taskType?: string | null;
  taskTemplateId?: string | null;
  simulatorBackend?: string | null;
  dataSourceLabel?: string | null;
  generationMode?: string | null;
  datasetFormat?: string | null;
  sourceFormat?: string | null;
  dataCount?: number | null;
  successfulEpisodes?: number | null;
  totalEpisodes?: number | null;
  validTrajectories?: number | null;
  generationRounds?: number | null;
  fileSizeBytes?: number | null;
  needsBuild?: boolean;
  episodeParsed?: boolean;
  datasetFile?: string | null;
  lerobotPath?: string | null;
  pi0Ready?: boolean;
  successRate?: number | null;
  enhancementMode?: string | null;
  trainable?: boolean;
  physicsEnhancementEnabled?: boolean;
  policyMode?: string | null;
  sourceDemoOrigin?: string | null;
  mimicgenGeneratedDemos?: number | null;
  rawDemoCount?: number | null;
  repairedDemoCount?: number | null;
  finalDemoCount?: number | null;
  demoCount?: number | null;
  pinnModelId?: string | null;
  pinnRepairValidationRate?: number | null;
  sourceDemoPath?: string | null;
  sourceDemoHash?: string | null;
  envName?: string | null;
  episodesRequested?: number | null;
  episodesGenerated?: number | null;
  datagenFailedTrials?: number | null;
  hasDatagenInfo?: boolean;
  objectPoseKeys?: string[] | null;
  totalSteps?: number | null;
  hasEpisodeMetadata?: boolean;
  successEpisodes?: number | null;
  hasObjectPoses?: boolean;
  validForTrainingEpisodes?: number | null;
  trainingFilterMode?: string | null;
  defaultTrainingFilterMode?: string | null;
  filteredDemoCount?: number | null;
  trainingBuildReady?: boolean;
  hasStageStatistics?: boolean;
  graspSuccessEpisodes?: number | null;
  liftSuccessEpisodes?: number | null;
  insertionSuccessEpisodes?: number | null;
  averageGraspAttempts?: number | null;
  actionSchema?: string | null;
  observationSchema?: string | null;
  controllerSchema?: string | null;
  trainedActionMode?: string | null;
  evalExecutor?: string | null;
  preferredPolicySchemaId?: string | null;
  imageKeys?: string[] | null;
  lowDimKeys?: string[] | null;
  actionDim?: number | null;
  robotType?: string | null;
  lerobotTaskInstruction?: string | null;
  lerobotStateDim?: number | null;
  lerobotActionDim?: number | null;
  builtDatasetPath?: string | null;
  availableFormats?: string[] | null;
  mainFormats?: string[] | null;
  directTrainable?: boolean;
  replayAvailable?: boolean;
  replayBackend?: string | null;
  needsMapping?: boolean;
  ilExportFailureReason?: string | null;
  dataOrganizationFormat?: string | null;
}

export interface ModelAsset {
  id: string;
  name: string;
  sourceTrainingJobId: string;
  sourceDatasetId: string | null;
  taskTemplateId: string | null;
  modelType: string;
  framework: string;
  checkpointPath: string;
  manifestPath: string;
  version: string;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
  updatedAt: string;
  /** 资源中心 / 训练预训练权重扩展字段 */
  displayName?: string | null;
  structureConfig?: Record<string, unknown> | null;
  resolvedModelParams?: Record<string, unknown> | null;
  checkpointMetricValue?: number | null;
  checkpointKind?: 'final' | 'best' | string | null;
  assetSource?: 'imported' | 'training' | string | null;
  modelTypeId?: string | null;
  trainingBackend?: string | null;
  backendType?: string | null;
  baseAlgorithm?: string | null;
  modelTypeName?: string | null;
  adapterId?: string | null;
  importMetadata?: Record<string, unknown> | null;
  validationResult?: Record<string, unknown> | null;
  datasetDisplayName?: string | null;
  checkpointMetricName?: string | null;
  checkpointEpoch?: number | null;
  taskType?: string | null;
  isPlaceholder?: boolean;
  canEvaluate?: boolean;
  fileExists?: boolean;
  displayStatus?: string | null;
  actionKey?: string | null;
  gripperActionKey?: string | null;
  actionDim?: number | null;
  trainedActionMode?: string | null;
  actionMode?: string | null;
  evalExecutor?: string | null;
  controllerType?: string | null;
  openpiEnvironment?: Record<string, unknown> | null;
}

export interface EvaluationJob {
  id: string;
  name: string;
  taskTemplateId: string | null;
  sceneId: string | null;
  datasetId: string | null;
  modelAssetId: string | null;
  evaluationMode: string;
  metricProfileId: string | null;
  status: BenchmarkEntityStatus | string;
  resultPath: string | null;
  replayRecordIds: string[];
  reportId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ReplayRecord {
  id: string;
  evaluationJobId: string;
  episodeId: string;
  videoPath: string;
  timelinePath: string | null;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
}

export interface Report {
  id: string;
  evaluationJobId: string;
  title: string;
  summary: string;
  reportPath: string;
  createdAt: string;
}

export interface RealDataImport {
  id: string;
  name: string;
  sourceFilePath: string;
  dataFormat: string;
  parsedSignals: string[];
  linkedTaskTemplateId: string | null;
  status: BenchmarkEntityStatus | string;
  createdAt: string;
  updatedAt: string;
}

/** 真机数据构建流程本地草稿（未写入 Dataset registry） */
export interface RealDataImportDraft {
  id: string;
  name: string;
  sourceFileName: string;
  dataFormat: string;
  parsedSignals: string[];
  linkedTaskTemplateId: string | null;
  sceneConfig?: Record<string, unknown>;
  perturbationConfig?: Record<string, unknown>;
  status: string;
  createdAt: string;
  updatedAt: string;
}
