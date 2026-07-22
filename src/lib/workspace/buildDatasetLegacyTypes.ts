/** 仿真数据「登记索引」mock 流程类型（与 HDF5 导入构建弹窗无关）。 */

export type BuildDatasetPurpose = '训练数据集' | '评测数据集' | '训练与评测';
export type DownstreamModelType =
  | 'ACT'
  | 'DT'
  | 'Diffusion Policy'
  | 'Robomimic'
  | 'LeRobot'
  | '自定义模型';
export type DataOrganizationFormat = 'HDF5' | 'NPZ' | 'LeRobot' | 'HDF5 + NPZ';
export type UsageScope = '全部成功轨迹' | '全部轨迹' | '自定义数量';
export type SplitMode = 'train_val_80_20' | 'none' | 'custom';

export interface BuildDatasetPayload {
  rawDataId: string;
  rawDataName: string;
  relatedTaskName: string;
  sourceJobId: string;
  purpose: BuildDatasetPurpose;
  downstreamModelType: DownstreamModelType;
  dataOrganizationFormat: DataOrganizationFormat;
  usageScope: UsageScope;
  customEpisodeCount?: number;
  outputName: string;
  includeTrajectory: boolean;
  includeImageObservation: boolean;
  includeStateAction: boolean;
  includeProcessVideo: boolean;
  includeRunLog: boolean;
  includeFailures: boolean;
  includeTimeline: boolean;
  splitMode: SplitMode;
  customTrainRatio?: number;
}

export type BuildDatasetSourceMode = 'global' | 'row';
