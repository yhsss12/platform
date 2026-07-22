/** 概览页数据与图表用类型 */

export type TimeRangeKey = '30m' | 'today' | '7d';

export interface DashboardKpi {
  projectCount: number;
  taskTotal: number;
  taskRunning: number;
  dataAssetCount: number;
  memberCount: number;
  todayNewTasks?: number;
  todayNewAssets?: number;
}

export interface FunnelSegment {
  label: string;
  value: number;
  color?: string;
}

export interface EventBucket {
  time: string;
  count: number;
  label: string;
}

export interface ProjectLoadItem {
  projectId: string;
  projectName: string;
  taskCount: number;
  dataCount: number;
}

export interface AssetPoint {
  time: string;
  cumulative: number;
  label: string;
}
