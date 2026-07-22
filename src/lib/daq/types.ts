import type { TaskFrequencyConfig } from '@/features/data-platform/models/frequencyConfigTypes';

/** 采集任务列表项（admin/projects 等隐藏页使用） */
export interface DaqTask {
  id: string;
  taskNumber: string;
  taskName: string;
  taskDescription?: string;
  owner?: string;
  creatorId?: string;
  creatorUsername?: string;
  creatorAccountId?: string;
  creatorDisplay?: string;
  deviceName?: string;
  deviceId?: string;
  episodeCount?: number;
  durationSec?: number;
  storagePath?: string;
  storageTypes?: string[];
  createdAt: string;
  updatedAt?: string;
  remark?: string;
  projectId?: string;
  collector?: string;
  collectorName?: string;
  collectionCount?: number;
  hasJobs?: boolean;
  jobCount?: number;
  completedCount?: number;
  deviceType?: string;
  cameraDataFormat?: string;
  frequencyConfig?: TaskFrequencyConfig;
}
