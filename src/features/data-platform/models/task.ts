import type { TaskStatus } from './status';

export interface Task {
  id: string;
  name: string;
  status: TaskStatus;
  createdAt: string;
  updatedAt: string;
  lastRunId?: string;
  boundDevices?: string[];
  configRef?: string;
  // Extended fields
  description?: string;
  owner?: string;
  deviceId?: string;
  deviceName?: string;
  /** 所属项目 ID（与项目管理对接） */
  projectId?: string;
  creatorId?: string;
  creatorUsername?: string;
  creatorAccountId?: string;
  episodeCount?: number;
  durationSec?: number;
  storagePath?: string;
  storageTypes?: string[];
  remark?: string;
  cameraDataFormat?: string;
}

