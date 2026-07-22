// Run 状态枚举
export type RunStatus = 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELED';

// Job 状态枚举
export type JobStatus = 'PENDING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELED';

// Task 状态枚举
export type TaskStatus = 'PENDING' | 'DRAFT' | 'READY' | 'RUNNING' | 'COMPLETED' | 'ARCHIVED';

// Dataset 状态枚举
export type DatasetStatus = 'ACTIVE' | 'ARCHIVED';

// Run 状态转移表
export const RUN_STATUS_TRANSITIONS: Record<RunStatus, RunStatus[]> = {
  QUEUED: ['RUNNING', 'CANCELED'],
  RUNNING: ['SUCCEEDED', 'FAILED', 'CANCELED'],
  SUCCEEDED: [],
  FAILED: [],
  CANCELED: [],
};

// Job 状态转移表
export const JOB_STATUS_TRANSITIONS: Record<JobStatus, JobStatus[]> = {
  PENDING: ['RUNNING', 'CANCELED'],
  RUNNING: ['SUCCEEDED', 'FAILED', 'CANCELED'],
  SUCCEEDED: [],
  FAILED: [],
  CANCELED: [],
};


