import type { JobStatus } from './status';

export interface JobTarget {
  runId?: string;
  datasetId?: string;
}

export interface JobStep {
  name: string;
  status: JobStatus;
  startedAt?: string;
  endedAt?: string;
}

export interface Job {
  id: string;
  type: string;
  status: JobStatus;
  progress: {
    percent: number;
    current?: number;
    total?: number;
  };
  target: JobTarget;
  steps?: JobStep[];
  logs?: string[];
  error?: string;
}


