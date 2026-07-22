import type { RunStatus } from './status';

export interface RunArtifact {
  type: string;
  path: string;
  bytes: number;
}

export interface Run {
  id: string;
  taskId: string;
  status: RunStatus;
  startedAt?: string;
  endedAt?: string;
  artifact: RunArtifact;
  durationSec?: number;
}


