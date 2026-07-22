import type { DatasetStatus } from './status';

export interface DatasetArtifactSummary {
  totalBytes: number;
  fileCount: number;
}

export interface Dataset {
  id: string;
  name: string;
  status: DatasetStatus;
  runIds: string[];
  artifactSummary?: DatasetArtifactSummary;
  createdAt: string;
  updatedAt: string;
}


