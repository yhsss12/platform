#!/usr/bin/env node
// Generate the repository model-schema report artifact.

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const ARTIFACTS_DIR = join(process.cwd(), 'artifacts', 'report');
const OUTPUT_FILE = join(ARTIFACTS_DIR, 'models.schema.json');

mkdirSync(ARTIFACTS_DIR, { recursive: true });

const schema = {
  timestamp: new Date().toISOString(),
  models: {
    Task: {
      id: 'string',
      name: 'string',
      status: 'TaskStatus (DRAFT | READY | RUNNING | COMPLETED | ARCHIVED)',
      createdAt: 'string (ISO 8601)',
      updatedAt: 'string (ISO 8601)',
      lastRunId: 'string (optional)',
      boundDevices: 'string[] (optional)',
      configRef: 'string (optional)',
    },
    Run: {
      id: 'string',
      taskId: 'string',
      status: 'RunStatus (QUEUED | RUNNING | SUCCEEDED | FAILED | CANCELED)',
      startedAt: 'string (ISO 8601, optional)',
      endedAt: 'string (ISO 8601, optional)',
      artifact: {
        type: 'string',
        path: 'string',
        bytes: 'number',
      },
      durationSec: 'number (optional)',
    },
    Dataset: {
      id: 'string',
      name: 'string',
      status: 'DatasetStatus (ACTIVE | ARCHIVED)',
      runIds: 'string[]',
      artifactSummary: {
        totalBytes: 'number',
        fileCount: 'number',
      },
      createdAt: 'string (ISO 8601)',
      updatedAt: 'string (ISO 8601)',
    },
    Job: {
      id: 'string',
      type: 'string',
      status: 'JobStatus (PENDING | RUNNING | SUCCEEDED | FAILED | CANCELED)',
      progress: {
        percent: 'number',
        current: 'number (optional)',
        total: 'number (optional)',
      },
      target: {
        runId: 'string (optional)',
        datasetId: 'string (optional)',
      },
      steps: [
        {
          name: 'string',
          status: 'JobStatus',
          startedAt: 'string (ISO 8601, optional)',
          endedAt: 'string (ISO 8601, optional)',
        },
      ],
      logs: 'string[] (optional)',
      error: 'string (optional)',
    },
  },
  statusTransitions: {
    Run: {
      QUEUED: ['RUNNING', 'CANCELED'],
      RUNNING: ['SUCCEEDED', 'FAILED', 'CANCELED'],
      SUCCEEDED: [],
      FAILED: [],
      CANCELED: [],
    },
    Job: {
      PENDING: ['RUNNING', 'CANCELED'],
      RUNNING: ['SUCCEEDED', 'FAILED', 'CANCELED'],
      SUCCEEDED: [],
      FAILED: [],
      CANCELED: [],
    },
  },
};

writeFileSync(OUTPUT_FILE, JSON.stringify(schema, null, 2));
console.log(`✅ Models schema generated: ${OUTPUT_FILE}`);

