import type { WorkspaceArtifactItem } from '@/lib/api/workspaceJobClient';

export interface TrainingArtifactDisplayItem {
  id: string;
  label: string;
  name: string;
}

const ARTIFACT_TYPE_LABELS: Record<string, string> = {
  checkpoint: 'Checkpoint',
  manifest: 'Model Manifest',
  log: 'Train Log',
  metrics: 'Metrics',
  result: 'Report',
  video: 'Video',
  other: 'Other',
};

function artifactTypeLabel(type: string): string {
  const key = type.trim().toLowerCase();
  return ARTIFACT_TYPE_LABELS[key] ?? type;
}

/** 基于 artifacts 数组构建详情页展示列表（顶部计数与下方列表共用） */
export function buildTrainingArtifactDisplayItems(
  artifacts: WorkspaceArtifactItem[]
): TrainingArtifactDisplayItem[] {
  const seen = new Set<string>();
  const items: TrainingArtifactDisplayItem[] = [];

  for (const artifact of artifacts) {
    const dedupeKey = `${artifact.artifactType}:${artifact.name}:${artifact.id}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    items.push({
      id: String(artifact.id),
      label: artifactTypeLabel(artifact.artifactType),
      name: artifact.name,
    });
  }

  return items.sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'));
}
