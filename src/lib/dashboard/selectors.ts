/** 概览页统一数据源（只读） */
import * as projectService from '@/lib/projects/projectService';
import { loadLabelTasks } from '@/features/data-platform/storage/labelTasks';
import { getAllActivities } from '@/lib/projects/projectActivity';
import type { Project } from '@/lib/projects/types';
import type { LabelTask } from '@/features/data-platform/models/labelTask';
import type { HDF5Dataset } from '@/features/data-platform/api/hdf5DatasetApi';
import type { ProjectActivity } from '@/lib/projects/projectActivity';

export interface DashboardData {
  projects: Project[];
  collectionTasks: Array<{ id: string; projectId?: string; status?: string; createdAt: string }>;
  labelTasks: LabelTask[];
  dataAssets: HDF5Dataset[];
  activities: ProjectActivity[];
  devices?: Array<{ id: string; status?: string; name?: string }>;
}

export function getAllProjects(): Project[] {
  if (typeof window === 'undefined') return [];
  return projectService.list();
}

export function getLabelTasks(): LabelTask[] {
  if (typeof window === 'undefined') return [];
  return loadLabelTasks();
}

export function getActivities(limit?: number): ProjectActivity[] {
  if (typeof window === 'undefined') return [];
  return getAllActivities(limit ?? 50);
}

export function getTotalMemberCount(projects: Project[]): number {
  const ids = new Set<string>();
  projects.forEach((p) => (p.members ?? []).forEach((m) => ids.add(m.id)));
  return ids.size;
}
