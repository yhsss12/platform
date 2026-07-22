import type { TaskBuildConfig } from '@/types/benchmark';

const STORAGE_KEY = 'workspace_task_build_configs';

function readAll(): TaskBuildConfig[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as TaskBuildConfig[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(items: TaskBuildConfig[]): void {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

export function listTaskBuildConfigs(): TaskBuildConfig[] {
  return readAll();
}

export function saveTaskBuildConfig(config: TaskBuildConfig): void {
  const items = readAll();
  const next = [config, ...items.filter((c) => c.id !== config.id)];
  writeAll(next.slice(0, 20));
}

export function getTaskBuildConfig(id: string): TaskBuildConfig | null {
  return readAll().find((c) => c.id === id) ?? null;
}

export function makeTaskBuildConfigId(): string {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 6);
  return `tbc_${ts}_${rand}`;
}
