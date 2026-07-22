import type { DaqTask } from '@/lib/daq/types';

const STORAGE_KEY = 'eai_daq_tasks_v1';

/**
 * 从 localStorage 加载任务列表
 */
export function loadDaqTasks(): DaqTask[] {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const tasks = JSON.parse(stored) as DaqTask[];
      if (Array.isArray(tasks)) {
        return tasks;
      }
    }
  } catch (error) {
    console.error('Failed to load daq tasks from localStorage:', error);
  }

  return [];
}

/**
 * 保存任务列表到 localStorage
 */
export function saveDaqTasks(tasks: DaqTask[]): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
  } catch (error) {
    console.error('Failed to save daq tasks to localStorage:', error);
  }
}

/**
 * 添加任务
 */
export function addDaqTask(task: DaqTask): void {
  const tasks = loadDaqTasks();
  tasks.push(task);
  saveDaqTasks(tasks);
}

/**
 * 更新任务
 */
export function updateDaqTask(id: string, patch: Partial<DaqTask>): void {
  const tasks = loadDaqTasks();
  const index = tasks.findIndex(t => t.id === id);
  if (index >= 0) {
    tasks[index] = {
      ...tasks[index],
      ...patch,
      updatedAt: new Date().toISOString(),
    };
    saveDaqTasks(tasks);
  }
}

/**
 * 删除任务
 */
export function removeDaqTask(id: string): void {
  const tasks = loadDaqTasks();
  const filtered = tasks.filter(t => t.id !== id);
  saveDaqTasks(filtered);
}


