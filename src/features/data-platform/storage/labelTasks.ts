import type { LabelTask } from '../models/labelTask';

const STORAGE_KEY = 'labelTasks';

/**
 * 获取种子数据（已移除，不再使用占位数据）
 */
function getSeedData(): LabelTask | null {
  // 不再返回占位数据，返回null表示没有种子数据
  return null;
}

/**
 * 确保所有任务都有稳定的 taskNo（1 起连续），按 createdAt 升序分配。
 * 返回 [排序并补齐 taskNo 的列表, 是否需要写回]。
 */
function ensureTaskNo(tasks: LabelTask[]): [LabelTask[], boolean] {
  if (tasks.length === 0) return [[], false];
  const sorted = [...tasks].sort(
    (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
  );
  let changed = false;
  const next: LabelTask[] = sorted.map((t, i) => {
    const seq = i + 1;
    if (t.taskNo != null && t.taskNo === seq) return t;
    changed = true;
    return { ...t, taskNo: seq };
  });
  return [next, changed];
}

/**
 * 从 localStorage 加载任务列表；自动补齐 taskNo 并写回（迁移）
 */
export function loadLabelTasks(): LabelTask[] {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const tasks = JSON.parse(stored) as LabelTask[];
      if (Array.isArray(tasks) && tasks.length > 0) {
        const [withTaskNo, needsSave] = ensureTaskNo(tasks);
        if (needsSave) saveLabelTasks(withTaskNo);
        return withTaskNo;
      }
    }
  } catch (error) {
    console.error('Failed to load label tasks from localStorage:', error);
  }

  return [];
}

/**
 * 初始化种子数据（如果 localStorage 为空）
 */
export function initLabelTasksIfNeeded(): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      // 不再初始化占位数据，直接保存空数组
      saveLabelTasks([]);
    }
  } catch (error) {
    console.error('Failed to init label tasks:', error);
  }
}

/**
 * 保存任务列表到 localStorage
 */
export function saveLabelTasks(tasks: LabelTask[]): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
  } catch (error) {
    console.error('Failed to save label tasks to localStorage:', error);
  }
}

