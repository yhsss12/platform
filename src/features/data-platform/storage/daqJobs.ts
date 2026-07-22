/** localStorage 缓存的采集作业（与旧采集页结构兼容） */
interface Job {
  id: string;
  jobNumber?: string;
  taskId: string;
  taskName?: string;
  deviceName?: string;
  deviceId?: string;
  collector?: string;
  collectionQuantity?: number;
  progress?: { current?: number; total?: number; percent?: number };
  status?: string;
  updatedAt?: string;
  validationReportJson?: string | null;
}

const STORAGE_KEY = 'eai_daq_jobs_v1';

/**
 * 从 localStorage 加载作业列表
 */
export function loadDaqJobs(): Job[] {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const jobs = JSON.parse(stored) as Job[];
      if (Array.isArray(jobs)) {
        return jobs;
      }
    }
  } catch (error) {
    console.error('Failed to load daq jobs from localStorage:', error);
  }

  return [];
}

/**
 * 保存作业列表到 localStorage
 */
export function saveDaqJobs(jobs: Job[]): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs));
  } catch (error) {
    console.error('Failed to save daq jobs to localStorage:', error);
  }
}

/**
 * 根据任务ID获取作业列表
 */
export function getJobsByTaskId(taskId: string): Job[] {
  const jobs = loadDaqJobs();
  return jobs.filter(j => j.taskId === taskId);
}

/**
 * 添加作业
 */
export function addDaqJob(job: Job): void {
  const jobs = loadDaqJobs();
  jobs.push(job);
  saveDaqJobs(jobs);
}

/**
 * 更新作业
 */
export function updateDaqJob(id: string, patch: Partial<Job>): void {
  const jobs = loadDaqJobs();
  const index = jobs.findIndex(j => j.id === id);
  if (index >= 0) {
    jobs[index] = {
      ...jobs[index],
      ...patch,
      updatedAt: new Date().toISOString(),
    };
    saveDaqJobs(jobs);
  }
}

/**
 * 删除作业
 */
export function removeDaqJob(id: string): void {
  const jobs = loadDaqJobs();
  const filtered = jobs.filter(j => j.id !== id);
  saveDaqJobs(filtered);
}

/**
 * 根据任务ID删除所有相关作业
 */
export function removeJobsByTaskId(taskId: string): void {
  const jobs = loadDaqJobs();
  const filtered = jobs.filter(j => j.taskId !== taskId);
  saveDaqJobs(filtered);
}


