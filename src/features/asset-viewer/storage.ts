/**
 * 保存任务描述到 localStorage
 */
export function saveTaskDescription(taskId: string, description: string): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(`label_runner_desc_${taskId}`, description);
  } catch (error) {
    console.error('Failed to save task description:', error);
  }
}

/**
 * 加载任务描述
 */
export function loadTaskDescription(taskId: string): string {
  if (typeof window === 'undefined') return '';
  try {
    return localStorage.getItem(`label_runner_desc_${taskId}`) || '';
  } catch (error) {
    console.error('Failed to load task description:', error);
    return '';
  }
}

/**
 * 保存 Agent 日志
 */
export function saveAgentLog(taskId: string, episodeName: string, logs: string): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(`label_runner_agentlog_${taskId}_${episodeName}`, logs);
  } catch (error) {
    console.error('Failed to save agent log:', error);
  }
}

/**
 * 加载 Agent 日志
 */
export function loadAgentLog(taskId: string, episodeName: string): string {
  if (typeof window === 'undefined') return '';
  try {
    return localStorage.getItem(`label_runner_agentlog_${taskId}_${episodeName}`) || '';
  } catch (error) {
    console.error('Failed to load agent log:', error);
    return '';
  }
}


