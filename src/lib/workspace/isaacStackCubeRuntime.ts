const STACK_CUBE_ISSUE_LABELS: Record<string, string> = {
  missing_isaaclab_root: 'Isaac Lab 运行节点未配置',
  runtime_disabled: 'Isaac Lab 运行节点未启用',
  missing_default_seed: '默认物块堆叠 Seed Demo 未就绪',
  gpu_unavailable: 'GPU / nvidia-smi 不可用',
  task_not_registered: '物块堆叠任务未注册',
  scripted_expert_script_missing: '专家策略脚本未部署',
};

export function formatIsaacStackCubeIssue(code: string): string {
  return STACK_CUBE_ISSUE_LABELS[code] ?? code;
}

export function formatIsaacStackCubeIssues(codes: string[]): string[] {
  return codes.map(formatIsaacStackCubeIssue);
}

export function formatScriptedExpertIssues(codes: string[]): string[] {
  return codes
    .filter((code) => code !== 'missing_default_seed')
    .map(formatIsaacStackCubeIssue);
}
