/** 物理代理模型 mock 数据与工具函数 */

export type PhysicsProxyMode = 'off' | 'pinn' | 'hybrid';

export type PhysicsProxyModelStatus = 'available' | 'validating';

export interface PhysicsProxyModelRow {
  id: string;
  name: string;
  proxyType: string;
  applicableTasks: string;
  physicalObjects: string;
  inputVariables: string;
  outputVariables: string;
  trainingMethod: string;
  errorMetric: string;
  speedup: string;
  status: PhysicsProxyModelStatus;
}

export const physicsProxyModelOptions = [
  'contact-force-pinn-v1',
  'elastic-deform-pinn-v1',
  'cable-shape-pinn-v1',
] as const;

export const physicsProxyModelsMock: PhysicsProxyModelRow[] = [
  {
    id: 'contact-force-pinn-v1',
    name: 'contact-force-pinn-v1',
    proxyType: '接触力',
    applicableTasks: '线缆穿杆',
    physicalObjects: '螺丝 / 工件 / 电批',
    inputVariables: '位姿、速度、接触深度、材料参数',
    outputVariables: '接触力、摩擦状态',
    trainingMethod: 'PINN-CAML',
    errorMetric: '3.8%',
    speedup: '12.5×',
    status: 'available',
  },
  {
    id: 'elastic-deform-pinn-v1',
    name: 'elastic-deform-pinn-v1',
    proxyType: '弹性形变',
    applicableTasks: '螺母装配',
    physicalObjects: '工件 / 夹具',
    inputVariables: '夹紧力、材料参数、接触边界',
    outputVariables: '位移场、应力分布',
    trainingMethod: 'PINN',
    errorMetric: '4.6%',
    speedup: '9.2×',
    status: 'available',
  },
  {
    id: 'cable-shape-pinn-v1',
    name: 'cable-shape-pinn-v1',
    proxyType: '柔性对象',
    applicableTasks: '线缆整理',
    physicalObjects: '线缆 / 接插件',
    inputVariables: '端点位姿、约束点、材料刚度',
    outputVariables: '线缆曲线、弯曲应力',
    trainingMethod: 'PINN-CAML',
    errorMetric: '5.1%',
    speedup: '8.7×',
    status: 'validating',
  },
];

export function findPhysicsProxyModel(id: string | null | undefined): PhysicsProxyModelRow | undefined {
  if (!id) return undefined;
  return physicsProxyModelsMock.find((m) => m.id === id);
}

export function physicsProxyStatusLabel(status: PhysicsProxyModelStatus): string {
  return status === 'available' ? '可用' : '验证中';
}

export function physicsProxyModeLabel(mode: PhysicsProxyMode): string {
  switch (mode) {
    case 'off':
      return '关闭';
    case 'pinn':
      return 'PINN 加速';
    case 'hybrid':
      return 'Hybrid';
  }
}

export function physicsProxyAcceleratedModeLabel(mode: PhysicsProxyMode): string {
  switch (mode) {
    case 'off':
      return 'MuJoCo';
    case 'pinn':
      return 'MuJoCo + PINN';
    case 'hybrid':
      return 'MuJoCo + PINN Hybrid';
  }
}

export function formatEvalConfigWithPhysicsProxy(row: {
  evalBackend: string;
  evalRounds: number;
  physicsProxyMode?: PhysicsProxyMode;
}): string {
  const mode = row.physicsProxyMode ?? 'off';
  const rounds = row.evalRounds ? `${row.evalRounds} 次` : '';
  if (mode === 'off') {
    if (!row.evalRounds) return row.evalBackend;
    return `${row.evalBackend} · ${rounds}`;
  }
  return `${physicsProxyAcceleratedModeLabel(mode)} · ${rounds}`;
}

export interface PhysicsProxyRuntimeState {
  mode: PhysicsProxyMode;
  modelId: string;
  physicalObjects: string;
  currentPrediction: string;
  errorEstimate: string;
  speedup: string;
  reviewStatus: string;
}

export function buildPhysicsProxyRuntimeState(
  mode: PhysicsProxyMode,
  modelId: string | null
): PhysicsProxyRuntimeState | null {
  if (mode === 'off' || !modelId) return null;
  const model = findPhysicsProxyModel(modelId);
  if (!model) return null;
  return {
    mode,
    modelId: model.id,
    physicalObjects: model.physicalObjects,
    currentPrediction: model.proxyType === '接触力' ? '接触力 12.4 N' : '局部物理响应预测中',
    errorEstimate: model.errorMetric.replace('%', '') === '3.8' ? '3.2%' : model.errorMetric,
    speedup: model.speedup,
    reviewStatus: mode === 'hybrid' ? '第 10 条触发高保真复核' : '—',
  };
}

export interface PhysicsProxyReportSection {
  accelerationMode: string;
  proxyModel: string;
  proxyObjects: string;
  errorEstimate: string;
  speedup: string;
  highFidelityReviewRatio: string;
  passedErrorThreshold: string;
}

export function buildPhysicsProxyReportSection(row: {
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string | null;
  physicsProxyError?: string;
  physicsProxySpeedup?: string;
  highFidelityReviewRatio?: number | string;
  physicsProxyErrorThreshold?: number;
}): PhysicsProxyReportSection | null {
  const mode = row.physicsProxyMode ?? 'off';
  if (mode === 'off') return null;
  const model = findPhysicsProxyModel(row.physicsProxyModel);
  const errorEstimate = row.physicsProxyError ?? model?.errorMetric ?? '—';
  const threshold = row.physicsProxyErrorThreshold ?? 5;
  const errorNum = parseFloat(errorEstimate.replace('%', ''));
  const passed = !Number.isNaN(errorNum) && errorNum <= threshold;
  const reviewRatio =
    row.highFidelityReviewRatio != null
      ? typeof row.highFidelityReviewRatio === 'number'
        ? `${row.highFidelityReviewRatio}%`
        : row.highFidelityReviewRatio
      : '—';

  return {
    accelerationMode: physicsProxyAcceleratedModeLabel(mode),
    proxyModel: row.physicsProxyModel ?? model?.id ?? '—',
    proxyObjects: model?.physicalObjects ?? '—',
    errorEstimate,
    speedup: row.physicsProxySpeedup ?? model?.speedup ?? '—',
    highFidelityReviewRatio: reviewRatio,
    passedErrorThreshold: passed ? '通过' : '未通过',
  };
}
