/** 训练节点 — 产品化展示与提交参数 */

export type TrainingDeviceValue = 'l20-172-18-0-73' | 'h20-local-placeholder' | 'l20' | 'h20';

export type TrainingNodeStatus =
  | 'available'
  | 'busy'
  | 'unreachable'
  | 'misconfigured'
  | 'placeholder';

export interface TrainingDeviceOption {
  label: string;
  value: TrainingDeviceValue;
  nodeId: string;
  deviceParam: string;
  deviceLabel: string;
  trainingNodeDisplayName?: string;
  description: string;
  status?: TrainingNodeStatus;
  statusLabel?: string;
  message?: string;
  selectable?: boolean;
  host?: string | null;
}

/** 后端不可用时的静态回退 */
export const TRAINING_DEVICE_FALLBACK_OPTIONS: TrainingDeviceOption[] = [
  {
    label: 'L20 · 172.18.0.73',
    value: 'l20-172-18-0-73',
    nodeId: 'l20-172-18-0-73',
    deviceParam: 'cuda',
    deviceLabel: 'L20 · 172.18.0.73',
    trainingNodeDisplayName: 'L20 · 172.18.0.73',
    description: 'NVIDIA L20 远程 GPU 训练节点',
    host: '172.18.0.73',
    status: 'unreachable',
    statusLabel: '不可用',
    selectable: false,
  },
  {
    label: 'L20 · 172.18.0.101',
    value: 'h20-local-placeholder',
    nodeId: 'h20-local-placeholder',
    deviceParam: 'cuda',
    deviceLabel: 'L20 · 172.18.0.101',
    trainingNodeDisplayName: 'L20 · 172.18.0.101',
    description: '本地 NVIDIA L20 GPU 训练节点',
    host: '172.18.0.101',
    status: 'available',
    statusLabel: '空闲',
    selectable: true,
  },
];

export const DEFAULT_TRAINING_DEVICE: TrainingDeviceValue = 'l20-172-18-0-73';

export const L20_TRAINING_NODE_ID = 'l20-172-18-0-73';
/** @deprecated 历史 node_id；实际为本机 L20 */
export const H20_TRAINING_NODE_ID = 'h20-local-placeholder';
export const LOCAL_TRAINING_NODE_ID = H20_TRAINING_NODE_ID;

export function normalizeTrainingNodeId(value?: string | null): string {
  const token = (value || '').trim().toLowerCase();
  if (token === 'l20') return L20_TRAINING_NODE_ID;
  if (token === 'h20') return H20_TRAINING_NODE_ID;
  return token || DEFAULT_TRAINING_DEVICE;
}

export function findTrainingDeviceOption(
  value?: string | null,
  options: TrainingDeviceOption[] = TRAINING_DEVICE_FALLBACK_OPTIONS
): TrainingDeviceOption {
  const normalized = normalizeTrainingNodeId(value);
  return (
    options.find((item) => item.nodeId === normalized || item.value === normalized) ??
    options[0]
  );
}

export function formatTrainingNodeStatusLabel(status?: TrainingNodeStatus | null): string {
  switch (status) {
    case 'available':
      return '空闲';
    case 'busy':
      return '忙碌';
    case 'unreachable':
      return '不可用';
    case 'misconfigured':
      return '配置异常';
    case 'placeholder':
      return '本机 GPU';
    default:
      return '未知';
  }
}

/** 与平台 StatusBadge 一致的淡色 tag 样式 */
export function trainingNodeStatusBadgeStyle(status?: TrainingNodeStatus | null): {
  bg: string;
  color: string;
} {
  switch (status) {
    case 'available':
      return { bg: '#ecfdf5', color: '#047857' };
    case 'placeholder':
      return { bg: '#f0fdf4', color: '#15803d' };
    case 'busy':
      return { bg: '#fef3c7', color: '#92400e' };
    case 'unreachable':
      return { bg: '#fee2e2', color: '#991b1b' };
    case 'misconfigured':
      return { bg: '#fff7ed', color: '#9a3412' };
    default:
      return { bg: '#f3f4f6', color: '#6b7280' };
  }
}

/** 展示训练节点名称；优先 API 返回的 displayName，兼容历史 L20/H20 简写 */
export function formatTrainingDeviceLabel(
  deviceLabel?: string | null,
  trainingNodeDisplayName?: string | null,
  trainingNodeId?: string | null
): string {
  const display = (trainingNodeDisplayName || deviceLabel || '').trim();
  if (display.includes('·')) return display;

  const nodeId = normalizeTrainingNodeId(trainingNodeId);
  const fallback = findTrainingDeviceOption(nodeId);
  if (fallback.trainingNodeDisplayName) return fallback.trainingNodeDisplayName;
  if (fallback.label.includes('·')) return fallback.label;

  const normalized = display.toUpperCase();
  if (normalized === 'H20' || normalized === 'L20') {
    return fallback.label;
  }
  return display || fallback.label;
}

/** @deprecated 请使用 TrainingNodeSelect；保留给旧引用 */
export function formatTrainingDeviceOptionLabel(option: TrainingDeviceOption): string {
  return option.label;
}

export function trainingDeviceSubmitParams(
  value?: TrainingDeviceValue | string | null,
  options: TrainingDeviceOption[] = TRAINING_DEVICE_FALLBACK_OPTIONS
): {
  device: string;
  deviceLabel: string;
  trainingNodeId: string;
} {
  const option = findTrainingDeviceOption(value, options);
  const display = option.trainingNodeDisplayName || option.deviceLabel || option.label;
  return {
    device: option.deviceParam,
    deviceLabel: display,
    trainingNodeId: option.nodeId,
  };
}

export function apiNodeToDeviceOption(node: {
  nodeId: string;
  label: string;
  deviceLabel: string;
  trainingNodeDisplayName?: string;
  description?: string;
  status: TrainingNodeStatus;
  statusLabel?: string;
  message?: string;
  selectable?: boolean;
  host?: string | null;
}): TrainingDeviceOption {
  const display = node.trainingNodeDisplayName || node.deviceLabel || node.label;
  return {
    label: display,
    value: normalizeTrainingNodeId(node.nodeId) as TrainingDeviceValue,
    nodeId: node.nodeId,
    deviceParam: 'cuda',
    deviceLabel: display,
    trainingNodeDisplayName: display,
    description: node.description || '',
    status: node.status,
    statusLabel: formatTrainingNodeStatusLabel(node.status),
    message: node.message,
    selectable: node.selectable,
    host: node.host,
  };
}

/** @deprecated 使用 API 动态节点列表；保留兼容旧引用 */
export const TRAINING_DEVICE_OPTIONS = TRAINING_DEVICE_FALLBACK_OPTIONS;
