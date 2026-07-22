/** 资源库首页 mock 数据 */

import type { LucideIcon } from 'lucide-react';
import {
  FileStack,
  Factory,
  Boxes,
  Bot,
  Brain,
  Gauge,
  Zap,
} from 'lucide-react';

export interface ResourceHubEntry {
  id: string;
  title: string;
  description: string;
  href: string;
  icon: LucideIcon;
  count: number;
  status: 'active' | 'draft';
}

export const resourceHubEntries: ResourceHubEntry[] = [
  {
    id: 'model-assets',
    title: '模型资产',
    description: '训练产出的 checkpoint 与 model_manifest，用于策略评测。',
    href: '/workspace/resources/model-assets',
    icon: Brain,
    count: 0,
    status: 'active',
  },
  {
    id: 'task-templates',
    title: '任务模板',
    description: '管理拧螺丝、装夹、上下料等工业任务模板。',
    href: '/workspace/resources/task-templates',
    icon: FileStack,
    count: 12,
    status: 'active',
  },
  {
    id: 'scenes',
    title: '场景',
    description: '管理精密装配工位、装夹工位、上下料工位等仿真场景。',
    href: '/workspace/resources/scenes',
    icon: Factory,
    count: 18,
    status: 'active',
  },
  {
    id: 'assets',
    title: '操作对象',
    description:
      '管理任务中的可交互对象，如工件、夹具、线缆、接插件、试管、移液枪、工具和容器等。',
    href: '/workspace/resources/assets',
    icon: Boxes,
    count: 46,
    status: 'active',
  },
  {
    id: 'robots',
    title: '机器人',
    description: '管理机械臂、双臂机器人、AGV 等用于任务执行的机器人本体。',
    href: '/workspace/resources/robots',
    icon: Bot,
    count: 8,
    status: 'active',
  },
  {
    id: 'policies',
    title: '模型类型',
    description: '管理 ACT、DP3、Diffusion Policy、OpenVLA 等模型类型及模型版本。',
    href: '/workspace/resources/policies',
    icon: Brain,
    count: 22,
    status: 'active',
  },
  {
    id: 'metrics',
    title: '评测指标',
    description: '管理成功率、耗时、碰撞次数、轨迹误差等评测指标。',
    href: '/workspace/resources/metrics',
    icon: Gauge,
    count: 6,
    status: 'active',
  },
  {
    id: 'physics-proxies',
    title: '物理代理模型',
    description:
      '管理接触、形变、材料响应等高成本物理过程的 PINN 代理模型，为数据生成和策略评测提供加速能力。',
    href: '/workspace/resources/physics-proxies',
    icon: Zap,
    count: 3,
    status: 'active',
  },
];
