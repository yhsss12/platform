import type { LucideIcon } from 'lucide-react';
import {
  Bot,
  Boxes,
  Brain,
  Cog,
  Cuboid,
  Factory,
  FileStack,
  Gauge,
  Shield,
} from 'lucide-react';

export type ResourceHubCountKey =
  | 'taskTemplates'
  | 'modelAssets'
  | 'metrics'
  | 'scenes'
  | 'objects'
  | 'robots'
  | 'policyAssets'
  | 'simAssets'
  | 'modelTypes'
  | 'physicsProxies'
  | 'craftConfig';

export interface ResourceHubEntryConfig {
  id: string;
  title: string;
  description: string;
  href: string;
  icon: LucideIcon;
  countKey: ResourceHubCountKey;
}

export interface ResourceHubSectionConfig {
  id: 'core' | 'simulation' | 'advanced';
  title: string;
  entries: ResourceHubEntryConfig[];
}

export type ResourceHubOverviewKey =
  | ResourceHubCountKey
  | 'simulationResources';

export interface ResourceHubOverviewItemConfig {
  id: string;
  title: string;
  countKey: ResourceHubOverviewKey;
  href?: string;
}

/** 顶部资源总览区：6 项汇总统计 */
export const RESOURCE_HUB_OVERVIEW_ITEMS: ResourceHubOverviewItemConfig[] = [
  {
    id: 'overview-task-templates',
    title: '任务模板',
    countKey: 'taskTemplates',
    href: '/workspace/resources/task-templates',
  },
  {
    id: 'overview-model-assets',
    title: '模型资产',
    countKey: 'modelAssets',
    href: '/workspace/resources/model-assets',
  },
  {
    id: 'overview-simulation-resources',
    title: '仿真资源',
    countKey: 'simulationResources',
    href: '/workspace/resources/scenes',
  },
  {
    id: 'overview-metrics',
    title: '评测指标',
    countKey: 'metrics',
    href: '/workspace/resources/metrics',
  },
  {
    id: 'overview-policy-assets',
    title: '策略资产',
    countKey: 'policyAssets',
    href: '/workspace/resources/policies',
  },
  {
    id: 'overview-craft-config',
    title: '工艺配置',
    countKey: 'craftConfig',
    href: '/workspace/resources/craft-config',
  },
];

const SIMULATION_RESOURCE_COUNT_KEYS = [
  'scenes',
  'robots',
  'objects',
  'simAssets',
] as const satisfies readonly ResourceHubCountKey[];

export function aggregateSimulationResourcesCount(
  counts: Partial<Record<ResourceHubCountKey, number | null>>
): number | null {
  const values = SIMULATION_RESOURCE_COUNT_KEYS.map((key) => counts[key]);
  if (values.some((value) => value === null || value === undefined)) {
    return null;
  }
  return (values as number[]).reduce((sum, value) => sum + value, 0);
}

export function resolveResourceHubOverviewCount(
  countKey: ResourceHubOverviewKey,
  counts: Partial<Record<ResourceHubCountKey, number | null>>
): number | null {
  if (countKey === 'simulationResources') {
    return aggregateSimulationResourcesCount(counts);
  }
  return counts[countKey] ?? null;
}

export const RESOURCE_HUB_SECTIONS: ResourceHubSectionConfig[] = [
  {
    id: 'core',
    title: '平台核心资源',
    entries: [
      {
        id: 'task-templates',
        title: '任务模板',
        description: '标准任务定义',
        href: '/workspace/resources/task-templates',
        icon: FileStack,
        countKey: 'taskTemplates',
      },
      {
        id: 'model-assets',
        title: '模型资产',
        description: '训练模型管理',
        href: '/workspace/resources/model-assets',
        icon: Brain,
        countKey: 'modelAssets',
      },
      {
        id: 'metrics',
        title: '评测指标',
        description: '评测口径配置',
        href: '/workspace/resources/metrics',
        icon: Gauge,
        countKey: 'metrics',
      },
    ],
  },
  {
    id: 'simulation',
    title: '仿真与执行资源',
    entries: [
      {
        id: 'scenes',
        title: '仿真场景',
        description: '场景环境管理',
        href: '/workspace/resources/scenes',
        icon: Factory,
        countKey: 'scenes',
      },
      {
        id: 'robots',
        title: '机器人',
        description: '机器人本体配置',
        href: '/workspace/resources/robots',
        icon: Bot,
        countKey: 'robots',
      },
      {
        id: 'assets',
        title: '操作对象',
        description: '工件对象管理',
        href: '/workspace/resources/assets',
        icon: Boxes,
        countKey: 'objects',
      },
      {
        id: 'policies',
        title: '策略资产',
        description: '专家策略管理',
        href: '/workspace/resources/policies',
        icon: Shield,
        countKey: 'policyAssets',
      },
      {
        id: 'sim-assets',
        title: '仿真资产',
        description: '仿真资源管理',
        href: '/workspace/resources/sim-assets',
        icon: Cuboid,
        countKey: 'simAssets',
      },
    ],
  },
  {
    id: 'advanced',
    title: '高级配置资源',
    entries: [
      {
        id: 'physics-proxies',
        title: '物理代理模型',
        description: '物理模型配置',
        href: '/workspace/resources/physics-proxies',
        icon: Cog,
        countKey: 'physicsProxies',
      },
      {
        id: 'model-types',
        title: '模型类型',
        description: '算法类型定义',
        href: '/workspace/resources/model-types',
        icon: Brain,
        countKey: 'modelTypes',
      },
      {
        id: 'craft-config',
        title: '工艺配置',
        description: '流程参数配置',
        href: '/workspace/resources/craft-config',
        icon: FileStack,
        countKey: 'craftConfig',
      },
    ],
  },
];
