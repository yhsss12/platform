'use client';

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Route,
  Timer,
  TrendingUp,
  Waves,
  XCircle,
  type LucideIcon,
} from 'lucide-react';
import type { GenericMetricGroup } from '@/lib/workspace/evaluationMetricRegistry';

const GROUP_ICONS: Record<string, LucideIcon> = {
  success_rate: CheckCircle2,
  mean_reward: TrendingUp,
  mean_episode_length: Timer,
  failure_count: XCircle,
  timeout_rate: Clock,
  episode_stability: Activity,
  trajectory_error: Route,
  collision_count: AlertTriangle,
  action_smoothness: Waves,
};

export function GenericMetricCard({
  group,
  onOpen,
}: {
  group: GenericMetricGroup;
  onOpen: (group: GenericMetricGroup) => void;
}) {
  const Icon = GROUP_ICONS[group.groupKey] ?? Activity;
  const taskCount = group.applicableTaskLabels.length;
  const modeCount = group.applicableEvaluationModeLabels.length;
  const implemented = group.status === 'implemented';

  return (
    <button type="button" className="metrics-library-card" onClick={() => onOpen(group)}>
      <div className="metrics-library-card-top">
        <div className="metrics-library-card-icon">
          <Icon size={20} strokeWidth={1.65} />
        </div>
        <span
          className={`metrics-library-status-badge ${implemented ? 'implemented' : 'planned'}`}
        >
          {implemented ? '已接入' : '规划中'}
        </span>
      </div>

      <div className="metrics-library-card-body">
        <h3 className="metrics-library-card-title">{group.displayName}</h3>
        <p className="metrics-library-card-desc">{group.description}</p>
      </div>

      <div className="metrics-library-card-footer">
        <div className="metrics-library-card-meta">
          <span>{taskCount} 个任务</span>
          <span>{modeCount} 种模式</span>
        </div>
        <span className="metrics-library-card-detail-link">详情 →</span>
      </div>
    </button>
  );
}
