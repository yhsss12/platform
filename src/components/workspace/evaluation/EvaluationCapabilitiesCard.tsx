'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  getEvaluationCapabilities,
  type EvaluationCapabilities,
  type EvaluationTaskType,
} from '@/lib/api/evaluationClient';

const cardStyle: React.CSSProperties = {
  marginTop: 16,
  padding: '14px 16px',
  borderRadius: 12,
  border: '1px solid #e5e7eb',
  backgroundColor: '#f8fafc',
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 12,
  fontSize: 13,
  lineHeight: 1.55,
  marginBottom: 6,
};

const MODE_LABELS: Record<string, string> = {
  policy_evaluation: '策略评测',
  episode_stability: 'episode 稳定性评测',
};

function modeLabel(mode: string): string {
  return MODE_LABELS[mode] ?? mode;
}

function yesNo(value: boolean | undefined, fallback = '—'): string {
  if (value === true) return '是';
  if (value === false) return '否';
  return fallback;
}

export function EvaluationCapabilitiesCard({
  taskType,
  taskLabel,
}: {
  taskType: EvaluationTaskType | null;
  taskLabel: string;
}) {
  const [capabilities, setCapabilities] = useState<EvaluationCapabilities | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!taskType) {
      setCapabilities(null);
      setFetchError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setFetchError(null);

    void getEvaluationCapabilities(taskType)
      .then((data) => {
        if (!cancelled) {
          setCapabilities(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCapabilities(null);
          setFetchError('评测能力信息暂时无法加载，不影响已有评测功能。');
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [taskType]);

  const display = useMemo(() => {
    if (!taskType || !capabilities) return null;

    const isDualArm = taskType === 'dual_arm_cable_manipulation';
    const isCable = taskType === 'cable_threading';
    const phase2Pending =
      isDualArm &&
      (capabilities.description.includes('Phase 2 待实现') ||
        capabilities.description.includes('Phase 1 仅完成'));

    return {
      modes: capabilities.supportedModes.map(modeLabel).join('、') || '—',
      policies:
        capabilities.supportedPolicyTypes && capabilities.supportedPolicyTypes.length > 0
          ? capabilities.supportedPolicyTypes.join(' / ')
          : '—',
      checkpoint: yesNo(capabilities.supportsCheckpoint),
      trainModel: yesNo(capabilities.supportsTrainModelEvaluation),
      video: capabilities.supportsVideo ? '是' : '否',
      resultArtifact: isCable
        ? `${capabilities.resultArtifact ?? 'eval.results.json'} / eval.mp4`
        : capabilities.resultArtifact ?? 'aggregate_result.json',
      status: phase2Pending ? 'Phase 2 待实现' : '已接入',
      description: isCable
        ? capabilities?.supportedPolicyTypes?.length
          ? `支持选择已训练模型资产进行策略评测；当前适配器支持：${capabilities.supportedPolicyTypes.join(' / ')}。`
          : '支持选择已训练模型资产进行策略评测。'
        : isDualArm
          ? capabilities.description ||
            '该任务支持 episode 稳定性评测，通过多 seed 重复运行完整 episode，聚合成功率、接触、拉伸和线缆形态指标。'
          : capabilities.description,
      phase2Pending,
    };
  }, [capabilities, taskType]);

  if (!taskType) return null;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#111827', marginBottom: 10 }}>评测能力</div>

      {loading ? (
        <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>正在加载任务评测能力…</p>
      ) : null}

      {fetchError ? (
        <p style={{ margin: loading ? '8px 0 0' : 0, fontSize: 12, color: '#b45309', lineHeight: 1.55 }}>
          {fetchError}
        </p>
      ) : null}

      {display ? (
        <>
          <div style={rowStyle}>
            <span style={{ color: '#6b7280', flexShrink: 0 }}>任务</span>
            <span style={{ color: '#111827', textAlign: 'right' }}>{taskLabel}</span>
          </div>
          <div style={rowStyle}>
            <span style={{ color: '#6b7280' }}>评测模式</span>
            <span style={{ color: '#111827', textAlign: 'right' }}>{display.modes}</span>
          </div>
          {taskType === 'cable_threading' ? (
            <div style={rowStyle}>
              <span style={{ color: '#6b7280' }}>支持策略</span>
              <span style={{ color: '#111827', textAlign: 'right' }}>{display.policies}</span>
            </div>
          ) : null}
          <div style={rowStyle}>
            <span style={{ color: '#6b7280' }}>支持 checkpoint</span>
            <span style={{ color: '#111827', textAlign: 'right' }}>{display.checkpoint}</span>
          </div>
          <div style={rowStyle}>
            <span style={{ color: '#6b7280' }}>支持训练模型评测</span>
            <span style={{ color: '#111827', textAlign: 'right' }}>{display.trainModel}</span>
          </div>
          <div style={rowStyle}>
            <span style={{ color: '#6b7280' }}>支持视频回放</span>
            <span style={{ color: '#111827', textAlign: 'right' }}>{display.video}</span>
          </div>
          <div style={rowStyle}>
            <span style={{ color: '#6b7280' }}>结果产物</span>
            <span style={{ color: '#111827', textAlign: 'right', fontSize: 12 }}>{display.resultArtifact}</span>
          </div>
          <div style={rowStyle}>
            <span style={{ color: '#6b7280' }}>当前接入状态</span>
            <span
              style={{
                color: display.phase2Pending ? '#b45309' : '#059669',
                textAlign: 'right',
                fontWeight: 500,
              }}
            >
              {display.status}
            </span>
          </div>
          <p style={{ margin: '10px 0 0', fontSize: 12, color: '#4b5563', lineHeight: 1.6 }}>
            {display.description}
          </p>
          {capabilities?.description && taskType === 'cable_threading' ? (
            <p style={{ margin: '8px 0 0', fontSize: 11, color: '#9ca3af', lineHeight: 1.55 }}>
              {capabilities.description}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

import { isCableThreadingTask } from '@/lib/workspace/cableThreading';
import { isDualArmCableTask } from '@/lib/workspace/dualArmCable';

export function resolveEvaluationTaskType(template: string): EvaluationTaskType | null {
  if (template === 'cable_threading_single_arm' || isCableThreadingTask(template)) {
    return 'cable_threading';
  }
  if (template === 'dual_arm_cable_manipulation' || isDualArmCableTask(template)) {
    return 'dual_arm_cable_manipulation';
  }
  return null;
}
