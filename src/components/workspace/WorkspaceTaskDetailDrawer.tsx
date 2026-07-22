'use client';

import { useEffect } from 'react';
import type { WorkspaceTask } from '@/lib/mock/workspaceTasksMock';
import {
  formatWorkspaceTaskObjects,
  taskStatusBadgeStatus,
  workspaceTaskStatusLabel,
} from '@/lib/mock/workspaceTasksMock';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { StatusBadge } from '@/components/workspace/workspaceUi';

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.4)',
  zIndex: 1500,
};

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 440,
  maxWidth: '100vw',
  backgroundColor: '#ffffff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

const scrollBody: React.CSSProperties = {
  padding: '20px 24px 24px',
  overflow: 'auto',
  flex: 1,
  minHeight: 0,
};

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 500,
          color: '#6b7280',
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 14, color: '#111827', lineHeight: 1.5 }}>{children}</div>
    </div>
  );
}

interface WorkspaceTaskDetailDrawerProps {
  task: WorkspaceTask | null;
  onClose: () => void;
}

export function WorkspaceTaskDetailDrawer({ task, onClose }: WorkspaceTaskDetailDrawerProps) {
  useEffect(() => {
    if (!task) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [task, onClose]);

  if (!task) return null;

  return (
    <>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <aside style={panelStyle} role="dialog" aria-modal aria-labelledby="workspace-task-drawer-title">
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 12,
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            flexShrink: 0,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <h2
              id="workspace-task-drawer-title"
              style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#111827' }}
            >
              {task.name}
            </h2>
            <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280', fontFamily: 'monospace' }}>
              {task.id}
            </div>
            <div style={{ marginTop: 8 }}>
              <StatusBadge
                status={taskStatusBadgeStatus(task.status)}
                label={workspaceTaskStatusLabel[task.status]}
              />
            </div>
          </div>
          <ModalCloseButton onClick={onClose} />
        </div>

        <div style={scrollBody}>
          <DetailRow label="任务 ID">
            <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{task.id}</span>
          </DetailRow>
          {task.backendTaskType ? (
            <DetailRow label="后端标识">
              <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{task.backendTaskType}</span>
            </DetailRow>
          ) : null}
          <DetailRow label="任务描述">{task.description}</DetailRow>
          {task.backendTaskType === 'dual_arm_cable_manipulation' ? (
            <>
              <DetailRow label="英文名称">Dual-arm Cable Manipulation</DetailRow>
              <DetailRow label="仿真后端">MuJoCo</DetailRow>
              <DetailRow label="末端执行器">Robotiq 2F-85</DetailRow>
              <DetailRow label="感知模块">Mask2Former</DetailRow>
              <DetailRow label="操控流程">
                感知 → 抓取规划 → 双臂 pick-stretch-place → 安全释放
              </DetailRow>
              <DetailRow label="已验证能力">
                MuJoCo headless · 完整 episode · 过程视频 · 结果 JSON · live/latest.jpg
              </DetailRow>
              <DetailRow label="暂未接入">
                HDF5 数据集 · robomimic/ACT/DT/Diffusion Policy 训练 · checkpoint 策略评测
              </DetailRow>
            </>
          ) : null}
          <DetailRow label="任务目标">{task.goal}</DetailRow>
          <DetailRow label="场景域">{task.domain}</DetailRow>
          <DetailRow label="任务类型">{task.type}</DetailRow>
          <DetailRow label="关联场景">{task.scene}</DetailRow>
          <DetailRow label="初始状态">{task.initialState}</DetailRow>
          <DetailRow label="操作对象">{formatWorkspaceTaskObjects(task.objects, 8)}</DetailRow>
          <DetailRow label="推荐机器人">{task.robot}</DetailRow>
          <DetailRow label="推荐策略">{task.policy}</DetailRow>
          <DetailRow label="评测指标">{task.metrics.join(' · ')}</DetailRow>
          <DetailRow label="数据状态">
            {task.trajectoryCount > 0 ? `${task.trajectoryCount} 条轨迹` : task.dataStatus}
          </DetailRow>
          <DetailRow label="评测结果">
            {task.successRate ?? task.evaluationStatus}
          </DetailRow>
          <DetailRow label="最近运行">{task.lastRunTime}</DetailRow>
          <DetailRow label="创建人">{task.creator}</DetailRow>
          {task.averageSteps != null ? (
            <DetailRow label="平均步数">{task.averageSteps} 步</DetailRow>
          ) : null}
          <DetailRow label="预估时长">{task.estimatedDuration}</DetailRow>
          {task.tags.length > 0 ? (
            <DetailRow label="标签">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {task.tags.map((tag) => (
                  <span
                    key={tag}
                    style={{
                      padding: '2px 8px',
                      fontSize: 12,
                      borderRadius: 4,
                      backgroundColor: '#f3f4f6',
                      color: '#374151',
                      border: '1px solid #e5e7eb',
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </DetailRow>
          ) : null}
        </div>
      </aside>
    </>
  );
}
