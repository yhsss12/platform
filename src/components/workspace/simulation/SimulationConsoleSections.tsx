'use client';

import { useEffect } from 'react';
import type {
  CurrentSimulation,
  NodeStatusItem,
  ObjectStatusItem,
  ProcessPrediction,
  RecentSimulationRun,
  RobotStatusItem,
  SimulationEventLog,
  SimulationRunStatus,
  SimulationStep,
} from '@/lib/mock/workspaceSimulationMock';
import {
  simulationStatusLabel,
} from '@/lib/mock/workspaceSimulationMock';
import { ModalCloseButton } from '@/components/common/ModalCloseButton';
import { StatusBadge, WS } from '@/components/workspace/workspaceUi';
import type { SimulationConsoleMode } from '@/lib/workspace/simulationConsole';
import {
  buildPhysicsProxyRuntimeState,
  physicsProxyAcceleratedModeLabel,
  physicsProxyModeLabel,
  type PhysicsProxyMode,
} from '@/lib/mock/physicsProxiesMock';

const card = { ...WS.card, padding: 16 } as const;

const evaluationStatusStripStyle: React.CSSProperties = {
  ...WS.card,
  display: 'flex',
  alignItems: 'center',
  flexWrap: 'wrap',
  gap: '6px 10px',
  padding: '12px 14px',
  borderRadius: 8,
  marginTop: 4,
  marginBottom: 12,
  width: '100%',
  boxSizing: 'border-box',
};

const compactPrimaryBtn: React.CSSProperties = {
  padding: '5px 10px',
  fontSize: 12,
  fontWeight: 500,
  borderRadius: 6,
  border: 'none',
  backgroundColor: '#2563eb',
  color: '#fff',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
};

const compactSecondaryBtn: React.CSSProperties = {
  padding: '5px 9px',
  fontSize: 12,
  fontWeight: 500,
  borderRadius: 6,
  border: '1px solid #d1d5db',
  backgroundColor: '#fff',
  color: '#374151',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
};

const metaMuted: React.CSSProperties = { color: '#9ca3af' };

function runStatusBadge(status: SimulationRunStatus | 'completed' | 'failed', label?: string) {
  const map: Record<string, 'running' | 'paused' | 'idle' | 'completed' | 'failed'> = {
    running: 'running',
    paused: 'paused',
    idle: 'idle',
    completed: 'completed',
    failed: 'failed',
  };
  const key = map[status] ?? 'idle';
  const text =
    label ??
    (status === 'completed'
      ? '已完成'
      : status === 'failed'
        ? '失败'
        : simulationStatusLabel[status as SimulationRunStatus] ?? status);
  return <StatusBadge status={key} label={text} />;
}

export type ConsoleRunContext = {
  mode: SimulationConsoleMode | null;
  modelVersion?: string;
  evalRounds?: string;
  generationCount?: string;
  simEnvironment?: string;
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string;
};

export type ConsoleSummaryOverride = {
  modeTitle?: string;
  taskLabel?: string;
  statusLabel?: string;
  progressText?: string;
  showProgressBar?: boolean;
  progressPercent?: number;
  metaItems?: { label: string; value: string }[];
};

export type EvaluationControlOptions = {
  disablePauseStopReset?: boolean;
  disableViewEvaluation?: boolean;
  disableViewEvaluationTitle?: string;
  onViewEvaluation?: () => void;
  viewEvaluationLabel?: string;
  disableViewReplay?: boolean;
  disableViewReplayTitle?: string;
  onViewReplay?: () => void;
};

function runStatusLabelForMode(
  runStatus: SimulationRunStatus,
  mode: SimulationConsoleMode | null
): string {
  if (mode === 'evaluation' && runStatus === 'running') return '评测中';
  if (runStatus === 'completed') return '已完成';
  if (runStatus === 'failed') return '失败';
  return simulationStatusLabel[runStatus] ?? runStatus;
}

export function SimulationTaskSummaryBar({
  sim,
  runStatus,
  context,
  onControl,
  onViewData,
  onViewEvaluation,
  onViewLogs,
  onViewRecords,
  summaryOverride,
  evaluationControlOptions,
  compact = false,
}: {
  sim: CurrentSimulation;
  runStatus: SimulationRunStatus;
  context: ConsoleRunContext;
  onControl: (action: string) => void;
  onViewData: () => void;
  onViewEvaluation: () => void;
  onViewLogs: () => void;
  onViewRecords: () => void;
  summaryOverride?: ConsoleSummaryOverride;
  evaluationControlOptions?: EvaluationControlOptions;
  /** 评测运行页精简：仅保留任务名与状态，不展示摘要元数据与顶部操作按钮 */
  compact?: boolean;
}) {
  const { mode } = context;
  const modeTitle =
    summaryOverride?.modeTitle ??
    (mode === 'data-generation' ? '数据生成运行' : mode === 'evaluation' ? '策略评测运行' : null);
  const statusLabel = summaryOverride?.statusLabel ?? runStatusLabelForMode(runStatus, mode);
  const simEnv = context.simEnvironment ?? 'MuJoCo';
  const taskLabel = summaryOverride?.taskLabel ?? sim.taskName;
  const progressText =
    summaryOverride?.progressText ??
    (summaryOverride?.showProgressBar === false ? '评测运行中' : `${sim.progressPercent}%`);
  const showProgressBar = summaryOverride?.showProgressBar ?? true;
  const progressPercent = summaryOverride?.progressPercent ?? sim.progressPercent;

  const metaItems =
    summaryOverride?.metaItems ??
    (mode === 'data-generation'
      ? [
          { label: '场景', value: sim.scene },
          { label: '机器人', value: sim.robot },
          { label: '仿真环境', value: simEnv },
          { label: '生成数量', value: context.generationCount ?? '50 条' },
          { label: '当前阶段', value: sim.currentStepLabel },
        ]
      : mode === 'evaluation'
        ? [
            { label: '场景', value: sim.scene },
            { label: '机器人', value: sim.robot },
            { label: '模型版本', value: context.modelVersion ?? 'ckpt-screw-act-50-e80' },
            { label: '评测环境', value: simEnv },
            { label: '评测次数', value: context.evalRounds ?? '—' },
            { label: '当前阶段', value: sim.currentStepLabel },
          ]
        : [
            { label: '场景', value: sim.scene },
            { label: '机器人', value: sim.robot },
            { label: '策略', value: sim.policy },
            { label: '运行时长', value: sim.runDuration },
            { label: '当前阶段', value: sim.currentStepLabel },
          ]);

  const controlButtons =
    mode === 'data-generation' ? (
      <>
        <button type="button" style={compactPrimaryBtn} onClick={() => onControl('启动')}>
          启动
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={() => onControl('暂停')}>
          暂停
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={() => onControl('停止')}>
          停止
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={() => onControl('重置')}>
          重置
        </button>
        <span style={{ width: 1, height: 18, backgroundColor: '#e5e7eb' }} />
        <button type="button" style={compactSecondaryBtn} onClick={onViewData}>
          查看数据
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={onViewLogs}>
          查看日志
        </button>
      </>
    ) : mode === 'evaluation' ? (
      <>
        <button
          type="button"
          style={{
            ...compactSecondaryBtn,
            opacity: evaluationControlOptions?.disablePauseStopReset ? 0.45 : 1,
            cursor: evaluationControlOptions?.disablePauseStopReset ? 'not-allowed' : 'pointer',
          }}
          disabled={evaluationControlOptions?.disablePauseStopReset}
          title={
            evaluationControlOptions?.disablePauseStopReset
              ? '当前任务暂无可用控制动作'
              : undefined
          }
          onClick={() => {
            if (!evaluationControlOptions?.disablePauseStopReset) onControl('暂停');
          }}
        >
          暂停
        </button>
        <button
          type="button"
          style={{
            ...compactSecondaryBtn,
            opacity: evaluationControlOptions?.disablePauseStopReset ? 0.45 : 1,
            cursor: evaluationControlOptions?.disablePauseStopReset ? 'not-allowed' : 'pointer',
          }}
          disabled={evaluationControlOptions?.disablePauseStopReset}
          title={
            evaluationControlOptions?.disablePauseStopReset
              ? '当前任务暂无可用控制动作'
              : undefined
          }
          onClick={() => {
            if (!evaluationControlOptions?.disablePauseStopReset) onControl('停止');
          }}
        >
          停止
        </button>
        <button
          type="button"
          style={{
            ...compactSecondaryBtn,
            opacity: evaluationControlOptions?.disablePauseStopReset ? 0.45 : 1,
            cursor: evaluationControlOptions?.disablePauseStopReset ? 'not-allowed' : 'pointer',
          }}
          disabled={evaluationControlOptions?.disablePauseStopReset}
          title={
            evaluationControlOptions?.disablePauseStopReset
              ? '当前任务暂无可用控制动作'
              : undefined
          }
          onClick={() => {
            if (!evaluationControlOptions?.disablePauseStopReset) onControl('重置');
          }}
        >
          重置
        </button>
        <span style={{ width: 1, height: 18, backgroundColor: '#e5e7eb' }} />
        <button
          type="button"
          style={{
            ...compactPrimaryBtn,
            opacity: evaluationControlOptions?.disableViewEvaluation ? 0.45 : 1,
            cursor: evaluationControlOptions?.disableViewEvaluation ? 'not-allowed' : 'pointer',
          }}
          disabled={evaluationControlOptions?.disableViewEvaluation}
          title={evaluationControlOptions?.disableViewEvaluationTitle}
          onClick={() => {
            if (evaluationControlOptions?.disableViewEvaluation) return;
            if (evaluationControlOptions?.onViewEvaluation) {
              evaluationControlOptions.onViewEvaluation();
              return;
            }
            onViewEvaluation();
          }}
        >
          {evaluationControlOptions?.viewEvaluationLabel ?? '查看报告'}
        </button>
        <button
          type="button"
          style={{
            ...compactSecondaryBtn,
            opacity: evaluationControlOptions?.disableViewReplay ? 0.45 : 1,
            cursor: evaluationControlOptions?.disableViewReplay ? 'not-allowed' : 'pointer',
          }}
          disabled={evaluationControlOptions?.disableViewReplay}
          title={evaluationControlOptions?.disableViewReplayTitle}
          onClick={() => {
            if (evaluationControlOptions?.disableViewReplay) return;
            if (evaluationControlOptions?.onViewReplay) {
              evaluationControlOptions.onViewReplay();
            }
          }}
        >
          查看回放
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={onViewLogs}>
          查看日志
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={onViewRecords}>
          查看评测记录
        </button>
      </>
    ) : (
      <>
        <button type="button" style={compactPrimaryBtn} onClick={() => onControl('启动')}>
          启动
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={() => onControl('暂停')}>
          暂停
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={() => onControl('停止')}>
          停止
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={() => onControl('重置')}>
          重置
        </button>
        <span style={{ width: 1, height: 18, backgroundColor: '#e5e7eb' }} />
        <button type="button" style={compactSecondaryBtn} onClick={onViewData}>
          查看数据
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={onViewEvaluation}>
          查看评测
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={onViewLogs}>
          查看日志
        </button>
        <button type="button" style={compactSecondaryBtn} onClick={onViewRecords}>
          查看记录
        </button>
      </>
    );

  return (
    <div
      style={
        compact
          ? evaluationStatusStripStyle
          : {
              ...card,
              padding: '10px 14px',
              marginBottom: 12,
              minHeight: 72,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }
      }
    >
      <div style={compact ? undefined : { flex: 1, minWidth: 0 }}>
        {!compact && modeTitle ? (
          <div style={{ fontSize: 12, fontWeight: 600, color: '#2563eb', marginBottom: 4 }}>{modeTitle}</div>
        ) : null}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '6px 10px',
            marginBottom: compact ? 0 : 4,
          }}
        >
          <span style={{ fontSize: 11, color: '#9ca3af' }}>当前任务</span>
          <span style={{ fontSize: 15, fontWeight: 600, color: '#111827' }}>{taskLabel}</span>
          {runStatusBadge(runStatus, statusLabel)}
          {!compact ? (
            <span style={{ fontSize: 12, color: '#2563eb', fontWeight: 600 }}>{progressText}</span>
          ) : null}
        </div>
        {!compact ? (
          <div
            style={{
              fontSize: 11,
              color: '#4b5563',
              lineHeight: 1.45,
              display: 'flex',
              flexWrap: 'wrap',
              gap: '2px 12px',
            }}
          >
            {metaItems.map((item) => (
              <span key={item.label}>
                <span style={metaMuted}>{item.label} </span>
                {item.value}
              </span>
            ))}
          </div>
        ) : null}
        {!compact && showProgressBar ? (
          <div
            style={{
              marginTop: 6,
              maxWidth: 420,
              height: 3,
              borderRadius: 2,
              backgroundColor: '#e5e7eb',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${progressPercent}%`,
                height: '100%',
                backgroundColor: '#2563eb',
                borderRadius: 2,
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        ) : null}
      </div>

      {!compact ? (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
            alignItems: 'center',
            justifyContent: 'flex-end',
            flexShrink: 0,
            maxWidth: '48%',
          }}
        >
          {controlButtons}
        </div>
      ) : null}
    </div>
  );
}

export function SimulationViewport({
  sim,
  mode,
  physicsProxyMode = 'off',
}: {
  sim: CurrentSimulation;
  mode: SimulationConsoleMode | null;
  physicsProxyMode?: PhysicsProxyMode;
}) {
  const backendLabel = physicsProxyAcceleratedModeLabel(physicsProxyMode);
  const cornerLabel =
    mode === 'data-generation'
      ? `${backendLabel} 数据生成运行`
      : mode === 'evaluation'
        ? `${backendLabel} 策略评测运行`
        : `${backendLabel} 运行视图`;

  const placeholderTitle =
    mode === 'data-generation'
      ? '仿真环境运行中'
      : mode === 'evaluation'
        ? '策略评测运行中'
        : '运行视图';

  const placeholderSubtitle =
    mode === 'data-generation'
      ? '正在采集运行轨迹与状态数据'
      : mode === 'evaluation'
        ? '正在加载模型版本并执行任务评测'
        : `场景：${sim.scene} · 机器人：${sim.robot}`;

  return (
    <div
      style={{
        ...card,
        padding: '10px 12px',
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        background: '#fff',
        border: '1px solid #e5e7eb',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 8,
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: '#374151' }}>3D 仿真视图</span>
        <span style={{ fontSize: 10, color: '#9ca3af' }}>{cornerLabel}</span>
      </div>

      <div
        style={{
          width: '100%',
          aspectRatio: '16 / 9',
          minHeight: 420,
          maxHeight: 'calc(100vh - 260px)',
          position: 'relative',
          borderRadius: 8,
          overflow: 'hidden',
          background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
          border: '1px solid #334155',
          color: '#e2e8f0',
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundImage: `
              linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px)
            `,
            backgroundSize: '28px 28px',
          }}
        >
          <div style={{ textAlign: 'center', padding: 16 }}>
            <div style={{ fontSize: 14, fontWeight: 500, color: '#cbd5e1', marginBottom: 6 }}>
              {placeholderTitle}
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>{placeholderSubtitle}</div>
          </div>
        </div>
        <div
          style={{
            position: 'absolute',
            bottom: 10,
            left: 12,
            right: 12,
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 10,
            fontFamily: 'ui-monospace, monospace',
            color: '#64748b',
            zIndex: 1,
          }}
        >
          <span>仿真时间 {sim.simTime}</span>
          <span>帧 #{sim.frame}</span>
        </div>
      </div>
    </div>
  );
}

/** 3D 视图 + 运行状态面板主区域 */
export function SimulationConsoleMain({
  sim,
  mode,
  robots,
  objects,
  nodes,
  prediction,
  steps,
  physicsProxyMode = 'off',
  physicsProxyModel,
}: {
  sim: CurrentSimulation;
  mode: SimulationConsoleMode | null;
  robots: RobotStatusItem[];
  objects: ObjectStatusItem[];
  nodes: NodeStatusItem[];
  prediction: ProcessPrediction;
  steps: SimulationStep[];
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr) 340px',
        gap: 12,
        alignItems: 'stretch',
      }}
    >
      <SimulationViewport sim={sim} mode={mode} physicsProxyMode={physicsProxyMode} />
      <SimulationStatusPanel
        mode={mode}
        robots={robots}
        objects={objects}
        nodes={nodes}
        prediction={prediction}
        steps={steps}
        physicsProxyMode={physicsProxyMode}
        physicsProxyModel={physicsProxyModel}
      />
    </div>
  );
}

function PanelSection({
  title,
  children,
  first,
}: {
  title: string;
  children: React.ReactNode;
  first?: boolean;
}) {
  return (
    <div
      style={{
        paddingTop: first ? 0 : 8,
        marginTop: first ? 0 : 8,
        borderTop: first ? 'none' : '1px solid #f3f4f6',
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 600, color: '#9ca3af', marginBottom: 4 }}>{title}</div>
      {children}
    </div>
  );
}

const stepColors: Record<
  SimulationStep['status'],
  { bg: string; border: string; text: string; dot: string }
> = {
  completed: { bg: '#f0fdf4', border: '#bbf7d0', text: '#065f46', dot: '#10b981' },
  running: { bg: '#eff6ff', border: '#93c5fd', text: '#1e40af', dot: '#2563eb' },
  pending: { bg: '#f9fafb', border: '#e5e7eb', text: '#6b7280', dot: '#d1d5db' },
  error: { bg: '#fef2f2', border: '#fecaca', text: '#991b1b', dot: '#ef4444' },
};

/** @deprecated 使用 SimulationStatusPanel */
export function SimulationStatusPanels(props: {
  mode?: SimulationConsoleMode | null;
  robots: RobotStatusItem[];
  objects: ObjectStatusItem[];
  nodes: NodeStatusItem[];
  prediction: ProcessPrediction;
  steps: SimulationStep[];
}) {
  return <SimulationStatusPanel mode={props.mode ?? null} {...props} />;
}

function stepStatusText(status: SimulationStep['status']): string {
  switch (status) {
    case 'completed':
      return '已完成';
    case 'running':
      return '当前';
    case 'error':
      return '失败';
    default:
      return '待执行';
  }
}

function SimulationTaskStepsList({ steps }: { steps: SimulationStep[] }) {
  return (
    <PanelSection title="任务步骤">
      {steps.map((step, i) => {
        const c = stepColors[step.status];
        const isLast = i === steps.length - 1;
        return (
          <div
            key={step.id}
            style={{
              display: 'grid',
              gridTemplateColumns: '12px 1fr auto',
              gap: '0 8px',
              alignItems: 'start',
              minHeight: isLast ? undefined : 22,
            }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%' }}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: 999,
                  backgroundColor: c.dot,
                  marginTop: 3,
                  flexShrink: 0,
                }}
              />
              {!isLast ? (
                <div style={{ width: 1, flex: 1, minHeight: 12, backgroundColor: '#e5e7eb', marginTop: 2 }} />
              ) : null}
            </div>
            <span
              style={{
                fontSize: 11,
                color: step.status === 'running' ? '#2563eb' : step.status === 'pending' ? '#9ca3af' : '#374151',
                lineHeight: 1.4,
                paddingBottom: isLast ? 0 : 4,
              }}
            >
              {step.name}
            </span>
            <span
              style={{
                fontSize: 10,
                color: c.text,
                lineHeight: 1.4,
                flexShrink: 0,
                paddingBottom: isLast ? 0 : 4,
              }}
            >
              {stepStatusText(step.status)}
            </span>
          </div>
        );
      })}
    </PanelSection>
  );
}

function PhysicsProxyStatusSection({
  physicsProxyMode,
  physicsProxyModel,
}: {
  physicsProxyMode: PhysicsProxyMode;
  physicsProxyModel?: string;
}) {
  const runtime = buildPhysicsProxyRuntimeState(physicsProxyMode, physicsProxyModel ?? null);

  return (
    <PanelSection title="物理代理状态">
      {!runtime ? (
        <div style={{ fontSize: 11, color: '#9ca3af' }}>未启用</div>
      ) : (
        <div style={{ fontSize: 11, color: '#374151', lineHeight: 1.55 }}>
          <div>
            <span style={{ color: '#9ca3af' }}>加速模式 </span>
            <span style={{ fontWeight: 600 }}>
              {runtime.mode === 'hybrid' ? 'PINN Hybrid' : physicsProxyModeLabel(runtime.mode)}
            </span>
          </div>
          <div style={{ marginTop: 4 }}>
            <span style={{ color: '#9ca3af' }}>代理模型 </span>
            {runtime.modelId}
          </div>
          <div style={{ marginTop: 4 }}>
            <span style={{ color: '#9ca3af' }}>代理对象 </span>
            {runtime.physicalObjects}
          </div>
          <div style={{ marginTop: 4 }}>
            <span style={{ color: '#9ca3af' }}>当前预测 </span>
            {runtime.currentPrediction}
          </div>
          <div style={{ marginTop: 4 }}>
            <span style={{ color: '#9ca3af' }}>误差估计 </span>
            <span style={{ fontWeight: 600, color: '#059669' }}>{runtime.errorEstimate}</span>
          </div>
          <div style={{ marginTop: 4 }}>
            <span style={{ color: '#9ca3af' }}>加速倍率 </span>
            <span style={{ fontWeight: 600, color: '#2563eb' }}>{runtime.speedup}</span>
          </div>
          {runtime.reviewStatus !== '—' ? (
            <div style={{ marginTop: 4 }}>
              <span style={{ color: '#9ca3af' }}>复核状态 </span>
              {runtime.reviewStatus}
            </div>
          ) : null}
        </div>
      )}
    </PanelSection>
  );
}

export function SimulationStatusPanel({
  mode,
  robots,
  objects,
  nodes,
  prediction,
  steps,
  physicsProxyMode = 'off',
  physicsProxyModel,
}: {
  mode: SimulationConsoleMode | null;
  robots: RobotStatusItem[];
  objects: ObjectStatusItem[];
  nodes: NodeStatusItem[];
  prediction: ProcessPrediction;
  steps: SimulationStep[];
  physicsProxyMode?: PhysicsProxyMode;
  physicsProxyModel?: string;
}) {
  return (
    <div
      style={{
        ...card,
        padding: '10px 12px',
        width: 340,
        maxWidth: 340,
        display: 'flex',
        flexDirection: 'column',
        alignSelf: 'stretch',
        height: '100%',
        minHeight: 0,
        maxHeight: 'calc(100vh - 260px)',
      }}
    >
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: '#111827',
          paddingBottom: 8,
          borderBottom: '1px solid #e5e7eb',
          flexShrink: 0,
        }}
      >
        运行状态
      </div>

      <div style={{ overflowY: 'auto', flex: 1, minHeight: 0, paddingTop: 4 }}>
        <PanelSection title="机器人状态" first>
          {robots.map((r) => (
            <div key={r.name} style={{ marginBottom: 6, fontSize: 11, lineHeight: 1.4 }}>
              <span style={{ fontWeight: 600, color: '#111827' }}>{r.name}</span>
              <span style={{ color: '#2563eb', marginLeft: 4 }}>{r.state}</span>
              <div style={{ color: '#6b7280' }}>{r.detail}</div>
            </div>
          ))}
        </PanelSection>

        <PanelSection title="操作对象状态">
          {objects.map((o) => (
            <div
              key={o.name}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 6,
                fontSize: 11,
                marginBottom: 2,
                lineHeight: 1.3,
              }}
            >
              <span style={{ color: '#111827' }}>{o.name}</span>
              <span style={{ color: '#6b7280', textAlign: 'right', flexShrink: 0 }}>{o.state}</span>
            </div>
          ))}
        </PanelSection>

        {mode === 'data-generation' ? (
          <>
            <PanelSection title="数据生成进度">
              <div style={{ fontSize: 11, color: '#374151', lineHeight: 1.5 }}>
                <div>
                  <span style={{ color: '#9ca3af' }}>状态 </span>
                  <span style={{ color: '#2563eb', fontWeight: 600 }}>采集中</span>
                </div>
                <div style={{ marginTop: 4 }}>
                  <span style={{ color: '#9ca3af' }}>已采集样本数 </span>
                  <span style={{ fontWeight: 600 }}>34 / 50</span>
                </div>
              </div>
            </PanelSection>
            <PanelSection title="当前输出">
              <div style={{ fontSize: 11, color: '#6b7280', lineHeight: 1.6 }}>
                轨迹 · 视频 · 状态日志
              </div>
            </PanelSection>
          </>
        ) : mode === 'evaluation' ? (
          <PanelSection title="评测进度">
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '4px 8px',
                fontSize: 11,
              }}
            >
              <div>
                <span style={{ color: '#9ca3af' }}>进度 </span>
                <span style={{ fontWeight: 600, color: '#374151' }}>{prediction.progressPercent}%</span>
              </div>
              <div>
                <span style={{ color: '#9ca3af' }}>成功概率 </span>
                <span style={{ fontWeight: 600, color: '#059669' }}>
                  {prediction.successProbability.toFixed(2)}
                </span>
              </div>
              <div>
                <span style={{ color: '#9ca3af' }}>风险等级 </span>
                <span
                  style={{
                    fontWeight: 600,
                    color:
                      prediction.failureRisk === '低'
                        ? '#059669'
                        : prediction.failureRisk === '中'
                          ? '#d97706'
                          : '#dc2626',
                  }}
                >
                  {prediction.failureRisk}
                </span>
              </div>
            </div>
            <div style={{ marginTop: 4, fontSize: 11, color: '#6b7280', lineHeight: 1.35 }}>
              当前阶段：{prediction.currentPhase}
            </div>
          </PanelSection>
        ) : (
          <>
            <PanelSection title="节点">
              {nodes.map((n) => (
                <div key={n.name} style={{ marginBottom: 4, fontSize: 11, lineHeight: 1.4 }}>
                  <span style={{ fontWeight: 600, color: '#111827' }}>{n.name}</span>
                  <span style={{ color: '#059669', marginLeft: 4 }}>{n.state}</span>
                  <div style={{ color: '#9ca3af' }}>
                    CPU {n.cpu} · GPU {n.gpu}
                  </div>
                </div>
              ))}
            </PanelSection>
            <PanelSection title="过程预测">
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '4px 8px',
                  fontSize: 11,
                }}
              >
                <div>
                  <span style={{ color: '#9ca3af' }}>进度 </span>
                  <span style={{ fontWeight: 600, color: '#374151' }}>{prediction.progressPercent}%</span>
                </div>
                <div>
                  <span style={{ color: '#9ca3af' }}>成功 </span>
                  <span style={{ fontWeight: 600, color: '#374151' }}>
                    {prediction.successProbability.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span style={{ color: '#9ca3af' }}>风险 </span>
                  <span
                    style={{
                      fontWeight: 600,
                      color:
                        prediction.failureRisk === '低'
                          ? '#059669'
                          : prediction.failureRisk === '中'
                            ? '#d97706'
                            : '#dc2626',
                    }}
                  >
                    {prediction.failureRisk}
                  </span>
                </div>
              </div>
              <div style={{ marginTop: 4, fontSize: 11, color: '#6b7280', lineHeight: 1.35 }}>
                阶段：{prediction.currentPhase}
              </div>
            </PanelSection>
          </>
        )}

        <PhysicsProxyStatusSection
          physicsProxyMode={physicsProxyMode}
          physicsProxyModel={physicsProxyModel}
        />

        <SimulationTaskStepsList steps={steps} />
      </div>
    </div>
  );
}

const logStatusColor: Record<SimulationEventLog['status'], string> = {
  info: '#6b7280',
  success: '#059669',
  warning: '#d97706',
  error: '#dc2626',
};

function logStatusLabel(status: SimulationEventLog['status']) {
  return status === 'success'
    ? '成功'
    : status === 'warning'
      ? '警告'
      : status === 'error'
        ? '错误'
        : '信息';
}

function EventLogTableBody({ rows }: { rows: SimulationEventLog[] }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
      <thead>
        <tr style={{ backgroundColor: '#f9fafb', position: 'sticky', top: 0, zIndex: 1 }}>
          {['时间', '类型', '内容', '状态'].map((h) => (
            <th
              key={h}
              style={{
                padding: '8px 12px',
                textAlign: 'left',
                fontWeight: 600,
                color: '#374151',
                borderBottom: '1px solid #e5e7eb',
                whiteSpace: 'nowrap',
              }}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
            <td
              style={{
                padding: '8px 12px',
                fontFamily: 'ui-monospace, monospace',
                fontSize: 12,
                color: '#6b7280',
                whiteSpace: 'nowrap',
              }}
            >
              {row.time}
            </td>
            <td style={{ padding: '8px 12px', color: '#374151' }}>{row.type}</td>
            <td style={{ padding: '8px 12px', color: '#111827' }}>{row.content}</td>
            <td
              style={{
                padding: '8px 12px',
                fontSize: 12,
                fontWeight: 500,
                color: logStatusColor[row.status],
              }}
            >
              {logStatusLabel(row.status)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const drawerOverlay: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  backgroundColor: 'rgba(15, 23, 42, 0.4)',
  zIndex: 1500,
};

const drawerPanel: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 560,
  maxWidth: '100vw',
  backgroundColor: '#ffffff',
  boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.12)',
  zIndex: 1501,
  display: 'flex',
  flexDirection: 'column',
  borderLeft: '1px solid #e5e7eb',
};

export function SimulationEventLogDrawer({
  open,
  logs,
  extraLines,
  onClose,
}: {
  open: boolean;
  logs: SimulationEventLog[];
  extraLines?: SimulationEventLog[];
  onClose: () => void;
}) {
  const all = [...logs, ...(extraLines ?? [])];

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <div style={drawerOverlay} onClick={onClose} aria-hidden />
      <aside style={drawerPanel} role="dialog" aria-modal aria-labelledby="sim-log-drawer-title">
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 20px',
            borderBottom: '1px solid #e5e7eb',
            flexShrink: 0,
          }}
        >
          <h2 id="sim-log-drawer-title" style={{ margin: 0, fontSize: 17, fontWeight: 600, color: '#111827' }}>
            运行日志
          </h2>
          <ModalCloseButton onClick={onClose} />
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: '0 0 16px' }}>
          <EventLogTableBody rows={all} />
        </div>
      </aside>
    </>
  );
}

/** 保留供其它场景复用；默认不在仿真页主视图展示 */
export function SimulationEventLogTable({
  logs,
  extraLines,
}: {
  logs: SimulationEventLog[];
  extraLines?: SimulationEventLog[];
}) {
  const all = [...logs, ...(extraLines ?? [])];
  return (
    <div style={{ ...card, marginBottom: 16 }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 12 }}>运行日志</div>
      <div
        style={{
          border: '1px solid #e5e7eb',
          borderRadius: 8,
          overflow: 'hidden',
          maxHeight: 220,
          overflowY: 'auto',
        }}
      >
        <EventLogTableBody rows={all} />
      </div>
    </div>
  );
}

/** 保留供其它场景复用；默认不在仿真页主视图展示 */
export function SimulationRecentRunsTable({ runs }: { runs: RecentSimulationRun[] }) {
  return (
    <div style={{ ...card }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 12 }}>
        最近仿真记录
      </div>
      <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>
        共 {runs.length} 条记录，请前往实验中心查看。
      </p>
    </div>
  );
}
