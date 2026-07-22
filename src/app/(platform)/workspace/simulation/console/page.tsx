'use client';

import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  SimulationConsoleMain,
  SimulationEventLogDrawer,
  SimulationTaskSummaryBar,
  type ConsoleRunContext,
} from '@/components/workspace/simulation/SimulationConsoleSections';
import {
  currentSimulation,
  dataGenerationSteps,
  evaluationConsoleSteps,
  eventLogs as initialEventLogs,
  nodeStatus,
  objectStatus,
  processPrediction,
  robotStatus,
  simulationSteps,
  type SimulationEventLog,
  type SimulationRunStatus,
} from '@/lib/mock/workspaceSimulationMock';
import {
  appendMockDataItem,
  completeDataGenerationItem,
  createDataItemFromSimulation,
  getActiveDataGenerationContext,
  getActiveDataGenerationItemId,
  getActiveSimulationRun,
  runToCurrentSimulation,
} from '@/lib/mock/workspaceMockFlowStore';
import {
  getSimulationConsolePageCopy,
  resolveSimulationConsoleMode,
} from '@/lib/workspace/simulationConsole';
import type { PhysicsProxyMode } from '@/lib/mock/physicsProxiesMock';
import { CableThreadingEvaluateConsole } from '@/components/workspace/simulation/CableThreadingEvaluateConsole';
import { CableThreadingGenerateConsole } from '@/components/workspace/simulation/CableThreadingGenerateConsole';
import { DualArmCableGenerateConsole } from '@/components/workspace/simulation/DualArmCableGenerateConsole';
import { IsaacLabGenerateConsole } from '@/components/workspace/simulation/IsaacLabGenerateConsole';
import { NutAssemblyGenerateConsole } from '@/components/workspace/simulation/NutAssemblyGenerateConsole';
import {
  ConsoleHeaderActions,
  type SimConsoleHeaderState,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import { isCableThreadingReplayMode } from '@/lib/workspace/cableThreading';
import { isDualArmCableReplayMode } from '@/lib/workspace/dualArmCable';
import { isIsaacBlockStackingReplayMode } from '@/lib/workspace/isaacBlockStacking';
import {
  isPendingLocalJobId,
  isValidCableThreadingGenerateJobId,
  isValidDualArmGenerateJobId,
  isValidIsaacGenerateJobId,
  isValidIsaacReplayJobId,
  isValidNutAssemblyGenerateJobId,
} from '@/lib/workspace/backendJobIds';
import { isNutAssemblyReplayMode } from '@/lib/workspace/nutAssembly';
import { resolveRunConsoleKind } from '@/lib/workspace/runConsoleAdapters';

function formatTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

const consoleErrorStyle: React.CSSProperties = {
  padding: '48px 24px',
  textAlign: 'center',
  color: '#6b7280',
  fontSize: 15,
};

function ConsoleJobIdError({
  message,
  backLabel,
  backHref,
}: {
  message: string;
  backLabel: string;
  backHref: string;
}) {
  const router = useRouter();
  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="运行控制台"
        actions={<SecondaryButton onClick={() => router.push(backHref)}>{backLabel}</SecondaryButton>}
      />
      <div style={consoleErrorStyle}>{message}</div>
    </ModulePageContainer>
  );
}

function RealJobConsoleShell({
  children,
}: {
  children: (props: { onHeaderStateChange: (state: SimConsoleHeaderState) => void }) => React.ReactNode;
}) {
  const router = useRouter();
  const [headerState, setHeaderState] = useState<SimConsoleHeaderState>({
    canViewReplay: false,
    openReplay: () => {},
  });

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="运行控制台"
        actions={
          <ConsoleHeaderActions
            canViewReplay={headerState.canViewReplay}
            onBackToData={() => router.push('/workspace/data')}
            onViewReplay={() => headerState.openReplay()}
          />
        }
      />
      {children({ onHeaderStateChange: setHeaderState })}
    </ModulePageContainer>
  );
}

function DataGenerationRunConsole({
  jobId,
  dataId,
  onHeaderStateChange,
}: {
  jobId: string;
  dataId?: string;
  onHeaderStateChange: (state: SimConsoleHeaderState) => void;
}) {
  const kind = resolveRunConsoleKind(jobId);

  if (kind === 'cable_threading') {
    return (
      <CableThreadingGenerateConsole
        jobId={jobId}
        dataId={dataId}
        onHeaderStateChange={onHeaderStateChange}
      />
    );
  }
  if (kind === 'dual_arm_cable') {
    return (
      <DualArmCableGenerateConsole
        jobId={jobId}
        dataId={dataId}
        onHeaderStateChange={onHeaderStateChange}
      />
    );
  }
  if (kind === 'isaac_block_stacking') {
    return <IsaacLabGenerateConsole jobId={jobId} onHeaderStateChange={onHeaderStateChange} />;
  }
  if (kind === 'nut_assembly') {
    return (
      <NutAssemblyGenerateConsole
        jobId={jobId}
        dataId={dataId}
        onHeaderStateChange={onHeaderStateChange}
      />
    );
  }

  return (
    <ConsoleJobIdError
      message={`无法识别数据生成 jobId：${jobId}。请从数据中心重新启动任务。`}
      backLabel="返回数据中心"
      backHref="/workspace/data"
    />
  );
}

function SimulationConsolePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const modeParam = searchParams.get('mode');
  const mode = resolveSimulationConsoleMode(modeParam);
  const pageCopy = getSimulationConsolePageCopy(modeParam);

  const taskTypeFromQuery = searchParams.get('taskType');
  const jobIdFromQuery = searchParams.get('jobId');
  const cableThreadingEvalId = searchParams.get('evalId');
  const taskFromQuery = searchParams.get('task');
  const dataIdFromQuery = searchParams.get('dataId');
  const checkpointFromQuery = searchParams.get('checkpoint');
  const backendFromQuery = searchParams.get('backend');
  const roundsFromQuery = searchParams.get('rounds');
  const physicsProxyModeFromQuery = (searchParams.get('physicsProxyMode') ?? 'off') as PhysicsProxyMode;
  const physicsProxyModelFromQuery = searchParams.get('physicsProxyModel') ?? undefined;

  const physicsProxyMode: PhysicsProxyMode =
    physicsProxyModeFromQuery === 'pinn' || physicsProxyModeFromQuery === 'hybrid'
      ? physicsProxyModeFromQuery
      : 'off';

  const [simBase, setSimBase] = useState(currentSimulation);
  const [runStatus, setRunStatus] = useState<SimulationRunStatus>(
    mode === 'data-generation' || mode === 'evaluation' ? 'running' : currentSimulation.status
  );
  const [userLogs, setUserLogs] = useState<SimulationEventLog[]>([]);
  const [logDrawerOpen, setLogDrawerOpen] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  useEffect(() => {
    const run = getActiveSimulationRun();
    if (run) {
      const sim = runToCurrentSimulation(run);
      setSimBase(sim);
      if (!mode) {
        setRunStatus(sim.status);
      }
    }
  }, [mode]);

  const showToast = useCallback((text: string) => {
    setToastMsg(text);
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  const appendUserLog = useCallback((content: string, type = '控制') => {
    setUserLogs((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}-${prev.length}`,
        time: formatTime(),
        type,
        content,
        status: 'info',
      },
    ]);
  }, []);

  const handleControl = useCallback(
    (action: string) => {
      appendUserLog(`用户操作：${action}`);
      showToast(`已执行：${action}`);
      if (action === '启动') setRunStatus('running');
      if (action === '暂停') setRunStatus('paused');
      if (action === '停止' || action === '重置') setRunStatus('idle');
    },
    [appendUserLog, showToast]
  );

  const handleViewData = useCallback(() => {
    if (mode === 'data-generation') {
      const itemId = dataIdFromQuery ?? getActiveDataGenerationItemId();
      if (itemId) {
        const ctx = getActiveDataGenerationContext();
        const updated = completeDataGenerationItem(itemId, {
          episodes: ctx?.episodes,
          seed: ctx?.seed,
        });
        if (updated) {
          appendUserLog(`数据生成完成：${updated.name}`, '数据');
          showToast(`数据「${updated.name}」已生成完成`);
          window.setTimeout(() => router.push('/workspace/data'), 800);
          return;
        }
      }
      const run = getActiveSimulationRun();
      if (run) {
        const item = createDataItemFromSimulation(run);
        appendMockDataItem(item);
        appendUserLog(`已写入数据：${item.name}`, '数据');
        showToast(`数据「${item.name}」已写入数据中心`);
        window.setTimeout(() => router.push('/workspace/data'), 800);
        return;
      }
    }
    router.push('/workspace/data');
  }, [appendUserLog, dataIdFromQuery, mode, router, showToast]);

  const handleViewEvaluation = useCallback(() => {
    router.push('/workspace/evaluation');
  }, [router]);

  const handleViewRecords = useCallback(() => {
    if (mode === 'evaluation') {
      router.push('/workspace/evaluation');
      return;
    }
    router.push('/workspace/experiments');
  }, [mode, router]);

  const runContext = useMemo<ConsoleRunContext>(
    () => ({
      mode,
      modelVersion: checkpointFromQuery ?? undefined,
      evalRounds: roundsFromQuery ? `${roundsFromQuery} 次` : undefined,
      generationCount: '50 条',
      simEnvironment: backendFromQuery?.toLowerCase() === 'mujoco' ? 'MuJoCo' : 'MuJoCo',
    }),
    [mode, checkpointFromQuery, backendFromQuery, roundsFromQuery]
  );

  const sim = useMemo(
    () => ({
      ...simBase,
      status: runStatus,
      taskName: taskFromQuery ?? simBase.taskName,
      currentStepLabel:
        mode === 'data-generation' ? '采集运行轨迹' : simBase.currentStepLabel,
      engine: 'MuJoCo',
    }),
    [simBase, runStatus, taskFromQuery, mode]
  );

  const steps = useMemo(() => {
    if (mode === 'data-generation') return dataGenerationSteps;
    if (mode === 'evaluation') return evaluationConsoleSteps;
    return simulationSteps;
  }, [mode]);

  const isCableThreadingEvalConsole =
    mode === 'evaluation' &&
    cableThreadingEvalId &&
    (isCableThreadingReplayMode(taskTypeFromQuery) ||
      cableThreadingEvalId.startsWith('ct_eval_'));

  if (isCableThreadingEvalConsole && cableThreadingEvalId) {
    return (
      <ModulePageContainer>
        <ModulePageHeader
          title="线缆穿杆策略评测"
          actions={
            <SecondaryButton onClick={() => router.push('/workspace/evaluation')}>
              返回评测中心
            </SecondaryButton>
          }
        />
        <CableThreadingEvaluateConsole evalJobId={cableThreadingEvalId} />
      </ModulePageContainer>
    );
  }

  const isDataGenerationConsole =
    mode === 'data-generation' ||
    (mode === 'replay' && isIsaacBlockStackingReplayMode(taskTypeFromQuery));

  if (isDataGenerationConsole && jobIdFromQuery) {
    const consoleKind = resolveRunConsoleKind(jobIdFromQuery);
    const taskTypeMatches =
      (consoleKind === 'cable_threading' && isCableThreadingReplayMode(taskTypeFromQuery)) ||
      (consoleKind === 'dual_arm_cable' && isDualArmCableReplayMode(taskTypeFromQuery)) ||
      (consoleKind === 'isaac_block_stacking' &&
        isIsaacBlockStackingReplayMode(taskTypeFromQuery)) ||
      (consoleKind === 'nut_assembly' && isNutAssemblyReplayMode(taskTypeFromQuery));

    if (!taskTypeMatches && mode === 'data-generation') {
      /* fall through to generic console */
    } else {
      if (isPendingLocalJobId(jobIdFromQuery)) {
        return (
          <ConsoleJobIdError
            message={`无效的后端 jobId：${jobIdFromQuery}。请从数据中心重新启动任务。`}
            backLabel="返回数据中心"
            backHref="/workspace/data"
          />
        );
      }

      const jobValid =
        (consoleKind === 'cable_threading' && isValidCableThreadingGenerateJobId(jobIdFromQuery)) ||
        (consoleKind === 'dual_arm_cable' && isValidDualArmGenerateJobId(jobIdFromQuery)) ||
        (consoleKind === 'isaac_block_stacking' &&
          (isValidIsaacGenerateJobId(jobIdFromQuery) || isValidIsaacReplayJobId(jobIdFromQuery))) ||
        (consoleKind === 'nut_assembly' && isValidNutAssemblyGenerateJobId(jobIdFromQuery));

      if (!jobValid) {
        const label =
          consoleKind === 'nut_assembly'
            ? '螺母装配'
            : consoleKind === 'dual_arm_cable'
            ? '双臂线缆'
            : consoleKind === 'isaac_block_stacking'
              ? 'Isaac Lab'
              : '线缆穿杆';
        return (
          <ConsoleJobIdError
            message={`无效的${label}后端 jobId：${jobIdFromQuery}。请从数据中心重新启动任务。`}
            backLabel="返回数据中心"
            backHref="/workspace/data"
          />
        );
      }

      return (
        <RealJobConsoleShell>
          {({ onHeaderStateChange }) => (
            <DataGenerationRunConsole
              jobId={jobIdFromQuery}
              dataId={dataIdFromQuery ?? undefined}
              onHeaderStateChange={onHeaderStateChange}
            />
          )}
        </RealJobConsoleShell>
      );
    }
  }

  if (isDataGenerationConsole && !jobIdFromQuery) {
    const missingLabel = isNutAssemblyReplayMode(taskTypeFromQuery)
      ? '螺母装配'
      : isDualArmCableReplayMode(taskTypeFromQuery)
      ? '双臂线缆'
      : isIsaacBlockStackingReplayMode(taskTypeFromQuery)
        ? 'Isaac Lab'
        : '单臂线缆穿杆';
    return (
      <ConsoleJobIdError
        message={`缺少${missingLabel}任务 jobId，无法打开运行记录。`}
        backLabel="返回数据中心"
        backHref="/workspace/data"
      />
    );
  }

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title={pageCopy.title}
        actions={
          <SecondaryButton onClick={() => router.push(pageCopy.backHref)}>
            {pageCopy.backLabel}
          </SecondaryButton>
        }
      />

      <SimulationTaskSummaryBar
        sim={sim}
        runStatus={runStatus}
        context={runContext}
        onControl={handleControl}
        onViewData={handleViewData}
        onViewEvaluation={handleViewEvaluation}
        onViewLogs={() => setLogDrawerOpen(true)}
        onViewRecords={handleViewRecords}
      />

      <SimulationConsoleMain
        sim={sim}
        mode={mode}
        robots={robotStatus}
        objects={objectStatus}
        nodes={nodeStatus}
        prediction={processPrediction}
        steps={steps}
        physicsProxyMode={physicsProxyMode}
        physicsProxyModel={physicsProxyModelFromQuery}
      />

      <SimulationEventLogDrawer
        open={logDrawerOpen}
        logs={initialEventLogs}
        extraLines={userLogs}
        onClose={() => setLogDrawerOpen(false)}
      />

      {toastMsg ? (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            borderRadius: 10,
            fontSize: 14,
            fontWeight: 500,
            zIndex: 1700,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            backgroundColor: 'rgba(17,24,39,0.92)',
            color: '#fff',
          }}
        >
          {toastMsg}
        </div>
      ) : null}
    </ModulePageContainer>
  );
}

export default function SimulationConsolePage() {
  return (
    <Suspense fallback={null}>
      <SimulationConsolePageContent />
    </Suspense>
  );
}
