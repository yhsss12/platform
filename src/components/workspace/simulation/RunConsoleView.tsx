'use client';

import { useRouter } from 'next/navigation';
import { CableThreadingLiveFrame } from '@/components/workspace/simulation/CableThreadingLiveFrame';
import { DualArmCableLiveFrame } from '@/components/workspace/simulation/DualArmCableLiveFrame';
import { IsaacLabLiveFrame } from '@/components/workspace/simulation/IsaacLabLiveFrame';
import {
  SimulationViewportMessage,
  SimulationViewportPlaceholder,
} from '@/components/workspace/simulation/SimulationViewport';
import {
  CollapsiblePanel,
  InfoRow,
  RunLogDrawer,
  RunSummaryCard,
  SideActionsRow,
  SidePanelSection,
  simConsoleCardStyle,
  SimulationRunConsoleLayout,
  SimulationViewportSection,
} from '@/components/workspace/simulation/SimulationRunConsoleLayout';
import type { RunConsoleViewModel } from '@/lib/workspace/runConsoleViewModel';

function RunConsoleViewport({
  vm,
  onFrameLoadedChange,
}: {
  vm: RunConsoleViewModel;
  onFrameLoadedChange: (loaded: boolean) => void;
}) {
  const { scene } = vm;

  if (scene.viewportMode === 'failed') {
    return (
      <SimulationViewportMessage>{scene.failedMessage ?? '仿真画面不可用，请查看日志。'}</SimulationViewportMessage>
    );
  }

  if (scene.viewportMode === 'cameras_disabled') {
    return (
      <SimulationViewportMessage>当前任务未启用相机输出，无法显示实时画面。</SimulationViewportMessage>
    );
  }

  if (scene.liveFrame) {
    const { kind, jobId, pollEnabled, status, frameCount, phase } = scene.liveFrame;
    if (kind === 'cable_threading') {
      return (
        <CableThreadingLiveFrame
          jobId={jobId}
          status={status === 'completed' ? 'completed' : 'running'}
          frameCount={frameCount}
          onFrameReadyChange={onFrameLoadedChange}
        />
      );
    }
    if (kind === 'dual_arm_cable') {
      return (
        <DualArmCableLiveFrame
          jobId={jobId}
          status={status === 'queued' ? 'queued' : status}
          phase={phase}
          onFrameReadyChange={onFrameLoadedChange}
        />
      );
    }
    if (kind === 'isaac_lab') {
      return (
        <>
          <IsaacLabLiveFrame
            jobId={jobId}
            enabled
            pollEnabled={pollEnabled}
            status={status}
            silentMode
            onFrameReadyChange={onFrameLoadedChange}
            onFrameUsableChange={onFrameLoadedChange}
          />
          {scene.viewportMode === 'init' ? (
            <SimulationViewportPlaceholder message={scene.initializingText} embedded />
          ) : null}
        </>
      );
    }
  }

  return <SimulationViewportPlaceholder message={scene.initializingText} />;
}

export function RunConsoleView({
  vm,
  logTail,
  logLoading,
  logDrawerOpen,
  onOpenLog,
  onCloseLog,
  onFrameLoadedChange,
}: {
  vm: RunConsoleViewModel;
  logTail: string;
  logLoading: boolean;
  logDrawerOpen: boolean;
  onOpenLog: () => void;
  onCloseLog: () => void;
  onFrameLoadedChange: (loaded: boolean) => void;
}) {
  const router = useRouter();

  return (
    <SimulationRunConsoleLayout
      summary={
        <RunSummaryCard
          taskName={vm.taskName}
          taskTypeLabel={vm.taskTypeLabel}
          runStatus={vm.status}
          progressPercent={vm.progress}
        />
      }
      viewport={
        <SimulationViewportSection
          frameStatusLine={vm.scene.frameStatusLine}
          accentColor={vm.scene.frameStatusAccent}
          backendLabel={vm.scene.backendLabel}
        >
          <RunConsoleViewport vm={vm} onFrameLoadedChange={onFrameLoadedChange} />
        </SimulationViewportSection>
      }
      sidePanel={
        <div style={simConsoleCardStyle}>
          <div style={{ fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 4 }}>任务配置</div>

          <SidePanelSection title="基本信息">
            {vm.sections.basicInfo.map((row) => (
              <InfoRow key={row.label} label={row.label} value={row.value} />
            ))}
          </SidePanelSection>

          <SidePanelSection title="资产配置">
            {vm.sections.assetConfig.map((row) => (
              <InfoRow key={row.label} label={row.label} value={row.value} />
            ))}
          </SidePanelSection>

          <SidePanelSection title="运行结果">
            {vm.sections.results.map((row) => (
              <InfoRow key={row.label} label={row.label} value={row.value} />
            ))}
          </SidePanelSection>

          <SideActionsRow
            onViewLog={onOpenLog}
            showViewDataRecord={vm.actions.showViewDataRecord}
            onViewDataRecord={() => router.push(vm.actions.backToDataCenterHref)}
          />

          {vm.sections.debug?.length ? (
            <CollapsiblePanel title="内部调试信息">
              {vm.sections.debug.map((row) => (
                <InfoRow key={row.label} label={row.label} value={row.value} />
              ))}
            </CollapsiblePanel>
          ) : null}
        </div>
      }
      logDrawer={
        <RunLogDrawer open={logDrawerOpen} logTail={logTail} loading={logLoading} onClose={onCloseLog} />
      }
    />
  );
}
