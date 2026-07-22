'use client';

import type { CSSProperties, ReactNode } from 'react';

export type { SimulationViewportFramePhase } from '@/lib/workspace/simulationViewportFrameLogic';
export {
  buildSimulationFrameStatusLine,
  formatSimulationRoundPart,
  resolveCableThreadingFramePhase,
  resolveDualArmFramePhase,
  resolveIsaacBlockStackingFramePhase,
} from '@/lib/workspace/simulationViewportFrameLogic';

export const SIMULATION_VIEWPORT_MAX_HEIGHT = 620;

/** 16:9 仿真画面容器 — placeholder 与真实帧共用，避免切换时尺寸跳变 */
export const simulationViewportShellStyle: CSSProperties = {
  width: '100%',
  maxWidth: '100%',
  aspectRatio: '16 / 9',
  maxHeight: SIMULATION_VIEWPORT_MAX_HEIGHT,
  borderRadius: 12,
  overflow: 'hidden',
  background: '#0f172a',
  border: '1px solid #334155',
  position: 'relative',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  margin: '0 auto',
};

export function SimulationViewportShell({ children }: { children: ReactNode }) {
  return <div style={simulationViewportShellStyle}>{children}</div>;
}

function ViewportLoadingIcon() {
  return (
    <span
      aria-hidden
      style={{
        fontSize: 26,
        lineHeight: 1,
        color: '#64748b',
        opacity: 0.45,
        flexShrink: 0,
      }}
    >
      ◎
    </span>
  );
}

export type SimulationBackendKind = 'mujoco' | 'generic';

export function simulationInitMessage(backend: SimulationBackendKind = 'mujoco'): string {
  return backend === 'mujoco' ? '正在初始化 MuJoCo 场景。' : '正在初始化仿真场景。';
}

/** 帧未就绪时的统一初始化占位（204 / warming_up / waiting_valid_frame 等） */
export function SimulationViewportPlaceholder({
  backend = 'mujoco',
  message,
  embedded = false,
}: {
  backend?: SimulationBackendKind;
  message?: string;
  embedded?: boolean;
}) {
  const text = message ?? simulationInitMessage(backend);

  const content = (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        padding: 16,
        maxWidth: '100%',
      }}
    >
      <ViewportLoadingIcon />
      <span
        style={{
          color: '#94a3b8',
          fontSize: 13,
          lineHeight: 1.55,
          textAlign: 'center',
        }}
      >
        {text}
      </span>
    </div>
  );

  if (embedded) {
    return <div style={simulationViewportEmbeddedFillStyle}>{content}</div>;
  }

  return <SimulationViewportShell>{content}</SimulationViewportShell>;
}

export const simulationViewportMediaStyle: CSSProperties = {
  width: '100%',
  height: '100%',
  objectFit: 'contain',
  display: 'block',
};

export const simulationViewportEmbeddedFillStyle: CSSProperties = {
  width: '100%',
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: '#0f172a',
};

export function SimulationViewportImage({
  src,
  alt,
  embedded = false,
}: {
  src: string;
  alt: string;
  embedded?: boolean;
}) {
  const image = <img src={src} alt={alt} style={simulationViewportMediaStyle} />;
  if (embedded) return image;
  return <SimulationViewportShell>{image}</SimulationViewportShell>;
}

export function SimulationViewportMessage({
  children,
  embedded = false,
}: {
  children: ReactNode;
  embedded?: boolean;
}) {
  const message = (
    <span
      style={{
        color: '#94a3b8',
        fontSize: 13,
        lineHeight: 1.55,
        textAlign: 'center',
        padding: 16,
      }}
    >
      {children}
    </span>
  );
  if (embedded) {
    return <div style={simulationViewportEmbeddedFillStyle}>{message}</div>;
  }
  return <SimulationViewportShell>{message}</SimulationViewportShell>;
}
