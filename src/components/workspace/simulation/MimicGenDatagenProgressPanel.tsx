'use client';

import type { MimicGenProgressPanelViewModel } from '@/lib/workspace/runConsoleViewModel';

const panelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 12,
  padding: 20,
  height: '100%',
  overflow: 'auto',
  background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
};

const titleStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 600,
  color: '#111827',
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 12,
  fontSize: 13,
  color: '#374151',
};

const labelStyle: React.CSSProperties = { color: '#6b7280', flexShrink: 0 };

const logBoxStyle: React.CSSProperties = {
  marginTop: 4,
  padding: 12,
  borderRadius: 8,
  background: '#111827',
  color: '#e5e7eb',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
  fontSize: 11,
  lineHeight: 1.5,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  maxHeight: 220,
  overflow: 'auto',
};

function formatElapsed(seconds: number | null): string {
  if (seconds == null || Number.isNaN(seconds)) return '—';
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins > 0 ? `${mins} 分 ${secs} 秒` : `${secs} 秒`;
}

export function MimicGenDatagenProgressPanel({ vm }: { vm: MimicGenProgressPanelViewModel }) {
  return (
    <div style={panelStyle}>
      <div style={titleStyle}>
        {vm.isFailed ? 'MimicGen 数据生成失败' : '正在进行 MimicGen 数据生成'}
      </div>
      <div style={{ fontSize: 13, color: '#4b5563' }}>{vm.message}</div>

      <div style={rowStyle}>
        <span style={labelStyle}>当前阶段</span>
        <span>{vm.stageLabel}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>generationMode</span>
        <span>{vm.generationMode}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>policyMode</span>
        <span>{vm.policyMode}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>Source Demo</span>
        <span style={{ textAlign: 'right', wordBreak: 'break-all' }}>{vm.sourceDemoPath}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>已运行时间</span>
        <span>{formatElapsed(vm.elapsedSeconds)}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>lastHeartbeatAt</span>
        <span>{vm.lastHeartbeatAt ?? '—'}</span>
      </div>
      <div style={rowStyle}>
        <span style={labelStyle}>demo 写入数</span>
        <span>
          {vm.episodesGenerated} / {vm.episodesRequested}
          {vm.datagenFailedTrials !== '—' ? `（失败 trials: ${vm.datagenFailedTrials}）` : ''}
        </span>
      </div>

      {vm.isFailed && vm.errorMessage ? (
        <div style={{ fontSize: 13, color: '#b91c1c' }}>{vm.errorMessage}</div>
      ) : null}

      <div>
        <div style={{ ...labelStyle, fontSize: 12, marginBottom: 6 }}>generate.log 最近输出</div>
        <div style={logBoxStyle}>{vm.logTail || '（暂无日志输出）'}</div>
        {vm.logStaleHint ? (
          <div style={{ marginTop: 6, fontSize: 12, color: '#b45309' }}>{vm.logStaleHint}</div>
        ) : null}
      </div>

      {vm.isFailed && vm.traceback ? (
        <div>
          <div style={{ ...labelStyle, fontSize: 12, marginBottom: 6 }}>错误 traceback</div>
          <div style={logBoxStyle}>{vm.traceback}</div>
        </div>
      ) : null}

      <div style={{ fontSize: 12, color: '#6b7280' }}>{vm.completionHint}</div>

      {vm.replayHref ? (
        <a href={vm.replayHref} style={{ fontSize: 13, color: '#2563eb' }}>
          打开 HDF5 轨迹回放
        </a>
      ) : null}
    </div>
  );
}
