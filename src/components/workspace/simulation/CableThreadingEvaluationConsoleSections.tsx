'use client';

import { CableThreadingLiveFrame } from '@/components/workspace/simulation/CableThreadingLiveFrame';
import { CableThreadingVideoPlayer } from '@/components/workspace/replay/CableThreadingVideoPlayer';
import { EvaluationReplayMetricsBlock } from '@/components/workspace/replay/EvaluationReplayMetricsBlock';
import { WS } from '@/components/workspace/workspaceUi';
import type {
  CableThreadingEvalConsoleViewModel,
  CableThreadingEvalRunDetailRow,
} from '@/lib/workspace/cableThreadingEvaluationRunAdapter';

const card = { ...WS.card, padding: 16 } as const;

function DetailRow({ row }: { row: CableThreadingEvalRunDetailRow }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: 10,
        fontSize: 11,
        lineHeight: 1.45,
        marginBottom: 4,
      }}
    >
      <span style={{ color: '#9ca3af', flexShrink: 0 }}>{row.label}</span>
      <span
        style={{
          color: '#111827',
          textAlign: 'right',
          wordBreak: 'break-all',
          overflowWrap: 'anywhere',
          maxWidth: '58%',
          minWidth: 0,
          fontFamily: row.mono ? 'ui-monospace, monospace' : undefined,
          fontSize: row.mono ? 10 : 11,
        }}
      >
        {row.value}
      </span>
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

export function CableThreadingEvaluationViewport({
  viewport,
}: {
  viewport: CableThreadingEvalConsoleViewModel['viewport'];
}) {
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
          marginBottom: 8,
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: '#374151' }}>MuJoCo 评测画面</span>
      </div>

      <div
        style={{
          width: '100%',
          aspectRatio: '16 / 9',
          position: 'relative',
          borderRadius: 8,
          overflow: 'hidden',
          background: '#020617',
        }}
      >
        {viewport.jobStatus === 'completed' && viewport.evalVideoExists ? (
          <div style={{ width: '100%', height: '100%' }}>
            <CableThreadingVideoPlayer videoJobId={viewport.frameJobId} />
          </div>
        ) : viewport.hasLiveFrame ? (
          <CableThreadingLiveFrame
            jobId={viewport.frameJobId}
            status={viewport.frameStatus}
            frameCount={viewport.frameCount}
            embedded
          />
        ) : (
          <div
            style={{
              width: '100%',
              height: '100%',
              background: '#0f172a',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 24,
              textAlign: 'center',
              fontSize: 13,
              color: '#94a3b8',
              lineHeight: 1.6,
            }}
          >
            {viewport.waitingMessage || '正在初始化 MuJoCo 场景…'}
          </div>
        )}
      </div>
    </div>
  );
}

export function CableThreadingEvaluationStatusPanel({
  view,
}: {
  view: CableThreadingEvalConsoleViewModel;
}) {
  return (
    <div
      style={{
        ...card,
        padding: '10px 12px',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignSelf: 'stretch',
        height: '100%',
        minHeight: 0,
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
        <PanelSection title="基础信息" first>
          {view.runDetailRows.map((row) => (
            <DetailRow key={row.label} row={row} />
          ))}
        </PanelSection>

        <PanelSection title="运行进度">
          {view.progressRows.map((row) => (
            <DetailRow key={row.label} row={row} />
          ))}
        </PanelSection>

        <PanelSection title="评测指标">
          <EvaluationReplayMetricsBlock {...view.evaluationMetrics} />
        </PanelSection>

        {view.results ? (
          <PanelSection title="评测结果">
            <DetailRow row={{ label: '成功率', value: view.results.successRate }} />
            <DetailRow row={{ label: 'ever success rate', value: view.results.everSuccessRate }} />
            {view.results.evalCsvPath ? (
              <DetailRow row={{ label: 'eval.csv', value: view.results.evalCsvPath, mono: true }} />
            ) : null}
            {view.results.resultsJsonPath ? (
              <DetailRow
                row={{ label: 'eval.results.json', value: view.results.resultsJsonPath, mono: true }}
              />
            ) : null}
            {view.results.failuresJsonPath ? (
              <DetailRow
                row={{ label: 'eval.failures.json', value: view.results.failuresJsonPath, mono: true }}
              />
            ) : null}
            {view.results.logPath ? (
              <DetailRow row={{ label: 'run.log', value: view.results.logPath, mono: true }} />
            ) : null}
          </PanelSection>
        ) : null}
      </div>
    </div>
  );
}
