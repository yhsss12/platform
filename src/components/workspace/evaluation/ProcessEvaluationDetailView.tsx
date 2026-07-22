'use client';

import { useState } from 'react';
import type { ProcessEvaluationDetail } from '@/lib/mock/workspaceEvaluationMock';
import { MockCurveChart } from '@/components/workspace/evaluation/MockCurveChart';
import { WS } from '@/components/workspace/workspaceUi';

const quadrant: React.CSSProperties = {
  ...WS.card,
  padding: 14,
  display: 'flex',
  flexDirection: 'column',
  minHeight: 0,
  minWidth: 0,
};

const sectionTitle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: '#111827',
  marginBottom: 10,
  flexShrink: 0,
};

const metaRow: React.CSSProperties = {
  fontSize: 11,
  color: '#6b7280',
  lineHeight: 1.5,
};

const compactBtn: React.CSSProperties = {
  padding: '4px 10px',
  fontSize: 12,
  borderRadius: 6,
  border: '1px solid #475569',
  backgroundColor: '#1e293b',
  color: '#e2e8f0',
  cursor: 'pointer',
};

function ReplayQuadrant({ detail }: { detail: ProcessEvaluationDetail }) {
  const [playing, setPlaying] = useState(false);
  const progress = detail.progressPercent;

  return (
    <div style={quadrant}>
      <div style={sectionTitle}>轨迹 / 视频回放</div>
      <div style={{ ...metaRow, marginBottom: 8, display: 'flex', flexWrap: 'wrap', gap: '4px 14px' }}>
        <span>
          <span style={{ color: '#9ca3af' }}>数据 </span>
          <span style={{ fontFamily: 'ui-monospace, monospace', color: '#374151' }}>{detail.dataName}</span>
        </span>
        <span>
          <span style={{ color: '#9ca3af' }}>任务 </span>
          {detail.taskName}
        </span>
      </div>
      <div
        style={{
          width: '100%',
          aspectRatio: '16 / 9',
          minHeight: 200,
          maxHeight: 360,
          borderRadius: 8,
          overflow: 'hidden',
          background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
          border: '1px solid #334155',
          position: 'relative',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              border: '2px solid #64748b',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 22,
              color: '#94a3b8',
              backgroundColor: 'rgba(15, 23, 42, 0.6)',
            }}
          >
            {playing ? '❚❚' : '▶'}
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', textAlign: 'center', padding: '0 16px' }}>
            轨迹 / 仿真视频回放 · 评测输入
          </div>
          <div style={{ fontSize: 11, color: '#64748b' }}>
            {detail.scene} · {detail.robot}
          </div>
        </div>
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            fontSize: 10,
            color: '#64748b',
            fontFamily: 'ui-monospace, monospace',
          }}
        >
          {detail.policy}
        </div>
        <div
          style={{
            position: 'absolute',
            bottom: 8,
            left: 10,
            right: 10,
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 10,
            fontFamily: 'ui-monospace, monospace',
            color: '#64748b',
          }}
        >
          <span>
            帧 {detail.currentFrame} / {detail.totalFrames}
          </span>
          <span>{detail.timestamp}</span>
        </div>
      </div>

      <div style={{ marginTop: 10, flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
          <span>播放进度 {progress}%</span>
          <span>FPS {detail.sampleFps}</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={progress}
          readOnly
          style={{ width: '100%', marginBottom: 8, accentColor: '#2563eb' }}
        />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
          <button type="button" style={compactBtn} onClick={() => setPlaying((p) => !p)}>
            {playing ? '暂停' : '播放'}
          </button>
          <button type="button" style={compactBtn}>
            上一帧
          </button>
          <button type="button" style={compactBtn}>
            下一帧
          </button>
        </div>
      </div>
    </div>
  );
}

function PredictionQuadrant({ detail }: { detail: ProcessEvaluationDetail }) {
  return (
    <div style={quadrant}>
      <div style={sectionTitle}>Progress &amp; Success Prediction</div>
      <p style={{ ...metaRow, margin: '0 0 12px' }}>基于回放逐帧预测 · 评测输出</p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 10,
          marginBottom: 14,
        }}
      >
        <div style={{ padding: 10, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 11, color: '#6b7280' }}>当前进度</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#2563eb' }}>{detail.progressPercent}%</div>
        </div>
        <div style={{ padding: 10, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 11, color: '#6b7280' }}>成功概率</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#059669' }}>
            {detail.successProbability.toFixed(2)}
          </div>
        </div>
        <div style={{ padding: 10, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 11, color: '#6b7280' }}>失败风险</div>
          <div
            style={{
              fontSize: 16,
              fontWeight: 600,
              color:
                detail.failureRisk === '低'
                  ? '#059669'
                  : detail.failureRisk === '中'
                    ? '#d97706'
                    : '#dc2626',
            }}
          >
            {detail.failureRisk}
          </div>
        </div>
        <div style={{ padding: 10, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 11, color: '#6b7280' }}>轨迹质量</div>
          <div style={{ fontSize: 16, fontWeight: 600, color: '#374151' }}>{detail.trajectoryQuality}</div>
        </div>
      </div>

      <MockCurveChart
        title="进度预测 Progress"
        points={detail.progressCurve}
        color="#2563eb"
        fillColor="rgba(37, 99, 235, 0.12)"
        currentPercent={detail.progressPercent}
      />
      <div style={{ height: 12 }} />
      <MockCurveChart
        title="成功概率 Success"
        points={detail.successCurve}
        color="#059669"
        fillColor="rgba(5, 150, 105, 0.12)"
        currentPercent={detail.progressPercent}
      />
    </div>
  );
}

function TaskContextQuadrant({ detail }: { detail: ProcessEvaluationDetail }) {
  return (
    <div style={quadrant}>
      <div style={sectionTitle}>任务描述与评测条件</div>
      <div style={{ overflowY: 'auto', flex: 1, fontSize: 13, color: '#374151', lineHeight: 1.6 }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>任务描述</div>
          <p style={{ margin: 0 }}>{detail.taskDescription}</p>
        </div>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>成功条件</div>
          <p style={{ margin: 0 }}>{detail.successCondition}</p>
        </div>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>过程评测输入</div>
          <p style={{ margin: 0 }}>{detail.inputs.join(' / ')}</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
          <div>
            <span style={{ color: '#9ca3af' }}>采样 FPS：</span>
            {detail.sampleFps}
          </div>
          <div>
            <span style={{ color: '#9ca3af' }}>逐帧预测：</span>
            {detail.frameWisePrediction ? '是' : '否'}
          </div>
          <div>
            <span style={{ color: '#9ca3af' }}>失败节点检测：</span>
            {detail.detectFailureNodes ? '是' : '否'}
          </div>
          <div>
            <span style={{ color: '#9ca3af' }}>场景域：</span>
            {detail.domain}
          </div>
        </div>
      </div>
    </div>
  );
}

const severityColor = { low: '#6b7280', medium: '#d97706', high: '#dc2626' } as const;

function FailureQuadrant({ detail }: { detail: ProcessEvaluationDetail }) {
  return (
    <div style={quadrant}>
      <div style={sectionTitle}>失败节点与关键帧</div>
      <div style={{ overflowY: 'auto', flex: 1, fontSize: 12, color: '#374151' }}>
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 6 }}>关键阶段</div>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
            {detail.stages.map((s) => (
              <li key={s.id}>
                {s.name}：{s.frameRange ?? '待执行'}
              </li>
            ))}
          </ul>
        </div>
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 6 }}>失败风险节点</div>
          {detail.failureNodes.map((n) => (
            <div
              key={n.id}
              style={{
                marginBottom: 8,
                padding: '8px 10px',
                backgroundColor: '#f9fafb',
                borderRadius: 6,
                borderLeft: `3px solid ${severityColor[n.severity]}`,
              }}
            >
              <div style={{ fontWeight: 600, color: '#111827' }}>{n.label}</div>
              <div style={{ color: '#6b7280', marginTop: 2 }}>
                {n.frame}，{n.description}
              </div>
            </div>
          ))}
        </div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 6 }}>异常片段</div>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7, color: '#6b7280' }}>
            {detail.anomalies.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export function ProcessEvaluationDetailView({ detail }: { detail: ProcessEvaluationDetail }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
        gap: 12,
        alignItems: 'stretch',
      }}
    >
      <ReplayQuadrant detail={detail} />
      <PredictionQuadrant detail={detail} />
      <TaskContextQuadrant detail={detail} />
      <FailureQuadrant detail={detail} />
    </div>
  );
}
