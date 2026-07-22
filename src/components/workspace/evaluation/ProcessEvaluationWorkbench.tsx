'use client';

import type { ProcessEvaluation } from '@/lib/mock/workspaceEvaluationMock';
import { MockCurveChart } from '@/components/workspace/evaluation/MockCurveChart';
import { WS } from '@/components/workspace/workspaceUi';

const card: React.CSSProperties = {
  ...WS.card,
  padding: 16,
  height: '100%',
  boxSizing: 'border-box',
};

const stageColors: Record<ProcessEvaluation['stages'][0]['status'], { bg: string; text: string }> = {
  completed: { bg: '#f0fdf4', text: '#065f46' },
  running: { bg: '#eff6ff', text: '#1e40af' },
  pending: { bg: '#f9fafb', text: '#6b7280' },
  risk: { bg: '#fef2f2', text: '#991b1b' },
};

const severityColor: Record<'low' | 'medium' | 'high', string> = {
  low: '#6b7280',
  medium: '#d97706',
  high: '#dc2626',
};

export function ProcessEvaluationWorkbench({ eval_: evalData }: { eval_: ProcessEvaluation }) {
  const framePercent = Math.round((evalData.currentFrame / evalData.totalFrames) * 100);

  return (
    <div style={{ marginBottom: 24 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 12,
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: '#111827' }}>
            过程评测工作台
          </h3>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#6b7280' }}>
            基于轨迹、视频与任务描述进行逐帧进度与成功概率预测
          </p>
        </div>
        <span
          style={{
            fontSize: 12,
            padding: '4px 10px',
            borderRadius: 6,
            backgroundColor: '#eff6ff',
            color: '#1e40af',
            border: '1px solid #bfdbfe',
          }}
        >
          当前数据：{evalData.dataName}
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gridTemplateRows: 'auto auto',
          gap: 16,
        }}
      >
        {/* 左上：轨迹 / 视频 */}
        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 10 }}>
            轨迹 / 视频回放
          </div>
          <div
            style={{
              background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
              borderRadius: 8,
              border: '1px solid #334155',
              minHeight: 160,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#94a3b8',
              fontSize: 13,
              marginBottom: 12,
              position: 'relative',
            }}
          >
            <div style={{ textAlign: 'center', padding: 16 }}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>任务过程回放</div>
              <div style={{ color: '#cbd5e1' }}>{evalData.dataName}</div>
            </div>
            <div
              style={{
                position: 'absolute',
                bottom: 8,
                left: 12,
                right: 12,
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 11,
                fontFamily: 'ui-monospace, monospace',
              }}
            >
              <span>帧 {evalData.currentFrame} / {evalData.totalFrames}</span>
              <span>{evalData.timestamp}</span>
            </div>
          </div>
          <div style={{ fontSize: 12, color: '#4b5563', lineHeight: 1.7 }}>
            <div>
              <strong>任务：</strong>
              {evalData.taskName}
            </div>
            <div>
              <strong>场景：</strong>
              {evalData.scene} · <strong>机器人：</strong>
              {evalData.robot}
            </div>
            <div>
              <strong>策略：</strong>
              {evalData.policy} · <strong>采样：</strong>
              {evalData.sampleFps} FPS
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
              播放进度 {framePercent}%
            </div>
            <div style={{ height: 4, backgroundColor: '#e5e7eb', borderRadius: 2 }}>
              <div
                style={{
                  width: `${framePercent}%`,
                  height: '100%',
                  backgroundColor: '#2563eb',
                  borderRadius: 2,
                }}
              />
            </div>
          </div>
        </div>

        {/* 右上：预测曲线 */}
        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 12 }}>
            Progress &amp; Success Prediction
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <MockCurveChart
              title="进度预测 (Progress)"
              points={evalData.progressCurve}
              color="#2563eb"
              fillColor="rgba(37, 99, 235, 0.12)"
              yMax={1}
              currentPercent={evalData.progressPercent}
            />
            <MockCurveChart
              title="成功概率 (Success)"
              points={evalData.successCurve}
              color="#059669"
              fillColor="rgba(5, 150, 105, 0.12)"
              yMax={1}
              currentPercent={Math.round(evalData.successProbability * 100)}
            />
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 10,
              marginTop: 16,
              paddingTop: 12,
              borderTop: '1px solid #f3f4f6',
            }}
          >
            {[
              { label: '当前进度', value: `${evalData.progressPercent}%`, color: '#2563eb' },
              {
                label: '成功概率',
                value: evalData.successProbability.toFixed(2),
                color: '#059669',
              },
              { label: '失败风险', value: evalData.failureRisk, color: '#374151' },
              { label: '轨迹质量', value: evalData.trajectoryQuality, color: '#7c3aed' },
            ].map((m) => (
              <div key={m.label}>
                <div style={{ fontSize: 11, color: '#6b7280' }}>{m.label}</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: m.color }}>{m.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* 左下：任务描述 */}
        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 10 }}>
            任务描述与评测条件
          </div>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 4 }}>
              任务描述
            </div>
            <p style={{ margin: 0, fontSize: 13, color: '#374151', lineHeight: 1.55 }}>
              {evalData.taskDescription}
            </p>
          </div>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 4 }}>
              成功条件
            </div>
            <p style={{ margin: 0, fontSize: 13, color: '#374151', lineHeight: 1.55 }}>
              {evalData.successCondition}
            </p>
          </div>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>过程评测输入</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
            {evalData.inputs.map((inp) => (
              <span
                key={inp}
                style={{
                  padding: '4px 8px',
                  fontSize: 12,
                  borderRadius: 4,
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                }}
              >
                {inp}
              </span>
            ))}
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 8,
              fontSize: 12,
              color: '#4b5563',
            }}
          >
            <div>采样 FPS：{evalData.sampleFps}</div>
            <div>逐帧预测：{evalData.frameWisePrediction ? '是' : '否'}</div>
            <div>失败节点检测：{evalData.detectFailureNodes ? '是' : '否'}</div>
            <div>场景域：{evalData.domain}</div>
          </div>
        </div>

        {/* 右下：失败节点 */}
        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 10 }}>
            失败节点与关键帧
          </div>
          <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 8 }}>
            关键阶段
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14 }}>
            {evalData.stages.map((st) => {
              const c = stageColors[st.status];
              return (
                <div
                  key={st.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '8px 10px',
                    borderRadius: 6,
                    backgroundColor: c.bg,
                    fontSize: 12,
                  }}
                >
                  <span style={{ fontWeight: 500, color: c.text }}>{st.name}</span>
                  <span style={{ color: '#6b7280' }}>{st.frameRange ?? '—'}</span>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 8 }}>
            失败风险节点
          </div>
          <ul style={{ margin: '0 0 12px', padding: 0, listStyle: 'none' }}>
            {evalData.failureNodes.map((fn) => (
              <li
                key={fn.id}
                style={{
                  padding: '10px 0',
                  borderBottom: '1px solid #f3f4f6',
                  fontSize: 13,
                }}
              >
                <div style={{ fontWeight: 600, color: severityColor[fn.severity] }}>{fn.label}</div>
                {fn.frame ? (
                  <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 2 }}>{fn.frame}</div>
                ) : null}
                <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>{fn.description}</div>
              </li>
            ))}
          </ul>
          {evalData.anomalies.length > 0 ? (
            <>
              <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 6 }}>
                异常片段
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#b45309' }}>
                {evalData.anomalies.map((a) => (
                  <li key={a} style={{ marginBottom: 4 }}>
                    {a}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
