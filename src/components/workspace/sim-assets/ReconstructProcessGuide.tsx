'use client';

import type { CSSProperties } from 'react';
import { Box, CheckCircle2, Download, ScanLine, Upload } from 'lucide-react';

const STEPS = [
  { title: '上传图片', icon: Upload },
  { title: '框选目标', icon: ScanLine },
  { title: '确认分割结果', icon: CheckCircle2 },
  { title: '生成三维模型', icon: Box },
  { title: '导出仿真资产', icon: Download },
] as const;

function StepRow({
  index,
  title,
  icon: Icon,
  active,
  isLast,
}: {
  index: number;
  title: string;
  icon: typeof Upload;
  active: boolean;
  isLast: boolean;
}) {
  const dotStyle: CSSProperties = {
    width: 22,
    height: 22,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 11,
    fontWeight: 600,
    flexShrink: 0,
    background: active ? '#2563eb' : '#e2e8f0',
    color: active ? '#fff' : '#64748b',
    border: active ? '2px solid #2563eb' : '2px solid #e2e8f0',
    boxSizing: 'border-box',
  };

  const rowStyle: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '10px 12px',
    borderRadius: 10,
    background: active ? '#eff6ff' : 'transparent',
    border: active ? '1px solid #bfdbfe' : '1px solid transparent',
  };

  const iconBoxStyle: CSSProperties = {
    width: 36,
    height: 36,
    borderRadius: 8,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    background: active ? '#fff' : '#f1f5f9',
    border: `1px solid ${active ? '#bfdbfe' : '#e2e8f0'}`,
    color: active ? '#2563eb' : '#64748b',
  };

  return (
    <div style={{ display: 'flex', gap: 12 }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 22, flexShrink: 0 }}>
        <div style={dotStyle}>{index + 1}</div>
        {!isLast ? (
          <div
            style={{
              flex: 1,
              width: 2,
              minHeight: 20,
              marginTop: 4,
              background: '#e2e8f0',
              borderRadius: 1,
            }}
          />
        ) : null}
      </div>
      <div style={{ flex: 1, paddingBottom: isLast ? 0 : 6 }}>
        <div style={rowStyle}>
          <div style={iconBoxStyle}>
            <Icon size={18} strokeWidth={1.75} />
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, color: active ? '#1e40af' : '#334155' }}>{title}</div>
        </div>
      </div>
    </div>
  );
}

export function ReconstructProcessGuide({ activeStep = 0 }: { activeStep?: number }) {
  return (
    <div>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#334155', marginBottom: 16 }}>操作流程</div>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {STEPS.map((step, index) => (
          <StepRow
            key={step.title}
            index={index}
            title={step.title}
            icon={step.icon}
            active={index === activeStep}
            isLast={index === STEPS.length - 1}
          />
        ))}
      </div>
    </div>
  );
}
