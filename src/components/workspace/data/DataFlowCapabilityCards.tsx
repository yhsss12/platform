'use client';

import { useState } from 'react';
import { DatabaseZap, type LucideIcon } from 'lucide-react';

interface PremiumActionCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  onClick: () => void;
  glowColor: string;
}

function PremiumActionCard({
  title,
  description,
  icon: Icon,
  onClick,
  glowColor,
}: PremiumActionCardProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: 'relative',
        flex: 1,
        minWidth: 0,
        padding: '12px 16px',
        borderRadius: 14,
        border: `1px solid ${hovered ? '#93c5fd' : '#dbeafe'}`,
        background: `linear-gradient(135deg, #ffffff 0%, #f6faff 45%, #eef6ff 100%)`,
        boxShadow: hovered
          ? '0 10px 28px rgba(37, 99, 235, 0.1)'
          : '0 6px 20px rgba(15, 23, 42, 0.04)',
        cursor: 'pointer',
        textAlign: 'left',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
      }}
    >
      <div
        aria-hidden
        style={{
          pointerEvents: 'none',
          position: 'absolute',
          top: -20,
          right: -20,
          width: 80,
          height: 80,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`,
          opacity: hovered ? 0.9 : 0.65,
          transition: 'opacity 0.2s ease',
        }}
      />

      <div
        style={{
          position: 'relative',
          flexShrink: 0,
          width: 36,
          height: 36,
          borderRadius: 9,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: hovered
            ? 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)'
            : 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
          border: `1px solid ${hovered ? '#93c5fd' : '#bfdbfe'}`,
          color: hovered ? '#1d4ed8' : '#2563eb',
          transition: 'background 0.2s ease, border-color 0.2s ease, color 0.2s ease',
        }}
      >
        <Icon size={17} strokeWidth={1.75} />
      </div>

      <div style={{ position: 'relative', flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 3 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a', lineHeight: 1.35 }}>{title}</div>
        <p
          style={{
            margin: 0,
            fontSize: 12,
            color: '#64748b',
            lineHeight: 1.5,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {description}
        </p>
      </div>
    </button>
  );
}

export function DataFlowCapabilityCards({
  onGenerate,
}: {
  onGenerate: () => void;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr',
        gap: 14,
        marginBottom: 16,
        maxWidth: 480,
      }}
    >
      <PremiumActionCard
        title="生成数据"
        description="基于任务模板和仿真环境生成任务数据，用于后续模型训练和策略评测。"
        icon={DatabaseZap}
        onClick={onGenerate}
        glowColor="rgba(59, 130, 246, 0.12)"
      />
    </div>
  );
}
