'use client';

import { useState } from 'react';
import { ArrowRight, DatabaseZap, Layers, type LucideIcon } from 'lucide-react';

interface DataCenterEntryCardProps {
  title: string;
  tag: string;
  tagTone: 'sim' | 'real';
  description: string;
  actionLabel?: string;
  icon: LucideIcon;
  onClick: () => void;
}

function DataCenterEntryCard({
  title,
  tag,
  tagTone,
  description,
  actionLabel,
  icon: Icon,
  onClick,
}: DataCenterEntryCardProps) {
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);

  const active = hovered || focused;

  const tagColors =
    tagTone === 'sim'
      ? { bg: '#f0f9ff', border: '#bae6fd', text: '#0369a1' }
      : { bg: '#f8fafc', border: '#e2e8f0', text: '#475569' };

  return (
    <button
      type="button"
      aria-label={title}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      style={{
        flex: 1,
        minWidth: 0,
        minHeight: 118,
        display: 'flex',
        flexDirection: 'column',
        padding: '16px 18px',
        borderRadius: 12,
        border: `1px solid ${active ? '#93c5fd' : '#e5e7eb'}`,
        background: active
          ? 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)'
          : 'linear-gradient(180deg, #ffffff 0%, #fafbfc 100%)',
        boxShadow: active
          ? '0 4px 14px rgba(37, 99, 235, 0.07)'
          : '0 1px 2px rgba(15, 23, 42, 0.04)',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease',
        outline: focused ? '2px solid #2563eb' : 'none',
        outlineOffset: 2,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
          marginBottom: 8,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <div
            style={{
              flexShrink: 0,
              width: 34,
              height: 34,
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: active ? '#eff6ff' : '#f1f5f9',
              border: `1px solid ${active ? '#bfdbfe' : '#e2e8f0'}`,
              color: active ? '#2563eb' : '#475569',
              transition: 'background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease',
            }}
          >
            <Icon size={16} strokeWidth={1.75} />
          </div>
          <h3
            style={{
              margin: 0,
              fontSize: 16,
              fontWeight: 600,
              color: '#111827',
              lineHeight: 1.35,
            }}
          >
            {title}
          </h3>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          <span
            style={{
              fontSize: 11,
              fontWeight: 500,
              padding: '3px 8px',
              borderRadius: 9999,
              backgroundColor: tagColors.bg,
              border: `1px solid ${tagColors.border}`,
              color: tagColors.text,
              whiteSpace: 'nowrap',
            }}
          >
            {tag}
          </span>
          {active ? (
            <ArrowRight
              size={15}
              strokeWidth={1.75}
              style={{
                color: '#2563eb',
                transition: 'transform 0.2s ease',
                transform: 'translateX(2px)',
              }}
              aria-hidden
            />
          ) : null}
        </div>
      </div>

      <p
        style={{
          margin: 0,
          fontSize: 13,
          color: '#6b7280',
          lineHeight: 1.55,
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {description}
      </p>
      {actionLabel ? (
        <span
          style={{
            marginTop: 10,
            fontSize: 12,
            fontWeight: 600,
            color: active ? '#2563eb' : '#475569',
          }}
        >
          {actionLabel}
        </span>
      ) : null}
    </button>
  );
}

export function DataCenterEntryCards({
  onStartGenerate,
  onStartBuild,
}: {
  onStartGenerate: () => void;
  onStartBuild: () => void;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
        gap: 14,
        marginBottom: 14,
      }}
    >
      <DataCenterEntryCard
        title="数据生成"
        tag="仿真数据"
        tagTone="sim"
        description="模板与专家策略生成仿真数据，并自动登记为可训练、可回放、可评测的数据集。"
        icon={DatabaseZap}
        onClick={onStartGenerate}
      />
      <DataCenterEntryCard
        title="数据构建"
        tag="真实数据"
        tagTone="real"
        description="将已导入的 HDF5 数据整理为标准训练数据集，支持字段映射、结构校验和训练数据注册。"
        icon={Layers}
        onClick={onStartBuild}
      />
    </div>
  );
}
