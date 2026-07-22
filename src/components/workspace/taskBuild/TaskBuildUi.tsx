'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import { buildRealDataBuildHref, buildTaskBuildTemplateHref } from '@/lib/workspace/taskBuildNavigation';

const cardStyle: React.CSSProperties = {
  padding: '24px 28px',
  borderRadius: 12,
  border: '1px solid #e5e7eb',
  backgroundColor: '#fff',
  display: 'flex',
  flexDirection: 'column',
  gap: 12,
  minHeight: 200,
};

export function TaskBuildPathSelector() {
  const router = useRouter();

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: 20,
        marginTop: 8,
      }}
    >
      <div style={cardStyle}>
        <div style={{ fontSize: 16, fontWeight: 600, color: '#111827' }}>使用标准任务模板</div>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280', lineHeight: 1.6 }}>
          从平台已登记的任务模板出发，确认场景与策略信息，关联数据集或模型资产，生成可跳转的任务配置。
        </p>
        <ul style={{ margin: '0 0 8px', paddingLeft: 18, fontSize: 13, color: '#4b5563', lineHeight: 1.55 }}>
          <li>线缆操作任务族（线缆穿杆、线缆整理）</li>
          <li>基于 TaskTemplate registry 真实数据</li>
          <li>可跳转数据生成、训练与评测</li>
        </ul>
        <div style={{ marginTop: 'auto', display: 'flex', gap: 8 }}>
          <PrimaryButton onClick={() => router.push(buildTaskBuildTemplateHref())}>
            开始配置
          </PrimaryButton>
          <SecondaryButton onClick={() => router.push('/workspace/resources/task-templates')}>
            浏览模板库
          </SecondaryButton>
        </div>
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 16, fontWeight: 600, color: '#111827' }}>基于真机数据构建</div>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280', lineHeight: 1.6 }}>
          基于真机数据配置仿真任务：解析数据结构、选择仿真模板并填写场景参数，保存为构建草稿。
        </p>
        <ul style={{ margin: '0 0 8px', paddingLeft: 18, fontSize: 13, color: '#4b5563', lineHeight: 1.55 }}>
          <li>数据结构解析与场景参数配置</li>
          <li>保存为 RealDataImport 构建草稿</li>
          <li>不写入 Dataset registry</li>
        </ul>
        <div style={{ marginTop: 'auto' }}>
          <SecondaryButton onClick={() => router.push(buildRealDataBuildHref())}>
            进入配置流程
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
}

export function TaskBuildStepper({
  steps,
  currentStep,
}: {
  steps: string[];
  currentStep: number;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 8,
        marginBottom: 20,
        padding: '12px 16px',
        borderRadius: 10,
        backgroundColor: '#f9fafb',
        border: '1px solid #e5e7eb',
      }}
    >
      {steps.map((label, index) => {
        const stepNum = index + 1;
        const active = stepNum === currentStep;
        const done = stepNum < currentStep;
        return (
          <div
            key={label}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 13,
              color: active ? '#1d4ed8' : done ? '#374151' : '#9ca3af',
              fontWeight: active ? 600 : 400,
            }}
          >
            <span
              style={{
                width: 22,
                height: 22,
                borderRadius: 9999,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 12,
                backgroundColor: active ? '#2563eb' : done ? '#e5e7eb' : '#f3f4f6',
                color: active ? '#fff' : '#6b7280',
              }}
            >
              {stepNum}
            </span>
            {label}
          </div>
        );
      })}
    </div>
  );
}

export function TaskBuildField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6 }}>{label}</div>
      {children}
    </div>
  );
}

export function TaskBuildBackLink({ href = '/workspace/task-build' }: { href?: string }) {
  return (
    <Link href={href} style={{ fontSize: 13, color: '#2563eb', textDecoration: 'none' }}>
      ← 返回任务构建入口
    </Link>
  );
}
