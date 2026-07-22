'use client';

import { Suspense, useCallback, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { ProcessEvaluationDetailView } from '@/components/workspace/evaluation/ProcessEvaluationDetailView';
import { getProcessEvaluationDetail } from '@/lib/mock/workspaceEvaluationMock';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';

function ProcessEvaluationDetailContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const dataParam = searchParams.get('data');
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const detail = useMemo(() => getProcessEvaluationDetail(dataParam), [dataParam]);

  const showToast = useCallback((text: string) => {
    setToastMsg(text);
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="过程评测详情"
        subtitle="基于任务执行视频、仿真轨迹和任务描述进行过程级评测，分析任务进度、成功概率、失败节点和轨迹质量。"
        actions={
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <SecondaryButton onClick={() => router.push('/workspace/evaluation')}>
              返回评测中心
            </SecondaryButton>
            <PrimaryButton onClick={() => showToast('过程评测报告导出已开始')}>导出报告</PrimaryButton>
          </div>
        }
      />

      <div
        role="status"
        style={{
          marginBottom: 16,
          padding: '12px 16px',
          borderRadius: 10,
          border: '1px solid #fcd34d',
          backgroundColor: '#fffbeb',
          color: '#92400e',
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        <strong style={{ fontWeight: 600 }}>内部实验页面，组内试用不开放</strong>
        {' — '}
        过程评测能力仍在开发中，当前环境不提供正式支持。请使用
        {' '}
        <button
          type="button"
          onClick={() => router.push('/workspace/evaluation')}
          style={{
            padding: 0,
            border: 'none',
            background: 'none',
            color: '#b45309',
            fontWeight: 600,
            cursor: 'pointer',
            textDecoration: 'underline',
          }}
        >
          评测中心
        </button>
        {' '}
        查看真实评测任务与回放。
      </div>

      <ProcessEvaluationDetailView detail={detail} />

      {toastMsg ? (
        <div
          style={{
            position: 'fixed',
            left: '50%',
            bottom: 24,
            transform: 'translateX(-50%)',
            padding: '10px 16px',
            borderRadius: 10,
            fontSize: 14,
            fontWeight: 500,
            zIndex: 1700,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            backgroundColor: 'rgba(17,24,39,0.92)',
            color: '#fff',
          }}
        >
          {toastMsg}
        </div>
      ) : null}
    </ModulePageContainer>
  );
}

export default function ProcessEvaluationDetailPage() {
  return (
    <Suspense fallback={null}>
      <ProcessEvaluationDetailContent />
    </Suspense>
  );
}
