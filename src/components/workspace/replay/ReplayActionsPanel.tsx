'use client';

import Link from 'next/link';
import type { ReplayAdapterResult } from '@/lib/workspace/replayAdapters';
import { REPLAY_DATA_CENTER_HREF } from '@/lib/workspace/replayPanelNavigation';

interface ReplayActionsPanelProps {
  adapter?: ReplayAdapterResult;
  generating?: boolean;
  onGenerateReplay?: () => void;
  variant?: 'panel' | 'inline' | 'footer';
  showDatasetDetail?: boolean;
  dataCenterHref?: string;
  backLabel?: string;
}

const outlineButtonStyle: React.CSSProperties = {
  padding: '6px 14px',
  borderRadius: 8,
  border: '1px solid #d1d5db',
  background: '#fff',
  color: '#374151',
  fontSize: 13,
  textDecoration: 'none',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
};

const primaryOutlineButtonStyle: React.CSSProperties = {
  ...outlineButtonStyle,
  border: '1px solid #2563eb',
  color: '#2563eb',
};

export function ReplayActionsPanel({
  adapter,
  generating,
  onGenerateReplay,
  variant = 'panel',
  showDatasetDetail = false,
  dataCenterHref = REPLAY_DATA_CENTER_HREF,
  backLabel = '返回数据中心',
}: ReplayActionsPanelProps) {
  const showGenerate =
    adapter?.canGenerateReplay && !adapter?.replayInProgress && onGenerateReplay;

  if (variant === 'inline') {
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
        {showGenerate ? (
          <button
            type="button"
            style={{
              ...primaryOutlineButtonStyle,
              background: '#2563eb',
              color: '#fff',
              cursor: generating ? 'not-allowed' : 'pointer',
            }}
            disabled={generating}
            onClick={onGenerateReplay}
          >
            {generating ? '正在创建…' : '生成回放视频'}
          </button>
        ) : null}
      </div>
    );
  }

  if (variant === 'footer') {
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'flex-end' }}>
        <Link href={dataCenterHref} style={outlineButtonStyle}>
          {backLabel}
        </Link>
        {showGenerate ? (
          <button
            type="button"
            style={{
              ...primaryOutlineButtonStyle,
              cursor: generating ? 'not-allowed' : 'pointer',
            }}
            disabled={generating}
            onClick={onGenerateReplay}
          >
            {generating ? '正在创建回放任务…' : '生成回放视频'}
          </button>
        ) : null}
        {showDatasetDetail && adapter?.datasetDetailHref ? (
          <Link href={adapter.datasetDetailHref} style={outlineButtonStyle}>
            查看数据集详情
          </Link>
        ) : null}
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <Link href={dataCenterHref} style={{ fontSize: 13, color: '#2563eb', textDecoration: 'none' }}>
        {backLabel}
      </Link>
      {showGenerate ? (
        <button
          type="button"
          style={{
            padding: 0,
            border: 'none',
            background: 'none',
            color: '#2563eb',
            fontSize: 13,
            textAlign: 'left',
            cursor: generating ? 'not-allowed' : 'pointer',
          }}
          disabled={generating}
          onClick={onGenerateReplay}
        >
          {generating ? '正在创建回放任务…' : '生成回放视频'}
        </button>
      ) : null}
      {showDatasetDetail && adapter?.datasetDetailHref ? (
        <Link href={adapter.datasetDetailHref} style={{ fontSize: 13, color: '#2563eb', textDecoration: 'none' }}>
          查看数据集详情
        </Link>
      ) : null}
    </div>
  );
}
