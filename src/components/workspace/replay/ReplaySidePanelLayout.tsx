'use client';

import type { ReactNode } from 'react';
import { simConsoleCardStyle } from '@/components/workspace/simulation/SimulationRunConsoleLayout';

export function ReplayPanelSectionTitle({ children }: { children: string }) {
  return (
    <div style={{ fontSize: 13, fontWeight: 600, color: '#111827', marginBottom: 8 }}>{children}</div>
  );
}

export function ReplaySidePanelLayout({
  children,
  footerActions,
  emptyLabel,
  title = '运行信息',
  fitContent = false,
}: {
  children?: ReactNode;
  footerActions?: ReactNode;
  emptyLabel?: string;
  title?: string;
  /** 无底部操作时随内容高度收口，避免侧栏底部留白 */
  fitContent?: boolean;
}) {
  return (
    <aside
      className="replay-run-info-panel replay-side-panel-layout"
      style={{
        ...simConsoleCardStyle,
        ...(fitContent ? { alignSelf: 'flex-start', maxHeight: 'none' } : {}),
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 4, flexShrink: 0 }}>
        {title}
      </div>
      {emptyLabel ? (
        <p style={{ margin: 0, fontSize: 13, color: '#9ca3af', lineHeight: 1.5 }}>{emptyLabel}</p>
      ) : (
        <>
          <div className="replay-run-info-scroll replay-side-panel-scroll">{children}</div>
          {footerActions ? (
            <div className="replay-side-panel-footer">{footerActions}</div>
          ) : null}
        </>
      )}
    </aside>
  );
}
