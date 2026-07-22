'use client';

import type { ReactNode } from 'react';
import { REPLAY_PAGE_STYLES } from '@/components/workspace/replay/ReplayWorkbench';

export interface DatasetReplayLayoutProps {
  /** Optional tab bar (content kind switcher). */
  tabBar?: ReactNode;
  /** Optional tag shown above player (e.g. video source label). */
  playerTag?: ReactNode;
  player: ReactNode;
  sidePanel?: ReactNode;
  /** Alerts below the main content row. */
  belowContent?: ReactNode;
  footer?: ReactNode;
}

/**
 * Unified dataset trajectory replay shell: white card, left player, right info panel.
 * Page-level title/subtitle live in ModulePageHeader — no duplicate headings inside the card.
 */
export function DatasetReplayLayout({
  tabBar,
  playerTag,
  player,
  sidePanel,
  belowContent,
  footer,
}: DatasetReplayLayoutProps) {
  return (
    <>
      <style>{REPLAY_PAGE_STYLES}</style>
      <div className="replay-page-stack">
        <section className="replay-workspace-card">
          <div className="replay-main-area">
            {tabBar ? <div style={{ marginBottom: 12 }}>{tabBar}</div> : null}
            <div className="replay-content-row">
              <div className="replay-player-column">
                {playerTag ? (
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
                    {playerTag}
                  </div>
                ) : null}
                <div className="replay-player-shell">
                  <div className="replay-player">
                    <div className="replay-player-media">{player}</div>
                  </div>
                </div>
              </div>
              {sidePanel}
            </div>
            {belowContent}
          </div>
          {footer}
        </section>
      </div>
    </>
  );
}
