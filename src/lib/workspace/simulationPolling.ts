'use client';

import { useEffect, useState } from 'react';

const TERMINAL_STATUSES = new Set([
  'completed',
  'failed',
  'canceled',
  'cancelled',
  'timeout',
]);

export function isTerminalSimJobStatus(status: string | null | undefined): boolean {
  return TERMINAL_STATUSES.has((status ?? '').trim().toLowerCase());
}

/** Returns false while the document/tab is hidden (pause polling). */
export function usePageVisibleForPolling(): boolean {
  const [visible, setVisible] = useState(
    () => typeof document === 'undefined' || !document.hidden
  );

  useEffect(() => {
    const onVisibilityChange = () => {
      setVisible(!document.hidden);
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);

  return visible;
}
