'use client';

import { useEffect, useRef } from 'react';

/**
 * Poll only while `shouldPoll` is true; avoids recreating interval when list data changes.
 */
export function usePollingRefresh(
  shouldPoll: boolean,
  refresh: () => void | Promise<void>,
  intervalMs: number
) {
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;

  useEffect(() => {
    if (!shouldPoll) return;
    const timer = window.setInterval(() => {
      void refreshRef.current();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [shouldPoll, intervalMs]);
}
