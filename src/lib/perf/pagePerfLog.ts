'use client';

import { useEffect, useRef } from 'react';

type PagePerfOptions = {
  loading?: boolean;
  apiRequestCount?: number;
};

/**
 * Dev/prod-safe page performance logging: mount time, API count, first ready paint.
 */
export function usePagePerfLog(pageName: string, options: PagePerfOptions = {}) {
  const mountAtRef = useRef<number>(0);
  const loggedReadyRef = useRef(false);

  useEffect(() => {
    mountAtRef.current = performance.now();
    loggedReadyRef.current = false;
    if (process.env.NODE_ENV !== 'production' || process.env.NEXT_PUBLIC_PERF_LOG === '1') {
      console.info(`[Perf] ${pageName} mount`);
    }
    return () => {
      if (process.env.NODE_ENV !== 'production' || process.env.NEXT_PUBLIC_PERF_LOG === '1') {
        const elapsed = performance.now() - mountAtRef.current;
        console.info(`[Perf] ${pageName} unmount after ${Math.round(elapsed)}ms`);
      }
    };
  }, [pageName]);

  useEffect(() => {
    if (options.loading !== false || loggedReadyRef.current) return;
    loggedReadyRef.current = true;
    const elapsed = performance.now() - mountAtRef.current;
    if (process.env.NODE_ENV !== 'production' || process.env.NEXT_PUBLIC_PERF_LOG === '1') {
      console.info(
        `[Perf] ${pageName} ready in ${Math.round(elapsed)}ms apiRequests=${options.apiRequestCount ?? 0}`
      );
    }
  }, [pageName, options.loading, options.apiRequestCount]);
}
