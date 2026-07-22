'use client';

import {
  createContext,
  useContext,
  useRef,
  useCallback,
  useState,
  type ReactNode,
} from 'react';
import { getFramesBatch } from '../api/labelApi';

interface FrameCacheContextValue {
  getCached: (episodeId: string, camera: string, frame: number) => string | null;
  /** MCAP 播放时写入缓存，供后续拖动进度条即时显示 */
  setCachedFrame: (episodeId: string, camera: string, frame: number, url: string) => void;
  preload: (
    episodeId: string,
    camera: string,
    start: number,
    totalToLoad: number,
    taskId: string
  ) => void;
  preloadProgress: number;  // 0-100
  isPreloading: boolean;
  preloadError: string | null;
  clearCache: () => void;
}

const FrameCacheContext = createContext<FrameCacheContextValue | null>(null);

const BATCH_SIZE = 20;
const MAX_CACHED_FRAMES = 500;  // 每相机最多缓存帧数，超出则 FIFO 清理
const MAIN_THREAD_YIELD_EVERY = 5;

export function FrameCacheProvider({ children }: { children: ReactNode }) {
  const cacheRef = useRef<Map<string, Map<number, string>>>(new Map());
  const [preloadProgress, setPreloadProgress] = useState(0);
  const [isPreloading, setIsPreloading] = useState(false);
  const [preloadError, setPreloadError] = useState<string | null>(null);
  const preloadAbortRef = useRef<AbortController | null>(null);
  const preloadSessionRef = useRef(0);

  const cacheKey = (episodeId: string, camera: string) => `${episodeId}::${camera}`;

  const getCached = useCallback(
    (episodeId: string, camera: string, frame: number): string | null => {
      const key = cacheKey(episodeId, camera);
      const map = cacheRef.current.get(key);
      return map?.get(frame) ?? null;
    },
    []
  );

  const setCachedFrame = useCallback(
    (episodeId: string, camera: string, frame: number, url: string) => {
      const key = cacheKey(episodeId, camera);
      let map = cacheRef.current.get(key);
      if (!map) {
        map = new Map();
        cacheRef.current.set(key, map);
      }
      if (map.size >= MAX_CACHED_FRAMES) {
        const firstKey = map.keys().next().value;
        if (firstKey != null) {
          const old = map.get(firstKey);
          if (old) URL.revokeObjectURL(old);
          map.delete(firstKey);
        }
      }
      map.set(frame, url);
    },
    []
  );

  const clearCache = useCallback(() => {
    preloadSessionRef.current += 1;
    preloadAbortRef.current?.abort();
    for (const map of cacheRef.current.values()) {
      for (const url of map.values()) {
        URL.revokeObjectURL(url);
      }
    }
    cacheRef.current.clear();
    setPreloadProgress(0);
    setIsPreloading(false);
    setPreloadError(null);
  }, []);

  const preload = useCallback(
    async (
      episodeId: string,
      camera: string,
      start: number,
      totalToLoad: number,  // 预加载帧数
      taskId: string
    ) => {
      // 切换 episode/camera 时先终止上一轮预加载，避免旧请求回流污染当前状态
      preloadAbortRef.current?.abort();
      preloadAbortRef.current = new AbortController();
      const mySession = ++preloadSessionRef.current;

      const key = cacheKey(episodeId, camera);
      let map = cacheRef.current.get(key);
      if (!map) {
        map = new Map();
        cacheRef.current.set(key, map);
      }

      setIsPreloading(true);
      setPreloadProgress(0);
      setPreloadError(null);

      let loaded = 0;
      const total = Math.min(totalToLoad, 500);
      const progressThrottleRef = { last: 0 };
      const setProgressThrottled = (pct: number) => {
        const now = Date.now();
        if (now - progressThrottleRef.last >= 150 || pct >= 100) {
          progressThrottleRef.last = now;
          setPreloadProgress(pct);
        }
      };

      try {
        for (let s = start; s < start + total; s += BATCH_SIZE) {
          if (preloadAbortRef.current?.signal.aborted) break;
          const count = Math.min(BATCH_SIZE, start + total - s);
          if (mySession !== preloadSessionRef.current) break;
          const res = await getFramesBatch(
            episodeId,
            camera,
            s,
            count,
            taskId,
            preloadAbortRef.current?.signal
          );
          if (mySession !== preloadSessionRef.current) break;
          for (let i = 0; i < res.frames.length; i++) {
            const frameIdx = res.start + i;
            const b64 = res.frames[i];
            const blob = base64ToBlob(b64, 'image/jpeg');
            const url = URL.createObjectURL(blob);
            if (map!.size >= MAX_CACHED_FRAMES) {
              const firstKey = map!.keys().next().value;
              if (firstKey != null) {
                const old = map!.get(firstKey);
                if (old) URL.revokeObjectURL(old);
                map!.delete(firstKey);
              }
            }
            map!.set(frameIdx, url);
            loaded++;
            // 分片让出主线程，避免 atob/Blob 转换长时间阻塞 UI
            if ((i + 1) % MAIN_THREAD_YIELD_EVERY === 0) {
              await new Promise<void>((resolve) => {
                setTimeout(() => resolve(), 0);
              });
              if (mySession !== preloadSessionRef.current) break;
            }
          }
          setProgressThrottled(Math.round((loaded / total) * 100));
          await new Promise<void>((resolve) => {
            setTimeout(() => resolve(), 0);
          });
          if (mySession !== preloadSessionRef.current) break;
        }
      } catch (e) {
        // 过期会话/已中止请求属于预期行为，不提示失败
        if (mySession !== preloadSessionRef.current) return;
        if ((e as Error)?.name !== 'AbortError') {
          const msg = e instanceof Error ? e.message : String(e);
          // 预加载是增强链路：切换过程中的短暂 404 不应打断主流程
          if (/not found|episode .* not found|camera .* not found/i.test(msg)) return;
          const friendly =
            /403|forbidden|无数据|无权|权限/i.test(msg)
              ? '当前账号无权访问该标注任务的数据'
              : `预加载失败：${msg || '请稍后重试'}`;
          setPreloadError(friendly);
          console.error('Frame preload error:', e);
        }
      } finally {
        setIsPreloading(false);
        setPreloadProgress(100);
      }
    },
    []
  );

  return (
    <FrameCacheContext.Provider
      value={{
        getCached,
        setCachedFrame,
        preload,
        preloadProgress,
        isPreloading,
        preloadError,
        clearCache,
      }}
    >
      {children}
    </FrameCacheContext.Provider>
  );
}

function base64ToBlob(b64: string, mime: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

// 无 Provider 时返回稳定引用，避免 ViewportTile 等依赖 getCached 的 effect 无限重跑
const getCachedStub = (): string | null => null;
const setCachedFrameStub = () => {};
const preloadStub = () => {};
const clearCacheStub = () => {};

const EMPTY_FRAME_CACHE = {
  getCached: getCachedStub,
  setCachedFrame: setCachedFrameStub,
  preload: preloadStub,
  preloadProgress: 0,
  isPreloading: false,
  preloadError: null,
  clearCache: clearCacheStub,
} as const;

export function useFrameCache() {
  const ctx = useContext(FrameCacheContext);
  if (!ctx) return EMPTY_FRAME_CACHE;
  return ctx;
}
