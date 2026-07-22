/**
 * 设备启动后预建 WebRTC，并在后台保活；实时采集页可复用同一 PeerConnection，减少首帧等待。
 */
import { createWebRtcSession } from './webrtcClient';
import { fetchStreamCameras } from './streamCameraApi';

export type WarmWebRtcSession = {
  pc: RTCPeerConnection;
  stream: MediaStream | null;
  deviceId: string;
  cameraId: string;
};

const WARM_POOL_KEY = '__fromWarmPool';

type PoolEntry = WarmWebRtcSession & {
  refCount: number;
};

const pool = new Map<string, PoolEntry>();
const activeDevices = new Set<string>();
const reconnectTimers = new Map<string, ReturnType<typeof setTimeout>>();
/** disconnected 常为 ICE 短暂抖动，延迟后再确认，避免与采集命令争抢 offer */
const WARM_RECONNECT_DEBOUNCE_MS = 8000;
const WARM_RECONNECT_DELAY_MS = 2500;

function poolKey(deviceId: string, cameraId: string): string {
  return `${deviceId}:${cameraId}`;
}

export function isWarmPoolPeer(pc: RTCPeerConnection | null | undefined): boolean {
  return Boolean(pc && (pc as RTCPeerConnection & { [WARM_POOL_KEY]?: boolean })[WARM_POOL_KEY]);
}

function markWarmPoolPeer(pc: RTCPeerConnection): void {
  (pc as RTCPeerConnection & { [WARM_POOL_KEY]?: boolean })[WARM_POOL_KEY] = true;
}

function notifyTrack(entry: PoolEntry, stream: MediaStream | null) {
  if (stream) entry.stream = stream;
}

async function buildSession(deviceId: string, cameraId: string): Promise<PoolEntry | null> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    return null;
  }
  try {
    let boundStream: MediaStream | null = null;
    const pc = await createWebRtcSession({
      deviceId,
      cameraId,
      onTrack: (event: RTCTrackEvent) => {
        const [s] = event.streams;
        if (s) boundStream = s;
      },
    });
    markWarmPoolPeer(pc);

    const entry: PoolEntry = {
      pc,
      stream: boundStream,
      deviceId,
      cameraId,
      refCount: 0,
    };

    pc.addEventListener('track', (event: RTCTrackEvent) => {
      const [s] = event.streams;
      if (s) notifyTrack(entry, s);
    });

    pc.addEventListener('connectionstatechange', () => {
      const st = pc.connectionState;
      if (st === 'failed' || st === 'closed') {
        scheduleWarmReconnect(deviceId, cameraId, { immediate: true });
        return;
      }
      if (st === 'disconnected') {
        scheduleWarmReconnect(deviceId, cameraId, { debounceMs: WARM_RECONNECT_DEBOUNCE_MS });
      }
    });

    // onTrack 可能略晚于 offer 完成，短延迟再读一次
    await new Promise((r) => setTimeout(r, 80));
    if (boundStream) entry.stream = boundStream;

    return entry;
  } catch (e) {
    console.warn('[devicePreviewWarmPool] create session failed', deviceId, cameraId, e);
    return null;
  }
}

function scheduleWarmReconnect(
  deviceId: string,
  cameraId: string,
  options?: { debounceMs?: number; immediate?: boolean },
) {
  if (!activeDevices.has(deviceId)) return;
  const k = poolKey(deviceId, cameraId);
  const existingTimer = reconnectTimers.get(k);
  if (existingTimer) clearTimeout(existingTimer);
  const delay = options?.immediate
    ? WARM_RECONNECT_DELAY_MS
    : (options?.debounceMs ?? WARM_RECONNECT_DEBOUNCE_MS);
  const t = setTimeout(() => {
    reconnectTimers.delete(k);
    void (async () => {
      if (!activeDevices.has(deviceId)) return;
      const old = pool.get(k);
      if (old) {
        const st = old.pc.connectionState;
        if (st === 'connected' || st === 'connecting') return;
        if (old.refCount > 0 && st !== 'failed' && st !== 'closed') return;
      }
      if (old) {
        try {
          old.pc.close();
        } catch {
          // ignore
        }
        pool.delete(k);
      }
      const next = await buildSession(deviceId, cameraId);
      if (next) pool.set(k, next);
    })();
  }, delay);
  reconnectTimers.set(k, t);
}

/** 设备已启动且隧道可用时调用；可重复调用（幂等补全缺失路） */
export async function startDevicePreviewWarm(
  deviceId: string | number,
  options?: { maxCameras?: number },
): Promise<void> {
  const did = String(deviceId);
  activeDevices.add(did);

  let cameras: Awaited<ReturnType<typeof fetchStreamCameras>> = [];
  try {
    cameras = await fetchStreamCameras(did);
  } catch (e) {
    console.warn('[devicePreviewWarmPool] list cameras failed', did, e);
    return;
  }

  const limit = Math.max(1, options?.maxCameras ?? 4);
  for (const cam of cameras.slice(0, limit)) {
    const k = poolKey(did, cam.id);
    const existing = pool.get(k);
    if (existing && existing.pc.connectionState !== 'closed' && existing.pc.connectionState !== 'failed') {
      continue;
    }
    if (existing) {
      try {
        existing.pc.close();
      } catch {
        // ignore
      }
      pool.delete(k);
    }
    const entry = await buildSession(did, cam.id);
    if (entry) pool.set(k, entry);
  }
}

/** 设备停止或不再需要保活 */
export function stopDevicePreviewWarm(deviceId: string | number): void {
  const did = String(deviceId);
  activeDevices.delete(did);
  for (const k of [...pool.keys()]) {
    if (!k.startsWith(`${did}:`)) continue;
    const t = reconnectTimers.get(k);
    if (t) clearTimeout(t);
    reconnectTimers.delete(k);
    const e = pool.get(k);
    if (e) {
      try {
        e.pc.close();
      } catch {
        // ignore
      }
    }
    pool.delete(k);
  }
}

/** 取出已预热的会话（引用计数 +1） */
export function acquireWarmWebRtcSession(
  deviceId: string | number,
  cameraId: string,
): WarmWebRtcSession | null {
  const k = poolKey(String(deviceId), cameraId);
  const e = pool.get(k);
  if (!e) return null;
  if (e.pc.connectionState === 'closed' || e.pc.connectionState === 'failed') {
    pool.delete(k);
    return null;
  }
  e.refCount += 1;
  return { pc: e.pc, stream: e.stream, deviceId: e.deviceId, cameraId: e.cameraId };
}

/** 归还引用；默认不关闭连接以便后台保活 */
export function releaseWarmWebRtcSession(
  deviceId: string | number,
  cameraId: string,
  options?: { forceClose?: boolean },
): void {
  const k = poolKey(String(deviceId), cameraId);
  const e = pool.get(k);
  if (!e) return;
  e.refCount = Math.max(0, e.refCount - 1);
  if (options?.forceClose || (!activeDevices.has(String(deviceId)) && e.refCount === 0)) {
    try {
      e.pc.close();
    } catch {
      // ignore
    }
    pool.delete(k);
  }
}

/** 关闭或归还：热池 PC 仅 release，其它 close */
export function disposePeerConnection(
  pc: RTCPeerConnection | null | undefined,
  deviceId: string | number | undefined,
  cameraId: string | undefined,
  options?: { forceClose?: boolean },
): void {
  if (!pc) return;
  const fromPool = isWarmPoolPeer(pc);
  const cam = (pc as RTCPeerConnection & { _cameraId?: string })._cameraId || cameraId;
  if (fromPool && deviceId != null && cam) {
    releaseWarmWebRtcSession(deviceId, cam, options);
    return;
  }
  try {
    pc.close();
  } catch {
    // ignore
  }
}

export function getWarmSessionStream(deviceId: string | number, cameraId: string): MediaStream | null {
  const e = pool.get(poolKey(String(deviceId), cameraId));
  return e?.stream ?? null;
}
