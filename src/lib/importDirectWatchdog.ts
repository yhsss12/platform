/**
 * 浏览器直传导入：同一会话内「无进度」与 complete 无返回的 watchdog（与 importDiagLog 配套）。
 */

import { importDiagLog } from '@/lib/importDiagLog';

/** 与 DataImportProgress 对齐，避免从 runner 循环 import */
export type DirectImportWatchdogProgress = {
  completedUnits: number;
  failedUnits: number;
  totalUnits: number;
  currentLabel?: string;
  phase?: string;
  uploadHintPercent?: number;
};

export const DIRECT_IMPORT_WATCHDOG_PREPARE_STALL_MS = 90_000;
export const DIRECT_IMPORT_WATCHDOG_UPLOAD_STALL_MS = 180_000;
export const DIRECT_IMPORT_WATCHDOG_COMPLETE_WALL_MS = 120_000;

export function combineAbortSignals(a?: AbortSignal, b?: AbortSignal): AbortSignal {
  if (!a) return b ?? new AbortController().signal;
  if (!b) return a;
  const c = new AbortController();
  const forward = () => {
    if (!c.signal.aborted) {
      try {
        c.abort();
      } catch {
        /* ignore */
      }
    }
  };
  if (a.aborted || b.aborted) {
    forward();
    return c.signal;
  }
  a.addEventListener('abort', forward, { once: true });
  b.addEventListener('abort', forward, { once: true });
  return c.signal;
}

/** merged 因 watchdog 子信号 abort 时，user 可能尚未 abort */
export async function raceWithAbort<T>(promise: Promise<T>, merged: AbortSignal): Promise<T> {
  if (merged.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const cleanup = () => {
      merged.removeEventListener('abort', onAbort);
    };
    merged.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (v) => {
        cleanup();
        resolve(v);
      },
      (e) => {
        cleanup();
        reject(e);
      },
    );
  });
}

type PhaseBucket = 'idle' | 'prepare' | 'upload' | 'complete';

function progressStamp(m: DirectImportWatchdogProgress): string {
  const hint =
    m.uploadHintPercent != null ? String(Math.round(m.uploadHintPercent * 100) / 100) : 'x';
  return `${m.completedUnits}|${m.failedUnits}|${m.phase ?? ''}|${hint}|${m.currentLabel ?? ''}`;
}

export class DirectImportWatchdog {
  private readonly userSignal: AbortSignal | undefined;
  private readonly watchdogAc = new AbortController();
  readonly signal: AbortSignal;
  private interval: ReturnType<typeof setInterval> | undefined;
  private disposed = false;
  private lastStamp = '';
  private lastMoveAt = Date.now();
  private phaseBucket: PhaseBucket = 'idle';
  private completeWallStarted: number | null = null;
  private lastFireReason: string | null = null;

  constructor(userSignal: AbortSignal | undefined) {
    this.userSignal = userSignal;
    this.signal = combineAbortSignals(userSignal, this.watchdogAc.signal);
    this.interval = setInterval(() => this.check(), 1000);
  }

  noteProgress(merged: DirectImportWatchdogProgress): void {
    if (this.disposed) return;
    const ph = (merged.phase || '').trim();
    if (ph === '准备上传') {
      this.phaseBucket = 'prepare';
      this.completeWallStarted = null;
    } else if (ph === '上传中') {
      this.phaseBucket = 'upload';
      this.completeWallStarted = null;
    } else if (ph === '写入登记') {
      if (this.phaseBucket !== 'complete') {
        this.phaseBucket = 'complete';
        this.completeWallStarted = Date.now();
      }
    } else {
      this.phaseBucket = 'idle';
      this.completeWallStarted = null;
    }

    const st = progressStamp(merged);
    if (st !== this.lastStamp) {
      this.lastStamp = st;
      this.lastMoveAt = Date.now();
    }
  }

  /** 在 completeDirectUpload 返回后调用，避免误判仍处于 complete 墙钟 */
  markCompletePhaseEnded(): void {
    this.completeWallStarted = null;
    if (this.phaseBucket === 'complete') {
      this.phaseBucket = 'idle';
    }
  }

  consumeLastFireReason(): string | null {
    const r = this.lastFireReason;
    this.lastFireReason = null;
    return r;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = undefined;
    }
  }

  private fire(message: string): void {
    if (this.disposed || this.watchdogAc.signal.aborted) return;
    this.lastFireReason = message;
    importDiagLog('import_watchdog_fired', {
      message,
      phaseBucket: this.phaseBucket,
      stallMs:
        this.phaseBucket === 'complete' && this.completeWallStarted != null
          ? Date.now() - this.completeWallStarted
          : Date.now() - this.lastMoveAt,
    });
    try {
      this.watchdogAc.abort();
    } catch {
      /* ignore */
    }
  }

  private check(): void {
    if (this.disposed || this.watchdogAc.signal.aborted) return;
    const now = Date.now();

    if (this.phaseBucket === 'complete' && this.completeWallStarted != null) {
      if (now - this.completeWallStarted >= DIRECT_IMPORT_WATCHDOG_COMPLETE_WALL_MS) {
        this.fire(
          `写入登记阶段超过 ${Math.round(DIRECT_IMPORT_WATCHDOG_COMPLETE_WALL_MS / 1000)} 秒无返回（upload-complete）`,
        );
        return;
      }
    }

    if (this.phaseBucket === 'prepare') {
      if (now - this.lastMoveAt >= DIRECT_IMPORT_WATCHDOG_PREPARE_STALL_MS) {
        this.fire(
          `准备上传阶段超过 ${Math.round(DIRECT_IMPORT_WATCHDOG_PREPARE_STALL_MS / 1000)} 秒无进度`,
        );
        return;
      }
    }

    if (this.phaseBucket === 'upload') {
      if (now - this.lastMoveAt >= DIRECT_IMPORT_WATCHDOG_UPLOAD_STALL_MS) {
        this.fire(
          `上传阶段超过 ${Math.round(DIRECT_IMPORT_WATCHDOG_UPLOAD_STALL_MS / 1000)} 秒无进度`,
        );
      }
    }
  }
}
