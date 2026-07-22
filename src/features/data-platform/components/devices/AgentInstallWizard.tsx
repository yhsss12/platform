'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import {
  getZeroInstallStatus,
  startZeroInstall,
  type ZeroInstallStartData,
} from '../../api/agentInstallApi';

/** 兼容非 HTTPS / 无 Clipboard API 环境（navigator.clipboard 常静默失败） */
async function copyTextToClipboard(text: string): Promise<boolean> {
  const t = String(text ?? '').trim();
  if (!t) return false;
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(t);
      return true;
    }
  } catch {
    // fallback below
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = t;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    ta.setSelectionRange(0, t.length);
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

interface Props {
  open: boolean;
  onClose: () => void;
}

type ZeroInstallStatusLog = {
  ts: string;
  level: string;
  message: string;
};

type ZeroInstallStatusData = {
  status: string;
  progress: number;
  stage: string;
  logs: ZeroInstallStatusLog[];
};

export default function AgentInstallWizard({ open, onClose }: Props) {
  const [step, setStep] = useState<1 | 2>(1);
  const [submitting, setSubmitting] = useState(false);
  const [task, setTask] = useState<ZeroInstallStatusData | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [zero, setZero] = useState<ZeroInstallStartData | null>(null);
  const [zeroToken, setZeroToken] = useState<string | null>(null);
  const [zeroCmdSnapshot, setZeroCmdSnapshot] = useState<string>('');
  const [copyHint, setCopyHint] = useState<string | null>(null);
  const copyHintTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (copyHintTimer.current) clearTimeout(copyHintTimer.current);
  }, []);

  useEffect(() => {
    if (!open) {
      setStep(1);
      setTask(null);
      setSubmitting(false);
      setZero(null);
      setZeroToken(null);
      setZeroCmdSnapshot('');
      setCopyHint(null);
      if (copyHintTimer.current) clearTimeout(copyHintTimer.current);
      if (pollTimer.current) clearInterval(pollTimer.current);
    }
  }, [open]);

  useEffect(() => {
    if (!open || !zeroToken) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await getZeroInstallStatus(zeroToken);
        if (cancelled) return;
        if (res.ok && res.data) {
          setTask(res.data as ZeroInstallStatusData);
          const st = String(res.data?.status || '').toLowerCase();
          if (st === 'success' || st === 'failed') {
            if (pollTimer.current) clearInterval(pollTimer.current);
          }
        }
      } catch {
        // ignore transient polling failures
      }
    };

    pollTimer.current = setInterval(poll, 1200);
    void poll();
    return () => {
      cancelled = true;
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, [zeroToken, open]);

  const setFailedTask = (message: string) => {
    setTask({
      status: 'failed',
      progress: 100,
      stage: 'init',
      logs: [{ ts: '', level: 'error', message }],
    });
  };

  const onStartZero = async () => {
    if (submitting) return;
    setSubmitting(true);
    setTask(null);
    setZero(null);
    setZeroToken(null);
    setZeroCmdSnapshot('');
    try {
      const res = await startZeroInstall();
      if (res.ok && res.data) {
        setZero(res.data);
        setZeroToken(res.data.token);
        setZeroCmdSnapshot(res.data.command || '');
        setStep(2);
      } else {
        setFailedTask(res.error || '创建安装会话失败');
      }
    } catch (error: any) {
      setFailedTask(error?.message || '创建安装会话失败');
    } finally {
      setSubmitting(false);
    }
  };

  const zeroCommand = useMemo(() => String(zero?.command || '').trim(), [zero?.command]);

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
      }}
    >
      <div
        style={{
          width: 780,
          maxWidth: '95vw',
          maxHeight: '86vh',
          background: '#fff',
          borderRadius: 10,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            padding: '14px 20px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ fontSize: 16, fontWeight: 600, color: '#111827' }}>Client 安装向导</div>
          <button
            onClick={onClose}
            style={{ border: 0, background: 'transparent', fontSize: 22, color: '#6b7280', cursor: 'pointer' }}
          >
            ×
          </button>
        </div>

        <div style={{ padding: 20, overflowY: 'auto' }}>
          {step === 1 && (
            <div>
              <div style={{ fontSize: 13, color: '#6b7280', lineHeight: 1.6 }}>
                点击生成命令后，把命令复制到目标 Linux 机器执行即可。执行过程中会实时回传安装进度到本弹窗，并在安装完成后配置 systemd 开机自启。
              </div>
              <div style={{ marginTop: 14, display: 'flex', gap: 10 }}>
                <button
                  type="button"
                  onClick={onStartZero}
                  disabled={submitting}
                  style={{
                    padding: '8px 14px',
                    borderRadius: 6,
                    border: 'none',
                    background: submitting ? '#9ca3af' : '#2563eb',
                    color: '#fff',
                    cursor: submitting ? 'not-allowed' : 'pointer',
                  }}
                >
                  {submitting ? '生成中…' : '生成一键安装命令'}
                </button>
                {zeroCommand ? (
                  <button
                    type="button"
                    onClick={async () => {
                      const ok = await copyTextToClipboard(zeroCommand);
                      if (copyHintTimer.current) clearTimeout(copyHintTimer.current);
                      setCopyHint(ok ? '已复制到剪贴板' : '复制失败，请手动选中命令复制');
                      copyHintTimer.current = setTimeout(() => setCopyHint(null), 2500);
                    }}
                    style={{
                      padding: '8px 14px',
                      borderRadius: 6,
                      border: '1px solid #d1d5db',
                      background: '#fff',
                      cursor: 'pointer',
                    }}
                  >
                    复制命令
                  </button>
                ) : null}
              </div>
              {copyHint ? (
                <div style={{ marginTop: 8, fontSize: 12, color: copyHint.startsWith('复制失败') ? '#b91c1c' : '#059669' }}>
                  {copyHint}
                </div>
              ) : null}
              {zeroCommand ? (
                <pre
                  style={{
                    marginTop: 12,
                    padding: 12,
                    background: '#0b1220',
                    color: '#e5e7eb',
                    borderRadius: 8,
                    overflowX: 'auto',
                    fontSize: 12,
                  }}
                >
                  {zeroCommand}
                </pre>
              ) : null}
            </div>
          )}

          {step === 2 && (
            <div>
              {(zeroCmdSnapshot || zeroCommand) ? (
                <div style={{ marginBottom: 14 }}>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      gap: 12,
                      marginBottom: 8,
                    }}
                  >
                    <div style={{ fontSize: 14, color: '#374151', fontWeight: 600 }}>
                      一键安装命令（在目标机执行）
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        type="button"
                        onClick={async () => {
                          const txt = zeroCmdSnapshot || zeroCommand;
                          const ok = await copyTextToClipboard(txt);
                          if (copyHintTimer.current) clearTimeout(copyHintTimer.current);
                          setCopyHint(ok ? '已复制到剪贴板' : '复制失败，请手动选中下方命令复制');
                          copyHintTimer.current = setTimeout(() => setCopyHint(null), 2500);
                        }}
                        style={{
                          padding: '6px 10px',
                          borderRadius: 6,
                          border: '1px solid #d1d5db',
                          background: '#fff',
                          cursor: 'pointer',
                          fontSize: 12,
                        }}
                      >
                        复制命令
                      </button>
                      <button
                        type="button"
                        onClick={async () => {
                          if (!zeroToken) return;
                          const res = await getZeroInstallStatus(zeroToken);
                          if (res.ok && res.data) {
                            setTask(res.data as ZeroInstallStatusData);
                          }
                        }}
                        style={{
                          padding: '6px 10px',
                          borderRadius: 6,
                          border: '1px solid #d1d5db',
                          background: '#fff',
                          cursor: 'pointer',
                          fontSize: 12,
                        }}
                      >
                        刷新状态
                      </button>
                    </div>
                  </div>
                  {copyHint ? (
                    <div style={{ marginBottom: 8, fontSize: 12, color: copyHint.startsWith('复制失败') ? '#b91c1c' : '#059669' }}>
                      {copyHint}
                    </div>
                  ) : null}
                  <pre
                    style={{
                      margin: 0,
                      padding: 12,
                      background: '#0b1220',
                      color: '#e5e7eb',
                      borderRadius: 8,
                      overflowX: 'auto',
                      fontSize: 12,
                    }}
                  >
                    {zeroCmdSnapshot || zeroCommand}
                  </pre>
                </div>
              ) : null}

              <div style={{ marginBottom: 12, fontSize: 14, color: '#374151' }}>等待目标机执行一键命令…</div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 13, color: '#6b7280' }}>进度</div>
                <div style={{ height: 10, background: '#f3f4f6', borderRadius: 8 }}>
                  <div
                    style={{
                      width: `${Math.min(100, Number(task?.progress ?? (zeroToken ? 5 : 0)))}%`,
                      height: '100%',
                      background: '#2563eb',
                      borderRadius: 8,
                    }}
                  />
                </div>
              </div>

              <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 6 }}>
                阶段：{String(task?.stage ?? (zeroToken ? 'waiting' : '-'))}
              </div>

              <div style={{ maxHeight: 320, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 8 }}>
                <pre style={{ margin: 0, padding: 12, fontFamily: 'monospace', fontSize: 12, whiteSpace: 'pre-wrap' }}>
                  {(Array.isArray(task?.logs) ? task?.logs : [])
                    .map((log) => `[${log.ts}] ${log.level}: ${log.message}`)
                    .join('\n')}
                </pre>
              </div>

              {String(task?.status || '').toLowerCase() === 'failed' && (
                <div style={{ marginTop: 12, fontSize: 13, color: '#991b1b' }}>
                  失败：请检查网络/权限（sudo）/端口占用，并重新执行命令
                </div>
              )}
              {String(task?.status || '').toLowerCase() === 'success' && (
                <div style={{ marginTop: 12, fontSize: 13, color: '#065f46' }}>
                  安装成功，已配置为开机自启（systemd）。可前往设备页连接测试。
                </div>
              )}
            </div>
          )}
        </div>

        <div
          style={{
            padding: 16,
            borderTop: '1px solid #e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontSize: 13, color: '#6b7280' }}>步骤 {step} / 2</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={onClose}
              style={{ padding: '8px 14px', borderRadius: 6, border: '1px solid #d1d5db', background: '#fff' }}
            >
              取消
            </button>
            {step === 2 ? (
              <>
                {String(task?.status || '').toLowerCase() === 'failed' ? (
                  <button
                    onClick={() => {
                      setTask(null);
                      setSubmitting(false);
                      setZeroToken(null);
                      setZero(null);
                      setZeroCmdSnapshot('');
                      setStep(1);
                    }}
                    style={{ padding: '8px 14px', borderRadius: 6, border: '1px solid #d1d5db', background: '#fff' }}
                  >
                    重试
                  </button>
                ) : null}
                <button
                  onClick={onClose}
                  style={{ padding: '8px 14px', borderRadius: 6, border: '1px solid #d1d5db', background: '#fff' }}
                >
                  关闭
                </button>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
