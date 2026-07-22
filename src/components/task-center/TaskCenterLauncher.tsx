'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useTaskCenter } from './TaskCenterContext';
import { useI18n } from '@/components/common/I18nProvider';

/** 拖拽位移超过该阈值（px）才视为“发生了拖拽”，否则算点击/双击 */
const DRAG_THRESHOLD_PX = 5;

function RobotAgentIcon({
  active,
  panelOpen,
}: {
  active: boolean;
  panelOpen: boolean;
}) {
  return (
    <svg
      width="56"
      height="56"
      viewBox="0 0 56 56"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={[
        'tc-robot',
        active ? 'is-active' : '',
        panelOpen ? 'is-open' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      aria-hidden
    >
      <defs>
        <radialGradient id="tc_head" cx="35%" cy="25%" r="80%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="55%" stopColor="#f3f6fb" />
          <stop offset="100%" stopColor="#e6edf7" />
        </radialGradient>
        <linearGradient id="tc_edge" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.95" />
          <stop offset="45%" stopColor="#cbd5e1" stopOpacity="0.65" />
          <stop offset="100%" stopColor="#94a3b8" stopOpacity="0.55" />
        </linearGradient>
        <linearGradient id="tc_mask" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#0b1220" stopOpacity="0.92" />
          <stop offset="55%" stopColor="#0b1220" stopOpacity="0.84" />
          <stop offset="100%" stopColor="#111827" stopOpacity="0.88" />
        </linearGradient>
        <linearGradient id="tc_eye" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#60a5fa" />
          <stop offset="40%" stopColor="#38bdf8" />
          <stop offset="100%" stopColor="#2563eb" />
        </linearGradient>
        <radialGradient id="tc_glow" cx="50%" cy="50%" r="55%">
          <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.55" />
          <stop offset="55%" stopColor="#2563eb" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
        </radialGradient>
        <filter id="tc_softShadow" x="-30%" y="-30%" width="160%" height="180%">
          <feDropShadow dx="0" dy="5" stdDeviation="6" floodColor="#0f172a" floodOpacity="0.22" />
          <feDropShadow dx="0" dy="10" stdDeviation="14" floodColor="#0f172a" floodOpacity="0.10" />
        </filter>
        <filter id="tc_glowBlur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* ambient glow (hover/active controlled by CSS opacity) */}
      <g className="tc-robot-glow">
        <circle cx="28" cy="29" r="18" fill="url(#tc_glow)" filter="url(#tc_glowBlur)" />
      </g>

      {/* base shadow */}
      <ellipse cx="28" cy="47" rx="14.5" ry="4.2" fill="#0f172a" opacity="0.14" />

      {/* body/base */}
      <g filter="url(#tc_softShadow)">
        <path
          d="M18.5 34.5C18.5 32.0147 20.5147 30 23 30H33C35.4853 30 37.5 32.0147 37.5 34.5V40.5C37.5 43.5376 35.0376 46 32 46H24C20.9624 46 18.5 43.5376 18.5 40.5V34.5Z"
          fill="url(#tc_head)"
          stroke="url(#tc_edge)"
          strokeWidth="1.2"
        />

        {/* tiny side fins/arms (minimal) */}
        <path
          d="M17.3 36.4C15.8 36.9 14.9 38.1 14.9 39.6C14.9 40.6 15.3 41.4 16.0 42.0"
          stroke="#cbd5e1"
          strokeWidth="1.4"
          strokeLinecap="round"
          opacity="0.85"
        />
        <path
          d="M38.7 36.4C40.2 36.9 41.1 38.1 41.1 39.6C41.1 40.6 40.7 41.4 40.0 42.0"
          stroke="#cbd5e1"
          strokeWidth="1.4"
          strokeLinecap="round"
          opacity="0.85"
        />
      </g>

      {/* head */}
      <g filter="url(#tc_softShadow)">
        <path
          d="M14.8 22.5C14.8 16.1837 19.9837 11 26.3 11H29.7C36.0163 11 41.2 16.1837 41.2 22.5V27.4C41.2 28.9464 39.9464 30.2 38.4 30.2H17.6C16.0536 30.2 14.8 28.9464 14.8 27.4V22.5Z"
          fill="url(#tc_head)"
          stroke="url(#tc_edge)"
          strokeWidth="1.2"
        />

        {/* face mask */}
        <path
          d="M18.0 21.8C18.0 19.4 20.0 17.4 22.4 17.4H33.6C36.0 17.4 38.0 19.4 38.0 21.8V24.5C38.0 26.9 36.0 28.9 33.6 28.9H22.4C20.0 28.9 18.0 26.9 18.0 24.5V21.8Z"
          fill="url(#tc_mask)"
          opacity="0.98"
        />

        {/* eyes/status bar */}
        <g className="tc-robot-eyes" filter="url(#tc_glowBlur)">
          <rect x="22.1" y="21.6" width="4.8" height="3.0" rx="1.5" fill="url(#tc_eye)" />
          <rect x="29.1" y="21.6" width="4.8" height="3.0" rx="1.5" fill="url(#tc_eye)" />
          {/* tiny status ticks */}
          <path d="M21.2 26.2H34.8" stroke="url(#tc_eye)" strokeWidth="1.2" strokeLinecap="round" opacity="0.55" />
          <path d="M23.2 26.2H26.2" stroke="url(#tc_eye)" strokeWidth="1.2" strokeLinecap="round" opacity="0.9" />
        </g>

        {/* small top indicator (not chat bubble) */}
        <circle cx="28" cy="13.8" r="1.6" fill="#93c5fd" opacity="0.9" />
        <circle cx="28" cy="13.8" r="3.2" fill="#60a5fa" opacity="0.12" />
      </g>
    </svg>
  );
}

export default function TaskCenterLauncher() {
  const { t } = useI18n();
  const {
    activeCount,
    panelOpen,
    setPanelOpen,
    assistantVisible,
    assistantPosition,
    setAssistantPosition,
    persistAssistantPosition,
    hideAssistant,
  } = useTaskCenter();

  const hasActive = activeCount > 0;

  const [hovered, setHovered] = useState(false);
  /** 拖拽过程中的位置仅存本地，不写 Context，避免整树重渲染 */
  const [dragPosition, setDragPosition] = useState<{ x: number; y: number } | null>(null);
  const draggingRef = useRef(false);
  const dragStartRef = useRef<{ pointerId: number; startX: number; startY: number; baseX: number; baseY: number } | null>(
    null
  );
  const movedRef = useRef(false);
  const latestDragPosRef = useRef<{ x: number; y: number } | null>(null);
  if (dragPosition) latestDragPosRef.current = dragPosition;

  const displayPosition = dragPosition ?? assistantPosition;

  const posStyle = useMemo(
    () => ({
      position: 'fixed' as const,
      left: displayPosition.x,
      top: displayPosition.y,
      zIndex: 1310,
      touchAction: 'none' as const,
      userSelect: 'none' as const,
      pointerEvents: 'auto' as const,
    }),
    [displayPosition.x, displayPosition.y]
  );

  useEffect(() => {
    const stop = () => {
      draggingRef.current = false;
      dragStartRef.current = null;
      movedRef.current = false;
      setDragPosition(null);
    };
    window.addEventListener('blur', stop);
    return () => window.removeEventListener('blur', stop);
  }, []);

  if (!assistantVisible) return null;

  return (
    <div
      style={{ ...posStyle, width: 64, height: 64 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        onClick={() => {
          if (movedRef.current) return;
          setPanelOpen(!panelOpen);
        }}
        onDoubleClick={() => {
          if (movedRef.current) return;
          setPanelOpen(true);
        }}
        title={t('backgroundTasks.launcherTitle')}
        aria-label={
          activeCount > 0
            ? t('backgroundTasks.launcherAriaLabel', { n: activeCount })
            : t('backgroundTasks.launcherTitle')
        }
        className={[
          'tc-launcher',
          hasActive ? 'has-active' : '',
          panelOpen ? 'is-open' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          (e.currentTarget as HTMLButtonElement).setPointerCapture(e.pointerId);
          draggingRef.current = true;
          movedRef.current = false;
          latestDragPosRef.current = null;
          setDragPosition(null);
          dragStartRef.current = {
            pointerId: e.pointerId,
            startX: e.clientX,
            startY: e.clientY,
            baseX: displayPosition.x,
            baseY: displayPosition.y,
          };
        }}
        onPointerMove={(e) => {
          if (!draggingRef.current) return;
          const st = dragStartRef.current;
          if (!st || st.pointerId !== e.pointerId) return;
          const dx = e.clientX - st.startX;
          const dy = e.clientY - st.startY;
          if (!movedRef.current && Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) movedRef.current = true;
          if (movedRef.current) {
            setDragPosition({ x: st.baseX + dx, y: st.baseY + dy });
          }
        }}
        onPointerUp={(e) => {
          const st = dragStartRef.current;
          if (!st || st.pointerId !== e.pointerId) return;
          const hadDrag = movedRef.current;
          const finalPos = hadDrag ? latestDragPosRef.current : null;
          draggingRef.current = false;
          dragStartRef.current = null;
          latestDragPosRef.current = null;
          setDragPosition(null);
          if (hadDrag && finalPos) {
            setAssistantPosition(finalPos);
            persistAssistantPosition(finalPos);
            requestAnimationFrame(() => {
              movedRef.current = false;
            });
          }
        }}
        onPointerCancel={() => {
          draggingRef.current = false;
          dragStartRef.current = null;
          movedRef.current = false;
          latestDragPosRef.current = null;
          setDragPosition(null);
        }}
      >
        <span className="tc-launcher-robot" aria-hidden>
          <RobotAgentIcon active={hasActive} panelOpen={panelOpen} />
        </span>

        {activeCount > 0 && (
          <span className="tc-badge" aria-hidden>
            {activeCount > 99 ? '99+' : activeCount}
          </span>
        )}
      </button>

      {/* hover close button：与主 button 平级，避免 button 嵌套导致 hydration 报错 */}
      {hovered && (
        <button
          type="button"
          className="tc-close"
          aria-label={t('backgroundTasks.closeRobotAriaLabel')}
          onClick={(e) => {
            e.stopPropagation();
            hideAssistant();
          }}
          onPointerDown={(e) => {
            e.stopPropagation();
          }}
        >
          ×
        </button>
      )}

      <style jsx>{`
        .tc-launcher {
          width: 64px;
          height: 64px;
          padding: 0;
          border: none;
          background: transparent;
          cursor: pointer;
          position: relative;
          display: grid;
          place-items: center;
          outline: none;
          transform: translateY(0px);
          transition: transform 140ms ease, filter 140ms ease;
          filter: drop-shadow(0 10px 26px rgba(15, 23, 42, 0.10));
        }

        .tc-launcher:active {
          cursor: grabbing;
        }

        .tc-launcher-robot {
          width: 64px;
          height: 64px;
          display: grid;
          place-items: center;
          transform-origin: 50% 70%;
          animation: tcFloat 3.8s ease-in-out infinite;
        }

        /* “容器感”极轻：仅在 hover/open 给一点玻璃感边缘 */
        .tc-launcher::before {
          content: '';
          position: absolute;
          inset: 6px;
          border-radius: 18px;
          border: 1px solid rgba(148, 163, 184, 0.0);
          background: rgba(255, 255, 255, 0.0);
          box-shadow: 0 0 0 rgba(0, 0, 0, 0);
          transition: border-color 140ms ease, background 140ms ease, box-shadow 140ms ease;
          pointer-events: none;
        }

        .tc-launcher:hover {
          transform: translateY(-2px);
          filter: drop-shadow(0 14px 34px rgba(15, 23, 42, 0.14));
        }
        .tc-launcher:hover::before {
          border-color: rgba(148, 163, 184, 0.55);
          background: rgba(248, 250, 252, 0.55);
          box-shadow: 0 18px 44px rgba(15, 23, 42, 0.10);
          backdrop-filter: blur(6px);
        }

        .tc-launcher.is-open {
          transform: translateY(-1px);
          filter: drop-shadow(0 16px 40px rgba(15, 23, 42, 0.16));
        }
        .tc-launcher.is-open::before {
          border-color: rgba(148, 163, 184, 0.60);
          background: rgba(248, 250, 252, 0.70);
          box-shadow: 0 22px 58px rgba(15, 23, 42, 0.12);
          backdrop-filter: blur(8px);
        }

        /* badge：右上角但不挡眼睛，整体更“分层” */
        .tc-badge {
          position: absolute;
          top: -2px;
          right: -3px;
          min-width: 16px;
          height: 16px;
          border-radius: 999px;
          background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
          color: #ffffff;
          font-size: 10px;
          font-weight: 800;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 0 5px;
          border: 2px solid rgba(255, 255, 255, 0.92);
          box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
          pointer-events: none;
        }

        .tc-close {
          position: absolute;
          top: -6px;
          right: 8px;
          width: 18px;
          height: 18px;
          border-radius: 999px;
          border: 1px solid rgba(148, 163, 184, 0.65);
          background: rgba(248, 250, 252, 0.85);
          color: #64748b;
          display: grid;
          place-items: center;
          font-size: 14px;
          line-height: 1;
          cursor: pointer;
          box-shadow: 0 10px 22px rgba(15, 23, 42, 0.10);
          transition: background 120ms ease, border-color 120ms ease, color 120ms ease, transform 120ms ease;
          z-index: 2;
        }
        .tc-close:hover {
          background: rgba(255, 255, 255, 0.95);
          border-color: rgba(148, 163, 184, 0.8);
          color: #334155;
          transform: translateY(-1px);
        }

        /* SVG 内部：眼睛/光晕的“工作状态”增强，克制但有存在感 */
        :global(.tc-robot .tc-robot-glow) {
          opacity: 0.28;
          transition: opacity 160ms ease;
        }
        .tc-launcher:hover :global(.tc-robot .tc-robot-glow) {
          opacity: 0.40;
        }
        .tc-launcher.has-active :global(.tc-robot .tc-robot-glow) {
          opacity: 0.34;
        }
        .tc-launcher.has-active:hover :global(.tc-robot .tc-robot-glow) {
          opacity: 0.48;
        }

        :global(.tc-robot .tc-robot-eyes) {
          opacity: 0.88;
          transform-origin: 28px 24px;
          transition: opacity 160ms ease, filter 160ms ease;
        }
        .tc-launcher:hover :global(.tc-robot .tc-robot-eyes) {
          opacity: 1;
        }
        .tc-launcher.has-active :global(.tc-robot .tc-robot-eyes) {
          opacity: 1;
          animation: tcWorkPulse 1.6s ease-in-out infinite;
        }
        .tc-launcher.is-open :global(.tc-robot .tc-robot-eyes) {
          opacity: 1;
        }

        /* 轻呼吸浮动（默认态就有一点点“装置在运行”） */
        @keyframes tcFloat {
          0%,
          100% {
            transform: translateY(0px) rotate(0deg);
          }
          50% {
            transform: translateY(-1.2px) rotate(-0.2deg);
          }
        }

        /* 进行中：眼睛更亮一点点（不是游戏风的强闪） */
        @keyframes tcWorkPulse {
          0%,
          100% {
            filter: drop-shadow(0 0 0 rgba(96, 165, 250, 0.0));
          }
          50% {
            filter: drop-shadow(0 0 6px rgba(96, 165, 250, 0.26));
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .tc-launcher-robot {
            animation: none;
          }
          .tc-launcher.has-active :global(.tc-robot .tc-robot-eyes) {
            animation: none;
          }
        }
      `}</style>
    </div>
  );
}
