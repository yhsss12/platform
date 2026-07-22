'use client';

import { useRef } from 'react';
import { useI18n } from '@/components/common/I18nProvider';

interface BottomTimelineProps {
  currentFrame: number;
  maxFrame: number;
  isPlaying: boolean;
  speed: number;
  onFrameChange: (frame: number) => void;
  onPlay: () => void;
  onPause: () => void;
  onPreviousFrame: () => void;
  onNextFrame: () => void;
  onSpeedChange: (speed: number) => void;
  onRefresh: () => void;
  /** MCAP: 首尾时间戳（纳秒），存在时进度条以时间显示 */
  startTimeNs?: number;
  endTimeNs?: number;
  /** 开始/结束拖动进度条时调用，父组件可用来在拖动期间忽略播放源的帧更新 */
  onSliderDragStart?: () => void;
  onSliderDragEnd?: () => void;
  /** 是否循环播放 */
  loopPlayback?: boolean;
  /** 切换循环播放 */
  onLoopPlaybackChange?: (loop: boolean) => void;
}

export default function BottomTimeline({
  currentFrame,
  maxFrame,
  isPlaying,
  speed,
  onFrameChange,
  onPlay,
  onPause,
  onPreviousFrame,
  onNextFrame,
  onSpeedChange,
  onRefresh,
  startTimeNs,
  endTimeNs,
  onSliderDragStart,
  onSliderDragEnd,
  loopPlayback = true,
  onLoopPlaybackChange,
}: BottomTimelineProps) {
  const { t } = useI18n();
  const percentage = maxFrame > 0 ? (currentFrame / maxFrame) * 100 : 0;
  const wasPlayingBeforeDragRef = useRef(false);

  // MCAP 时间戳模式：根据当前帧计算时间（秒）
  const useTimestamp =
    typeof startTimeNs === 'number' &&
    typeof endTimeNs === 'number' &&
    startTimeNs >= 0 &&
    endTimeNs > startTimeNs;
  const totalDurationSec = useTimestamp ? (endTimeNs! - startTimeNs!) / 1e9 : 0;
  const currentTimeSec =
    useTimestamp && maxFrame > 0
      ? ((endTimeNs! - startTimeNs!) * currentFrame) / Math.max(1, maxFrame) / 1e9
      : 0;

  return (
    <div
      style={{
        height: '84px',
        backgroundColor: '#ffffff',
        borderTop: '1px solid #e5e7eb',
        display: 'flex',
        flexDirection: 'column',
        padding: '12px 16px',
        gap: '12px',
      }}
    >
      {/* Slider - 拖动时暂停，松手后恢复播放 */}
      <div style={{ position: 'relative', width: '100%' }}>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={currentFrame}
          onPointerDown={(e) => {
            (e.currentTarget as HTMLInputElement).setPointerCapture(e.pointerId);
            onSliderDragStart?.();
            if (isPlaying) {
              wasPlayingBeforeDragRef.current = true;
              onPause();
            }
          }}
          onPointerUp={(e) => {
            const el = e.currentTarget as HTMLInputElement;
            if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
            onSliderDragEnd?.();
            if (wasPlayingBeforeDragRef.current) {
              wasPlayingBeforeDragRef.current = false;
              onPlay();
            }
          }}
          onPointerCancel={(e) => {
            const el = e.currentTarget as HTMLInputElement;
            if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
            onSliderDragEnd?.();
            wasPlayingBeforeDragRef.current = false;
          }}
          onChange={(e) => onFrameChange(parseInt(e.target.value, 10))}
          style={{
            width: '100%',
            height: '6px',
            appearance: 'none',
            background: `linear-gradient(to right, #2563eb 0%, #2563eb ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`,
            borderRadius: '3px',
            outline: 'none',
            cursor: 'pointer',
          }}
        />
        <style jsx>{`
          input[type='range']::-webkit-slider-thumb {
            appearance: none;
            width: 16px;
            height: 16px;
            background: #2563eb;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid #ffffff;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
          }
          input[type='range']::-moz-range-thumb {
            width: 16px;
            height: 16px;
            background: #2563eb;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid #ffffff;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
          }
        `}</style>
      </div>

      {/* 控制按钮和信息 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* 播放控制按钮 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={onPreviousFrame}
            style={{
              width: '32px',
              height: '32px',
              padding: 0,
              backgroundColor: '#f9fafb',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#f9fafb';
            }}
          >
            |&lt;
          </button>
          <button
            onClick={isPlaying ? onPause : onPlay}
            style={{
              width: '32px',
              height: '32px',
              padding: 0,
              backgroundColor: '#2563eb',
              border: 'none',
              borderRadius: '6px',
              color: '#ffffff',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#1d4ed8';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#2563eb';
            }}
          >
            {isPlaying ? '❚❚' : '▶'}
          </button>
          <button
            onClick={onNextFrame}
            style={{
              width: '32px',
              height: '32px',
              padding: 0,
              backgroundColor: '#f9fafb',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#f9fafb';
            }}
          >
            &gt;|
          </button>
        </div>

        {/* 右侧信息区 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span
            style={{
              fontSize: '12px',
              color: '#6b7280',
            }}
          >
            {useTimestamp
              ? `index = ${currentFrame}    t = ${currentTimeSec.toFixed(3)} / ${totalDurationSec.toFixed(3)} s`
              : `${t('labelExecutePage.previewFrame')}: ${currentFrame}/${maxFrame}`}
          </span>
          <select
            value={speed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            style={{
              height: '32px',
              padding: '0 24px 0 8px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#111827',
              fontSize: '12px',
              outline: 'none',
              cursor: 'pointer',
              appearance: 'none',
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M6 9L1 4h10z'/%3E%3C/svg%3E")`,
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 8px center',
              transition: 'all 0.2s',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#2563eb';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = '#d1d5db';
            }}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
          </select>
          {onLoopPlaybackChange && (
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '12px',
                color: '#6b7280',
                cursor: 'pointer',
              }}
              title={t('labelExecutePage.previewLoopTitle')}
            >
              <input
                type="checkbox"
                checked={loopPlayback}
                onChange={(e) => onLoopPlaybackChange(e.target.checked)}
                style={{ cursor: 'pointer' }}
              />
              {t('labelExecutePage.previewLoop')}
            </label>
          )}
          <button
            onClick={onRefresh}
            style={{
              width: '32px',
              height: '32px',
              padding: 0,
              backgroundColor: '#f9fafb',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#f9fafb';
            }}
            title={t('labelExecutePage.previewRefreshTitle')}
          >
            ↻
          </button>
        </div>
      </div>
    </div>
  );
}


