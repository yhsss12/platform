'use client';

import ViewportTile from './ViewportTile';

interface CenterPanelProps {
  viewportTopics: (string | null)[];
  onViewportTopicChange: (index: number, topic: string | null) => void;
  onAddSlot: () => void;
  onRemoveSlot: (index: number) => void;
  currentFrame: number;
  selectedEpisode: string | null;
  selectedEpisodeInfo: any;
  taskId?: string;
  /** 按数据资产查看时传入，用于走 /api/data-assets/frames */
  assetId?: string;
  isPlaying?: boolean;
  currentTimeSec?: number;
  onFrameIndexChange?: (index: number) => void;
  playbackFps?: number;
  loopPlayback?: boolean;
  onPlaybackLooped?: () => void;
}

export default function CenterPanel({
  viewportTopics,
  onViewportTopicChange,
  onAddSlot,
  onRemoveSlot,
  currentFrame,
  selectedEpisode,
  selectedEpisodeInfo,
  taskId,
  assetId,
  isPlaying = false,
  currentTimeSec,
  onFrameIndexChange,
  playbackFps = 15,
  loopPlayback = true,
  onPlaybackLooped,
}: CenterPanelProps) {
  const canAddMore = viewportTopics.length < 9;
  const totalCells = viewportTopics.length + (canAddMore ? 1 : 0);
  const rows = Math.ceil(totalCells / 3) || 1;

  return (
    <div
      style={{
        height: '100%',
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
        gap: '8px',
        backgroundColor: '#f6f7f9',
        padding: '8px',
        overflow: 'hidden',
      }}
    >
      {viewportTopics.map((topic, index) => (
        <ViewportTile
          key={index}
          topic={topic}
          onTopicChange={(t) => onViewportTopicChange(index, t)}
          currentFrame={currentFrame}
          showClose
          onClose={() => onRemoveSlot(index)}
          selectedEpisode={selectedEpisode}
          cameras={selectedEpisodeInfo?.cameras || []}
          taskId={taskId}
          assetId={assetId}
          isMcap={!!(selectedEpisodeInfo?.startTimeNs != null)}
          isPlaying={isPlaying}
          currentTimeSec={currentTimeSec}
          onFrameIndexChange={onFrameIndexChange}
          playbackFps={playbackFps}
          loopPlayback={loopPlayback}
          onPlaybackLooped={onPlaybackLooped}
        />
      ))}
      {canAddMore && (
        <button
          type="button"
          onClick={onAddSlot}
          style={{
            minHeight: '120px',
            backgroundColor: '#f3f4f6',
            border: '2px dashed #d1d5db',
            borderRadius: '8px',
            color: '#9ca3af',
            fontSize: '32px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#e5e7eb';
            e.currentTarget.style.borderColor = '#9ca3af';
            e.currentTarget.style.color = '#6b7280';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = '#f3f4f6';
            e.currentTarget.style.borderColor = '#d1d5db';
            e.currentTarget.style.color = '#9ca3af';
          }}
        >
          +
        </button>
      )}
    </div>
  );
}

