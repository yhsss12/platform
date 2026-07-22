'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { CableThreadingVideoPlayer } from '@/components/workspace/replay/CableThreadingVideoPlayer';
import { EvaluationTrajectorySelector } from '@/components/workspace/replay/EvaluationTrajectorySelector';
import {
  buildRepresentativeVideoHint,
  findTrajectoryIndexByRound,
  getEpisodeIndex,
  getTrajectoryLabel,
  normalizeReplayTrajectoryItems,
  type EvaluationReplayInfo,
  type EvaluationReplayUriItem,
} from '@/lib/workspace/evaluationReplayInfo';

export interface EvaluationReplayVideoSectionProps {
  evalJobId: string;
  replayInfo: EvaluationReplayInfo;
  initialEpisode?: number;
  footerLabel?: string;
}

export function EvaluationReplayVideoSection({
  evalJobId,
  replayInfo,
  initialEpisode,
  footerLabel = 'MuJoCo 评测画面',
}: EvaluationReplayVideoSectionProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(0);

  const replayVideoOptions = useMemo(
    () => normalizeReplayTrajectoryItems(replayInfo),
    [replayInfo]
  );

  useEffect(() => {
    setSelectedVideoIndex(0);
  }, [evalJobId]);

  useEffect(() => {
    if (replayVideoOptions.length <= 1) return;
    const urlRound =
      (initialEpisode && initialEpisode > 0 ? initialEpisode : null) ??
      (Number(searchParams.get('episode') ?? '0') || 0);
    if (urlRound > 0) {
      setSelectedVideoIndex(findTrajectoryIndexByRound(replayVideoOptions, urlRound));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalJobId, replayVideoOptions.length, initialEpisode]);

  useEffect(() => {
    if (selectedVideoIndex >= replayVideoOptions.length) {
      setSelectedVideoIndex(0);
    }
  }, [replayVideoOptions.length, selectedVideoIndex]);

  const handleSelectVideo = useCallback(
    (index: number) => {
      setSelectedVideoIndex(index);
      const item = replayVideoOptions[index];
      if (!item || replayVideoOptions.length <= 1) return;
      const round = getEpisodeIndex(item, index);
      const params = new URLSearchParams(searchParams.toString());
      params.set('episode', String(round));
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [replayVideoOptions, searchParams, pathname, router]
  );

  const selectedVideoOption: EvaluationReplayUriItem | null =
    replayVideoOptions[selectedVideoIndex] ?? replayVideoOptions[0] ?? null;
  const videoApiPath = selectedVideoOption?.uri ?? replayInfo.replayUri ?? null;
  const replayVideoHint = useMemo(() => buildRepresentativeVideoHint(replayInfo), [replayInfo]);
  const showTrajectorySelector = replayVideoOptions.length > 1;
  const representativeHint =
    replayInfo.isRepresentativeVideo || replayVideoOptions.length === 1
      ? replayInfo.warning ?? replayVideoHint
      : replayInfo.warning ?? replayVideoHint;

  return (
    <>
      {showTrajectorySelector ? (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 10,
          }}
        >
          <span style={{ fontSize: 14, color: '#64748b', flexShrink: 0 }}>选择轨迹：</span>
          <EvaluationTrajectorySelector
            replayItems={replayVideoOptions}
            selectedIndex={selectedVideoIndex}
            onSelect={handleSelectVideo}
          />
        </div>
      ) : null}
      <div className="replay-player-shell">
        <div className="replay-player">
          <div className="replay-player-media">
            {videoApiPath ? (
              <CableThreadingVideoPlayer videoJobId={evalJobId} videoApiPath={videoApiPath} />
            ) : (
              <div
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#94a3b8',
                  fontSize: 13,
                  padding: 24,
                  textAlign: 'center',
                }}
              >
                评测视频尚未生成。
              </div>
            )}
          </div>
        </div>
      </div>
      {representativeHint ? (
        <p style={{ margin: '8px 0 0', fontSize: 12, color: '#64748b', textAlign: 'center' }}>
          {representativeHint}
        </p>
      ) : null}
      <div
        style={{
          marginTop: 8,
          fontSize: 12,
          color: '#64748b',
          textAlign: 'center',
        }}
      >
        {selectedVideoOption ? getTrajectoryLabel(selectedVideoOption, selectedVideoIndex) : footerLabel}
      </div>
    </>
  );
}
