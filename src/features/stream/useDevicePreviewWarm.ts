'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  acquireWarmWebRtcSession,
  getWarmSessionStream,
  releaseWarmWebRtcSession,
  startDevicePreviewWarm,
} from './devicePreviewWarmPool';
import { fetchStreamCameras } from './streamCameraApi';

/**
 * 设备详情页：设备运行后预建 WebRTC 并绑定首路相机到 video。
 */
export function useDevicePreviewWarm(
  deviceId: string | undefined,
  enabled: boolean,
) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [previewCameraId, setPreviewCameraId] = useState<string | null>(null);
  const [hasStream, setHasStream] = useState(false);
  const [warmError, setWarmError] = useState<string | null>(null);
  const acquiredRef = useRef<{ deviceId: string; cameraId: string } | null>(null);

  const attachStream = useCallback((stream: MediaStream | null) => {
    const el = videoRef.current;
    if (!el || !stream) {
      setHasStream(false);
      return;
    }
    el.srcObject = stream;
    void el.play().catch(() => {});
    setHasStream(true);
  }, []);

  useEffect(() => {
    if (!enabled || !deviceId) {
      if (acquiredRef.current) {
        releaseWarmWebRtcSession(acquiredRef.current.deviceId, acquiredRef.current.cameraId);
        acquiredRef.current = null;
      }
      setPreviewCameraId(null);
      setHasStream(false);
      setWarmError(null);
      return;
    }

    let cancelled = false;
    const did = deviceId;

    const run = async () => {
      setWarmError(null);
      try {
        await startDevicePreviewWarm(did, { maxCameras: 4 });
        if (cancelled) return;

        const cameras = await fetchStreamCameras(did);
        const firstId = cameras[0]?.id;
        if (!firstId) {
          setWarmError('暂无可用相机');
          return;
        }

        if (acquiredRef.current) {
          releaseWarmWebRtcSession(acquiredRef.current.deviceId, acquiredRef.current.cameraId);
          acquiredRef.current = null;
        }

        let session = acquireWarmWebRtcSession(did, firstId);
        if (!session) {
          await startDevicePreviewWarm(did, { maxCameras: 4 });
          session = acquireWarmWebRtcSession(did, firstId);
        }
        if (cancelled || !session) {
          if (!cancelled) setWarmError('WebRTC 预热失败');
          return;
        }

        acquiredRef.current = { deviceId: did, cameraId: firstId };
        setPreviewCameraId(firstId);

        const stream = session.stream || getWarmSessionStream(did, firstId);
        if (stream) {
          attachStream(stream);
        } else {
          session.pc.addEventListener(
            'track',
            (ev: RTCTrackEvent) => {
              const [s] = ev.streams;
              if (s) attachStream(s);
            },
            { once: true },
          );
        }
      } catch (e) {
        if (!cancelled) {
          setWarmError(e instanceof Error ? e.message : String(e));
        }
      }
    };

    void run();
    const poll = window.setInterval(() => {
      if (cancelled || !acquiredRef.current) return;
      const s = getWarmSessionStream(acquiredRef.current.deviceId, acquiredRef.current.cameraId);
      if (s && videoRef.current && videoRef.current.srcObject !== s) {
        attachStream(s);
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(poll);
      if (acquiredRef.current) {
        releaseWarmWebRtcSession(acquiredRef.current.deviceId, acquiredRef.current.cameraId);
        acquiredRef.current = null;
      }
    };
  }, [enabled, deviceId, attachStream]);

  return {
    videoRef,
    previewCameraId,
    hasStream,
    warmError,
  };
}
