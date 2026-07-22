export interface CreateWebRtcSessionParams {
  deviceId?: string;
  agentId?: string;
  cameraId?: string;
  runId?: string;
  scenarioId?: string;
  onTrack?: (event: RTCTrackEvent) => void;
}

function waitIceGatheringComplete(pc: RTCPeerConnection, timeoutMs: number): Promise<void> {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') {
      resolve();
      return;
    }
    const timer = globalThis.setTimeout(() => {
      pc.removeEventListener('icegatheringstatechange', onState);
      resolve();
    }, Math.max(0, timeoutMs));
    const onState = () => {
      if (pc.iceGatheringState === 'complete') {
        globalThis.clearTimeout(timer);
        pc.removeEventListener('icegatheringstatechange', onState);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', onState);
  });
}

/**
 * 浏览器侧 WebRTC 客户端占位实现：
 * - 创建 RTCPeerConnection
 * - 生成 offer 并通过 /api/webrtc/offer 发送到平台，再由平台转发给 Agent
 * - 等待 answer 并设置为 remote description
 *
 * 媒体轨道的真正来源（摄像头视频）需要在 Agent 端实现 webrtc offer 处理逻辑后才能看到画面。
 */
export async function createWebRtcSession(params: CreateWebRtcSessionParams) {
  const pc = new RTCPeerConnection({
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
    ],
  });

  if (params.onTrack) {
    pc.addEventListener('track', params.onTrack);
  }

  const offer = await pc.createOffer({
    offerToReceiveVideo: true,
    offerToReceiveAudio: false,
  } as any);
  await pc.setLocalDescription(offer);

  await waitIceGatheringComplete(pc, 2500);

  const local = pc.localDescription!;
  const payload: any = {
    sdp: local.sdp ?? '',
    type: local.type,
  };
  if (params.deviceId) payload.device_id = Number(params.deviceId);
  if (params.agentId) payload.agent_id = params.agentId;
  if (params.cameraId) payload.camera_id = params.cameraId;
  if (params.runId) payload.run_id = params.runId;
  if (params.scenarioId) payload.scenario_id = params.scenarioId;

  const res = await fetch('/api/webrtc/offer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`webrtc offer failed: ${await res.text()}`);
  }
  const data = await res.json();
  if (!data?.sdp || !data?.type) {
    throw new Error('invalid webrtc answer from server');
  }

  const answer: RTCSessionDescriptionInit = {
    type: data.type as RTCSdpType,
    sdp: data.sdp as string,
  };
  await pc.setRemoteDescription(answer);

  return pc;
}
