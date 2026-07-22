export type StreamCameraInfo = {
  id: string;
  name: string;
  url?: string;
};

/** 平台 `/api/stream/list`：按 device_id 解析采集端相机列表 */
export async function fetchStreamCameras(deviceId: string | number): Promise<StreamCameraInfo[]> {
  const params = new URLSearchParams();
  params.set('device_id', String(deviceId));
  const res = await fetch(`/api/stream/list?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`stream list failed: ${res.status}`);
  }
  const data = await res.json();
  if (!Array.isArray(data)) return [];
  return data.map((c: { id?: string; name?: string; url?: string }) => ({
    id: String(c.id || ''),
    name: String(c.name || c.id || ''),
    url: c.url,
  })).filter((c) => c.id);
}
