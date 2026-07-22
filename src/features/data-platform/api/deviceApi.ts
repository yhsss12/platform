/**
 * 设备管理 API
 */
import { apiGet, apiPost, apiPut, apiDelete, ApiResponse } from './client';
import type { RobotDevice, ROS2Config, DeviceTestResult, DeviceDriverType, DeviceStatus } from '../models/device';

/**
 * 后端设备响应格式（临时类型定义）
 */
interface BackendDeviceResponse {
  id: number;
  name: string;
  vendor?: string;
  model?: string;
  device_type: string;
  created_at: string;
  updated_at: string;
  runtime_status?: string;
  hardware_uuid?: string;
  hostname?: string;
  agent_ip?: string;
  agent_port?: number;
  agent_status?: string;
  camera_list?: string[];
  location?: {
    country?: string;
    region?: string;
    city?: string;
    lat?: number;
    lon?: number;
    timezone?: string;
    isp?: string;
    note?: string;
  };
  ros2_config?: {
    mode: string;
    local_bind_ip?: string;
    domain_id: number;
    discovery_protocol: string;
    initial_announcements_count: number;
    initial_announcements_period_sec: number;
    profile_path?: string;
    peer_ips: string[];
  };
  launch_config?: {
    id: number;
    script_path: string;
    script_args?: string;
    stop_script_path?: string;
    stop_script_args?: string;
    env_vars?: Record<string, string>;
  };
  collect_script_compress?: string | null;
  collect_script_raw?: string | null;
  team_id?: string | null;
  team_name?: string | null;
  /** 采集端隧道是否在线；false 时列表应显示未连接（覆盖历史「测试成功」缓存） */
  agent_tunnel_connected?: boolean | null;
  last_test_result?: {
    status: string;
    node_count?: number;
    nodes_sample?: string[];
    topic_count?: number;
    topics_sample?: string[];
    error_type?: string;
    error_message?: string;
    tested_at: string;
  };
}

export interface OnlineAgentItem {
  agent_id: string;
  name?: string;
  host?: string;
  port?: number;
  online?: boolean;
  runtime_status?: string;
  camera_list?: string[];
  tunnel_last_seen_ts?: number | null;
  seconds_since_tunnel_seen?: number | null;
  tunnel_stale?: boolean;
}

/**
 * 将后端设备格式转换为前端格式
 */
function backendDeviceToRobotDevice(backend: BackendDeviceResponse): RobotDevice {
  // 转换设备类型（只支持 ROS 和 ROS2）
  const deviceTypeMap: Record<string, DeviceDriverType> = {
    'ROS': 'ROS',
    'ROS2': 'ROS2',
  };
  const deviceType = deviceTypeMap[backend.device_type] || 'ROS2';

  // 构建 ROS2 配置
  let ros2Config: ROS2Config | undefined;
  if (backend.ros2_config) {
    ros2Config = {
      mode: backend.ros2_config.mode as any,
      localBindIp: backend.ros2_config.local_bind_ip,
      domainId: backend.ros2_config.domain_id,
      peerIps: backend.ros2_config.peer_ips || [],
      discoveryProtocol: backend.ros2_config.discovery_protocol,
      initialAnnouncementsCount: backend.ros2_config.initial_announcements_count,
      initialAnnouncementsPeriodSec: backend.ros2_config.initial_announcements_period_sec,
      profilePath: backend.ros2_config.profile_path,
    };
  }

  // 构建启动配置
  let launchConfig;
  if (backend.launch_config) {
    launchConfig = {
      id: backend.launch_config.id,
      scriptPath: backend.launch_config.script_path,
      scriptArgs: backend.launch_config.script_args,
      stopScriptPath: backend.launch_config.stop_script_path,
      stopScriptArgs: backend.launch_config.stop_script_args,
      envVars: backend.launch_config.env_vars
    };
  }

  // 转换测试结果
  let lastTestResult: DeviceTestResult | undefined;
  if (backend.last_test_result) {
    lastTestResult = {
      status: backend.last_test_result.status === 'success' ? 'success' : backend.last_test_result.status === 'fail' ? 'fail' : 'untested',
      nodeCount: backend.last_test_result.node_count,
      nodesSample: backend.last_test_result.nodes_sample,
      topicCount: backend.last_test_result.topic_count,
      topicsSample: backend.last_test_result.topics_sample,
      errorType: backend.last_test_result.error_type,
      errorMessage: backend.last_test_result.error_message,
      testedAt: backend.last_test_result.tested_at,
    };
  }

  // 转换状态：绑定采集端 Agent 的设备以隧道为准，Agent 离线时一律未连接
  let status: DeviceStatus = 'DISCONNECTED';
  const tunnel = backend.agent_tunnel_connected;
  const tunnelKnown = tunnel === true || tunnel === false;
  if (tunnelKnown && tunnel === false) {
    status = 'DISCONNECTED';
  } else if (backend.last_test_result) {
    if (backend.last_test_result.status === 'success') {
      status = 'CONNECTED';
    } else if (backend.last_test_result.status === 'fail') {
      if (backend.last_test_result.error_type === 'STOPPED') {
        status = 'DISCONNECTED';
      } else {
        status = 'ERROR';
      }
    }
  }

  return {
    id: String(backend.id),
    name: backend.name,
    vendor: backend.vendor,
    model: backend.model,
    deviceType,
    ros2Config,
    launchConfig,
    hardwareUuid: backend.hardware_uuid,
    hostname: backend.hostname,
    agentIp: backend.agent_ip,
    agentPort: backend.agent_port,
    agentStatus: backend.agent_status,
    cameraList: backend.camera_list,
    location: backend.location,
    collectScriptCompress: backend.collect_script_compress ?? undefined,
    collectScriptRaw: backend.collect_script_raw ?? undefined,
    teamId: backend.team_id ?? undefined,
    teamName: backend.team_name ?? undefined,
    lastTestResult,
    status,
    runtimeStatus: backend.runtime_status as any,
    createdAt: backend.created_at,
    updatedAt: backend.updated_at,
  };
}

export interface DeviceCreateRequest {
  name: string;
  vendor?: string;
  model?: string;
  deviceType: string;
  hardwareUuid?: string;
  hostname?: string;
  agentIp?: string;
  agentPort?: number;
  ros2Config?: {
    mode: string;
    localBindIp?: string;
    domainId: number;
    discoveryProtocol: string;
    initialAnnouncementsCount: number;
    initialAnnouncementsPeriodSec: number;
    peerIps: string[];
  };
  launchConfig?: {
    scriptPath: string;
    scriptArgs?: string;
    stopScriptPath?: string;
    stopScriptArgs?: string;
    envVars?: Record<string, string>;
  };
  collectScriptCompress?: string | null;
  collectScriptRaw?: string | null;
}

export interface DeviceUpdateRequest {
  name?: string;
  vendor?: string;
  model?: string;
  deviceType?: string;
  hardwareUuid?: string;
  hostname?: string;
  agentIp?: string;
  agentPort?: number;
  ros2Config?: {
    mode?: string;
    localBindIp?: string;
    domainId?: number;
    discoveryProtocol?: string;
    initialAnnouncementsCount?: number;
    initialAnnouncementsPeriodSec?: number;
    peerIps?: string[];
  };
  launchConfig?: {
    scriptPath?: string;
    scriptArgs?: string;
    stopScriptPath?: string;
    stopScriptArgs?: string;
    envVars?: Record<string, string>;
  };
  collectScriptCompress?: string | null;
  collectScriptRaw?: string | null;
}

export interface TestConnectionResponse {
  success: boolean;
  result: DeviceTestResult;
  message?: string;
}

/**
 * 获取设备列表
 */
export async function listDevices(): Promise<ApiResponse<RobotDevice[]>> {
  const response = await apiGet<BackendDeviceResponse[]>('/api/devices');
  if (response.ok && response.data) {
    return {
      ...response,
      data: response.data.map(backendDeviceToRobotDevice),
    };
  }
  return response as unknown as ApiResponse<RobotDevice[]>;
}

/**
 * 获取设备详情
 */
export async function getDevice(deviceId: string | number): Promise<ApiResponse<RobotDevice>> {
  const response = await apiGet<BackendDeviceResponse>(`/api/devices/${deviceId}`);
  if (response.ok && response.data) {
    return {
      ...response,
      data: backendDeviceToRobotDevice(response.data),
    };
  }
  return response as unknown as ApiResponse<RobotDevice>;
}

/**
 * 创建设备
 */
export async function createDevice(device: DeviceCreateRequest): Promise<ApiResponse<RobotDevice>> {
  // 转换前端字段名为后端字段名
  const backendRequest: any = {
    name: device.name,
    vendor: device.vendor,
    model: device.model,
    device_type: device.deviceType,
  };
  if (device.hardwareUuid !== undefined) backendRequest.hardware_uuid = device.hardwareUuid;
  if (device.hostname !== undefined) backendRequest.hostname = device.hostname;
  if (device.agentIp !== undefined) backendRequest.agent_ip = device.agentIp;
  if (device.agentPort !== undefined) backendRequest.agent_port = device.agentPort;
  
  if (device.ros2Config) {
    backendRequest.ros2_config = {
      mode: device.ros2Config.mode,
      local_bind_ip: device.ros2Config.localBindIp,  // 转换字段名
      domain_id: device.ros2Config.domainId,        // 转换字段名
      discovery_protocol: device.ros2Config.discoveryProtocol,
      initial_announcements_count: device.ros2Config.initialAnnouncementsCount,
      initial_announcements_period_sec: device.ros2Config.initialAnnouncementsPeriodSec,
      peer_ips: device.ros2Config.peerIps || [],
    };
  }

  if (device.launchConfig) {
    backendRequest.launch_config = {
      script_path: device.launchConfig.scriptPath,
      script_args: device.launchConfig.scriptArgs,
      stop_script_path: device.launchConfig.stopScriptPath,
      stop_script_args: device.launchConfig.stopScriptArgs,
      env_vars: device.launchConfig.envVars
    };
  }
  if (device.collectScriptCompress !== undefined) {
    backendRequest.collect_script_compress = device.collectScriptCompress;
  }
  if (device.collectScriptRaw !== undefined) {
    backendRequest.collect_script_raw = device.collectScriptRaw;
  }
  
  const response = await apiPost<BackendDeviceResponse>('/api/devices', backendRequest);
  if (response.ok && response.data) {
    return {
      ...response,
      data: backendDeviceToRobotDevice(response.data),
    };
  }
  return response as unknown as ApiResponse<RobotDevice>;
}

/**
 * 更新设备
 */
export async function updateDevice(
  deviceId: string | number,
  device: DeviceUpdateRequest
): Promise<ApiResponse<RobotDevice>> {
  // 转换前端字段名为后端字段名
  const backendRequest: any = {};
  
  if (device.name !== undefined) backendRequest.name = device.name;
  if (device.vendor !== undefined) backendRequest.vendor = device.vendor;
  if (device.model !== undefined) backendRequest.model = device.model;
  if (device.deviceType !== undefined) backendRequest.device_type = device.deviceType;
  if (device.hardwareUuid !== undefined) backendRequest.hardware_uuid = device.hardwareUuid;
  if (device.hostname !== undefined) backendRequest.hostname = device.hostname;
  if (device.agentIp !== undefined) backendRequest.agent_ip = device.agentIp;
  if (device.agentPort !== undefined) backendRequest.agent_port = device.agentPort;
  
  if (device.ros2Config) {
    backendRequest.ros2_config = {};
    if (device.ros2Config.mode !== undefined) backendRequest.ros2_config.mode = device.ros2Config.mode;
    if (device.ros2Config.localBindIp !== undefined) backendRequest.ros2_config.local_bind_ip = device.ros2Config.localBindIp;  // 转换字段名
    if (device.ros2Config.domainId !== undefined) backendRequest.ros2_config.domain_id = device.ros2Config.domainId;            // 转换字段名
    if (device.ros2Config.discoveryProtocol !== undefined) backendRequest.ros2_config.discovery_protocol = device.ros2Config.discoveryProtocol;
    if (device.ros2Config.initialAnnouncementsCount !== undefined) backendRequest.ros2_config.initial_announcements_count = device.ros2Config.initialAnnouncementsCount;
    if (device.ros2Config.initialAnnouncementsPeriodSec !== undefined) backendRequest.ros2_config.initial_announcements_period_sec = device.ros2Config.initialAnnouncementsPeriodSec;
    if (device.ros2Config.peerIps !== undefined) backendRequest.ros2_config.peer_ips = device.ros2Config.peerIps;
  }

  if (device.launchConfig) {
    backendRequest.launch_config = {};
    if (device.launchConfig.scriptPath !== undefined) backendRequest.launch_config.script_path = device.launchConfig.scriptPath;
    if (device.launchConfig.scriptArgs !== undefined) backendRequest.launch_config.script_args = device.launchConfig.scriptArgs;
    if (device.launchConfig.stopScriptPath !== undefined) backendRequest.launch_config.stop_script_path = device.launchConfig.stopScriptPath;
    if (device.launchConfig.stopScriptArgs !== undefined) backendRequest.launch_config.stop_script_args = device.launchConfig.stopScriptArgs;
    if (device.launchConfig.envVars !== undefined) backendRequest.launch_config.env_vars = device.launchConfig.envVars;
  }
  if (device.collectScriptCompress !== undefined) backendRequest.collect_script_compress = device.collectScriptCompress;
  if (device.collectScriptRaw !== undefined) backendRequest.collect_script_raw = device.collectScriptRaw;
  
  const response = await apiPut<BackendDeviceResponse>(`/api/devices/${deviceId}`, backendRequest);
  if (response.ok && response.data) {
    return {
      ...response,
      data: backendDeviceToRobotDevice(response.data),
    };
  }
  return response as unknown as ApiResponse<RobotDevice>;
}

/**
 * 删除设备
 */
export async function deleteDevice(deviceId: string | number): Promise<ApiResponse<void>> {
  return apiDelete<void>(`/api/devices/${deviceId}`);
}

/**
 * 测试设备连接
 */
export async function testDeviceConnection(
  deviceId: string | number
): Promise<ApiResponse<TestConnectionResponse>> {
  return apiPost<TestConnectionResponse>(`/api/devices/${deviceId}/test-connection`, {});
}

/**
 * 启动设备
 */
export async function launchDevice(deviceId: string | number): Promise<ApiResponse<any>> {
  return apiPost(`/api/devices/${deviceId}/launch`, {});
}

/**
 * 停止设备
 */
export async function stopDevice(deviceId: string | number): Promise<ApiResponse<any>> {
  return apiPost(`/api/devices/${deviceId}/stop`, {});
}

/**
 * 设备主动添加（中心平台通过 IP/Port 探测 Agent）
 */
export async function connectDevice(
  ip: string,
  port: number,
  meta?: {
    name?: string;
    vendor?: string;
    model?: string;
    deviceType?: DeviceDriverType;
    launchConfig?: {
      scriptPath?: string;
      scriptArgs?: string;
      envVars?: Record<string, string>;
    };
  },
): Promise<ApiResponse<RobotDevice>> {
  const payload: any = { ip, port };
  if (meta?.name !== undefined) payload.name = meta.name;
  if (meta?.vendor !== undefined) payload.vendor = meta.vendor;
  if (meta?.model !== undefined) payload.model = meta.model;
  if (meta?.deviceType !== undefined) payload.device_type = meta.deviceType;
  if (meta?.launchConfig?.scriptPath) {
    payload.launch_config = {
      script_path: meta.launchConfig.scriptPath,
      script_args: meta.launchConfig.scriptArgs || '',
      env_vars: meta.launchConfig.envVars || {},
    };
  }

  const response = await apiPost<BackendDeviceResponse>('/api/devices/connect', payload);
  if (response.ok && response.data) {
    return {
      ...response,
      data: backendDeviceToRobotDevice(response.data),
    };
  }
  return response as unknown as ApiResponse<RobotDevice>;
}

/**
 * 在线 Agent 列表（用于“添加设备”下拉）
 */
export async function listOnlineAgents(): Promise<ApiResponse<OnlineAgentItem[]>> {
  return apiGet<OnlineAgentItem[]>('/api/devices/agents/online');
}

/**
 * 通过在线 agent_id 绑定设备（无需手填 IP/Port）
 */
export async function connectDeviceByAgent(
  agentId: string,
  meta?: {
    name?: string;
    vendor?: string;
    model?: string;
    deviceType?: DeviceDriverType;
    hostname?: string;
    ros2Config?: {
      mode: string;
      localBindIp?: string;
      domainId: number;
      discoveryProtocol: string;
      initialAnnouncementsCount: number;
      initialAnnouncementsPeriodSec: number;
      peerIps: string[];
    };
    launchConfig?: {
      scriptPath?: string;
      scriptArgs?: string;
      stopScriptPath?: string;
      stopScriptArgs?: string;
      envVars?: Record<string, string>;
    };
    collectScriptCompress?: string | null;
    collectScriptRaw?: string | null;
  },
): Promise<ApiResponse<RobotDevice>> {
  const payload: any = { agent_id: agentId };
  if (meta?.name !== undefined) payload.name = meta.name;
  if (meta?.vendor !== undefined) payload.vendor = meta.vendor;
  if (meta?.model !== undefined) payload.model = meta.model;
  if (meta?.deviceType !== undefined) payload.device_type = meta.deviceType;
  if (meta?.hostname !== undefined) payload.hostname = meta.hostname;
  if (meta?.ros2Config !== undefined) {
    payload.ros2_config = {
      mode: meta.ros2Config.mode,
      local_bind_ip: meta.ros2Config.localBindIp,
      domain_id: meta.ros2Config.domainId,
      discovery_protocol: meta.ros2Config.discoveryProtocol,
      initial_announcements_count: meta.ros2Config.initialAnnouncementsCount,
      initial_announcements_period_sec: meta.ros2Config.initialAnnouncementsPeriodSec,
      peer_ips: meta.ros2Config.peerIps || [],
    };
  }
  if (meta?.launchConfig?.scriptPath) {
    payload.launch_config = {
      script_path: meta.launchConfig.scriptPath,
      script_args: meta.launchConfig.scriptArgs || '',
      stop_script_path: meta.launchConfig.stopScriptPath || '',
      stop_script_args: meta.launchConfig.stopScriptArgs || '',
      env_vars: meta.launchConfig.envVars || {},
    };
  }
  if (meta?.collectScriptCompress !== undefined) payload.collect_script_compress = meta.collectScriptCompress;
  if (meta?.collectScriptRaw !== undefined) payload.collect_script_raw = meta.collectScriptRaw;
  const response = await apiPost<BackendDeviceResponse>('/api/devices/connect-agent', payload);
  if (response.ok && response.data) {
    return {
      ...response,
      data: backendDeviceToRobotDevice(response.data),
    };
  }
  return response as unknown as ApiResponse<RobotDevice>;
}

export type ScanCollectScriptResult = {
  script_path: string;
  defaults: {
    camera_hz: number;
    gripper_hz: number;
    joint_hz: number;
    force_hz: number;
  };
  topics: Array<{
    topic: string;
    label: string;
    group: string;
    default_min_hz: number;
  }>;
};

/** 在采集端扫描设备配置的采集脚本，解析频率检测话题与默认阈值 */
export async function scanDeviceCollectScript(
  deviceId: string,
  cameraDataFormat: string,
): Promise<ApiResponse<ScanCollectScriptResult>> {
  return apiPost<ScanCollectScriptResult>(`/api/devices/${deviceId}/scan-collect-script`, {
    camera_data_format: cameraDataFormat,
  });
}
