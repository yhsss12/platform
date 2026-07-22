// 设备类型（历史数据里存在非 ROS* 类型，需兼容）
export type DeviceDriverType = "ROS" | "ROS2" | "PLCNEXT" | "OPCUA" | "HTTP" | "MOCK";
export type DeviceStatus = "DISCONNECTED" | "CONNECTING" | "CONNECTED" | "ERROR";

// 运行时状态（云端通过 Agent 推断）
export type DeviceRuntimeStatus =
  | "OFFLINE"
  | "ONLINE_IDLE"
  | "LAUNCHING"
  | "READY"
  | "COLLECTING"
  | "ERROR";

// ROS2 配置类型
export type ConnectionMode = "fastdds_tailscale_peer" | "lan_multicast";

export type ROS2Config = {
  mode: ConnectionMode;
  localBindIp?: string;
  domainId?: number;
  peerIps: string[];
  discoveryProtocol?: string;
  initialAnnouncementsCount?: number;
  initialAnnouncementsPeriodSec?: number;
  profilePath?: string;
};

// 启动脚本配置
export type LaunchConfig = {
  id?: number;
  scriptPath: string;
  scriptArgs?: string;
  stopScriptPath?: string;
  stopScriptArgs?: string;
  envVars?: Record<string, string>;
};

// 设备测试结果类型
export type DeviceTestResult = {
  status: "untested" | "success" | "fail";
  nodeCount?: number;
  nodesSample?: string[];
  topicCount?: number;
  topicsSample?: string[];
  errorType?: string;
  errorMessage?: string;
  testedAt?: string;
};

export type RobotDevice = {
  id: string;
  name: string;          // 必填：设备名称
  vendor?: string;       // 可选
  model?: string;        // 可选
  deviceType: DeviceDriverType;  // ROS 或 ROS2

  /** 设备所属团队（用于可见性过滤与展示） */
  teamId?: string | null;
  teamName?: string | null;

  // 采集端硬件/Agent 元数据（用于“设备主动添加”）
  hardwareUuid?: string;
  hostname?: string;
  agentIp?: string;
  agentPort?: number;
  agentStatus?: string;
  cameraList?: string[];

  // IP GeoIP 定位结果（用于“所在地”展示）
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

  // ROS2 配置（仅当 deviceType 为 ROS2 时使用）
  ros2Config?: ROS2Config;

  // 启动脚本配置
  launchConfig?: LaunchConfig;

  /** 数据采集脚本（采集端绝对路径）；与任务「相机数据格式」压缩/原图对应，未配置时用平台默认路径 */
  collectScriptCompress?: string | null;
  collectScriptRaw?: string | null;

  // 旧版兼容字段（保留用于向后兼容）
  driverType?: DeviceDriverType;
  connection?: {
    host?: string;
    port?: number;
    namespace?: string;
    domainId?: number;
    endpoint?: string;
    username?: string;
    password?: string;
    plcIp?: string;
    sshPort?: number;
    rdpPort?: number;
    token?: string;
  };

  status: DeviceStatus;
  runtimeStatus?: DeviceRuntimeStatus;
  tags?: string[];       // 可选：如 ["RM65", "双臂", "实验室A"]
  lastTestResult?: DeviceTestResult;  // 最新测试结果
  createdAt: string;
  updatedAt: string;
};

export interface GetConnSummaryOptions {
  /** 设备 ID 前缀文案，如「设备ID」 */
  deviceIdLabel?: string;
  /** 「未配置」的文案（与列表页 i18n 一致） */
  notConfiguredLabel?: string;
}

/**
 * 获取设备连接信息摘要（含采集端地址、ROS2 配置；末尾附设备 ID 便于区分）
 */
export function getConnSummary(device: RobotDevice, options?: GetConnSummaryOptions): string {
  const notConfigured = options?.notConfiguredLabel ?? '未配置';
  // 连接信息列：与设备详情页保持一致，展示同一字段的“设备ID值”
  // 设备详情页渲染的是 device.hardwareUuid（常见为 MAC 地址/采集端硬件标识）
  const deviceIdValue = (device.hardwareUuid || '').trim() || (device.id || '').trim();
  if (!deviceIdValue) return notConfigured;
  return deviceIdValue;
}

/**
 * 获取设备状态（优先使用测试结果）
 */
export function getDeviceStatus(device: RobotDevice): DeviceStatus {
  // 如果有测试结果，根据测试结果判断状态
  if (device.lastTestResult) {
    if (device.lastTestResult.status === 'success') {
      return 'CONNECTED';
    } else if (device.lastTestResult.status === 'fail') {
      return 'ERROR';
    }
  }
  return device.status || 'DISCONNECTED';
}

