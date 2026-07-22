import type { RobotDevice } from '../models/device';

const STORAGE_KEY = 'eai_devices_v1';

/**
 * 派发设备变更事件
 */
function dispatchDevicesChanged(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('eai:devices_changed'));
  }
}

/**
 * 获取种子数据（8个设备）
 */
function getSeedData(): RobotDevice[] {
  const now = new Date().toISOString();
  return [
    {
      id: 'device-001',
      name: 'Galaxea R1 Pro',
      vendor: '一湃智能',
      model: 'R1 Pro',
      deviceType: 'ROS2',
      driverType: 'ROS2',
      connection: {
        host: '127.0.0.1',
        port: 11311,
        namespace: '/galaxea',
        domainId: 0,
      },
      status: 'DISCONNECTED',
      tags: ['RM65', '双臂', '实验室A'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-002',
      name: 'PLCnext-172.18.0.108',
      vendor: 'Phoenix Contact',
      model: 'PLCnext',
      deviceType: 'PLCNEXT',
      driverType: 'PLCNEXT',
      connection: {
        plcIp: '172.18.0.108',
        sshPort: 22,
        rdpPort: 3389,
        username: 'admin',
        password: '',
      },
      status: 'DISCONNECTED',
      tags: ['PLC', '产线A'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-003',
      name: 'RealMan RM65',
      vendor: '一湃智能',
      model: 'RM65',
      deviceType: 'ROS2',
      driverType: 'ROS2',
      connection: {
        host: '192.168.1.100',
        port: 11311,
        namespace: '/rm65',
        domainId: 0,
      },
      status: 'DISCONNECTED',
      tags: ['RM65', '双臂', '实验室B'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-004',
      name: 'OPCUA-Server-01',
      vendor: 'Siemens',
      model: 'S7-1500',
      deviceType: 'OPCUA',
      driverType: 'OPCUA',
      connection: {
        endpoint: 'opc.tcp://192.168.1.200:4840',
        username: 'admin',
        password: '',
      },
      status: 'DISCONNECTED',
      tags: ['OPCUA', '产线B'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-005',
      name: 'HTTP-API-Device',
      vendor: 'Custom',
      model: 'API Gateway',
      deviceType: 'HTTP',
      driverType: 'HTTP',
      connection: {
        endpoint: 'https://api.example.com',
        token: 'token123',
      },
      status: 'DISCONNECTED',
      tags: ['HTTP', '云端'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-006',
      name: 'Mock-Device-01',
      vendor: 'Test',
      model: 'Mock',
      deviceType: 'MOCK',
      driverType: 'MOCK',
      connection: {},
      status: 'DISCONNECTED',
      tags: ['测试', '模拟'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-007',
      name: 'Galaxea R2',
      vendor: '一湃智能',
      model: 'R2',
      deviceType: 'ROS2',
      driverType: 'ROS2',
      connection: {
        host: '192.168.1.101',
        port: 11311,
        namespace: '/galaxea_r2',
        domainId: 0,
      },
      status: 'DISCONNECTED',
      tags: ['RM65', '单臂', '实验室C'],
      createdAt: now,
      updatedAt: now,
    },
    {
      id: 'device-008',
      name: 'PLCnext-192.168.1.50',
      vendor: 'Phoenix Contact',
      model: 'PLCnext',
      deviceType: 'PLCNEXT',
      driverType: 'PLCNEXT',
      connection: {
        plcIp: '192.168.1.50',
        sshPort: 22,
        rdpPort: 3389,
        username: 'admin',
        password: '',
      },
      status: 'DISCONNECTED',
      tags: ['PLC', '产线C'],
      createdAt: now,
      updatedAt: now,
    },
  ];
}

/**
 * 从 localStorage 加载设备列表
 */
export function loadDevices(): RobotDevice[] {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const devices = JSON.parse(stored) as RobotDevice[];
      if (Array.isArray(devices) && devices.length > 0) {
        return devices;
      }
    }
  } catch (error) {
    console.error('Failed to load devices from localStorage:', error);
  }

  // 如果没有数据，返回种子数据（但不立即保存，由调用方决定）
  return getSeedData();
}

/**
 * 初始化种子数据（如果 localStorage 为空）
 */
export function initDevicesIfNeeded(): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      const seedData = getSeedData();
      saveDevices(seedData);
    }
  } catch (error) {
    console.error('Failed to init devices:', error);
  }
}

/**
 * 保存设备列表到 localStorage
 */
export function saveDevices(devices: RobotDevice[]): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(devices));
    dispatchDevicesChanged();
  } catch (error) {
    console.error('Failed to save devices to localStorage:', error);
  }
}

/**
 * 添加设备
 */
export function addDevice(device: RobotDevice): void {
  const devices = loadDevices();
  devices.push(device);
  saveDevices(devices);
}

/**
 * 更新设备
 */
export function updateDevice(id: string, patch: Partial<RobotDevice>): void {
  const devices = loadDevices();
  const index = devices.findIndex(d => d.id === id);
  if (index >= 0) {
    devices[index] = {
      ...devices[index],
      ...patch,
      updatedAt: new Date().toISOString(),
    };
    saveDevices(devices);
  }
}

/**
 * 删除设备
 */
export function removeDevice(id: string): void {
  const devices = loadDevices();
  const filtered = devices.filter(d => d.id !== id);
  saveDevices(filtered);
}

/**
 * 根据ID获取设备
 */
export function getDeviceById(id: string): RobotDevice | undefined {
  const devices = loadDevices();
  return devices.find(d => d.id === id);
}

/**
 * 重置设备列表为种子数据
 */
export function resetDevicesToSeed(): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const seedData = getSeedData();
    saveDevices(seedData);
  } catch (error) {
    console.error('Failed to reset devices to seed:', error);
  }
}

