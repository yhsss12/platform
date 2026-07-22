'use client';

import React, { useState, useEffect } from 'react';
import type { RobotDevice, DeviceDriverType, ROS2Config, DeviceTestResult } from '../../models/device';
import { testDeviceConnection, createDevice, updateDevice, listOnlineAgents, connectDeviceByAgent, type OnlineAgentItem } from '../../api/deviceApi';
import { useI18n } from '@/components/common/I18nProvider';
import PathPickerModal from '@/features/data-platform/components/label/PathPickerModal';

/** 暂时隐藏设备表单中的 ROS2 设置；恢复时改为 true */
const SHOW_ROS2_SETTINGS = false;

function formatLaunchEnvVarsJson(env?: Record<string, string> | null): string {
  const src = env && typeof env === 'object' ? env : {};
  const keys = Object.keys(src).sort((a, b) => a.localeCompare(b));
  const sorted: Record<string, string> = {};
  for (const k of keys) sorted[k] = String((src as any)[k] ?? '');
  return JSON.stringify(sorted, null, 2);
}

interface AddDeviceModalProps {
  device?: RobotDevice | null;
  open: boolean;
  onClose: () => void;
  onSave: (device: RobotDevice) => void;
  onOpenAgentWizard?: () => void;
}

export default function AddDeviceModal({ device, open, onClose, onSave, onOpenAgentWizard }: AddDeviceModalProps) {
  const [formData, setFormData] = useState<Partial<RobotDevice>>({
    name: '',
    vendor: '',
    model: '',
    hardwareUuid: '',
    hostname: '',
    agentIp: '',
    agentPort: 9100,
    deviceType: 'ROS2',
    ros2Config: {
        mode: 'fastdds_tailscale_peer',
        localBindIp: '',
        domainId: 0,
        discoveryProtocol: 'SIMPLE',
        initialAnnouncementsCount: 5,
        initialAnnouncementsPeriodSec: 1,
        peerIps: [],
      },
    launchConfig: {
      scriptPath: '',
      scriptArgs: '',
      stopScriptPath: '',
      stopScriptArgs: '',
      envVars: {},
    },
    collectScriptCompress: '',
    collectScriptRaw: '',
    tags: [],
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [tagInput, setTagInput] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<DeviceTestResult | null>(null);
  const [onlineAgents, setOnlineAgents] = useState<OnlineAgentItem[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('');
  const [connectingByHandshake, setConnectingByHandshake] = useState(false);
  const [connectedDeviceMeta, setConnectedDeviceMeta] = useState<RobotDevice | null>(null);
  const [toastMsg, setToastMsg] = useState<{ text: string; isError?: boolean } | null>(null);
  const [scriptPickerField, setScriptPickerField] = useState<'launch' | 'stop' | 'collectCompress' | 'collectRaw' | null>(
    null
  );
  const [launchEnvVarsText, setLaunchEnvVarsText] = useState('{}');
  const { t } = useI18n();
  const deviceIdForPathPicker = (connectedDeviceMeta?.id || device?.id || (formData.id as string) || '').trim();
  const agentIdForPathPicker = (selectedAgentId || '').trim();

  const isEditMode = !!device;

  useEffect(() => {
    if (open) {
      if (device) {
        // 编辑模式：填充现有数据
        const deviceType = device.deviceType || device.driverType || 'ROS2';
        const defaultRos2Config = {
          mode: 'fastdds_tailscale_peer',
          localBindIp: '',
          domainId: 0,
          discoveryProtocol: 'SIMPLE',
          initialAnnouncementsCount: 5,
          initialAnnouncementsPeriodSec: 1,
          peerIps: [],
        };

        const nextLaunchConfig = device.launchConfig
          ? { ...device.launchConfig }
          : {
              scriptPath: '',
              scriptArgs: '',
              stopScriptPath: '',
              stopScriptArgs: '',
              envVars: {},
            };

        setFormData({
          name: device.name || '',
          vendor: device.vendor || '',
          model: device.model || '',
          hardwareUuid: device.hardwareUuid || '',
          hostname: device.hostname || '',
          agentIp: device.agentIp || '',
          agentPort: device.agentPort ?? 9100,
          deviceType,
          ros2Config: ((deviceType === 'ROS' || deviceType === 'ROS2') 
            ? (device.ros2Config ? { 
                ...device.ros2Config,
                peerIps: Array.isArray(device.ros2Config.peerIps) ? device.ros2Config.peerIps : [],
              } : defaultRos2Config)
            : undefined) as any,
          launchConfig: nextLaunchConfig,
          collectScriptCompress: device.collectScriptCompress ?? '',
          collectScriptRaw: device.collectScriptRaw ?? '',
          tags: device.tags || [],
        });
        setLaunchEnvVarsText(formatLaunchEnvVarsJson((nextLaunchConfig.envVars as any) || {}));
        const lc: any = nextLaunchConfig;
        const shouldExpandAdvanced =
          Boolean(String(lc?.scriptArgs || '').trim()) ||
          Boolean(String(lc?.stopScriptArgs || '').trim()) ||
          (lc?.envVars && typeof lc.envVars === 'object' && Object.keys(lc.envVars).length > 0) ||
          Boolean(String(device.collectScriptCompress || '').trim()) ||
          Boolean(String(device.collectScriptRaw || '').trim());
        setShowAdvanced(shouldExpandAdvanced);
        setTestResult(device.lastTestResult || null);
      } else {
        // 新建模式：重置表单
        setFormData({
          name: '',
          vendor: '',
          model: '',
          hardwareUuid: '',
          hostname: '',
          agentIp: '',
          agentPort: 9100,
          deviceType: 'ROS2',
          ros2Config: {
            mode: 'fastdds_tailscale_peer',
            localBindIp: '',
            domainId: 0,
            discoveryProtocol: 'SIMPLE',
            initialAnnouncementsCount: 5,
            initialAnnouncementsPeriodSec: 1,
            peerIps: [],
          },
          launchConfig: {
            scriptPath: '/home/sia/workspace/test/start_all_ros_topics.sh', // 默认填入脚本路径
            scriptArgs: '',
            stopScriptPath: '/home/sia/workspace/test/stop_all_ros_topics.sh',
            stopScriptArgs: '',
            envVars: {},
          },
          collectScriptCompress: '/home/sia/workspace/test/collect_data_compressed.sh',
          collectScriptRaw: '/home/sia/workspace/test/collect_data_uncompressed.sh',
          tags: [],
        });
        setLaunchEnvVarsText('{}');
        setShowAdvanced(false);
        setTestResult(null);
        setConnectedDeviceMeta(null);
        setSelectedAgentId('');
      }
      setErrors({});
      setTagInput('');
    }
  }, [open, device]);

  useEffect(() => {
    if (!open || isEditMode) return;
    let cancelled = false;
    const loadAgents = async () => {
      try {
        const res = await listOnlineAgents();
        if (!cancelled && res.ok && Array.isArray(res.data)) {
          setOnlineAgents(res.data);
          if (!selectedAgentId && res.data.length > 0) {
            const preferred = res.data.find((a) => !a.tunnel_stale) || res.data[0];
            setSelectedAgentId(preferred.agent_id);
          }
        }
      } catch {
        if (!cancelled) setOnlineAgents([]);
      }
    };
    void loadAgents();
    return () => {
      cancelled = true;
    };
  }, [open, isEditMode]);

  useEffect(() => {
    if (!open || isEditMode) return;
    if (!selectedAgentId) return;
    const picked = onlineAgents.find((a) => a.agent_id === selectedAgentId);
    setFormData((prev) => {
      const next: any = { ...prev, hardwareUuid: selectedAgentId };
      if (picked?.host && !String(next.agentIp || '').trim()) next.agentIp = picked.host;
      if (picked?.port && !Number(next.agentPort || 0)) next.agentPort = picked.port;
      if (picked?.name && !String(next.hostname || '').trim()) next.hostname = picked.name;
      return next;
    });
  }, [open, isEditMode, selectedAgentId, onlineAgents]);

  const validate = (data: Partial<RobotDevice> = formData): Record<string, string> => {
    const newErrors: Record<string, string> = {};
    const src = data;

    if (!src.name?.trim()) {
      newErrors.name = '设备名称不能为空';
    }

    if (!src.deviceType) {
      newErrors.deviceType = '请选择设备类型';
    }

    const hw = (src.hardwareUuid || '').trim();
    if (!hw) {
      newErrors.hardwareUuid = '请先选择并连接在线 Client';
    } else if (/\s/.test(hw)) {
      newErrors.hardwareUuid = '设备唯一标识不能包含空格';
    }

    const ip = (src.agentIp || '').trim();
    const port = Number(src.agentPort ?? 0) || 0;
    if (ip) {
      if (port < 1 || port > 65535) {
        newErrors.agentPort = '端口范围应为 1~65535';
      }
    }

    // ROS/ROS2 配置验证（SHOW_ROS2_SETTINGS 控制显隐）
    if (SHOW_ROS2_SETTINGS && (src.deviceType === 'ROS' || src.deviceType === 'ROS2')) {
      const config = src.ros2Config as any;
      if (!config) {
        newErrors.ros2Config = 'ROS2配置不能为空';
      } else {
        const domainId = Number(config.domainId ?? 0);
        if (!Number.isFinite(domainId) || domainId < 0 || domainId > 232) {
          newErrors.ros2DomainId = 'Domain ID 范围应为 0~232';
        }
        const dp = String(config.discoveryProtocol || 'SIMPLE');
        if (!['SIMPLE', 'SUPER_CLIENT', 'CLIENT'].includes(dp)) {
          newErrors.ros2DiscoveryProtocol = '发现协议仅支持 SIMPLE/SUPER_CLIENT/CLIENT';
        }
        const mode = String(config.mode || '');
        if (!['fastdds_tailscale_peer', 'lan_multicast'].includes(mode)) {
          newErrors.ros2Mode = '连接模式仅支持 fastdds_tailscale_peer/lan_multicast';
        }
        const peers = Array.isArray(config.peerIps) ? config.peerIps : [];
        if (peers.some((x: any) => !String(x || '').trim())) {
          newErrors.ros2PeerIps = 'Peer IP 不能为空（可删除空行）';
        }
      }
    }

    setErrors(newErrors);
    return newErrors;
  };

  const parseLaunchEnvVarsText = (text: string): { ok: true; value: Record<string, string> } | { ok: false; message: string } => {
    const trimmed = String(text || '').trim();
    if (!trimmed) return { ok: true, value: {} };
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      return { ok: false, message: '启动环境变量 JSON 格式不正确' };
    }
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { ok: false, message: '启动环境变量必须是 JSON 对象（键为变量名，值为字符串）' };
    }
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (!k.trim()) continue;
      if (v === null || v === undefined) {
        out[k] = '';
        continue;
      }
      if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
        out[k] = String(v);
        continue;
      }
      return { ok: false, message: `环境变量「${k}」的值必须是字符串/数字/布尔（或空值）` };
    }
    return { ok: true, value: out };
  };

  const handleSave = async () => {
    const envParse = parseLaunchEnvVarsText(launchEnvVarsText);
    if (!envParse.ok) {
      setErrors((prev) => ({ ...prev, launchEnvVars: envParse.message }));
      return;
    }

    const mergedForm: Partial<RobotDevice> = {
      ...formData,
      launchConfig: formData.launchConfig
        ? { ...formData.launchConfig, envVars: envParse.value }
        : formData.launchConfig,
    };

    const newErrors = validate(mergedForm);
    if (Object.keys(newErrors).length > 0) {
      console.error('验证失败:', newErrors);
      return;
    }

    try {
      const now = new Date().toISOString();

      // 构建请求数据
      const requestData: any = {
        name: mergedForm.name!,
        vendor: mergedForm.vendor,
        model: mergedForm.model,
        deviceType: mergedForm.deviceType!,
      };
      if ((mergedForm.hardwareUuid || '').trim()) requestData.hardwareUuid = (mergedForm.hardwareUuid || '').trim();
      if ((mergedForm.hostname || '').trim()) requestData.hostname = (mergedForm.hostname || '').trim();
      if ((mergedForm.agentIp || '').trim()) requestData.agentIp = (mergedForm.agentIp || '').trim();
      if (Number(mergedForm.agentPort ?? 0)) requestData.agentPort = Number(mergedForm.agentPort ?? 0);

      // 如果是 ROS 或 ROS2 类型，添加 ros2Config
      if ((mergedForm.deviceType === 'ROS' || mergedForm.deviceType === 'ROS2') && mergedForm.ros2Config) {
        const config = mergedForm.ros2Config as any;

        requestData.ros2Config = {
          mode: config.mode,
          localBindIp: (config.localBindIp || '').trim() || undefined,
          domainId: Number(config.domainId ?? 0) || 0,
          discoveryProtocol: config.discoveryProtocol || 'SIMPLE',
          initialAnnouncementsCount: config.initialAnnouncementsCount || 5,
          initialAnnouncementsPeriodSec: config.initialAnnouncementsPeriodSec || 1,
          peerIps: Array.isArray(config.peerIps)
            ? config.peerIps.map((x: any) => String(x || '').trim()).filter(Boolean)
            : [],
        };
      }

      // 启动脚本配置：如果表单中填写了脚本路径，则一并保存
      if (mergedForm.launchConfig?.scriptPath) {
        requestData.launchConfig = {
          scriptPath: mergedForm.launchConfig.scriptPath,
          scriptArgs: mergedForm.launchConfig.scriptArgs || '',
          stopScriptPath: mergedForm.launchConfig.stopScriptPath || '',
          stopScriptArgs: mergedForm.launchConfig.stopScriptArgs || '',
          envVars: mergedForm.launchConfig.envVars || {},
        };
      }

      requestData.collectScriptCompress = (mergedForm.collectScriptCompress || '').trim() || null;
      requestData.collectScriptRaw = (mergedForm.collectScriptRaw || '').trim() || null;

      let response;
      if (device?.id) {
        // 更新设备
        response = await updateDevice(device.id, requestData);
      } else {
        // 创建设备
        response = await createDevice(requestData);
      }

      if (response.ok && response.data) {
        const deviceData: RobotDevice = {
          ...response.data,
          tags: mergedForm.tags || [],
          lastTestResult: testResult || undefined,
          createdAt: device?.createdAt || now,
          updatedAt: now,
        };
        onSave(deviceData);
        onClose();
      } else {
        setToastMsg({ text: t('deviceForm.saveFailed'), isError: true });
      }
    } catch (error) {
      console.error('保存设备失败:', error);
      setToastMsg({ text: t('deviceForm.saveFailed'), isError: true });
    }
  };

  const handleConnectAdd = async () => {
    if (isEditMode) return;

    const deviceName = (formData.name || '').trim();
    if (!deviceName) {
      setToastMsg({ text: '设备名称不能为空', isError: true });
      return;
    }

    const envParseForConnect = parseLaunchEnvVarsText(launchEnvVarsText);
    if (!envParseForConnect.ok) {
      setErrors((prev) => ({ ...prev, launchEnvVars: envParseForConnect.message }));
      setToastMsg({ text: envParseForConnect.message, isError: true });
      return;
    }

    setConnectingByHandshake(true);
    setToastMsg(null);
    setConnectedDeviceMeta(null);
    setTestResult(null);

    try {
      const selected = onlineAgents.find((a) => a.agent_id === selectedAgentId);
      if (!selectedAgentId || !selected || selected.tunnel_stale) {
        setToastMsg({ text: '请先完成 Client 安装并确保已与平台建立通信连接', isError: true });
        return;
      }
      const scriptPath = formData.launchConfig?.scriptPath?.trim();
      const meta = {
        name: deviceName,
        vendor: formData.vendor || undefined,
        model: formData.model || undefined,
        deviceType: (formData.deviceType as DeviceDriverType) || 'ROS2',
        hostname: (formData.hostname || '').trim() || undefined,
        ros2Config: (formData.deviceType === 'ROS' || formData.deviceType === 'ROS2') ? (formData.ros2Config as any) : undefined,
        collectScriptCompress: (formData.collectScriptCompress || '').trim() || null,
        collectScriptRaw: (formData.collectScriptRaw || '').trim() || null,
        launchConfig: scriptPath
          ? {
              scriptPath,
              scriptArgs: formData.launchConfig?.scriptArgs || '',
              stopScriptPath: formData.launchConfig?.stopScriptPath || '',
              stopScriptArgs: formData.launchConfig?.stopScriptArgs || '',
              envVars: envParseForConnect.value,
            }
          : undefined,
      };
      if (!selectedAgentId) {
        setToastMsg({ text: '请先选择在线 Client', isError: true });
        return;
      }
      const response = await connectDeviceByAgent(selectedAgentId, meta);
      if (response.ok && response.data) {
        setConnectedDeviceMeta(response.data);
        setTestResult(response.data.lastTestResult ?? null);
        onSave(response.data);
      setToastMsg({ text: '绑定成功', isError: false });
        onClose();
      } else {
        setToastMsg({ text: (response.error || '连接失败'), isError: true });
      }
    } catch (error) {
      setToastMsg({ text: error instanceof Error ? error.message : '连接失败', isError: true });
    } finally {
      setConnectingByHandshake(false);
    }
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags?.includes(tagInput.trim())) {
      setFormData({
        ...formData,
        tags: [...(formData.tags || []), tagInput.trim()],
      });
      setTagInput('');
    }
  };

  const handleRemoveTag = (tag: string) => {
    setFormData({
      ...formData,
      tags: formData.tags?.filter(t => t !== tag) || [],
    });
  };


  const handleTestConnection = async () => {
    if (!device?.id) {
      setToastMsg({ text: '请先保存设备后再测试连接', isError: true });
      return;
    }

    setTesting(true);
    setTestResult(null);

    try {
      const response = await testDeviceConnection(device.id);
      if (response.ok && response.data) {
        setTestResult(response.data.result);
      } else {
        setTestResult({
          status: 'fail',
          errorMessage: response.error || '测试失败',
          testedAt: new Date().toISOString(),
        });
      }
    } catch (error) {
      setTestResult({
        status: 'fail',
        errorMessage: error instanceof Error ? error.message : '测试失败',
        testedAt: new Date().toISOString(),
      });
    } finally {
      setTesting(false);
    }
  };

  const renderROS2Config = () => {
    if (!SHOW_ROS2_SETTINGS) return null;
    // ROS 和 ROS2 都显示配置
    if (formData.deviceType !== 'ROS' && formData.deviceType !== 'ROS2') return null;

    const cfg = (formData.ros2Config || {
      mode: 'fastdds_tailscale_peer',
      localBindIp: '',
      domainId: 0,
      discoveryProtocol: 'SIMPLE',
      initialAnnouncementsCount: 5,
      initialAnnouncementsPeriodSec: 1,
      peerIps: [],
    }) as any;

    const peerIpsText = Array.isArray(cfg.peerIps) ? cfg.peerIps.join('\n') : '';

    return (
      <>
        <h4
          style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#111827',
            margin: '0 0 12px 0',
          }}
        >
          ROS2 配置
        </h4>

        <div style={{ marginBottom: 12 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            连接模式
          </label>
          <select
            value={cfg.mode || 'fastdds_tailscale_peer'}
            onChange={(e) =>
              setFormData({
                ...formData,
                ros2Config: { ...cfg, mode: e.target.value, peerIps: Array.isArray(cfg.peerIps) ? cfg.peerIps : [] },
              })
            }
            style={{
              width: '100%',
              height: 40,
              padding: '0 12px',
              backgroundColor: '#ffffff',
              border: errors.ros2Mode ? '1px solid #ef4444' : '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
              outline: 'none',
              boxSizing: 'border-box',
            }}
          >
            <option value="fastdds_tailscale_peer">fastdds_tailscale_peer</option>
            <option value="lan_multicast">lan_multicast</option>
          </select>
          {errors.ros2Mode && <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.ros2Mode}</div>}
        </div>

        <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
              Domain ID
            </label>
            <input
              type="number"
              value={Number(cfg.domainId ?? 0)}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  ros2Config: { ...cfg, domainId: Number(e.target.value), peerIps: Array.isArray(cfg.peerIps) ? cfg.peerIps : [] },
                })
              }
              style={{
                width: '100%',
                height: 40,
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.ros2DomainId ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: 6,
                fontSize: 14,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
            {errors.ros2DomainId && <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.ros2DomainId}</div>}
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
              发现协议
            </label>
            <select
              value={cfg.discoveryProtocol || 'SIMPLE'}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  ros2Config: { ...cfg, discoveryProtocol: e.target.value, peerIps: Array.isArray(cfg.peerIps) ? cfg.peerIps : [] },
                })
              }
              style={{
                width: '100%',
                height: 40,
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.ros2DiscoveryProtocol ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: 6,
                fontSize: 14,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            >
              <option value="SIMPLE">SIMPLE</option>
              <option value="SUPER_CLIENT">SUPER_CLIENT</option>
              <option value="CLIENT">CLIENT</option>
            </select>
            {errors.ros2DiscoveryProtocol && (
              <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.ros2DiscoveryProtocol}</div>
            )}
          </div>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            Local Bind IP（可选）
          </label>
          <input
            type="text"
            value={cfg.localBindIp || ''}
            onChange={(e) =>
              setFormData({
                ...formData,
                ros2Config: { ...cfg, localBindIp: e.target.value, peerIps: Array.isArray(cfg.peerIps) ? cfg.peerIps : [] },
              })
            }
            placeholder="例如 192.168.1.10"
            style={{
              width: '100%',
              height: 40,
              padding: '0 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 14,
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
              初始公告次数
            </label>
            <input
              type="number"
              value={Number(cfg.initialAnnouncementsCount ?? 5)}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  ros2Config: { ...cfg, initialAnnouncementsCount: Number(e.target.value), peerIps: Array.isArray(cfg.peerIps) ? cfg.peerIps : [] },
                })
              }
              style={{
                width: '100%',
                height: 40,
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: 6,
                fontSize: 14,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
              初始公告周期（秒）
            </label>
            <input
              type="number"
              value={Number(cfg.initialAnnouncementsPeriodSec ?? 1)}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  ros2Config: { ...cfg, initialAnnouncementsPeriodSec: Number(e.target.value), peerIps: Array.isArray(cfg.peerIps) ? cfg.peerIps : [] },
                })
              }
              style={{
                width: '100%',
                height: 40,
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: 6,
                fontSize: 14,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>
        </div>

        <div style={{ marginBottom: 20 }}>
          <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            Peer IPs（每行一个，可选）
          </label>
          <textarea
            value={peerIpsText}
            onChange={(e) => {
              const arr = e.target.value
                .split('\n')
                .map((x) => x.trim())
                .filter(Boolean);
              setFormData({ ...formData, ros2Config: { ...cfg, peerIps: arr } });
            }}
            rows={4}
            placeholder="10.0.0.2&#10;10.0.0.3"
            style={{
              width: '100%',
              padding: 10,
              backgroundColor: '#ffffff',
              border: errors.ros2PeerIps ? '1px solid #ef4444' : '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 13,
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              whiteSpace: 'pre-wrap',
            }}
          />
          {errors.ros2PeerIps && <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.ros2PeerIps}</div>}
        </div>

      </>
    );
  };

  const renderLaunchConfig = () => {
    if (!formData.launchConfig) return null;

    return (
      <div style={{ marginTop: '24px' }}>
        <h4
          style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#111827',
            margin: '0 0 12px 0',
          }}
        >
          启动脚本
        </h4>
        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: '#374151',
              marginBottom: '6px',
            }}
          >
            启动脚本路径
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              placeholder="/home/sia/workspace/test/start_all_ros_topics.sh"
              style={{
                flex: 1,
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                fontFamily:
                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              }}
              value={formData.launchConfig?.scriptPath || ''}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  launchConfig: {
                    ...(formData.launchConfig || {
                      scriptPath: '',
                      scriptArgs: '',
                      stopScriptPath: '',
                      stopScriptArgs: '',
                      envVars: {},
                    }),
                    scriptPath: e.target.value,
                  },
                })
              }
            />
            <button
              type="button"
              onClick={() => setScriptPickerField('launch')}
              style={{
                width: 90,
                height: 40,
                borderRadius: 6,
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                color: '#374151',
                fontSize: 14,
                cursor: 'pointer',
              }}
            >
              选择
            </button>
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
            该路径会在采集端 Agent 所在机器上执行，请确保脚本存在且有可执行权限。
          </div>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: '#374151',
              marginBottom: '6px',
            }}
          >
            启动脚本参数（可选）
          </label>
          <input
            type="text"
            placeholder='例如：--foo bar（会直接拼到启动命令后面）'
            style={{
              width: '100%',
              height: '40px',
              padding: '0 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px',
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            }}
            value={formData.launchConfig?.scriptArgs || ''}
            onChange={(e) =>
              setFormData({
                ...formData,
                launchConfig: {
                  ...(formData.launchConfig || {
                    scriptPath: '',
                    scriptArgs: '',
                    stopScriptPath: '',
                    stopScriptArgs: '',
                    envVars: {},
                  }),
                  scriptArgs: e.target.value,
                },
              })
            }
          />
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: '#374151',
              marginBottom: '6px',
            }}
          >
            停止脚本路径
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              type="text"
              placeholder="/home/sia/workspace/test/stop_all_ros_topics.sh"
              style={{
                flex: 1,
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                fontFamily:
                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              }}
              value={formData.launchConfig?.stopScriptPath || ''}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  launchConfig: {
                    ...(formData.launchConfig || {
                      scriptPath: '',
                      scriptArgs: '',
                      stopScriptPath: '',
                      stopScriptArgs: '',
                      envVars: {},
                    }),
                    stopScriptPath: e.target.value,
                  },
                })
              }
            />
            <button
              type="button"
              onClick={() => setScriptPickerField('stop')}
              style={{
                width: 90,
                height: 40,
                borderRadius: 6,
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                color: '#374151',
                fontSize: 14,
                cursor: 'pointer',
              }}
            >
              选择
            </button>
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
            留空则后端回退为默认停止脚本；建议为每台设备配置独立停止脚本路径。
          </div>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: '#374151',
              marginBottom: '6px',
            }}
          >
            停止脚本参数（可选）
          </label>
          <input
            type="text"
            placeholder='例如：--force'
            style={{
              width: '100%',
              height: '40px',
              padding: '0 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px',
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            }}
            value={formData.launchConfig?.stopScriptArgs || ''}
            onChange={(e) =>
              setFormData({
                ...formData,
                launchConfig: {
                  ...(formData.launchConfig || {
                    scriptPath: '',
                    scriptArgs: '',
                    stopScriptPath: '',
                    stopScriptArgs: '',
                    envVars: {},
                  }),
                  stopScriptArgs: e.target.value,
                },
              })
            }
          />
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: '#374151',
              marginBottom: '6px',
            }}
          >
            启动/停止脚本环境变量（JSON 对象，可选）
          </label>
          <textarea
            value={launchEnvVarsText}
            onChange={(e) => {
              setLaunchEnvVarsText(e.target.value);
              if (errors.launchEnvVars) {
                setErrors((prev) => {
                  const next = { ...prev };
                  delete next.launchEnvVars;
                  return next;
                });
              }
            }}
            rows={8}
            placeholder={`{
  "PATH": "/path/to/conda/env/bin:/usr/bin:/bin",
  "CONDA_DEFAULT_ENV": "rm75"
}`}
            style={{
              width: '100%',
              padding: 10,
              backgroundColor: '#ffffff',
              border: errors.launchEnvVars ? '1px solid #ef4444' : '1px solid #d1d5db',
              borderRadius: 6,
              fontSize: 13,
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              whiteSpace: 'pre-wrap',
            }}
          />
          <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280', lineHeight: 1.5 }}>
            这些变量会在采集端执行启动/停止脚本时注入进程环境（可与系统环境合并）。留空或填写 <code style={{ fontFamily: 'inherit' }}>{'{}'}</code> 表示不额外注入。
          </div>
          {errors.launchEnvVars && <div style={{ color: '#ef4444', fontSize: 12, marginTop: 6 }}>{errors.launchEnvVars}</div>}
        </div>
      </div>
    );
  };

  const renderCollectScriptConfig = () => (
    <div style={{ marginTop: '24px' }}>
      <h4
        style={{
          fontSize: '16px',
          fontWeight: '600',
          color: '#111827',
          margin: '0 0 12px 0',
        }}
      >
        数据采集脚本
      </h4>
      <p style={{ fontSize: 12, color: '#6b7280', margin: '0 0 12px 0', lineHeight: 1.5 }}>
        在「数据采集」任务中点击开始采集时，按任务的「相机数据格式」选用下方路径之一在采集端执行。留空则使用平台默认脚本路径。
      </p>
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '6px' }}>
          压缩图像采集脚本（对应任务选「压缩」）
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="text"
            placeholder="/home/sia/workspace/test/collect_data_compressed.sh"
            style={{
              flex: 1,
              height: '40px',
              padding: '0 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px',
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            }}
            value={formData.collectScriptCompress ?? ''}
            onChange={(e) => setFormData({ ...formData, collectScriptCompress: e.target.value })}
          />
          <button
            type="button"
            onClick={() => setScriptPickerField('collectCompress')}
            style={{
              width: 90,
              height: 40,
              borderRadius: 6,
              border: '1px solid #d1d5db',
              backgroundColor: '#ffffff',
              color: '#374151',
              fontSize: 14,
              cursor: 'pointer',
            }}
          >
            选择
          </button>
        </div>
      </div>
      <div>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '6px' }}>
          原始图像采集脚本（对应任务选「原始」）
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="text"
            placeholder="/home/sia/workspace/test/collect_data_uncompressed.sh"
            style={{
              flex: 1,
              height: '40px',
              padding: '0 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '14px',
              outline: 'none',
              boxSizing: 'border-box',
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            }}
            value={formData.collectScriptRaw ?? ''}
            onChange={(e) => setFormData({ ...formData, collectScriptRaw: e.target.value })}
          />
          <button
            type="button"
            onClick={() => setScriptPickerField('collectRaw')}
            style={{
              width: 90,
              height: 40,
              borderRadius: 6,
              border: '1px solid #d1d5db',
              backgroundColor: '#ffffff',
              color: '#374151',
              fontSize: 14,
              cursor: 'pointer',
            }}
          >
            选择
          </button>
        </div>
      </div>
    </div>
  );

  // Render Connection Test - Hidden as requested
  const renderConnectionTest = () => {
    return null;
  };

  if (!open) return null;

  return (
    <>
    {toastMsg && (
      <div
        style={{
          position: 'fixed',
          left: '50%',
          bottom: 24,
          transform: 'translateX(-50%)',
          padding: '10px 16px',
          borderRadius: 10,
          fontSize: 14,
          fontWeight: 500,
          zIndex: 1700,
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          backgroundColor: toastMsg.isError ? '#fef2f2' : 'rgba(17,24,39,0.92)',
          color: toastMsg.isError ? '#b91c1c' : '#fff',
        }}
      >
        {toastMsg.text}
      </div>
    )}
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
    }} onClick={onClose}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        width: '800px',
        maxHeight: '90vh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15)',
      }} onClick={(e) => e.stopPropagation()}>
        {/* 头部 */}
        <div style={{
          padding: '20px 24px',
          borderBottom: '1px solid #e5e7eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <h3 style={{
            fontSize: '18px',
            fontWeight: '600',
            color: '#111827',
            margin: 0,
          }}>{device ? t('deviceForm.titleEdit') : t('deviceForm.titleCreate')}</h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#6b7280',
              fontSize: '20px',
              cursor: 'pointer',
              padding: '4px',
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* 内容区域 */}
        <div style={{
          flex: 1,
          padding: '24px',
          overflowY: 'auto',
        }}>
          {/* 新模式下：设备ID绑定由“后端主动添加/握手”完成，不再要求采集端预配置 DEVICES */}

          {/* 连接检测结果展示（最近一次 or 本次测试） */}
          {(testResult || device?.lastTestResult) && (
            <div
              style={{
                marginBottom: 20,
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                backgroundColor: '#f9fafb',
                fontSize: 12,
                color: '#374151',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 6 }}>
                {t('devicesPage.lastConnectionTestTitle') || '最近一次连接检测结果'}
              </div>
              {(() => {
                const r = testResult || device?.lastTestResult;
                if (!r) return null;
                const ok = r.status === 'success';
                return (
                  <>
                    <div style={{ marginBottom: 4 }}>
                      状态：
                      <span
                        style={{
                          fontWeight: 600,
                          color: ok ? '#16a34a' : '#b91c1c',
                        }}
                      >
                        {ok ? '成功' : '失败'}
                      </span>
                    </div>
                    <div style={{ marginBottom: 4 }}>
                      节点数：{r.nodeCount ?? '-'}；话题数：{r.topicCount ?? '-'}
                    </div>
                    {r.nodesSample && r.nodesSample.length > 0 && (
                      <div style={{ marginBottom: 4 }}>
                        节点示例：
                        <code style={{ fontFamily: 'monospace' }}>
                          {r.nodesSample.slice(0, 3).join(', ')}
                          {r.nodeCount && r.nodeCount > r.nodesSample.length ? ' ...' : ''}
                        </code>
                      </div>
                    )}
                    {r.topicsSample && r.topicsSample.length > 0 && (
                      <div style={{ marginBottom: 4 }}>
                        话题示例：
                        <code style={{ fontFamily: 'monospace' }}>
                          {r.topicsSample.slice(0, 3).join(', ')}
                          {r.topicCount && r.topicCount > r.topicsSample.length ? ' ...' : ''}
                        </code>
                      </div>
                    )}
                    {r.errorMessage && (
                      <div style={{ marginTop: 4, color: '#b91c1c' }}>
                        错误信息：{r.errorMessage}
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          )}

          {/* 采集端 Client 主动添加（只在新增模式下显示） */}
          {!isEditMode && (
            <div
              style={{
                marginBottom: 20,
                padding: '12px 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                backgroundColor: '#f9fafb',
                fontSize: 12,
                color: '#374151',
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: 6 }}>设备主动添加</div>
              <div style={{ marginBottom: 10, color: '#6b7280' }}>
                请选择在线 Client 直接绑定设备并拉取元数据。
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                <button
                  type="button"
                  onClick={async () => {
                    const res = await listOnlineAgents();
                    if (res.ok && Array.isArray(res.data)) {
                      setOnlineAgents(res.data);
                      if (!selectedAgentId && res.data.length > 0) {
                        setSelectedAgentId(res.data[0].agent_id);
                      }
                    }
                  }}
                  style={{
                    padding: '6px 10px',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    background: '#fff',
                    cursor: 'pointer',
                    fontSize: 12,
                  }}
                >
                  刷新在线 Client
                </button>
              </div>

              <div style={{ marginBottom: 12 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                  在线 Client
                </label>
                <select
                  value={selectedAgentId}
                  onChange={(e) => setSelectedAgentId(e.target.value)}
                  style={{
                    width: '100%',
                    height: '38px',
                    padding: '0 12px',
                    backgroundColor: '#ffffff',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    fontSize: 14,
                    outline: 'none',
                    boxSizing: 'border-box',
                  }}
                >
                  <option value="">请选择在线 Client</option>
                  {onlineAgents.map((a) => (
                    <option key={a.agent_id} value={a.agent_id} disabled={!!a.tunnel_stale}>
                      {a.name || a.agent_id} ({a.agent_id}) {a.runtime_status ? `· ${a.runtime_status}` : ''}{a.tunnel_stale ? ' · 通信超时' : ''}
                    </option>
                  ))}
                </select>
                <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
                  {onlineAgents.length > 0 ? `当前在线 ${onlineAgents.length} 个` : '当前无在线 Client'}
                </div>
                {onlineAgents.length === 0 ? (
                  <div style={{ marginTop: 8, fontSize: 12, color: '#991b1b', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                    <span>添加设备前必须先安装并连接 Client。</span>
                    {onOpenAgentWizard ? (
                      <button
                        type="button"
                        onClick={onOpenAgentWizard}
                        style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #d1d5db', background: '#fff', cursor: 'pointer', fontSize: 12 }}
                      >
                        安装 Client
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>

              <button
                type="button"
                onClick={handleConnectAdd}
                disabled={connectingByHandshake || !selectedAgentId || !!onlineAgents.find((a) => a.agent_id === selectedAgentId)?.tunnel_stale}
                style={{
                  padding: '10px 16px',
                  backgroundColor: connectingByHandshake ? '#9ca3af' : '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontSize: 14,
                  cursor: connectingByHandshake || !selectedAgentId || !!onlineAgents.find((a) => a.agent_id === selectedAgentId)?.tunnel_stale ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                  width: '100%',
                }}
              >
                {connectingByHandshake ? '绑定中...' : '绑定设备'}
              </button>

              {connectedDeviceMeta && (
                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>连接成功（元数据）</div>
                  <div style={{ marginBottom: 4 }}>
                    {t('devicesPage.deviceId')}：
                    <span style={{ fontFamily: 'monospace' }}>{connectedDeviceMeta.id ?? '-'}</span>
                  </div>
                  <div style={{ marginBottom: 4 }}>
                    Client 地址：
                    <span style={{ fontFamily: 'monospace' }}>
                      {connectedDeviceMeta.agentIp ?? '-'}
                      {connectedDeviceMeta.agentPort ? `:${connectedDeviceMeta.agentPort}` : ''}
                    </span>
                  </div>
                  <div style={{ marginBottom: 4 }}>
                    所在地：
                    <span style={{ fontFamily: 'monospace' }}>
                      {connectedDeviceMeta.location?.city ||
                        connectedDeviceMeta.location?.region ||
                        connectedDeviceMeta.location?.country
                        ? [
                            connectedDeviceMeta.location?.city,
                            connectedDeviceMeta.location?.region,
                            connectedDeviceMeta.location?.country,
                          ]
                            .filter(Boolean)
                            .join(', ')
                        : (connectedDeviceMeta.location?.note || connectedDeviceMeta.hostname || '-')}
                    </span>
                  </div>
                  <div style={{ marginBottom: 4 }}>
                    Client 状态：{connectedDeviceMeta.agentStatus ?? '-'}
                  </div>
                  <div style={{ marginBottom: 4 }}>
                    Camera List：
                    <div style={{ marginTop: 6 }}>
                      <code
                        style={{
                          display: 'block',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          fontFamily: 'monospace',
                          backgroundColor: '#eef2ff',
                          borderRadius: 6,
                          padding: '10px 12px',
                        }}
                      >
                        {(connectedDeviceMeta.cameraList ?? []).join(', ') || '-'}
                      </code>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 基础信息 */}
          <h4 style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#111827',
            margin: '0 0 20px 0',
          }}>{t('deviceForm.basicSectionTitle')}</h4>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
              {t('deviceForm.nameLabel')} <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="text"
              placeholder={t('deviceForm.namePlaceholder')}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.name ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              value={formData.name || ''}
              onChange={(e) => {
                setFormData({ ...formData, name: e.target.value });
                if (errors.name) setErrors({ ...errors, name: '' });
              }}
            />
            {errors.name && <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>{t('deviceForm.nameRequiredError')}</div>}
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
              {t('deviceForm.vendorLabel')}
            </label>
            <input
              type="text"
              placeholder={t('deviceForm.vendorPlaceholder')}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              value={formData.vendor || ''}
              onChange={(e) => setFormData({ ...formData, vendor: e.target.value })}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
              {t('deviceForm.modelLabel')}
            </label>
            <input
              type="text"
              placeholder={t('deviceForm.modelPlaceholder')}
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              value={formData.model || ''}
              onChange={(e) => setFormData({ ...formData, model: e.target.value })}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
              Client ID（设备唯一标识） <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="text"
              placeholder="选择在线 Client 后自动填充"
              readOnly
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.hardwareUuid ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
                fontFamily:
                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              }}
              value={(formData.hardwareUuid || '').trim()}
            />
            {errors.hardwareUuid && <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>{errors.hardwareUuid}</div>}
          </div>

          <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 8 }}>
                Hostname（可选）
              </label>
              <input
                type="text"
                placeholder="robot-001"
                readOnly
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                value={formData.hostname || ''}
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 8 }}>
                Client IP（可选）
              </label>
              <input
                type="text"
                placeholder="192.168.1.10"
                readOnly
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box',
                  fontFamily:
                    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                }}
                value={formData.agentIp || ''}
              />
            </div>
            <div style={{ width: 160 }}>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#374151', marginBottom: 8 }}>
                Client Port（可选）
              </label>
              <input
                type="number"
                placeholder="9100"
                readOnly
                style={{
                  width: '100%',
                  height: '40px',
                  padding: '0 12px',
                  backgroundColor: '#ffffff',
                  border: errors.agentPort ? '1px solid #ef4444' : '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                value={String(formData.agentPort ?? '')}
              />
              {errors.agentPort && <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.agentPort}</div>}
            </div>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
              {t('deviceForm.typeLabel')} <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <select
              style={{
                width: '100%',
                height: '40px',
                padding: '0 12px',
                backgroundColor: '#ffffff',
                border: errors.deviceType ? '1px solid #ef4444' : '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              value={formData.deviceType || 'ROS2'}
              onChange={(e) => {
                const newType = e.target.value as DeviceDriverType;
                setFormData({
                  ...formData,
                  deviceType: newType,
                  ros2Config: ((newType === 'ROS' || newType === 'ROS2') ? (formData.ros2Config || {
                    mode: 'fastdds_tailscale_peer',
                    localBindIp: '',
                    domainId: 0,
                    discoveryProtocol: 'SIMPLE',
                    initialAnnouncementsCount: 5,
                    initialAnnouncementsPeriodSec: 1,
                    peerIps: [],
                  }) : undefined),
                });
                if (errors.deviceType) setErrors({ ...errors, deviceType: '' });
              }}
            >
              <option value="ROS">ROS</option>
              <option value="ROS2">ROS2</option>
            </select>
            {errors.deviceType && <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '4px' }}>{t('deviceForm.typeRequiredError')}</div>}
          </div>

          {/* ROS2 配置（SHOW_ROS2_SETTINGS 控制显隐） */}
          {renderROS2Config()}

          {/* 高级设置（默认收起） */}
          <div style={{ marginTop: 24 }}>
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                backgroundColor: '#f9fafb',
                cursor: 'pointer',
                fontWeight: 700,
                color: '#111827',
              }}
            >
              <span>高级设置</span>
              <span style={{ fontWeight: 600, color: '#374151' }}>{showAdvanced ? '收起' : '展开'}</span>
            </button>

            {showAdvanced ? (
              <div style={{ marginTop: 12 }}>
                {/* 启动配置 */}
                {renderLaunchConfig()}
                {/* 数据采集脚本（与采集任务联动） */}
                {renderCollectScriptConfig()}
              </div>
            ) : null}
          </div>

          {/* 连接测试 */}
          {renderConnectionTest()}

          {/* 标签 */}
          <div style={{ marginTop: '24px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
              {t('deviceForm.tagsLabel')}
            </label>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <input
                type="text"
                placeholder={t('deviceForm.tagPlaceholder')}
                style={{
                  flex: 1,
                  height: '40px',
                  padding: '0 12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '14px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddTag();
                  }
                }}
              />
              <button
                type="button"
                onClick={handleAddTag}
                style={{
                  padding: '0 16px',
                  height: '40px',
                  backgroundColor: '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontSize: '14px',
                  cursor: 'pointer',
                }}
              >
                {t('deviceForm.addTag')}
              </button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {formData.tags?.map((tag) => (
                <span
                  key={tag}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 12px',
                    backgroundColor: '#f3f4f6',
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: '#374151',
                  }}
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => handleRemoveTag(tag)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#6b7280',
                      cursor: 'pointer',
                      fontSize: '14px',
                      padding: 0,
                      lineHeight: 1,
                    }}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* 底部按钮 */}
        <div style={{
          padding: '16px 24px',
          borderTop: '1px solid #e5e7eb',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '12px',
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '10px 20px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
            }}
          >
            {isEditMode ? t('deviceForm.cancel') : '关闭'}
          </button>
          {isEditMode && (
            <button
              onClick={handleSave}
              style={{
                padding: '10px 20px',
                backgroundColor: '#2563eb',
                border: 'none',
                borderRadius: '6px',
                color: '#ffffff',
                fontSize: '14px',
                cursor: 'pointer',
                fontWeight: '500',
              }}
            >
              {t('deviceForm.save')}
            </button>
          )}
        </div>
      </div>
    </div>
    <PathPickerModal
      open={scriptPickerField !== null}
      onClose={() => setScriptPickerField(null)}
      title={
        scriptPickerField === 'launch'
          ? '选择启动脚本'
          : scriptPickerField === 'stop'
            ? '选择停止脚本'
            : scriptPickerField === 'collectCompress'
              ? '选择压缩采集脚本'
              : scriptPickerField === 'collectRaw'
                ? '选择原始采集脚本'
                : '选择脚本'
      }
      mode="files"
      allowAllFiles
      source="agent"
      agentId={agentIdForPathPicker || undefined}
      deviceId={deviceIdForPathPicker || undefined}
      onConfirm={(items) => {
        const firstFile = items.find((it) => it.type === 'file');
        if (!firstFile) {
          setToastMsg({ text: '请选择脚本文件', isError: true });
          return;
        }
        setFormData((prev) => {
          const next: Partial<RobotDevice> = { ...prev };
          if (scriptPickerField === 'launch') {
            next.launchConfig = {
              ...(prev.launchConfig || { scriptPath: '', scriptArgs: '', stopScriptPath: '', stopScriptArgs: '', envVars: {} }),
              scriptPath: firstFile.path,
            };
          } else if (scriptPickerField === 'stop') {
            next.launchConfig = {
              ...(prev.launchConfig || { scriptPath: '', scriptArgs: '', stopScriptPath: '', stopScriptArgs: '', envVars: {} }),
              stopScriptPath: firstFile.path,
            };
          } else if (scriptPickerField === 'collectCompress') {
            next.collectScriptCompress = firstFile.path;
          } else if (scriptPickerField === 'collectRaw') {
            next.collectScriptRaw = firstFile.path;
          }
          return next;
        });
        setScriptPickerField(null);
      }}
    />
    </>
  );
}
