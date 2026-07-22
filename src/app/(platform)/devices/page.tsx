'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import Link from 'next/link';
import type { RobotDevice, DeviceStatus } from '@/features/data-platform/models/device';
import { listDevices, updateDevice, deleteDevice, testDeviceConnection, stopDevice, launchDevice, type DeviceUpdateRequest } from '@/features/data-platform/api/deviceApi';
import { getConnSummary } from '@/features/data-platform/models/device';
import AddDeviceModal from '@/features/data-platform/components/devices/AddDeviceModal';
import { useI18n } from '@/components/common/I18nProvider';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import { useAuthStore } from '@/store/authStore';
import { canMutateDevices } from '@/lib/api/roleLabels';
import { formatDateTimeMinute } from '@/utils/format';
import AgentInstallWizard from '@/features/data-platform/components/devices/AgentInstallWizard';
import {
  startDevicePreviewWarm,
  stopDevicePreviewWarm,
} from '@/features/stream/devicePreviewWarmPool';

const RUNTIME_NODE_TYPE_KEYS = [
  'typeSimulation',
  'typeEvaluation',
  'typeDataProcessing',
  'typeGpu',
] as const;

type RuntimeNodeTypeKey = (typeof RUNTIME_NODE_TYPE_KEYS)[number];

function runtimeNodeTypeKey(deviceId: string): RuntimeNodeTypeKey {
  let h = 0;
  for (let i = 0; i < deviceId.length; i++) h = (h + deviceId.charCodeAt(i)) | 0;
  return RUNTIME_NODE_TYPE_KEYS[Math.abs(h) % RUNTIME_NODE_TYPE_KEYS.length];
}

function runtimeNodeTypeLabel(device: RobotDevice, t: (key: string) => string): string {
  return t(`devicesPage.${runtimeNodeTypeKey(device.id)}`);
}

function getStatusTagMeta(status: DeviceStatus, t: (key: string) => string) {
  switch (status) {
    case 'DISCONNECTED':
      return { bg: '#f3f4f6', color: '#374151', text: t('devicesPage.statusDisconnected') };
    case 'CONNECTING':
      return { bg: '#dbeafe', color: '#1e40af', text: t('devicesPage.statusConnecting') };
    case 'CONNECTED':
      return { bg: '#d1fae5', color: '#065f46', text: t('devicesPage.statusConnected') };
    case 'ERROR':
    default:
      return { bg: '#fee2e2', color: '#991b1b', text: t('devicesPage.statusError') };
  }
}

export default function DevicesPage() {
  const { t } = useI18n();
  const authUser = useAuthStore((s) => s.user);
  const authHydrated = useAuthStore((s) => s.isHydrated);
  const canMutate = authHydrated && canMutateDevices(authUser?.role);
  const [devices, setDevices] = useState<RobotDevice[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchText, setSearchText] = useState('');
  const [runtimeNodeTypeFilter, setRuntimeNodeTypeFilter] = useState<RuntimeNodeTypeKey | ''>('');
  const [statusFilter, setStatusFilter] = useState<DeviceStatus | ''>('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingDevice, setEditingDevice] = useState<RobotDevice | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [connectingDeviceId, setConnectingDeviceId] = useState<string | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [toastMsg, setToastMsg] = useState<{ text: string; isError?: boolean } | null>(null);
  const [devicePendingAction, setDevicePendingAction] = useState<RobotDevice | null>(null);
  const [actionType, setActionType] = useState<'delete' | 'stop' | null>(null);
  const [showAgentWizard, setShowAgentWizard] = useState(false);

  const showToast = useCallback((text: string, isError?: boolean) => {
    setToastMsg({ text, isError });
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  useEffect(() => {
    if (canMutate) return;
    setShowAddModal(false);
    setShowEditModal(false);
    setEditingDevice(null);
    if (actionType === 'delete') {
      setDevicePendingAction(null);
      setActionType(null);
    }
  }, [canMutate, actionType]);

  // 初始化并加载设备列表
  useEffect(() => {
    const loadDeviceList = async () => {
      try {
        setLoading(true);
        const response = await listDevices();
        if (response.ok && response.data) {
          setDevices(response.data);
        } else {
          console.error('加载设备列表失败:', response.error);
          setDevices([]);
        }
      } catch (error) {
        console.error('加载设备列表异常:', error);
        setDevices([]);
      } finally {
        setLoading(false);
      }
    };
    loadDeviceList();

    // 轮询设备状态（每5秒）
    const intervalId = setInterval(async () => {
      try {
        // 静默更新，不显示 loading
        const response = await listDevices();
        if (response.ok && response.data) {
          setDevices(prev => {
            // 简单比较是否需要更新，避免频繁渲染
            // 这里简化处理，直接更新
            return response.data!;
          });
        }
      } catch (error) {
        console.error('轮询设备列表异常:', error);
      }
    }, 5000);

    return () => clearInterval(intervalId);
  }, []);

  // 过滤设备
  const filteredDevices = useMemo(() => {
    let filtered = [...devices];

    // 搜索过滤
    if (searchText) {
      filtered = filtered.filter(device =>
        device.name.toLowerCase().includes(searchText.toLowerCase())
      );
    }

    // 类型过滤
    if (runtimeNodeTypeFilter) {
      filtered = filtered.filter(
        (device) => runtimeNodeTypeKey(device.id) === runtimeNodeTypeFilter
      );
    }

    // 状态过滤
    if (statusFilter) {
      filtered = filtered.filter(device => device.status === statusFilter);
    }

    // 按更新时间排序（最新的在前）
    filtered.sort((a, b) => {
      return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
    });

    return filtered;
  }, [devices, searchText, runtimeNodeTypeFilter, statusFilter]);

  const handleAddDevice = async (device: RobotDevice) => {
    // AddDeviceModal 内部已经调用了 API，这里只需要更新本地状态
    // 重新加载设备列表以获取最新数据（包括生成的配置文件路径）
    try {
      const listResponse = await listDevices();
      if (listResponse.ok && listResponse.data) {
        setDevices(listResponse.data);
      }
    } catch (error) {
      console.error('刷新设备列表异常:', error);
      // 刷新失败时仍保持弹窗，让用户可继续查看连接/元数据结果
    }
  };

  const handleEditDevice = (device: RobotDevice) => {
    setEditingDevice(device);
    setShowEditModal(true);
  };

  const handleUpdateDevice = async (device: RobotDevice) => {
    try {
      // 转换为后端格式
      const deviceType = device.deviceType || device.driverType || 'ROS2';
      const updateRequest: DeviceUpdateRequest = {
        name: device.name,
        vendor: device.vendor,
        model: device.model,
        deviceType,
        ros2Config: deviceType === 'ROS2' && device.ros2Config ? {
          mode: device.ros2Config.mode,
          localBindIp: device.ros2Config.localBindIp,
          domainId: device.ros2Config.domainId,
          discoveryProtocol: device.ros2Config.discoveryProtocol,
          initialAnnouncementsCount: device.ros2Config.initialAnnouncementsCount,
          initialAnnouncementsPeriodSec: device.ros2Config.initialAnnouncementsPeriodSec,
          peerIps: device.ros2Config.peerIps,
        } : undefined,
      };

      const response = await updateDevice(device.id, updateRequest);
      if (response.ok && response.data) {
        // 重新加载设备列表
        const listResponse = await listDevices();
        if (listResponse.ok && listResponse.data) {
          setDevices(listResponse.data);
        }
        setShowEditModal(false);
        setEditingDevice(null);
        showToast('设备已更新');
      } else {
        showToast(`更新设备失败: ${response.error || '未知错误'}`, true);
      }
    } catch (error) {
      console.error('更新设备异常:', error);
      showToast(`更新设备失败: ${error instanceof Error ? error.message : String(error)}`, true);
    }
  };

  const openDeleteDeviceConfirm = (device: RobotDevice) => {
    setDevicePendingAction(device);
    setActionType('delete');
  };

  const handleConfirmDeleteDevice = useCallback(async () => {
    if (!devicePendingAction) return;
    try {
      const response = await deleteDevice(devicePendingAction.id);
      if (response.ok) {
        const listResponse = await listDevices();
        if (listResponse.ok && listResponse.data) {
          setDevices(listResponse.data);
        }
        showToast('设备已删除');
      } else {
        showToast(`删除设备失败: ${response.error || '未知错误'}`, true);
      }
    } catch (error) {
      console.error('删除设备异常:', error);
      showToast(`删除设备失败: ${error instanceof Error ? error.message : String(error)}`, true);
    } finally {
      setDevicePendingAction(null);
      setActionType(null);
    }
  }, [devicePendingAction, showToast]);

  const handleConnect = async (device: RobotDevice) => {
    if (device.status === 'CONNECTED' || device.status === 'CONNECTING') {
      return;
    }

    try {
      setConnectingDeviceId(device.id);

      // 1) 先通过 /launch 启动设备（由 Agent 拉起 ROS2 节点）
      const launchResp = await launchDevice(device.id);
      if (!launchResp.ok) {
        const msg = launchResp.error || '启动设备失败';
        setLaunchError(`启动设备失败: ${msg}`);
      } else {
        void startDevicePreviewWarm(device.id, { maxCameras: 4 });
        // 2) 再调用 /test-connection，仅用于在采集端读取节点/话题数并写入最近一次测试结果
        const response = await testDeviceConnection(device.id);

        if (response.ok && response.data?.success) {
          const listResponse = await listDevices();
          if (listResponse.ok && listResponse.data) {
            setDevices(listResponse.data);
          }
        } else {
          const errorMsg = response.data?.message || response.error || '连接失败';
          setLaunchError(`连接失败: ${errorMsg}`);
          const listResponse = await listDevices();
          if (listResponse.ok && listResponse.data) {
            setDevices(listResponse.data);
          }
        }
      }
    } catch (error) {
      console.error('启动设备异常:', error);
      setLaunchError(`启动设备失败: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setConnectingDeviceId(null);
    }
  };

  const openStopDeviceConfirm = (device: RobotDevice) => {
    setDevicePendingAction(device);
    setActionType('stop');
  };

  const handleConfirmStopDevice = useCallback(async () => {
    if (!devicePendingAction) return;
    try {
      const response = await stopDevice(devicePendingAction.id);
      if (response.ok) {
        stopDevicePreviewWarm(devicePendingAction.id);
        const listResponse = await listDevices();
        if (listResponse.ok && listResponse.data) {
          setDevices(listResponse.data);
        }
        showToast('设备已停止');
      } else {
        showToast(`停止设备失败: ${response.error || '未知错误'}`, true);
      }
    } catch (error) {
      console.error('停止设备异常:', error);
      showToast(`停止设备失败: ${error instanceof Error ? error.message : String(error)}`, true);
    } finally {
      setDevicePendingAction(null);
      setActionType(null);
    }
  }, [devicePendingAction, showToast]);

  const handleResetFilters = () => {
    setSearchText('');
    setRuntimeNodeTypeFilter('');
    setStatusFilter('');
  };

  if (loading) {
    return (
      <div style={{ padding: '24px', textAlign: 'center', color: '#6b7280' }}>
        {t('common.loading')}
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', backgroundColor: '#f6f7f9', minHeight: '100vh' }}>
      {/* 页面头部 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: '24px',
        paddingBottom: '16px',
        borderBottom: '1px solid #e5e7eb',
        gap: 16,
      }}>
        <div>
          <h2 style={{
            fontSize: '20px',
            fontWeight: '600',
            color: '#111827',
            margin: 0,
          }}>{t('devicesPage.title')}</h2>
          <p style={{ margin: '8px 0 0', fontSize: 14, color: '#6b7280', lineHeight: 1.55, maxWidth: 640 }}>
            {t('devicesPage.subtitle')}
          </p>
        </div>
        {canMutate ? (
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              style={{
                padding: '10px 20px',
                backgroundColor: '#2563eb',
                border: 'none',
                borderRadius: '6px',
                color: '#ffffff',
                fontSize: '14px',
                cursor: 'pointer',
                fontWeight: '500',
                boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                transition: 'all 0.2s',
              }}
              onClick={() => setShowAddModal(true)}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#1d4ed8'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#2563eb'; }}
            >
              + {t('devicesPage.addDevice')}
            </button>
            <button
              style={{
                padding: '10px 20px',
                backgroundColor: '#10b981',
                border: 'none',
                borderRadius: '6px',
                color: '#ffffff',
                fontSize: '14px',
                cursor: 'pointer',
                fontWeight: '500',
                boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                transition: 'all 0.2s',
              }}
              onClick={() => setShowAgentWizard(true)}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#059669'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#10b981'; }}
              title="远程安装采集端 Client"
            >
              安装 Client
            </button>
          </div>
        ) : null}
      </div>

      {/* 搜索和筛选 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 24px',
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        marginBottom: '16px',
        gap: '16px',
      }}>
        {/* 搜索框 */}
        <div style={{ position: 'relative', flex: 1, maxWidth: '300px' }}>
          <input
            type="text"
            placeholder={t('devicesPage.searchPlaceholder')}
            style={{
              width: '100%',
              padding: '8px 12px 8px 36px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#111827',
              fontSize: '14px',
              outline: 'none',
              boxSizing: 'border-box',
            }}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
          <svg
            style={{
              position: 'absolute',
              left: '12px',
              top: '50%',
              transform: 'translateY(-50%)',
              width: '16px',
              height: '16px',
              fill: '#6b7280',
            }}
            viewBox="0 0 24 24"
          >
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
          </svg>
        </div>

        {/* 筛选下拉 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <select
            style={{
              padding: '8px 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#111827',
              fontSize: '14px',
              outline: 'none',
              minWidth: '120px',
            }}
            value={runtimeNodeTypeFilter}
            onChange={(e) => setRuntimeNodeTypeFilter(e.target.value as RuntimeNodeTypeKey | '')}
          >
            <option value="">{t('devicesPage.typeFilter')}</option>
            {RUNTIME_NODE_TYPE_KEYS.map((key) => (
              <option key={key} value={key}>
                {t(`devicesPage.${key}`)}
              </option>
            ))}
          </select>

          <select
            style={{
              padding: '8px 12px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#111827',
              fontSize: '14px',
              outline: 'none',
              minWidth: '120px',
            }}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as DeviceStatus | '')}
          >
            <option value="">{t('devicesPage.statusFilter')}</option>
            <option value="CONNECTED">{t('devicesPage.statusConnected')}</option>
            <option value="DISCONNECTED">{t('devicesPage.statusDisconnected')}</option>
            <option value="CONNECTING">{t('devicesPage.statusConnecting')}</option>
            <option value="ERROR">{t('devicesPage.statusError')}</option>
          </select>

          <button
            style={{
              padding: '8px 16px',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              color: '#374151',
              fontSize: '14px',
              cursor: 'pointer',
              outline: 'none',
              transition: 'all 0.2s',
            }}
            onClick={handleResetFilters}
          >
            {t('devicesPage.reset')}
          </button>
        </div>
      </div>

      {/* 设备表格 */}
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        overflow: 'hidden',
      }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse' as const,
        }}>
          <thead>
            <tr style={{
              backgroundColor: '#f9fafb',
              borderBottom: '1px solid #e5e7eb',
            }}>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('devicesPage.tableName')}</th>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('devicesPage.tableType')}</th>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('devicesPage.tableConn')}</th>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('devicesPage.tableTeam')}</th>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('devicesPage.tableStatus')}</th>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('devicesPage.tableUpdatedAt')}</th>
              <th style={{ padding: '12px', textAlign: 'left', fontSize: '13px', fontWeight: '600', color: '#374151' }}>{t('common.actions')}</th>
            </tr>
          </thead>
          <tbody>
            {filteredDevices.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ padding: '40px', textAlign: 'center', color: '#6b7280' }}>
                  {t('devicesPage.empty') || t('common.noData')}
                </td>
              </tr>
            ) : (
              filteredDevices.map((device) => {
                const statusStyle = getStatusTagMeta(device.status, t);
                return (
                  <tr
                    key={device.id}
                    style={{
                      borderBottom: '1px solid #f3f4f6',
                      transition: 'background-color 0.15s',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#f9fafb';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <td style={{ padding: '12px', fontSize: '14px', color: '#111827', fontWeight: '500' }}>
                      <Link
                        href={`/devices/${device.id}`}
                        style={{
                          color: '#1d4ed8',
                          textDecoration: 'none',
                          fontWeight: 600,
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.textDecoration = 'underline';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.textDecoration = 'none';
                        }}
                      >
                        {device.name}
                      </Link>
                    </td>
                    <td style={{ padding: '12px', fontSize: '14px', color: '#111827' }}>
                      {runtimeNodeTypeLabel(device, t)}
                    </td>
                    <td
                      style={{
                        padding: '12px',
                        fontSize: '14px',
                        color: '#6b7280',
                        wordBreak: 'break-word',
                        maxWidth: 420,
                      }}
                    >
                      {getConnSummary(device, {
                        notConfiguredLabel: t('devicesPage.connNotConfigured'),
                      })}
                    </td>
                    <td style={{ padding: '12px', fontSize: '14px', color: '#6b7280', wordBreak: 'break-word' }}>
                      {(device.teamName || device.teamId || '').toString().trim() || '—'}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span style={{
                        display: 'inline-block',
                        padding: '4px 12px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        fontWeight: '500',
                        backgroundColor: statusStyle.bg,
                        color: statusStyle.color,
                      }}>
                        {statusStyle.text}
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontSize: '14px', color: '#6b7280' }}>
                      {formatDateTimeMinute(device.updatedAt)}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          onClick={() =>
                            device.status === 'CONNECTED' ? openStopDeviceConfirm(device) : handleConnect(device)
                          }
                          disabled={device.status === 'CONNECTING' || connectingDeviceId === device.id}
                          style={{
                            padding: '4px 12px',
                            backgroundColor:
                              device.status === 'CONNECTED'
                                ? '#ef4444'
                                : device.status === 'CONNECTING' || connectingDeviceId === device.id
                                ? '#d1d5db'
                                : '#2563eb',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#ffffff',
                            fontSize: '12px',
                            cursor:
                              device.status === 'CONNECTING' || connectingDeviceId === device.id
                                ? 'not-allowed'
                                : 'pointer',
                            opacity:
                              device.status === 'CONNECTING' || connectingDeviceId === device.id ? 0.6 : 1,
                          }}
                        >
                          {device.status === 'CONNECTED'
                            ? '停止设备'
                            : device.status === 'CONNECTING' || connectingDeviceId === device.id
                            ? '启动中...'
                            : '启动设备'}
                        </button>
                        {canMutate ? (
                          <>
                            <button
                              onClick={() => handleEditDevice(device)}
                              style={{
                                padding: '4px 12px',
                                backgroundColor: '#ffffff',
                                border: '1px solid #d1d5db',
                                borderRadius: '4px',
                                color: '#374151',
                                fontSize: '12px',
                                cursor: 'pointer',
                              }}
                            >
                              {t('common.edit')}
                            </button>
                            <button
                              onClick={() => openDeleteDeviceConfirm(device)}
                              style={{
                                padding: '4px 12px',
                                backgroundColor: '#ef4444',
                                border: 'none',
                                borderRadius: '4px',
                                color: '#ffffff',
                                fontSize: '12px',
                                cursor: 'pointer',
                              }}
                            >
                              {t('common.delete')}
                            </button>
                          </>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {canMutate ? (
        <>
          <AddDeviceModal
            open={showAddModal}
            onClose={() => setShowAddModal(false)}
            onSave={handleAddDevice}
            onOpenAgentWizard={() => setShowAgentWizard(true)}
          />
          <AddDeviceModal
            device={editingDevice}
            open={showEditModal}
            onClose={() => {
              setShowEditModal(false);
              setEditingDevice(null);
            }}
            onSave={handleUpdateDevice}
            onOpenAgentWizard={() => setShowAgentWizard(true)}
          />
          <AgentInstallWizard open={showAgentWizard} onClose={() => setShowAgentWizard(false)} />
        </>
      ) : null}

      {/* 启动错误弹窗 */}
      {launchError && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            width: '600px',
            maxWidth: '90vw',
            maxHeight: '80vh',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          }}>
            <div style={{
              padding: '16px 24px',
              borderBottom: '1px solid #e5e7eb',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#ef4444' }}>
                启动失败
              </h3>
              <button
                onClick={() => setLaunchError(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '24px',
                  color: '#6b7280',
                  cursor: 'pointer',
                  padding: '0 4px',
                }}
              >
                ×
              </button>
            </div>
            <div style={{
              padding: '24px',
              overflowY: 'auto',
              flex: 1,
            }}>
              <pre style={{
                margin: 0,
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
                fontFamily: 'monospace',
                fontSize: '14px',
                color: '#374151',
                backgroundColor: '#f3f4f6',
                padding: '12px',
                borderRadius: '6px',
              }}>
                {launchError}
              </pre>
            </div>
            <div style={{
              padding: '16px 24px',
              borderTop: '1px solid #e5e7eb',
              display: 'flex',
              justifyContent: 'flex-end',
            }}>
              <button
                onClick={() => setLaunchError(null)}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  color: '#374151',
                  fontSize: '14px',
                  cursor: 'pointer',
                  fontWeight: '500',
                }}
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={!!devicePendingAction && !!actionType && (actionType !== 'delete' || canMutate)}
        title={actionType === 'delete' ? '删除设备' : '停止设备'}
        description={
          devicePendingAction
            ? actionType === 'delete'
              ? `确定要删除设备「${devicePendingAction.name}」吗？删除后不可恢复。`
              : `确定要停止设备「${devicePendingAction.name}」吗？`
            : ''
        }
        confirmText="确认"
        cancelText="取消"
        onCancel={() => {
          setDevicePendingAction(null);
          setActionType(null);
        }}
        onConfirm={actionType === 'delete' ? handleConfirmDeleteDevice : handleConfirmStopDevice}
      />

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
    </div>
  );
}
