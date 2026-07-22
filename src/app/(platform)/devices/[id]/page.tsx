'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'next/navigation';
import { AlertPanel, type AlertItem } from '@/components/dashboard/AlertPanel';
import { getDevice, listOnlineAgents, type OnlineAgentItem } from '@/features/data-platform/api/deviceApi';
import type { RobotDevice } from '@/features/data-platform/models/device';
import devicePreviewPlaceholder from '@/picture/2efd61f1d58c53b9266da2dbf5e470df.jpg';

type MetricPoint = {
  t: string;
  cpu: number;
  memory: number;
  disk: number;
};

type PreviewMode = 'joint' | 'ft';

/** 末端 wrench 六维分量键（与采集端 heartbeat 一致） */
type FtCompKey = 'Fx' | 'Fy' | 'Fz' | 'Mx' | 'My' | 'Mz';

type JointState = {
  name: string;
  position: number | null;
  velocity: number | null;
  effort: number | null;
  temperature: number | null;
  status: 'normal' | 'warn' | 'error' | 'unknown';
};

function clamp(v: number, low: number, high: number) {
  return Math.max(low, Math.min(high, v));
}

function numOrNull(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === 'string' && v.trim() === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function boolish(v: unknown): boolean | null {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return v !== 0;
  if (typeof v === 'string') {
    const s = v.trim().toLowerCase();
    if (['true', '1', 'yes', 'on', 'running', 'ready', 'online'].includes(s)) return true;
    if (['false', '0', 'no', 'off', 'stopped', 'idle', 'offline'].includes(s)) return false;
  }
  return null;
}

function toGb(value: number, unit: 'mb' | 'gb'): number {
  return unit === 'mb' ? value / 1024 : value;
}

function formatGb(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return '--';
  if (v >= 100) return `${v.toFixed(0)} GiB`;
  return `${v.toFixed(1)} GiB`;
}

function seriesPath(values: number[], width: number, height: number, min = 0, max = 100): string {
  if (!values.length) return '';
  const span = Math.max(1e-9, max - min);
  const stepX = width / Math.max(1, values.length - 1);
  return values
    .map((v, i) => {
      const x = i * stepX;
      const y = height - ((clamp(v, min, max) - min) / span) * height;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
}

function miniSparkPath(values: number[], width: number, height: number): string {
  if (values.length < 2) return '';
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return '';
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  }
  return seriesPath(values, width, height, lo, hi);
}

function formatScalar(v: number | null, digits = 4): string {
  if (v === null || !Number.isFinite(v)) return '--';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(1);
  if (a >= 100) return v.toFixed(2);
  if (a >= 1) return v.toFixed(digits);
  return v.toFixed(Math.min(5, digits + 1));
}

function normalizeJointName(rawName: unknown, idx: number): string {
  // 统一关节名字，确保第二级“joint1/joint2...”能稳定命中数据。
  // 例如某些 payload 可能是 `joint4`，也可能是 `left_joint4`。
  const s = String(rawName ?? '').trim();
  const m = s.match(/(\d+)/);
  if (m && m[1]) {
    const n = Number(m[1]);
    if (Number.isFinite(n) && n > 0) return `joint${n}`;
  }
  return s || `joint${idx + 1}`;
}

/** 心跳里该话题的关节 payload 是否包含可用的 effort/力矩（仅用于提示，不用于自动切换/覆盖） */
function jointPayloadHasEffort(payload: Record<string, unknown> | undefined): boolean {
  if (!payload) return false;
  const efforts = payload.joint_efforts;
  if (Array.isArray(efforts)) {
    for (const x of efforts) {
      if (numOrNull(x) !== null) return true;
    }
  }
  const rawJoints = payload.joints;
  if (Array.isArray(rawJoints)) {
    for (const x of rawJoints) {
      const obj = (x || {}) as Record<string, unknown>;
      if (numOrNull(obj.effort) !== null || numOrNull(obj.torque) !== null) return true;
    }
  }
  return false;
}

type JointHistorySeries = {
  pos: number[];
  posTs: string[];
  vel: number[];
  velTs: string[];
  effort: number[];
  effortTs: string[];
};
type JointMetricKey = 'position' | 'velocity' | 'effort';

type ArmMode = 'master' | 'slave';

/** 主臂关节/命令侧：话题名含 cmd、master、gello（其余含 joint 的一律视为从臂关节状态） */
function isMasterTopicName(topic: string): boolean {
  const tl = topic.toLowerCase();
  return tl.includes('cmd') || tl.includes('master') || tl.includes('gello');
}

function jointGradeStyle(status: JointState['status']): { label: string; color: string; bg: string } {
  switch (status) {
    case 'normal':
      return { label: '正常', color: '#059669', bg: '#ecfdf5' };
    case 'warn':
      return { label: '预警', color: '#d97706', bg: '#fffbeb' };
    case 'error':
      return { label: '异常', color: '#dc2626', bg: '#fef2f2' };
    default:
      return { label: '未知', color: '#6b7280', bg: '#f3f4f6' };
  }
}

export default function DeviceDetailPage() {
  const params = useParams<{ id: string }>();
  const deviceId = String(params?.id || '').trim();
  const [device, setDevice] = useState<RobotDevice | null>(null);
  const [agents, setAgents] = useState<OnlineAgentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [initialLoaded, setInitialLoaded] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [history, setHistory] = useState<MetricPoint[]>([]);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [jointHistoryByName, setJointHistoryByName] = useState<Record<string, JointHistorySeries>>({});
  const [jointPanelOpen, setJointPanelOpen] = useState(true);
  const [previewMode, setPreviewMode] = useState<PreviewMode>('joint');
  const [armMode, setArmMode] = useState<ArmMode>('master');
  const [selectedJointTopic, setSelectedJointTopic] = useState('');
  const [selectedJointName, setSelectedJointName] = useState('');
  const [selectedJointMetric, setSelectedJointMetric] = useState<JointMetricKey>('position');

  const [ftHistoryByComp, setFtHistoryByComp] = useState<Record<string, { v: number[]; ts: string[] }>>({});
  const [selectedFtTopic, setSelectedFtTopic] = useState('');
  /** 点击某一维在下方展开大趋势；再次点击同卡收起 */
  const [selectedFtDetail, setSelectedFtDetail] = useState<FtCompKey | null>(null);

  const [rosTopicsView, setRosTopicsView] = useState<string[]>([]);
  const [rosRefreshing, setRosRefreshing] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load(initial = false) {
      if (!deviceId) return;
      try {
        if (initial && !initialLoaded) {
          setLoading(true);
        }
        const [devResp, agentsResp] = await Promise.all([getDevice(deviceId), listOnlineAgents()]);
        if (cancelled) return;
        if (!devResp.ok || !devResp.data) {
          setErr(devResp.error || '加载设备详情失败');
          if (!device) setDevice(null);
        } else {
          setErr(null);
          setDevice(devResp.data);
        }
        setAgents(agentsResp.ok && agentsResp.data ? agentsResp.data : []);
      } catch (e) {
        if (cancelled) return;
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) {
          setLoading(false);
          setInitialLoaded(true);
        }
      }
    }

    load(true);
    const iv = setInterval(() => load(false), 5000);
    return () => {
      cancelled = true;
      clearInterval(iv);
    };
  }, [deviceId, initialLoaded]);

  const agentRuntime = useMemo(() => {
    if (!device?.hardwareUuid) return null;
    const byId = agents.find((a) => a.agent_id === device.hardwareUuid);
    if (byId) {
      return byId as (OnlineAgentItem & {
        tunnel_stale?: boolean;
        heartbeat?: Record<string, unknown>;
        seconds_since_tunnel_seen?: number;
      });
    }
    const host = String(device.agentIp || '').trim();
    const port = Number(device.agentPort || 0);
    if (host && Number.isFinite(port) && port > 0) {
      const byAddr = agents.find((a) => String(a.host || '').trim() === host && Number(a.port || 0) === port);
      if (byAddr) {
        return byAddr as (OnlineAgentItem & {
          tunnel_stale?: boolean;
          heartbeat?: Record<string, unknown>;
          seconds_since_tunnel_seen?: number;
        });
      }
    }
    return null as any;
  }, [agents, device?.hardwareUuid, device?.agentIp, device?.agentPort]);

  const agentRuntimeWithExtras = agentRuntime as (OnlineAgentItem & {
      tunnel_stale?: boolean;
      heartbeat?: Record<string, unknown>;
      seconds_since_tunnel_seen?: number;
    }) | null;

  const tunnelStale = Boolean(agentRuntimeWithExtras && agentRuntimeWithExtras.tunnel_stale);

  const liveMetrics = useMemo(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const outq = Math.max(0, Number(hb.outq_size || 0));
    const collectRunning = Boolean(hb.collect_running);
    const deviceRunning = Boolean(hb.device_running);
    const cpuRaw = numOrNull(hb.cpu_percent) ?? numOrNull(hb.cpu_usage) ?? numOrNull(hb.cpu);
    const memUsedMb = numOrNull(hb.mem_used_mb) ?? numOrNull(hb.memory_used_mb);
    const memTotalMb = numOrNull(hb.mem_total_mb) ?? numOrNull(hb.memory_total_mb);
    const diskUsedGb = numOrNull(hb.disk_used_gb);
    const diskTotalGb = numOrNull(hb.disk_total_gb);
    const memRatio = memUsedMb !== null && memTotalMb && memTotalMb > 0 ? (memUsedMb / memTotalMb) * 100 : null;
    const diskRatio = diskUsedGb !== null && diskTotalGb && diskTotalGb > 0 ? (diskUsedGb / diskTotalGb) * 100 : null;
    const memoryRaw = memRatio ?? numOrNull(hb.mem_percent) ?? numOrNull(hb.memory_percent);
    const diskRaw = diskRatio ?? numOrNull(hb.disk_percent) ?? numOrNull(hb.disk_usage);
    const cpu = clamp(cpuRaw ?? 0, 0, 100);
    const memory = clamp(memoryRaw ?? 0, 0, 100);
    const disk = clamp(diskRaw ?? 0, 0, 100);

    // 网络/心跳质量：后端从 HEARTBEAT 的 envelope ts_ms 估算延迟
    const tunnelLatencyMs = numOrNull(hb.tunnel_latency_ms);
    const heartbeatIntervalMs = numOrNull(hb.heartbeat_interval_ms);
    const heartbeatMissedIntervals = numOrNull(hb.heartbeat_missed_intervals);

    if (tunnelStale) {
      // stale 后置空：前端不要展示“最后心跳”的资源统计
      return {
        cpu: Number.NaN,
        memory: Number.NaN,
        disk: Number.NaN,
        outq: Number.NaN,
        collectRunning: false,
        deviceRunning: false,
        tunnelLatencyMs: null,
        heartbeatIntervalMs: null,
        heartbeatMissedIntervals: null,
      };
    }

    return {
      cpu,
      memory,
      disk,
      outq,
      collectRunning,
      deviceRunning,
      tunnelLatencyMs,
      heartbeatIntervalMs,
      heartbeatMissedIntervals,
    };
  }, [agentRuntime, tunnelStale]);

const storageStats = useMemo(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;

    const memUsedMb = numOrNull(hb.mem_used_mb) ?? numOrNull(hb.memory_used_mb) ?? numOrNull(hb.used_memory_mb);
    const memTotalMb = numOrNull(hb.mem_total_mb) ?? numOrNull(hb.memory_total_mb) ?? numOrNull(hb.total_memory_mb);
    const memUsedGbRaw = numOrNull(hb.mem_used_gb) ?? numOrNull(hb.memory_used_gb) ?? numOrNull(hb.used_memory_gb);
    const memTotalGbRaw = numOrNull(hb.mem_total_gb) ?? numOrNull(hb.memory_total_gb) ?? numOrNull(hb.total_memory_gb);
    const memUsedGb = memUsedGbRaw ?? (memUsedMb !== null ? toGb(memUsedMb, 'mb') : null);
    const memTotalGb = memTotalGbRaw ?? (memTotalMb !== null ? toGb(memTotalMb, 'mb') : null);
    const memUsedFromRatio = memTotalGb !== null ? (memTotalGb * liveMetrics.memory) / 100 : null;

    const diskFreeGbRaw = numOrNull(hb.disk_free_gb) ?? numOrNull(hb.disk_available_gb) ?? numOrNull(hb.free_disk_gb);
    const diskTotalGbRaw = numOrNull(hb.disk_total_gb) ?? numOrNull(hb.total_disk_gb);
    const diskFreeMb = numOrNull(hb.disk_free_mb) ?? numOrNull(hb.disk_available_mb) ?? numOrNull(hb.free_disk_mb);
    const diskTotalMb = numOrNull(hb.disk_total_mb) ?? numOrNull(hb.total_disk_mb);
    const diskFreeGb = diskFreeGbRaw ?? (diskFreeMb !== null ? toGb(diskFreeMb, 'mb') : null);
    const diskTotalGb = diskTotalGbRaw ?? (diskTotalMb !== null ? toGb(diskTotalMb, 'mb') : null);
    const diskFreeFromRatio = diskTotalGb !== null ? diskTotalGb * (1 - liveMetrics.disk / 100) : null;

    return {
      memoryUsedLabel: formatGb(memUsedGb ?? memUsedFromRatio),
      memoryExtraLabel: memTotalGb !== null ? ` / ${formatGb(memTotalGb)}` : '',
      diskFreeLabel: formatGb(diskFreeGb ?? diskFreeFromRatio),
      diskTotalLabel: diskTotalGb !== null ? `（总计 ${formatGb(diskTotalGb)}）` : '',
    };
  }, [agentRuntime, liveMetrics.memory, liveMetrics.disk]);

  const jointStateMap = useMemo<Record<string, Record<string, unknown>>>(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const byTopic = (hb.joint_states_by_topic || {}) as Record<string, unknown>;
    const out: Record<string, Record<string, unknown>> = {};
    for (const [k, v] of Object.entries(byTopic)) {
      if (v && typeof v === 'object') out[k] = v as Record<string, unknown>;
    }
    return out;
  }, [agentRuntime]);

  const sampledJointTopics = useMemo<string[]>(() => Object.keys(jointStateMap).sort(), [jointStateMap]);

  const ftStateMap = useMemo<Record<string, Record<string, unknown>>>(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const byTopic = (hb.ft_states_by_topic || {}) as Record<string, unknown>;
    const out: Record<string, Record<string, unknown>> = {};
    for (const [k, v] of Object.entries(byTopic)) {
      if (v && typeof v === 'object') out[k] = v as Record<string, unknown>;
    }
    return out;
  }, [agentRuntime]);

  const sampledFtTopics = useMemo<string[]>(() => Object.keys(ftStateMap).sort(), [ftStateMap]);

  const ftSelectedPayload = useMemo(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const activeTopic = String(hb.ft_active_topic || '').trim();
    const hasSelectedTopic = Boolean(selectedFtTopic);
    const selectedPayload = hasSelectedTopic ? ftStateMap[selectedFtTopic] : undefined;
    const payload = (selectedPayload
      || (activeTopic ? ftStateMap[activeTopic] : undefined)
      || ((Array.isArray(hb.ft_force) || Array.isArray(hb.ft_torque))
        ? { force: hb.ft_force, torque: hb.ft_torque }
        : undefined)
      || {}) as Record<string, unknown>;

    const forceArr = Array.isArray(payload.force) ? payload.force : [];
    const torqueArr = Array.isArray(payload.torque) ? payload.torque : [];

    return {
      force: [numOrNull(forceArr[0]), numOrNull(forceArr[1]), numOrNull(forceArr[2])],
      torque: [numOrNull(torqueArr[0]), numOrNull(torqueArr[1]), numOrNull(torqueArr[2])],
    };
  }, [agentRuntime, ftStateMap, selectedFtTopic]);

  const ftAllCompValues = useMemo<Record<FtCompKey, number | null>>(() => {
    return {
      Fx: ftSelectedPayload.force[0],
      Fy: ftSelectedPayload.force[1],
      Fz: ftSelectedPayload.force[2],
      Mx: ftSelectedPayload.torque[0],
      My: ftSelectedPayload.torque[1],
      Mz: ftSelectedPayload.torque[2],
    };
  }, [ftSelectedPayload]);

  const ftHasWrenchData = useMemo(() => {
    if (!selectedFtTopic.trim()) return false;
    return Object.values(ftAllCompValues).some((v) => v !== null && Number.isFinite(v));
  }, [ftAllCompValues, selectedFtTopic]);

  const ftMaxAbs = useMemo(() => {
    let m = 0;
    for (const v of Object.values(ftAllCompValues)) {
      if (v === null || !Number.isFinite(v)) continue;
      m = Math.max(m, Math.abs(v));
    }
    return m;
  }, [ftAllCompValues]);

  /** 六维整体分级（任意分量超阈即升级），阈值可按现场标定再调 */
  const ftOverallStatus = useMemo<JointState['status']>(() => {
    if (!ftHasWrenchData) return 'unknown';
    const warnAbs = 10;
    const errorAbs = 30;
    if (ftMaxAbs >= errorAbs) return 'error';
    if (ftMaxAbs >= warnAbs) return 'warn';
    return 'normal';
  }, [ftHasWrenchData, ftMaxAbs]);

  const ftOverallGrade = useMemo(() => jointGradeStyle(ftOverallStatus), [ftOverallStatus]);

  const jointStates = useMemo<JointState[]>(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const activeTopic = String(hb.joint_active_topic || '').trim();
    const hasSelectedTopic = Boolean(selectedJointTopic);
    const selectedPayload = hasSelectedTopic ? jointStateMap[selectedJointTopic] : undefined;
    // 优先使用选中话题；选中话题未采样时，回退到 active_topic 对应数据，再回退到顶层 joints（兼容旧 payload）。
    const jointPayload = (selectedPayload
      || (activeTopic ? jointStateMap[activeTopic] : undefined)
      || (hb.joints || hb.joint_positions ? hb : undefined)
      || {}) as Record<string, unknown>;

    const getStatusByTemp = (temp: number | null): JointState['status'] => {
      if (temp === null) return 'unknown';
      if (temp >= 85) return 'error';
      if (temp >= 75) return 'warn';
      return 'normal';
    };

    const rawJoints = jointPayload.joints;
    const effortArr = Array.isArray(jointPayload.joint_efforts) ? jointPayload.joint_efforts : [];
    if (Array.isArray(rawJoints)) {
      return rawJoints.map((x, idx) => {
        const obj = (x || {}) as Record<string, unknown>;
        const temperature = numOrNull(obj.temperature) ?? numOrNull(obj.temp);
        const rawStatus = String(obj.status || '').toLowerCase();
        let status: JointState['status'] = getStatusByTemp(temperature);
        if (rawStatus.includes('error') || rawStatus.includes('fault')) status = 'error';
        else if (rawStatus.includes('warn')) status = 'warn';
        else if (rawStatus.includes('ok') || rawStatus.includes('normal')) status = 'normal';
        const effortFromObj = numOrNull(obj.effort) ?? numOrNull(obj.torque);
        const effortFromArr = numOrNull(effortArr[idx]);
        return {
          name: normalizeJointName(obj.name, idx),
          position: numOrNull(obj.position) ?? numOrNull(obj.pos),
          velocity: numOrNull(obj.velocity) ?? numOrNull(obj.vel),
          effort: effortFromObj !== null ? effortFromObj : effortFromArr,
          temperature,
          status,
        };
      });
    }

    const positions = Array.isArray(jointPayload.joint_positions) ? jointPayload.joint_positions : [];
    const velocities = Array.isArray(jointPayload.joint_velocities) ? jointPayload.joint_velocities : [];
    const efforts = Array.isArray(jointPayload.joint_efforts) ? jointPayload.joint_efforts : [];
    const temperatures = Array.isArray(jointPayload.joint_temperatures) ? jointPayload.joint_temperatures : [];
    const count = Math.max(positions.length, velocities.length, efforts.length, temperatures.length);
    if (count <= 0) return [];

    return Array.from({ length: count }).map((_, idx) => {
      const temperature = numOrNull(temperatures[idx]);
      return {
        name: `joint${idx + 1}`,
        position: numOrNull(positions[idx]),
        velocity: numOrNull(velocities[idx]),
        effort: numOrNull(efforts[idx]),
        temperature,
        status: getStatusByTemp(temperature),
      };
    });
  }, [agentRuntime, selectedJointTopic, jointStateMap]);

  useEffect(() => {
    setJointHistoryByName({});
  }, [selectedJointTopic]);

  useEffect(() => {
    setJointHistoryByName((prev) => {
      const next: Record<string, JointHistorySeries> = { ...prev };
      const cap = 36;
      const nowLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      for (const j of jointStates) {
        const cur = next[j.name] || { pos: [], posTs: [], vel: [], velTs: [], effort: [], effortTs: [] };
        next[j.name] = {
          pos: j.position !== null ? [...cur.pos, j.position].slice(-cap) : cur.pos,
          posTs: j.position !== null ? [...cur.posTs, nowLabel].slice(-cap) : cur.posTs,
          vel: j.velocity !== null ? [...cur.vel, j.velocity].slice(-cap) : cur.vel,
          velTs: j.velocity !== null ? [...cur.velTs, nowLabel].slice(-cap) : cur.velTs,
          effort: j.effort !== null ? [...cur.effort, j.effort].slice(-cap) : cur.effort,
          effortTs: j.effort !== null ? [...cur.effortTs, nowLabel].slice(-cap) : cur.effortTs,
        };
      }
      const keep = new Set(jointStates.map((x) => x.name));
      for (const k of Object.keys(next)) {
        if (!keep.has(k)) delete next[k];
      }
      return next;
    });
  }, [jointStates]);

  useEffect(() => {
    setFtHistoryByComp({});
  }, [selectedFtTopic]);

  useEffect(() => {
    setSelectedFtDetail(null);
  }, [selectedFtTopic]);

  useEffect(() => {
    if (tunnelStale) {
      setFtHistoryByComp({});
      return;
    }
    setFtHistoryByComp((prev) => {
      const next: Record<string, { v: number[]; ts: string[] }> = { ...prev };
      const cap = 36;
      const nowLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const hasAny = Object.values(ftAllCompValues).some((v) => v !== null && Number.isFinite(v));
      if (!hasAny) return prev;

      for (const [k, v] of Object.entries(ftAllCompValues)) {
        if (v === null || !Number.isFinite(v)) continue;
        const cur = next[k] || { v: [], ts: [] };
        next[k] = {
          v: [...cur.v, v].slice(-cap),
          ts: [...cur.ts, nowLabel].slice(-cap),
        };
      }
      return next;
    });
  }, [ftAllCompValues, tunnelStale]);

  const rosTopicNames = useMemo<string[]>(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const arr = hb.ros_topic_names;
    if (!Array.isArray(arr)) return [];
    return Array.from(new Set(arr.map((x) => String(x || '').trim()).filter(Boolean))).sort();
  }, [agentRuntime]);

  useEffect(() => {
    if (!rosTopicsView.length && rosTopicNames.length) {
      setRosTopicsView(rosTopicNames);
    }
  }, [rosTopicNames, rosTopicsView.length]);

  const refreshRosTopics = useCallback(async () => {
    if (!deviceId) return;
    try {
      setRosRefreshing(true);
      const resp = await listOnlineAgents();
      if (resp.ok && resp.data) {
        setAgents(resp.data);
      }
      setRosTopicsView(rosTopicNames);
    } finally {
      setRosRefreshing(false);
    }
  }, [deviceId, rosTopicNames]);

  const jointTopicOptions = useMemo<string[]>(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const byHeartbeat = Array.isArray(hb.joint_topics)
      ? hb.joint_topics.map((x) => String(x || '').trim()).filter(Boolean)
      : [];
    const byPattern = rosTopicNames.filter((t) => t.toLowerCase().includes('joint'));
    // 让“已采样到数据的话题”排在前面，优先可见、可选
    const merged = Array.from(new Set([...sampledJointTopics, ...byHeartbeat, ...byPattern]));
    return merged.sort((a, b) => {
      const aOk = sampledJointTopics.includes(a) ? 0 : 1;
      const bOk = sampledJointTopics.includes(b) ? 0 : 1;
      if (aOk !== bOk) return aOk - bOk;
      return a.localeCompare(b);
    });
  }, [agentRuntime, rosTopicNames, sampledJointTopics]);

  const filteredJointTopicsByArm = useMemo<string[]>(() => {
    if (!jointTopicOptions.length) return [];
    const masters = jointTopicOptions.filter(isMasterTopicName);
    const slaves = jointTopicOptions.filter((t) => !isMasterTopicName(t));
    if (armMode === 'master') {
      return masters.length ? masters : jointTopicOptions;
    }
    return slaves.length ? slaves : jointTopicOptions;
  }, [armMode, jointTopicOptions]);

  useEffect(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const activeTopic = String(hb.joint_active_topic || '').trim();
    const candidates = filteredJointTopicsByArm;
    if (!candidates.length) {
      setSelectedJointTopic('');
      return;
    }
    if (selectedJointTopic && candidates.includes(selectedJointTopic)) return;
    const sampledInArm = sampledJointTopics.filter((t) => candidates.includes(t));
    if (sampledInArm.length > 0) {
      setSelectedJointTopic(sampledInArm[0]);
      return;
    }
    if (activeTopic && candidates.includes(activeTopic)) {
      setSelectedJointTopic(activeTopic);
      return;
    }
    setSelectedJointTopic(candidates[0]);
  }, [filteredJointTopicsByArm, selectedJointTopic, sampledJointTopics, agentRuntime]);

  const ftTopicOptions = useMemo<string[]>(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const hasForce = (t: string) => String(t || '').toLowerCase().includes('force');
    const byHeartbeat = Array.isArray(hb.ft_topics)
      ? hb.ft_topics.map((x) => String(x || '').trim()).filter((x) => Boolean(x) && hasForce(x))
      : [];
    const byPattern = rosTopicNames.filter((t) => {
      return hasForce(t);
    });
    // 让“已采样到值的话题”排在前面，优先可见、可选
    const sampledForce = sampledFtTopics.filter((t) => hasForce(t));
    const merged = Array.from(new Set([...sampledForce, ...byHeartbeat, ...byPattern]));
    return merged.sort((a, b) => {
      const aOk = sampledFtTopics.includes(a) ? 0 : 1;
      const bOk = sampledFtTopics.includes(b) ? 0 : 1;
      if (aOk !== bOk) return aOk - bOk;
      return a.localeCompare(b);
    });
  }, [agentRuntime, rosTopicNames, sampledFtTopics]);

  /** 仅「从臂」展示末端六维力；主臂不展示。凡话题名含 force 的一律归入从臂末端力列表（含左/右臂等）。 */
  const filteredFtTopicsByArm = useMemo<string[]>(() => {
    if (!ftTopicOptions.length) return [];
    if (armMode === 'master') return [];
    return ftTopicOptions;
  }, [armMode, ftTopicOptions]);

  useEffect(() => {
    if (armMode === 'master') setPreviewMode('joint');
  }, [armMode]);

  useEffect(() => {
    const hb = (agentRuntime?.heartbeat || {}) as Record<string, unknown>;
    const activeTopic = String(hb.ft_active_topic || '').trim();
    const candidates = filteredFtTopicsByArm;
    if (!candidates.length) {
      setSelectedFtTopic('');
      return;
    }
    if (selectedFtTopic && candidates.includes(selectedFtTopic)) return;
    const sampledInArm = sampledFtTopics.filter((t) => candidates.includes(t));
    if (sampledInArm.length > 0) {
      setSelectedFtTopic(sampledInArm[0]);
      return;
    }
    if (activeTopic && candidates.includes(activeTopic)) {
      setSelectedFtTopic(activeTopic);
      return;
    }
    setSelectedFtTopic(candidates[0]);
  }, [filteredFtTopicsByArm, selectedFtTopic, sampledFtTopics, agentRuntime]);

  const jointNameOptions = useMemo<string[]>(
    () => Array.from(new Set(jointStates.map((j) => j.name).filter(Boolean))).sort(),
    [jointStates],
  );

  useEffect(() => {
    if (!jointNameOptions.length) {
      setSelectedJointName('');
      return;
    }
    if (selectedJointName && jointNameOptions.includes(selectedJointName)) return;
    setSelectedJointName(jointNameOptions[0]);
  }, [jointNameOptions, selectedJointName]);

  const isDeviceStarted = useMemo(() => {
    const hb = ((agentRuntimeWithExtras?.heartbeat || {}) as Record<string, unknown>) || {};
    const hbEmpty = !hb || Object.keys(hb).length === 0;
    const hbDeviceRunning = boolish(hb.device_running);
    const hbCollectRunning = boolish(hb.collect_running);
    const hbOnline = boolish(hb.online);
    const runtimeReady = ['ONLINE_IDLE', 'LAUNCHING', 'READY', 'COLLECTING'].includes(String(device?.runtimeStatus || ''));
    const statusReady = ['CONNECTED', 'CONNECTING'].includes(String(device?.status || ''));
    const tunnelReady = Boolean(agentRuntimeWithExtras && !agentRuntimeWithExtras.tunnel_stale);
    const rosTopicReady = rosTopicNames.length > 0;
    const hasBoundAgent = Boolean(agentRuntimeWithExtras);

    if (!hbEmpty) {
      if (hbDeviceRunning === true) return true;
      if (hbCollectRunning === true) return true;
      if (hbOnline !== null) return hbOnline || runtimeReady || statusReady || tunnelReady || rosTopicReady || hasBoundAgent;
    }
    return runtimeReady || statusReady || tunnelReady || rosTopicReady || hasBoundAgent || liveMetrics.deviceRunning;
  }, [agentRuntimeWithExtras, device?.runtimeStatus, device?.status, liveMetrics.deviceRunning, rosTopicNames.length]);

  /** 设备详情页「预览」仅以平台设备状态为准：未连接/连接中等一律黑屏，避免历史心跳残留导致误显「已启动」 */
  const showDevicePreviewLive = device?.status === 'CONNECTED' && isDeviceStarted;

  const selectedJointSeries = useMemo<number[]>(() => {
    if (!selectedJointName) return [];
    const h = jointHistoryByName[selectedJointName];
    if (!h) return [];
    if (selectedJointMetric === 'position') return h.pos;
    if (selectedJointMetric === 'velocity') return h.vel;
    return h.effort;
  }, [jointHistoryByName, selectedJointMetric, selectedJointName]);

  const selectedJointTsSeries = useMemo<string[]>(() => {
    if (!selectedJointName) return [];
    const h = jointHistoryByName[selectedJointName];
    if (!h) return [];
    if (selectedJointMetric === 'position') return h.posTs;
    if (selectedJointMetric === 'velocity') return h.velTs;
    return h.effortTs;
  }, [jointHistoryByName, selectedJointMetric, selectedJointName]);

  const selectedJointCurrent = useMemo<number | null>(() => {
    const j = jointStates.find((x) => x.name === selectedJointName);
    if (!j) return null;
    if (selectedJointMetric === 'position') return j.position;
    if (selectedJointMetric === 'velocity') return j.velocity;
    return j.effort;
  }, [jointStates, selectedJointMetric, selectedJointName]);

  const selectedJointStatus = useMemo<JointState['status']>(() => {
    const j = jointStates.find((x) => x.name === selectedJointName);
    return j?.status || 'unknown';
  }, [jointStates, selectedJointName]);

  const selectedJointGrade = useMemo(() => jointGradeStyle(selectedJointStatus), [selectedJointStatus]);

  useEffect(() => {
    if (tunnelStale) {
      setHistory([]);
      return;
    }
    const pushPoint = () => {
      const nowLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setHistory((prev) => {
        const next = [
          ...prev,
          {
            t: nowLabel,
            cpu: liveMetrics.cpu,
            memory: liveMetrics.memory,
            disk: liveMetrics.disk,
          },
        ].slice(-16);
        return next.map((x) => ({
          ...x,
          cpu: clamp(x.cpu, 0, 100),
          memory: clamp(x.memory, 0, 100),
          disk: clamp(x.disk, 0, 100),
        }));
      });
    };
    pushPoint();
    const iv = setInterval(pushPoint, 5000);
    return () => clearInterval(iv);
  }, [tunnelStale, liveMetrics.cpu, liveMetrics.memory, liveMetrics.disk]);

  const alerts = useMemo<AlertItem[]>(() => {
    if (!device) return [];
    const arr: AlertItem[] = [];
    if (!agentRuntime) {
      arr.push({ id: 'a-offline', type: 'device', message: '设备未匹配到在线 Agent，控制和预览不可用。' });
    } else {
      if (agentRuntime.tunnel_stale) {
        arr.push({ id: 'a-stale', type: 'device', message: '采集端隧道心跳超时，连接可能不稳定。' });
      }
      if (liveMetrics.outq > 120) {
        arr.push({ id: 'a-outq', type: 'device', message: `隧道发送队列积压较高（outq=${liveMetrics.outq}），可能影响实时性。` });
      }
    }
    if (device.status === 'ERROR') {
      arr.push({ id: 'a-status', type: 'device', message: `设备状态异常：${device.lastTestResult?.errorMessage || '请检查日志'}` });
    }
    if (!device.launchConfig?.scriptPath) {
      arr.push({ id: 'a-launch', type: 'task', message: '设备未配置启动脚本，无法远程启动。' });
    }
    return arr;
  }, [device, agentRuntime, liveMetrics.outq]);

  if (loading) {
    return <div style={{ padding: 24, color: '#6b7280' }}>加载设备详情中...</div>;
  }
  if (err || !device) {
    return (
      <div style={{ padding: 24 }}>
        <div style={{ color: '#b91c1c', marginBottom: 12 }}>加载失败：{err || '设备不存在'}</div>
        <Link href="/devices" style={{ color: '#1d4ed8' }}>返回设备列表</Link>
      </div>
    );
  }

  const statusColor = device.status === 'CONNECTED' ? '#10b981' : device.status === 'ERROR' ? '#ef4444' : '#f59e0b';
  const statusTextCN =
    device.status === 'CONNECTED'
      ? '已连接'
      : device.status === 'CONNECTING'
      ? '连接中'
      : device.status === 'DISCONNECTED'
      ? '未连接'
      : device.status === 'ERROR'
      ? '异常'
      : (device.status || '');
  const chartInnerW = 800;
  const chartInnerH = 200;
  const chartLeft = 64;
  const chartTop = 34;
  const yTicks = [0, 20, 40, 60, 80, 100];
  const cpuPath = seriesPath(history.map((x) => x.cpu), chartInnerW, chartInnerH, 0, 100);
  const memPath = seriesPath(history.map((x) => x.memory), chartInnerW, chartInnerH, 0, 100);
  const diskPath = seriesPath(history.map((x) => x.disk), chartInnerW, chartInnerH, 0, 100);
  const xTickIndexes = history.length > 1
    ? Array.from(new Set([0, Math.floor((history.length - 1) / 3), Math.floor((history.length - 1) * 2 / 3), history.length - 1]))
    : [0];
  const hoverX = hoveredIdx === null
    ? null
    : chartLeft + (chartInnerW * (history.length <= 1 ? 0 : hoveredIdx / (history.length - 1)));
  const hoveredPoint = hoveredIdx === null ? null : history[hoveredIdx] || null;
  const valueToChartY = (v: number) => chartTop + chartInnerH - (clamp(v, 0, 100) / 100) * chartInnerH;
  const jointMetricLabel = selectedJointMetric === 'position' ? 'Position' : selectedJointMetric === 'velocity' ? 'Velocity' : 'Effort';
  const jointMetricColor = selectedJointMetric === 'position' ? '#1d4ed8' : selectedJointMetric === 'velocity' ? '#7c3aed' : '#ea580c';
  // 切换关节/话题的瞬间，历史缓存可能还没填充；此时用当前值做兜底，让折线图不至于完全空白。
  const jointChartValues = selectedJointSeries.length
    ? selectedJointSeries
    : selectedJointCurrent === null
      ? []
      : [selectedJointCurrent, selectedJointCurrent];

  const jointNowLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const jointChartTsValues = selectedJointSeries.length
    ? selectedJointTsSeries
    : selectedJointCurrent === null
      ? []
      : [jointNowLabel, jointNowLabel];

  const jointXTickIndexes = jointChartValues.length > 1
    ? Array.from(new Set([0, Math.floor((jointChartValues.length - 1) / 3), Math.floor((jointChartValues.length - 1) * 2 / 3), jointChartValues.length - 1]))
    : [0];
  const jointChartMin = jointChartValues.length ? Math.min(...jointChartValues) : -1;
  const jointChartMax = jointChartValues.length ? Math.max(...jointChartValues) : 1;
  const jointChartPad = Math.max((jointChartMax - jointChartMin) * 0.1, 0.05);
  const jointYMin = jointChartMin - jointChartPad;
  const jointYMax = jointChartMax + jointChartPad;
  const jointLinePath = jointChartValues.length ? seriesPath(jointChartValues, 760, 220, jointYMin, jointYMax) : '';

  const FT_FORCE_KEYS: FtCompKey[] = ['Fx', 'Fy', 'Fz'];
  const FT_TORQUE_KEYS: FtCompKey[] = ['Mx', 'My', 'Mz'];
  const ftCardColors: Record<FtCompKey, string> = {
    Fx: '#1d4ed8',
    Fy: '#2563eb',
    Fz: '#3b82f6',
    Mx: '#9a3412',
    My: '#c2410c',
    Mz: '#ea580c',
  };

  const ftDetailChartValues: number[] = (() => {
    if (!selectedFtDetail) return [];
    const h = ftHistoryByComp[selectedFtDetail]?.v || [];
    const cur = ftAllCompValues[selectedFtDetail];
    if (h.length) return h;
    if (cur !== null && Number.isFinite(cur)) return [cur, cur];
    return [];
  })();

  const ftDetailChartTsValues: string[] = (() => {
    if (!selectedFtDetail) return [];
    const h = ftHistoryByComp[selectedFtDetail];
    if (h?.v?.length) return h.ts;
    return [];
  })();
  const ftDetailChartMin = ftDetailChartValues.length ? Math.min(...ftDetailChartValues) : -1;
  const ftDetailChartMax = ftDetailChartValues.length ? Math.max(...ftDetailChartValues) : 1;
  const ftDetailPad = Math.max((ftDetailChartMax - ftDetailChartMin) * 0.1, 0.05);
  const ftDetailYMin = ftDetailChartMin - ftDetailPad;
  const ftDetailYMax = ftDetailChartMax + ftDetailPad;
  const ftDetailLinePath = ftDetailChartValues.length
    ? seriesPath(ftDetailChartValues, 760, 220, ftDetailYMin, ftDetailYMax)
    : '';
  const ftDetailCurrent = selectedFtDetail ? ftAllCompValues[selectedFtDetail] : null;
  const ftDetailLabel = selectedFtDetail
    ? (selectedFtDetail.startsWith('F') ? `力 ${selectedFtDetail}` : `力矩 ${selectedFtDetail}`)
    : '';
  const ftDetailColor = selectedFtDetail ? ftCardColors[selectedFtDetail] : '#6b7280';

  return (
    <div style={{ padding: 20, background: '#f3f6fb', minHeight: '100vh' }}>
      <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 10 }}>
        <Link href="/devices" style={{ color: '#6b7280', textDecoration: 'none' }}>设备列表</Link>
        {' / '}
        <span>{device.name}</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h1 style={{ margin: 0, fontSize: 28, color: '#111827' }}>{device.name}</h1>
          <span style={{ background: '#dbeafe', color: '#1d4ed8', borderRadius: 999, padding: '3px 10px', fontSize: 12 }}>
            在线监控
          </span>
        </div>
        <div style={{ fontSize: 13, color: '#374151' }}>
          状态：<span style={{ color: statusColor, fontWeight: 700 }}>{statusTextCN}</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 14 }}>
        <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, overflow: 'hidden' }}>
          <div style={{ background: '#0b4b95', color: '#fff', padding: '10px 12px', fontSize: 14 }}>
            设备预览 <span style={{ color: '#7dd3fc' }}>{showDevicePreviewLive ? '已启动' : '未启动'}</span>
          </div>
          <div
            style={{
              height: 220,
              position: 'relative',
              background: '#000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#e5e7eb',
              overflow: 'hidden',
            }}
          >
            {showDevicePreviewLive ? (
              <img
                src={devicePreviewPlaceholder.src}
                width={devicePreviewPlaceholder.width}
                height={devicePreviewPlaceholder.height}
                alt={`${device.name} 预览示意`}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  objectPosition: 'top left',
                  display: 'block',
                }}
              />
            ) : (
              <div style={{ textAlign: 'center', color: '#d1d5db', fontSize: 14, lineHeight: 1.6, padding: 16 }}>
                <div style={{ fontSize: 17, fontWeight: 700, color: '#f9fafb' }}>未启动设备</div>
                <div style={{ marginTop: 6, color: '#9ca3af', fontSize: 12 }}>设备已连接且运行后将显示示意预览图</div>
              </div>
            )}
          </div>
          <div style={{ padding: 12, fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
            <div><b>OS:</b> Ubuntu 22.04 LTS</div>
            <div><b>主机名:</b> {device.hostname || 'unknown'}</div>
            <div><b>内核:</b> 5.x</div>
            <div>
              <b>位置:</b>{' '}
              {device.location?.city || device.location?.region || '-'} /{' '}
              {device.location?.country || device.location?.note || '-'}
            </div>
            <div>
              <b>设备ID:</b> {device.hardwareUuid || '-'}
            </div>
          </div>
          <div style={{ borderTop: '1px solid #e5e7eb', padding: 12, background: '#f8fafc' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#374151' }}>ROS 话题</div>
              <button
                onClick={refreshRosTopics}
                disabled={rosRefreshing}
                title="刷新话题"
                style={{
                  border: '1px solid #d1d5db',
                  borderRadius: 6,
                  background: '#fff',
                  color: '#374151',
                  cursor: rosRefreshing ? 'not-allowed' : 'pointer',
                  padding: '2px 8px',
                  fontSize: 12,
                }}
              >
                {rosRefreshing ? '刷新中...' : '↻'}
              </button>
            </div>
            {rosTopicsView.length > 0 ? (
                <div style={{ maxHeight: 140, overflow: 'auto', fontSize: 12, color: '#334155', lineHeight: 1.6 }}>
                  {rosTopicsView.map((topic) => (
                    <div key={topic} style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace' }}>
                      {topic}
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 12, color: '#6b7280' }}>暂无 ROS 话题，点击右上角刷新。</div>
            )}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateRows: 'auto auto 1fr', gap: 12 }}>
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, overflow: 'hidden' }}>
            <div
              style={{
                background: '#f8fafc',
                borderBottom: '1px solid #e5e7eb',
                padding: '10px 12px',
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 700, color: '#374151' }}>一体机状态</div>
              <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 400, marginTop: 2 }}>
                采集端 CPU / 内存 / 磁盘与隧道网络质量
              </div>
            </div>
            <div style={{ padding: 12 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 10 }}>
                {[
                  { label: 'CPU使用率', value: Number.isFinite(liveMetrics.cpu) ? `${liveMetrics.cpu.toFixed(1)}%` : '--', sub: `采集: ${liveMetrics.collectRunning ? '运行中' : '空闲'}` },
                  { label: '内存使用', value: Number.isFinite(liveMetrics.memory) ? `${liveMetrics.memory.toFixed(1)}%` : '--', sub: `已用 ${storageStats.memoryUsedLabel}${storageStats.memoryExtraLabel}` },
                  { label: '磁盘使用', value: Number.isFinite(liveMetrics.disk) ? `${liveMetrics.disk.toFixed(1)}%` : '--', sub: `可用 ${storageStats.diskFreeLabel} ${storageStats.diskTotalLabel}`.trim() },
                  {
                    label: '网络延迟',
                    value: liveMetrics.tunnelLatencyMs === null ? '--' : `${liveMetrics.tunnelLatencyMs.toFixed(0)} ms`,
                    sub:
                      liveMetrics.heartbeatIntervalMs === null
                        ? '心跳间隔：--'
                        : `心跳间隔：${(liveMetrics.heartbeatIntervalMs / 1000).toFixed(1)}s / 丢包估计：${liveMetrics.heartbeatMissedIntervals ?? 0}`,
                  },
                ].map((m) => (
                  <div key={m.label} style={{ background: '#f8fafc', border: '1px solid #e5e7eb', borderRadius: 10, padding: 12 }}>
                    <div style={{ fontSize: 13, color: '#6b7280' }}>{m.label}</div>
                    <div style={{ fontSize: 36, color: '#1e3a8a', lineHeight: 1.1, fontWeight: 700 }}>{m.value}</div>
                    <div style={{ fontSize: 12, color: '#4b5563' }}>{m.sub}</div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ padding: '0 12px 12px' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#374151', marginBottom: 8 }}>资源趋势</div>
            <svg width="100%" viewBox="0 0 920 300" preserveAspectRatio="xMidYMid meet" style={{ height: 280 }}>
              <rect x="0" y="0" width="920" height="300" fill="#f8fbff" />
              <text x="8" y="14" fontSize="12" fill="#6b7280">使用率 (%)</text>
              {yTicks.map((tick) => {
                const y = chartTop + chartInnerH - (tick / 100) * chartInnerH;
                return (
                  <g key={`yt-${tick}`}>
                    <line x1={chartLeft} y1={y} x2={chartLeft + chartInnerW} y2={y} stroke="#e5e7eb" strokeWidth="1" />
                    <text x={20} y={y + 4} fontSize="12" fill="#6b7280">{tick}%</text>
                  </g>
                );
              })}
              <line x1={chartLeft} y1={chartTop} x2={chartLeft} y2={chartTop + chartInnerH} stroke="#9ca3af" strokeWidth="1" />
              <line x1={chartLeft} y1={chartTop + chartInnerH} x2={chartLeft + chartInnerW} y2={chartTop + chartInnerH} stroke="#9ca3af" strokeWidth="1" />
              <g transform={`translate(${chartLeft},${chartTop})`}>
                <path d={cpuPath} fill="none" stroke="#1d4ed8" strokeWidth="2.2" />
                <path d={memPath} fill="none" stroke="#16a34a" strokeWidth="2.2" />
                <path d={diskPath} fill="none" stroke="#f59e0b" strokeWidth="2.2" />
              </g>
              {hoverX !== null && hoveredPoint && (
                <>
                  <line
                    x1={hoverX}
                    y1={chartTop}
                    x2={hoverX}
                    y2={chartTop + chartInnerH}
                    stroke="#9ca3af"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                  />
                  <text x={hoverX + 6} y={chartTop - 6} fontSize="12" fill="#374151">
                    {hoveredPoint.t}
                  </text>
                  <text x={chartLeft + chartInnerW + 8} y={chartTop + 8} fontSize="12" fill="#1d4ed8">
                    CPU {hoveredPoint.cpu.toFixed(1)}%
                  </text>
                  <text x={chartLeft + chartInnerW + 8} y={chartTop + 24} fontSize="12" fill="#16a34a">
                    MEM {hoveredPoint.memory.toFixed(1)}%
                  </text>
                  <text x={chartLeft + chartInnerW + 8} y={chartTop + 40} fontSize="12" fill="#f59e0b">
                    DISK {hoveredPoint.disk.toFixed(1)}%
                  </text>
                  <circle cx={hoverX} cy={valueToChartY(hoveredPoint.cpu)} r="4.5" fill="#1d4ed8" stroke="#ffffff" strokeWidth="1.5" />
                  <circle cx={hoverX} cy={valueToChartY(hoveredPoint.memory)} r="4.5" fill="#16a34a" stroke="#ffffff" strokeWidth="1.5" />
                  <circle cx={hoverX} cy={valueToChartY(hoveredPoint.disk)} r="4.5" fill="#f59e0b" stroke="#ffffff" strokeWidth="1.5" />
                </>
              )}
              {xTickIndexes.map((idx) => {
                const x = chartLeft + (chartInnerW * (history.length <= 1 ? 0 : idx / (history.length - 1)));
                const label = history[idx]?.t || '';
                return (
                  <g key={`xt-${idx}`}>
                    <line x1={x} y1={chartTop + chartInnerH} x2={x} y2={chartTop + chartInnerH + 4} stroke="#9ca3af" strokeWidth="1" />
                    <text x={x - 18} y={chartTop + chartInnerH + 20} fontSize="12" fill="#6b7280">{label}</text>
                  </g>
                );
              })}
              <rect
                x={chartLeft}
                y={chartTop}
                width={chartInnerW}
                height={chartInnerH}
                fill="transparent"
                style={{ cursor: 'crosshair' }}
                onMouseMove={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const x = e.clientX - rect.left;
                  const ratio = clamp(x / rect.width, 0, 1);
                  const idx = Math.round(ratio * Math.max(0, history.length - 1));
                  setHoveredIdx(idx);
                }}
                onMouseLeave={() => setHoveredIdx(null)}
              />
            </svg>
            <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#4b5563', padding: '4px 6px 0' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: '#1d4ed8', display: 'inline-block' }} />
                CPU使用率
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: '#16a34a', display: 'inline-block' }} />
                内存使用率
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: '#f59e0b', display: 'inline-block' }} />
                磁盘使用率
              </span>
            </div>
            </div>
          </div>

          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, overflow: 'hidden' }}>
            <button
              onClick={() => setJointPanelOpen((v) => !v)}
              style={{
                width: '100%',
                border: 'none',
                background: '#f8fafc',
                borderBottom: jointPanelOpen ? '1px solid #e5e7eb' : 'none',
                padding: '10px 12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                cursor: 'pointer',
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2, textAlign: 'left' }}>
                <span style={{ fontSize: 14, fontWeight: 700, color: '#374151' }}>机器人状态</span>
                <span style={{ fontSize: 11, color: '#6b7280', fontWeight: 400 }}>
                  主臂仅关节；从臂含关节与末端六维力/力矩
                </span>
              </div>
              <span style={{ fontSize: 12, color: '#6b7280', whiteSpace: 'nowrap' }}>{jointPanelOpen ? '收起' : '展开'}</span>
            </button>
            {jointPanelOpen && (
              <div style={{ padding: 12 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                  <button
                    onClick={() => setArmMode('master')}
                    style={{
                      flex: 1,
                      borderRadius: 999,
                      padding: '6px 10px',
                      border: '1px solid #d1d5db',
                      background: armMode === 'master' ? '#eef2ff' : '#ffffff',
                      color: armMode === 'master' ? '#4f46e5' : '#374151',
                      fontWeight: 600,
                      cursor: 'pointer',
                      fontSize: 12,
                    }}
                  >
                    主臂
                  </button>
                  <button
                    onClick={() => setArmMode('slave')}
                    style={{
                      flex: 1,
                      borderRadius: 999,
                      padding: '6px 10px',
                      border: '1px solid #d1d5db',
                      background: armMode === 'slave' ? '#f1f5f9' : '#ffffff',
                      color: armMode === 'slave' ? '#0f172a' : '#4b5563',
                      fontWeight: 600,
                      cursor: 'pointer',
                      fontSize: 12,
                    }}
                  >
                    从臂
                  </button>
                </div>
                {armMode === 'slave' ? (
                  <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
                    <button
                      onClick={() => setPreviewMode('joint')}
                      style={{
                        flex: 1,
                        borderRadius: 8,
                        padding: '8px 10px',
                        border: '1px solid #d1d5db',
                        background: previewMode === 'joint' ? '#dbeafe' : '#fff',
                        color: previewMode === 'joint' ? '#1d4ed8' : '#374151',
                        fontWeight: 700,
                        cursor: 'pointer',
                        fontSize: 12,
                      }}
                    >
                      关节
                    </button>
                    <button
                      onClick={() => setPreviewMode('ft')}
                      style={{
                        flex: 1,
                        borderRadius: 8,
                        padding: '8px 10px',
                        border: '1px solid #d1d5db',
                        background: previewMode === 'ft' ? '#fffbeb' : '#fff',
                        color: previewMode === 'ft' ? '#b45309' : '#374151',
                        fontWeight: 700,
                        cursor: 'pointer',
                        fontSize: 12,
                      }}
                    >
                      末端力/力矩
                    </button>
                  </div>
                ) : null}

                {armMode === 'master' || previewMode === 'joint' ? (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10, marginBottom: 12 }}>
                      <div>
                        <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>一级：话题名称</div>
                        <select
                          value={selectedJointTopic}
                          onChange={(e) => setSelectedJointTopic(e.target.value)}
                          style={{ width: '100%', fontSize: 12, border: '1px solid #d1d5db', borderRadius: 6, padding: '6px 8px' }}
                        >
                          {filteredJointTopicsByArm.length > 0
                            ? filteredJointTopicsByArm.map((topic) => (
                                <option key={topic} value={topic}>{topic}</option>
                              ))
                            : <option value="">暂无话题</option>}
                        </select>
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>二级：关节点</div>
                        <select
                          value={selectedJointName}
                          onChange={(e) => setSelectedJointName(e.target.value)}
                          style={{ width: '100%', fontSize: 12, border: '1px solid #d1d5db', borderRadius: 6, padding: '6px 8px' }}
                        >
                          {jointNameOptions.length > 0 ? jointNameOptions.map((name) => (
                            <option key={name} value={name}>{name}</option>
                          )) : <option value="">暂无关节</option>}
                        </select>
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>三级：指标</div>
                        <select
                          value={selectedJointMetric}
                          onChange={(e) => setSelectedJointMetric(e.target.value as JointMetricKey)}
                          style={{ width: '100%', fontSize: 12, border: '1px solid #d1d5db', borderRadius: 6, padding: '6px 8px' }}
                        >
                          <option value="position">Position</option>
                          <option value="velocity">Velocity</option>
                          <option value="effort">Effort</option>
                        </select>
                      </div>
                    </div>
                    {selectedJointMetric === 'effort' && selectedJointTopic && !jointPayloadHasEffort(jointStateMap[selectedJointTopic]) ? (
                      <div
                        style={{
                          fontSize: 12,
                          color: '#92400e',
                          background: '#fffbeb',
                          border: '1px solid #fcd34d',
                          borderRadius: 8,
                          padding: '8px 10px',
                          marginBottom: 10,
                          lineHeight: 1.65,
                        }}
                      >
                        当前话题在心跳中<b>未携带 effort/力矩</b>（或为空）。双臂场景常见只有一侧发布力矩；你可用{' '}
                        <span style={{ fontFamily: 'ui-monospace, monospace' }}>ros2 topic echo /left/joint_states</span>{' '}
                        对比 <span style={{ fontFamily: 'ui-monospace, monospace' }}>/right/joint_states</span>。
                        若左臂确无 <span style={{ fontFamily: 'ui-monospace, monospace' }}>effort</span>，请在下拉框切换到带力矩的话题查看。
                      </div>
                    ) : null}
                    {jointStates.length === 0 || !selectedJointName ? (
                      <div style={{ fontSize: 13, color: '#6b7280' }}>
                        {filteredJointTopicsByArm.length > 0
                          ? '当前话题暂无关节实时数据，请切换其它 joint 话题或检查该话题消息结构。'
                          : '暂无关节实时数据（等待采集端扫描并上报 joint 话题）。'}
                      </div>
                    ) : (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                          <div style={{ fontSize: 13, color: '#374151' }}>
                            当前选择：<b>{selectedJointTopic || '-'}</b> / <b>{selectedJointName}</b> / <b>{jointMetricLabel}</b>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <div style={{ fontSize: 13, color: jointMetricColor, fontWeight: 700 }}>
                                当前值：{selectedJointCurrent === null ? '--' : selectedJointCurrent.toFixed(4)}
                              </div>
                          </div>
                        </div>
                        <svg width="100%" viewBox="0 0 860 290" preserveAspectRatio="xMidYMid meet" style={{ height: 280 }}>
                          <rect x="0" y="0" width="860" height="290" fill="#f8fbff" />
                          <line x1={64} y1={24} x2={64} y2={244} stroke="#9ca3af" strokeWidth="1" />
                          <line x1={64} y1={244} x2={824} y2={244} stroke="#9ca3af" strokeWidth="1" />
                          <text x={12} y={10} fontSize="12" fill="#6b7280">{jointMetricLabel}</text>
                          <text x={14} y={262} fontSize="10" fill="#6b7280">{jointYMin.toFixed(3)}</text>
                          <text x={14} y={48} fontSize="10" fill="#6b7280">{jointYMax.toFixed(3)}</text>
                          {jointXTickIndexes.map((idx) => {
                            const x = 64 + (760 * (jointChartValues.length <= 1 ? 0 : idx / (jointChartValues.length - 1)));
                            const label = jointChartTsValues[idx] || '';
                            return (
                              <g key={`jxt-${idx}`}>
                                <line x1={x} y1={244} x2={x} y2={248} stroke="#9ca3af" strokeWidth="1" />
                                <text x={x} y={268} fontSize="10" fill="#6b7280" textAnchor="middle">{label}</text>
                              </g>
                            );
                          })}
                          <g transform="translate(64,24)">
                            <path d={jointLinePath} fill="none" stroke={jointMetricColor} strokeWidth="2.2" />
                          </g>
                        </svg>
                        <div style={{ fontSize: 12, color: '#6b7280' }}>
                          展示最近 {jointChartValues.length} 个采样点（无数据时图表为空）。
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>末端 wrench 数据源（ROS 话题）</div>
                      <select
                        value={selectedFtTopic}
                        onChange={(e) => setSelectedFtTopic(e.target.value)}
                        style={{ width: '100%', fontSize: 12, border: '1px solid #d1d5db', borderRadius: 6, padding: '8px 10px', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
                      >
                        {filteredFtTopicsByArm.length > 0
                          ? filteredFtTopicsByArm.map((topic) => (
                              <option key={topic} value={topic}>{topic}</option>
                            ))
                          : <option value="">暂无话题</option>}
                      </select>
                      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginTop: 8 }}>
                        <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                          六维向量：<b style={{ color: '#334155' }}>[Fx Fy Fz Mx My Mz]</b>
                          <span style={{ marginLeft: 8 }}>单位与话题定义一致（常见为 N / N·m）</span>
                          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                            RealMan 等：`force_fx~fz` 为力，`force_mx~mz` 为力矩，采集端已映射为上述六维。
                          </div>
                        </div>
                        <span style={{ background: '#f8fafc', color: '#334155', borderRadius: 999, padding: '3px 10px', fontSize: 11, fontWeight: 600 }}>
                          整体{!ftHasWrenchData ? '：无数据' : `：max≈${formatScalar(ftMaxAbs, 3)}`}
                        </span>
                      </div>
                    </div>

                    {!ftHasWrenchData ? (
                      <div style={{ fontSize: 13, color: '#6b7280', lineHeight: 1.6 }}>
                        {filteredFtTopicsByArm.length > 0
                          ? '当前话题未解析到 wrench（force/torque 的 x,y,z）数据，请切换其它 ft 话题或在采集端确认 `ros2 topic echo` 输出格式。'
                          : '暂无末端 wrench 话题（等待采集端扫描并上报含 wrench / ft / force_torque 的话题）。'}
                      </div>
                    ) : (
                      <div>
                        <div style={{ marginBottom: 10 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: '#1e3a8a', marginBottom: 6 }}>力分量</div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
                            {FT_FORCE_KEYS.map((k) => {
                              const v = ftAllCompValues[k];
                              const histV = ftHistoryByComp[k]?.v || [];
                              const spark = histV.length >= 2 ? histV : v !== null && Number.isFinite(v) ? [v, v] : [];
                              const sp = spark.length >= 2 ? miniSparkPath(spark, 116, 28) : '';
                              const sel = selectedFtDetail === k;
                              return (
                                <button
                                  key={k}
                                  type="button"
                                  onClick={() => setSelectedFtDetail(sel ? null : k)}
                                  style={{
                                    textAlign: 'left',
                                    cursor: 'pointer',
                                    borderRadius: 10,
                                    padding: '10px 12px',
                                    border: sel ? `2px solid ${ftCardColors[k]}` : '1px solid #e5e7eb',
                                    background: sel ? '#eff6ff' : '#fff',
                                    boxShadow: sel ? '0 1px 6px rgba(37,99,235,0.12)' : 'none',
                                  }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                                    <span style={{ fontSize: 13, fontWeight: 700, color: '#1e293b' }}>{k}</span>
                                    <span style={{ width: 8, height: 8, borderRadius: 999, background: ftCardColors[k] }} />
                                  </div>
                                  <div style={{ fontSize: 20, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em' }}>
                                    {formatScalar(v)}
                                  </div>
                                  <svg width="100%" viewBox="0 0 120 32" preserveAspectRatio="none" style={{ display: 'block', marginTop: 8, height: 32 }}>
                                    <rect x="0" y="0" width="120" height="32" fill={sel ? '#dbeafe' : '#f8fafc'} rx="4" />
                                    {sp ? <path d={sp} transform="translate(2,2)" fill="none" stroke={ftCardColors[k]} strokeWidth="1.6" /> : (
                                      <text x="60" y="20" fontSize="10" fill="#94a3b8" textAnchor="middle">趋势积累中</text>
                                    )}
                                  </svg>
                                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>点击展开该分量趋势</div>
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        <div style={{ marginBottom: 4 }}>
                          <div style={{ fontSize: 13, fontWeight: 700, color: '#9a3412', marginBottom: 6 }}>力矩分量</div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
                            {FT_TORQUE_KEYS.map((k) => {
                              const v = ftAllCompValues[k];
                              const histV = ftHistoryByComp[k]?.v || [];
                              const spark = histV.length >= 2 ? histV : v !== null && Number.isFinite(v) ? [v, v] : [];
                              const sp = spark.length >= 2 ? miniSparkPath(spark, 116, 28) : '';
                              const sel = selectedFtDetail === k;
                              return (
                                <button
                                  key={k}
                                  type="button"
                                  onClick={() => setSelectedFtDetail(sel ? null : k)}
                                  style={{
                                    textAlign: 'left',
                                    cursor: 'pointer',
                                    borderRadius: 10,
                                    padding: '10px 12px',
                                    border: sel ? `2px solid ${ftCardColors[k]}` : '1px solid #e5e7eb',
                                    background: sel ? '#fff7ed' : '#fff',
                                    boxShadow: sel ? '0 1px 6px rgba(194,65,12,0.12)' : 'none',
                                  }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                                    <span style={{ fontSize: 13, fontWeight: 700, color: '#1e293b' }}>{k}</span>
                                    <span style={{ width: 8, height: 8, borderRadius: 999, background: ftCardColors[k] }} />
                                  </div>
                                  <div style={{ fontSize: 20, fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em' }}>
                                    {formatScalar(v)}
                                  </div>
                                  <svg width="100%" viewBox="0 0 120 32" preserveAspectRatio="none" style={{ display: 'block', marginTop: 8, height: 32 }}>
                                    <rect x="0" y="0" width="120" height="32" fill={sel ? '#ffedd5' : '#f8fafc'} rx="4" />
                                    {sp ? <path d={sp} transform="translate(2,2)" fill="none" stroke={ftCardColors[k]} strokeWidth="1.6" /> : (
                                      <text x="60" y="20" fontSize="10" fill="#94a3b8" textAnchor="middle">趋势积累中</text>
                                    )}
                                  </svg>
                                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>点击展开该分量趋势</div>
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        {selectedFtDetail && ftDetailChartValues.length > 0 ? (
                          <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #e5e7eb' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                              <div style={{ fontSize: 13, color: '#374151' }}>
                                选中分量：<b style={{ color: ftDetailColor }}>{ftDetailLabel}</b>
                              </div>
                              <div style={{ fontSize: 13, color: ftDetailColor, fontWeight: 700 }}>
                                当前：{formatScalar(ftDetailCurrent)}
                              </div>
                            </div>
                            <svg width="100%" viewBox="0 0 860 290" preserveAspectRatio="xMidYMid meet" style={{ height: 260 }}>
                              <rect x="0" y="0" width="860" height="290" fill="#f8fafc" />
                              <line x1={64} y1={24} x2={64} y2={244} stroke="#9ca3af" strokeWidth="1" />
                              <line x1={64} y1={244} x2={824} y2={244} stroke="#9ca3af" strokeWidth="1" />
                              <text x={12} y={10} fontSize="12" fill="#6b7280">{ftDetailLabel}</text>
                              <text x={14} y={262} fontSize="10" fill="#6b7280">{ftDetailYMin.toFixed(4)}</text>
                              <text x={14} y={48} fontSize="10" fill="#6b7280">{ftDetailYMax.toFixed(4)}</text>
                              {(() => {
                                if (ftDetailChartValues.length <= 1) return null;
                                const idxs = Array.from(new Set([
                                  0,
                                  Math.floor((ftDetailChartValues.length - 1) / 3),
                                  Math.floor((ftDetailChartValues.length - 1) * 2 / 3),
                                  ftDetailChartValues.length - 1,
                                ]));
                                return idxs.map((idx) => {
                                  const x = 64 + (760 * (idx / (ftDetailChartValues.length - 1)));
                                  const label = ftDetailChartTsValues[idx] || '';
                                  return (
                                    <g key={`fxt-${idx}`}>
                                      <line x1={x} y1={244} x2={x} y2={248} stroke="#9ca3af" strokeWidth="1" />
                                      <text x={x} y={268} fontSize="10" fill="#6b7280" textAnchor="middle">{label}</text>
                                    </g>
                                  );
                                });
                              })()}
                              <g transform="translate(64,24)">
                                <path d={ftDetailLinePath} fill="none" stroke={ftDetailColor} strokeWidth="2.2" />
                              </g>
                            </svg>
                            <div style={{ fontSize: 12, color: '#6b7280' }}>
                              最近 {ftDetailChartValues.length} 个采样点 · 再点此卡片可收起
                            </div>
                          </div>
                        ) : null}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#374151', marginBottom: 10 }}>最近日志事件</div>
              <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
                <div>- runtimeStatus: {device.runtimeStatus || '-'}</div>
                <div>- 最后测试: {device.lastTestResult?.status || 'untested'}</div>
                <div>- 错误信息: {device.lastTestResult?.errorMessage || '无'}</div>
                <div>- 更新时间: {new Date(device.updatedAt).toLocaleString()}</div>
              </div>
            </div>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 12 }}>
              <AlertPanel alerts={alerts} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
