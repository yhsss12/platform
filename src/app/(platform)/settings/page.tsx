'use client';

import {
  ModulePageContainer,
  ModulePageFilterCard,
  ModulePageHeader,
  ModulePageTableCard,
} from '@/components/layout/ModulePageLayout';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';

const MOCK_SETTINGS = [
  { key: 'sim.max_concurrent_jobs', label: '最大并发仿真实验数', value: '10', category: '仿真' },
  { key: 'sim.default_physics_hz', label: '默认物理仿真频率', value: '240 Hz', category: '仿真' },
  { key: 'eval.default_metric_set', label: '默认评测指标集', value: '装配成功率指标集', category: '评测' },
  { key: 'eval.benchmark_rounds', label: 'Benchmark 默认轮次', value: '20', category: '评测' },
  { key: 'resource.scene_quota', label: '场景库配额', value: '30', category: '资源' },
  { key: 'notify.on_sim_failed', label: '仿真失败通知', value: '开启', category: '通知' },
];

export default function SettingsPage() {
  return (
    <ModulePageContainer>
      <ModulePageHeader
        title="系统配置"
        subtitle="管理平台级运行与评测参数"
        actions={
          <>
            <SecondaryButton onClick={() => window.alert('已恢复默认配置')}>重置</SecondaryButton>
            <PrimaryButton onClick={() => window.alert('设置已保存到当前会话')}>保存</PrimaryButton>
          </>
        }
      />
      <ModulePageFilterCard>
        <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
          以下配置影响全平台运行调度与 Benchmark 行为。
        </p>
      </ModulePageFilterCard>
      <ModulePageTableCard>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <thead>
            <tr style={{ backgroundColor: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
              {['分类', '配置项', '键', '当前值'].map((h) => (
                <th key={h} style={{ textAlign: 'left', padding: '12px 16px', fontWeight: 600, color: '#374151' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOCK_SETTINGS.map((row) => (
              <tr key={row.key} style={{ borderBottom: '1px solid #f3f4f6' }}>
                <td style={{ padding: '12px 16px', color: '#6b7280' }}>{row.category}</td>
                <td style={{ padding: '12px 16px', color: '#111827' }}>{row.label}</td>
                <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: '#6b7280', fontSize: 13 }}>
                  {row.key}
                </td>
                <td style={{ padding: '12px 16px', color: '#374151' }}>{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ModulePageTableCard>
    </ModulePageContainer>
  );
}
