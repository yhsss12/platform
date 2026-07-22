'use client';

import type { CreateTaskFormState } from '@/lib/mock/workspaceTaskCreateOptions';
import {
  CREATE_TASK_STEPS,
  createTaskControlModeOptions,
  createTaskDataTypeOptions,
  createTaskDomainOptions,
  createTaskExportFormatOptions,
  createTaskMetricOptions,
  createTaskObjectOptions,
  createTaskPolicyOptions,
  createTaskRobotOptions,
  createTaskSceneOptions,
  createTaskTypeOptions,
} from '@/lib/mock/workspaceTaskCreateOptions';

const cardStyle: React.CSSProperties = {
  backgroundColor: '#ffffff',
  borderRadius: 12,
  border: '1px solid #e5e7eb',
  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
  padding: '24px',
  marginBottom: 16,
};

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 13,
  fontWeight: 500,
  color: '#374151',
  marginBottom: 6,
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  fontSize: 14,
  color: '#111827',
  border: '1px solid #d1d5db',
  borderRadius: 6,
  outline: 'none',
  boxSizing: 'border-box',
};

const hintStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#9ca3af',
  margin: '0 0 16px',
};

function FieldGroup({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 18 }}>
      <label style={labelStyle}>
        {label}
        {required ? <span style={{ color: '#dc2626', marginLeft: 2 }}>*</span> : null}
      </label>
      {children}
    </div>
  );
}

function CheckboxGrid({
  options,
  selected,
  onChange,
}: {
  options: readonly string[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  const toggle = (opt: string) => {
    if (selected.includes(opt)) onChange(selected.filter((x) => x !== opt));
    else onChange([...selected, opt]);
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
        gap: 8,
      }}
    >
      {options.map((opt) => {
        const checked = selected.includes(opt);
        return (
          <label
            key={opt}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 10px',
              fontSize: 13,
              borderRadius: 6,
              border: checked ? '1px solid #93c5fd' : '1px solid #e5e7eb',
              backgroundColor: checked ? '#eff6ff' : '#fff',
              cursor: 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={() => toggle(opt)}
              style={{ width: 14, height: 14 }}
            />
            {opt}
          </label>
        );
      })}
    </div>
  );
}

function StepSection({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div style={cardStyle}>
      <h3 style={{ margin: '0 0 4px', fontSize: 16, fontWeight: 600, color: '#111827' }}>
        {title}
      </h3>
      <p style={hintStyle}>{hint}</p>
      {children}
    </div>
  );
}

interface CreateTaskWizardFormProps {
  step: number;
  form: CreateTaskFormState;
  onChange: (patch: Partial<CreateTaskFormState>) => void;
}

export function CreateTaskWizardForm({ step, form, onChange }: CreateTaskWizardFormProps) {
  if (step === 1) {
    const meta = CREATE_TASK_STEPS[0];
    return (
      <StepSection title={meta.title} hint={meta.hint}>
        <FieldGroup label="任务名称" required>
          <input
            type="text"
            value={form.name}
            onChange={(e) => onChange({ name: e.target.value })}
            placeholder="请输入任务名称，例如：线缆穿杆"
            style={inputStyle}
          />
        </FieldGroup>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <FieldGroup label="任务领域" required>
            <select
              value={form.domain}
              onChange={(e) => onChange({ domain: e.target.value })}
              style={inputStyle}
            >
              {createTaskDomainOptions.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </FieldGroup>
          <FieldGroup label="任务类型" required>
            <select
              value={form.type}
              onChange={(e) => onChange({ type: e.target.value })}
              style={inputStyle}
            >
              {createTaskTypeOptions.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </FieldGroup>
        </div>
        <FieldGroup label="任务描述">
          <textarea
            value={form.description}
            onChange={(e) => onChange({ description: e.target.value })}
            rows={3}
            style={{ ...inputStyle, resize: 'vertical', minHeight: 72 }}
          />
        </FieldGroup>
        <FieldGroup label="任务目标">
          <textarea
            value={form.goal}
            onChange={(e) => onChange({ goal: e.target.value })}
            rows={3}
            style={{ ...inputStyle, resize: 'vertical', minHeight: 72 }}
          />
        </FieldGroup>
      </StepSection>
    );
  }

  if (step === 2) {
    const meta = CREATE_TASK_STEPS[1];
    return (
      <StepSection title={meta.title} hint={meta.hint}>
        <FieldGroup label="场景" required>
          <select
            value={form.scene}
            onChange={(e) => onChange({ scene: e.target.value })}
            style={inputStyle}
          >
            {createTaskSceneOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </FieldGroup>
        <FieldGroup label="操作对象" required>
          <CheckboxGrid
            options={createTaskObjectOptions}
            selected={form.objects}
            onChange={(objects) => onChange({ objects })}
          />
        </FieldGroup>
        <FieldGroup label="初始状态">
          <textarea
            value={form.initialState}
            onChange={(e) => onChange({ initialState: e.target.value })}
            rows={3}
            style={{ ...inputStyle, resize: 'vertical', minHeight: 72 }}
          />
        </FieldGroup>
        <FieldGroup label="环境扰动（可选）">
          <input
            type="text"
            value={form.environmentDisturbance}
            onChange={(e) => onChange({ environmentDisturbance: e.target.value })}
            placeholder="位置扰动、光照变化、对象姿态扰动"
            style={inputStyle}
          />
        </FieldGroup>
      </StepSection>
    );
  }

  if (step === 3) {
    const meta = CREATE_TASK_STEPS[2];
    return (
      <StepSection title={meta.title} hint={meta.hint}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <FieldGroup label="机器人" required>
            <select
              value={form.robot}
              onChange={(e) => onChange({ robot: e.target.value })}
              style={inputStyle}
            >
              {createTaskRobotOptions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </FieldGroup>
          <FieldGroup label="策略模型" required>
            <select
              value={form.policy}
              onChange={(e) => onChange({ policy: e.target.value })}
              style={inputStyle}
            >
              {createTaskPolicyOptions.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </FieldGroup>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <FieldGroup label="控制方式">
            <select
              value={form.controlMode}
              onChange={(e) => onChange({ controlMode: e.target.value })}
              style={inputStyle}
            >
              {createTaskControlModeOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </FieldGroup>
          <FieldGroup label="运行轮次">
            <input
              type="number"
              min={1}
              max={999}
              value={form.runRounds}
              onChange={(e) =>
                onChange({ runRounds: Math.max(1, Number(e.target.value) || 1) })
              }
              style={inputStyle}
            />
          </FieldGroup>
        </div>
      </StepSection>
    );
  }

  if (step === 4) {
    const meta = CREATE_TASK_STEPS[3];
    return (
      <StepSection title={meta.title} hint={meta.hint}>
        <FieldGroup label="评测指标">
          <CheckboxGrid
            options={createTaskMetricOptions}
            selected={form.metrics}
            onChange={(metrics) => onChange({ metrics })}
          />
        </FieldGroup>
        <FieldGroup label="成功条件">
          <textarea
            value={form.successCondition}
            onChange={(e) => onChange({ successCondition: e.target.value })}
            rows={3}
            style={{ ...inputStyle, resize: 'vertical', minHeight: 72 }}
          />
        </FieldGroup>
        <FieldGroup label="是否生成数据">
          <label
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              fontSize: 14,
              color: '#374151',
              cursor: 'pointer',
            }}
          >
            <input
              type="checkbox"
              checked={form.generateData}
              onChange={(e) => onChange({ generateData: e.target.checked })}
              style={{ width: 16, height: 16 }}
            />
            仿真运行后自动生成训练与评测数据
          </label>
        </FieldGroup>
        {form.generateData ? (
          <>
            <FieldGroup label="数据类型">
              <CheckboxGrid
                options={createTaskDataTypeOptions}
                selected={form.dataTypes}
                onChange={(dataTypes) => onChange({ dataTypes })}
              />
            </FieldGroup>
            <FieldGroup label="导出格式">
              <select
                value={form.exportFormat}
                onChange={(e) => onChange({ exportFormat: e.target.value })}
                style={inputStyle}
              >
                {createTaskExportFormatOptions.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </FieldGroup>
          </>
        ) : null}
      </StepSection>
    );
  }

  const meta = CREATE_TASK_STEPS[4];
  const rows: { label: string; value: string }[] = [
    { label: '任务名称', value: form.name || '（未填写）' },
    { label: '任务领域', value: form.domain },
    { label: '任务类型', value: form.type },
    { label: '任务描述', value: form.description },
    { label: '任务目标', value: form.goal },
    { label: '场景', value: form.scene },
    { label: '操作对象', value: form.objects.join('、') || '—' },
    { label: '初始状态', value: form.initialState },
    {
      label: '环境扰动',
      value: form.environmentDisturbance || '无',
    },
    { label: '机器人', value: form.robot },
    { label: '策略模型', value: form.policy },
    { label: '控制方式', value: form.controlMode },
    { label: '运行轮次', value: String(form.runRounds) },
    { label: '评测指标', value: form.metrics.join('、') || '—' },
    { label: '成功条件', value: form.successCondition },
    {
      label: '数据生成',
      value: form.generateData
        ? `是 · ${form.dataTypes.join('、')} · ${form.exportFormat}`
        : '否',
    },
  ];

  return (
    <StepSection title={meta.title} hint={meta.hint}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: '12px 24px',
        }}
      >
        {rows.map((row) => (
          <div key={row.label}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{row.label}</div>
            <div style={{ fontSize: 14, color: '#111827', lineHeight: 1.5 }}>{row.value}</div>
          </div>
        ))}
      </div>
      <p style={{ marginTop: 20, marginBottom: 0, fontSize: 13, color: '#6b7280' }}>
        确认无误后，请点击页面右上角或底部的「保存任务」或「保存并启动仿真」。
      </p>
    </StepSection>
  );
}

export function CreateTaskStepper({
  step,
  onStepChange,
}: {
  step: number;
  onStepChange: (n: number) => void;
}) {
  return (
    <nav
      aria-label="新建任务步骤"
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 8,
        marginBottom: 20,
        padding: '12px 16px',
        backgroundColor: '#fff',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
      }}
    >
      {CREATE_TASK_STEPS.map((s) => {
        const active = step === s.id;
        const done = step > s.id;
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => onStepChange(s.id)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 14px',
              fontSize: 13,
              fontWeight: active ? 600 : 500,
              color: active ? '#1d4ed8' : done ? '#059669' : '#6b7280',
              backgroundColor: active ? '#eff6ff' : done ? '#f0fdf4' : 'transparent',
              border: active
                ? '1px solid #bfdbfe'
                : done
                  ? '1px solid #bbf7d0'
                  : '1px solid transparent',
              borderRadius: 8,
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                width: 22,
                height: 22,
                borderRadius: '50%',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 12,
                fontWeight: 600,
                backgroundColor: active ? '#2563eb' : done ? '#10b981' : '#e5e7eb',
                color: active || done ? '#fff' : '#6b7280',
              }}
            >
              {done ? '✓' : s.id}
            </span>
            {s.title}
          </button>
        );
      })}
    </nav>
  );
}
