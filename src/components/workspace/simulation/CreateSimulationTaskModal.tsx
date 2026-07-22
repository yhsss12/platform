'use client';

import { useEffect, useState } from 'react';
import { PrimaryButton, SecondaryButton } from '@/components/workspace/workspaceUi';
import {
  WorkspaceCenteredModal,
  workspaceModalFieldLabel,
  workspaceModalSectionLabel,
  workspaceModalSelectStyle,
} from '@/components/workspace/WorkspaceCenteredModal';
import {
  evaluationTemplateOptions,
  onlinePolicyOptions,
  onlineRobotOptions,
  onlineSceneOptions,
} from '@/lib/mock/workspaceEvaluationRecordsMock';

export interface CreateSimulationTaskPayload {
  template: string;
  scene: string;
  robot: string;
  policy: string;
  rounds: number;
  seed: number;
  generateData: boolean;
  saveVideo: boolean;
  autoEvaluate: boolean;
}

export function CreateSimulationTaskModal({
  open,
  onClose,
  onSave,
  onStart,
}: {
  open: boolean;
  onClose: () => void;
  onSave: (payload: CreateSimulationTaskPayload) => void;
  onStart: (payload: CreateSimulationTaskPayload) => void;
}) {
  const [template, setTemplate] = useState<string>(evaluationTemplateOptions[0]);
  const [scene, setScene] = useState<string>(onlineSceneOptions[0]);
  const [robot, setRobot] = useState<string>(onlineRobotOptions[0]);
  const [policy, setPolicy] = useState<string>(onlinePolicyOptions[0]);
  const [rounds, setRounds] = useState(10);
  const [seed, setSeed] = useState(42);
  const [generateData, setGenerateData] = useState(true);
  const [saveVideo, setSaveVideo] = useState(true);
  const [autoEvaluate, setAutoEvaluate] = useState(false);

  useEffect(() => {
    if (!open) return;
    setTemplate(evaluationTemplateOptions[0]);
    setScene(onlineSceneOptions[0]);
    setRobot(onlineRobotOptions[0]);
    setPolicy(onlinePolicyOptions[0]);
    setRounds(10);
    setSeed(42);
    setGenerateData(true);
    setSaveVideo(true);
    setAutoEvaluate(false);
  }, [open]);

  const payload = (): CreateSimulationTaskPayload => ({
    template,
    scene,
    robot,
    policy,
    rounds,
    seed,
    generateData,
    saveVideo,
    autoEvaluate,
  });

  return (
    <WorkspaceCenteredModal
      open={open}
      title="新建运行任务"
      titleId="create-sim-task-title"
      width={800}
      onClose={onClose}
      footer={
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 8,
          }}
        >
          <SecondaryButton onClick={onClose}>取消</SecondaryButton>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <SecondaryButton onClick={() => onSave(payload())}>保存任务</SecondaryButton>
            <PrimaryButton onClick={() => onStart(payload())}>启动任务</PrimaryButton>
          </div>
        </div>
      }
    >
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#6b7280', lineHeight: 1.55 }}>
        基于资源中心的任务模板创建运行任务。此处不会新建或修改任务模板。
      </p>

      <div style={workspaceModalSectionLabel}>任务配置</div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
          gap: '0 16px',
        }}
      >
        <div>
          <label style={workspaceModalFieldLabel}>任务模板</label>
          <select
            style={workspaceModalSelectStyle}
            value={template}
            onChange={(e) => setTemplate(e.target.value)}
          >
            {evaluationTemplateOptions.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>场景</label>
          <select style={workspaceModalSelectStyle} value={scene} onChange={(e) => setScene(e.target.value)}>
            {onlineSceneOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>机器人</label>
          <select style={workspaceModalSelectStyle} value={robot} onChange={(e) => setRobot(e.target.value)}>
            {onlineRobotOptions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>策略模型</label>
          <select style={workspaceModalSelectStyle} value={policy} onChange={(e) => setPolicy(e.target.value)}>
            {onlinePolicyOptions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>运行轮次</label>
          <input
            type="number"
            min={1}
            value={rounds}
            onChange={(e) => setRounds(Number(e.target.value) || 1)}
            style={workspaceModalSelectStyle}
          />
        </div>
        <div>
          <label style={workspaceModalFieldLabel}>seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={workspaceModalSelectStyle}
          />
        </div>
      </div>

      <div style={{ ...workspaceModalSectionLabel, marginTop: 8 }}>运行选项</div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 8 }}>
        <input type="checkbox" checked={generateData} onChange={(e) => setGenerateData(e.target.checked)} />
        是否生成数据
      </label>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 8 }}>
        <input type="checkbox" checked={saveVideo} onChange={(e) => setSaveVideo(e.target.checked)} />
        是否保存视频
      </label>
      <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
        <input type="checkbox" checked={autoEvaluate} onChange={(e) => setAutoEvaluate(e.target.checked)} />
        是否完成后自动评测
      </label>
    </WorkspaceCenteredModal>
  );
}
