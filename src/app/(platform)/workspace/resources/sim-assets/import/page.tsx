'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import {
  ModulePageContainer,
  ModulePageHeader,
} from '@/components/layout/ModulePageLayout';
import { ModelAssetFileUploadZone } from '@/components/workspace/resources/ModelAssetFileUploadZone';
import {
  FormField,
  SimAssetBackLink,
  SimAssetToast,
  formControlStyle,
} from '@/components/workspace/sim-assets/simAssetUi';
import { PrimaryButton, SecondaryButton, SectionCard, WS } from '@/components/workspace/workspaceUi';
import type { SimAssetTargetEngine, SimAssetType } from '@/types/simAsset';

const ACCEPT_EXTENSIONS =
  '.xml,.obj,.stl,.glb,.usd,.usda,.usdc,.ply';

export default function SimAssetImportPage() {
  const router = useRouter();
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [assetType, setAssetType] = useState<SimAssetType>('object');
  const [targetEngine, setTargetEngine] = useState<SimAssetTargetEngine>('mujoco');
  const [description, setDescription] = useState('');
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    if (!toastMsg) return;
    const timer = window.setTimeout(() => setToastMsg(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toastMsg]);

  const handleImport = useCallback(() => {
    setToastMsg('导入功能将在下一阶段接入后端。');
  }, []);

  return (
    <ModulePageContainer>
      <SimAssetBackLink href="/workspace/resources/sim-assets" label="返回仿真资产" />

      <ModulePageHeader
        title="导入仿真资产"
        subtitle="上传已有 MuJoCo / Isaac Sim / 通用 3D 文件，并注册到仿真资产库中。"
      />

      <SectionCard style={{ maxWidth: 720 }}>
        <FormField label="资产名称">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="例如：装配工位场景 v1"
            style={formControlStyle}
          />
        </FormField>

        <FormField label="资产类型">
          <select
            value={assetType}
            onChange={(e) => setAssetType(e.target.value as SimAssetType)}
            style={formControlStyle}
          >
            <option value="scene">场景资产</option>
            <option value="object">操作对象</option>
            <option value="robot">机器人资产</option>
            <option value="fixture">工装/夹具</option>
          </select>
        </FormField>

        <FormField label="目标平台">
          <select
            value={targetEngine}
            onChange={(e) => setTargetEngine(e.target.value as SimAssetTargetEngine)}
            style={formControlStyle}
          >
            <option value="mujoco">MuJoCo</option>
            <option value="isaac">Isaac Sim</option>
            <option value="generic">通用</option>
          </select>
        </FormField>

        <FormField
          label="文件上传"
          hint="支持 .xml, .obj, .stl, .glb, .usd, .usda, .usdc, .ply"
        >
          <ModelAssetFileUploadZone
            accept={ACCEPT_EXTENSIONS}
            emptyTitle="点击或拖拽上传仿真资产文件"
            emptySubtitle="支持 MuJoCo XML、USD、GLB、OBJ、STL、PLY 等格式"
            file={file}
            onFileChange={setFile}
            onInvalidFile={(msg) => setToastMsg(msg)}
          />
        </FormField>

        <FormField label="描述">
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            placeholder="可选：用途、尺寸、碰撞体说明等"
            style={{ ...formControlStyle, resize: 'vertical', minHeight: 96 }}
          />
        </FormField>

        <div style={{ display: 'flex', gap: 12, marginTop: WS.gap }}>
          <SecondaryButton onClick={() => router.push('/workspace/resources/sim-assets')}>
            取消
          </SecondaryButton>
          <PrimaryButton onClick={handleImport}>导入资产</PrimaryButton>
        </div>
      </SectionCard>

      <SimAssetToast message={toastMsg} />
    </ModulePageContainer>
  );
}
