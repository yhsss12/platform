import type { WorkspaceDataItem } from '@/lib/mock/workspaceDataMock';
import type { DualArmIlExportProbeResponse } from '@/lib/api/dualArmCableClient';
import { DUAL_ARM_IL_BUILD_DISABLED_HINT } from '@/lib/workspace/datasetTrainingAccess';

export function applyDualArmIlProbeToItem(
  item: WorkspaceDataItem,
  probe: DualArmIlExportProbeResponse
): WorkspaceDataItem {
  const hdf5Built = probe.hdf5Exists && probe.manifestExists;
  return {
    ...item,
    ilExportReady: probe.exportReady,
    ilExportProbed: true,
    ilExportFailureReason: probe.failureReason ?? undefined,
    trainable: probe.trainable,
    datasetBuildSupported: probe.exportReady,
    datasetBuildStatus: hdf5Built ? 'built' : item.datasetBuildStatus,
    hdf5Path: probe.hdf5Exists ? probe.hdf5Path ?? item.hdf5Path : item.hdf5Path,
    datasetManifestPath: probe.manifestExists
      ? probe.manifestPath ?? item.datasetManifestPath
      : item.datasetManifestPath,
    qualityStatus: hdf5Built ? '可训练（torch_bc）' : probe.exportReady ? '可构建' : '不可构建',
    trainingBackendPending: false,
  };
}

export function dualArmIlBuildDisabledReason(item: WorkspaceDataItem): string {
  if (item.ilExportFailureReason?.includes('step-level actions')) {
    return DUAL_ARM_IL_BUILD_DISABLED_HINT;
  }
  if (item.ilExportFailureReason) return item.ilExportFailureReason;
  return DUAL_ARM_IL_BUILD_DISABLED_HINT;
}

export function canBuildDualArmIlDataset(item: WorkspaceDataItem): boolean {
  return item.status === 'completed' && item.ilExportReady === true && !item.trainable;
}

export function shouldShowDualArmIlBuildAction(item: WorkspaceDataItem): boolean {
  return item.status === 'completed' && item.ilExportProbed === true && !item.trainable;
}
