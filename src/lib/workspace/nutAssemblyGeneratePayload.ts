import type { GenerateDataPayload } from '@/lib/workspace/generateDataPayloadTypes';
import type {
  NutAssemblyGenerateRequest,
  NutAssemblyGenerationMode,
  NutAssemblySourceDemoSelection,
} from '@/lib/api/nutAssemblyClient';
import { NUT_ASSEMBLY_DEFAULTS } from '@/lib/workspace/nutAssembly';
import { buildNutAssemblyPhysicsEnhancementPayload } from '@/lib/workspace/nutAssemblyPhysicsEnhancement';
import type { GenerationPath } from '@/lib/workspace/generateDataTypes';
import { NUT_ASSEMBLY_PATH_DEFAULTS } from '@/lib/workspace/generateDataTaskParams';
import {
  isNutAssemblyBuiltInDefaultDemoDatasetId,
  resolveNutAssemblyEffectiveSourceDemoDatasetId,
} from '@/lib/workspace/nutAssemblySeedDatasets';

function resolveGenerationPath(payload: GenerateDataPayload): GenerationPath {
  return payload.generationPath ?? NUT_ASSEMBLY_PATH_DEFAULTS.generationPath;
}

function mapPathToBackendMode(path: GenerationPath): NutAssemblyGenerationMode {
  if (path === 'expert_policy') return 'robosuite_rollout';
  return 'mimicgen_datagen';
}

function resolveEpisodes(payload: GenerateDataPayload, path: GenerationPath): number {
  if (path === 'expert_policy') {
    return payload.generationCount ?? payload.episodes ?? NUT_ASSEMBLY_DEFAULTS.episodes;
  }
  return payload.targetCount ?? payload.episodes ?? NUT_ASSEMBLY_DEFAULTS.episodes;
}

function hasCustomSourceDemo(payload: GenerateDataPayload): boolean {
  const datasetId = payload.sourceDemoDatasetId?.trim();
  if (isNutAssemblyBuiltInDefaultDemoDatasetId(datasetId)) {
    return false;
  }
  return Boolean(datasetId || payload.nutAssemblySourceDemoPath?.trim());
}

function resolveSourceDemoSelection(
  payload: GenerateDataPayload,
  path: GenerationPath
): NutAssemblySourceDemoSelection | undefined {
  if (path === 'demo_augmentation') {
    const datasetId = resolveNutAssemblyEffectiveSourceDemoDatasetId(
      path,
      payload.sourceDemoDatasetId ?? ''
    );
    if (isNutAssemblyBuiltInDefaultDemoDatasetId(datasetId)) {
      return 'official';
    }
    return hasCustomSourceDemo(payload) ? 'custom' : 'official';
  }
  if (path === 'expert_seed_then_augmentation') {
    if (payload.useExistingSeedDataset && hasCustomSourceDemo(payload)) {
      return 'custom';
    }
    return 'official';
  }
  return undefined;
}

export function buildNutAssemblyGenerateRequest(
  payload: GenerateDataPayload,
  taskConfigId?: string | null
): NutAssemblyGenerateRequest {
  const generationPath = resolveGenerationPath(payload);
  const generationMode = mapPathToBackendMode(generationPath);
  const enablePinnRepair = Boolean(
    payload.enablePinnRepair ?? payload.nutAssemblyPinnRepairEnabled
  );
  const horizon =
    payload.maxSteps ?? payload.nutAssemblyHorizon ?? NUT_ASSEMBLY_DEFAULTS.horizon;
  const sourceDemoDatasetId =
    generationPath === 'demo_augmentation'
      ? resolveNutAssemblyEffectiveSourceDemoDatasetId(
          generationPath,
          payload.sourceDemoDatasetId ?? ''
        )
      : payload.sourceDemoDatasetId?.trim() || undefined;

  return {
    taskTemplateId: 'nut_assembly_single_arm',
    episodes: resolveEpisodes(payload, generationPath),
    seed: payload.seed ?? NUT_ASSEMBLY_DEFAULTS.seed,
    renderVideo: payload.nutAssemblyRenderVideo ?? payload.saveProcessVideo ?? NUT_ASSEMBLY_DEFAULTS.renderVideo,
    envName: payload.nutAssemblyEnvName ?? NUT_ASSEMBLY_DEFAULTS.envName,
    outputName: payload.outputName?.trim() || NUT_ASSEMBLY_DEFAULTS.outputName,
    horizon,
    generationMode,
    generationPath,
    generationCount: payload.generationCount,
    maxSteps: payload.maxSteps ?? horizon,
    sourceDemoSelection: resolveSourceDemoSelection(payload, generationPath),
    sourceDemoPath: payload.nutAssemblySourceDemoPath?.trim() || undefined,
    sourceDemoDatasetId,
    augmentationAlgorithm: payload.augmentationAlgorithm ?? NUT_ASSEMBLY_PATH_DEFAULTS.augmentationAlgorithm,
    seedGenerationCount: payload.seedGenerationCount,
    seedKeepCount: payload.seedKeepCount,
    targetCount: payload.targetCount,
    autoSelectBestSeeds: payload.autoSelectBestSeeds,
    replayValidation: payload.replayValidation,
    expertPolicy: payload.expertPolicy,
    successFilter: payload.successFilter,
    keepFailedTrajectories: payload.keepFailedTrajectories,
    enablePinnRepair,
    taskConfigId: taskConfigId ?? undefined,
    physicsEnhancement: buildNutAssemblyPhysicsEnhancementPayload(enablePinnRepair),
  };
}

export function nutAssemblyUsesMimicgenProgress(path: GenerationPath | null | undefined): boolean {
  return path === 'demo_augmentation' || path === 'expert_seed_then_augmentation';
}
