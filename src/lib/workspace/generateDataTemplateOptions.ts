import { listTaskTemplates, type TaskTemplateDto } from '@/lib/api/taskTemplatesClient';
import {
  getStaticGenerateDataTemplateOptions,
  isDatasetGenerationEnabled,
  resolveTaskTemplateCapabilities,
} from '@/lib/workspace/taskTemplateCapabilities';

export const GENERATE_DATA_TEMPLATE_EMPTY_HINT =
  '暂无可用于数据生成的任务模板，请先在任务模板库中启用或接入生成适配器。';

export {
  getStaticGenerateDataTemplateOptions,
  getGenerateDataTemplateOptions,
  STATIC_GENERATE_DATA_TEMPLATE_LABELS,
} from '@/lib/workspace/taskTemplateCapabilities';

function isApiTemplateEligible(template: TaskTemplateDto): boolean {
  if (template.supportsDataGeneration !== true && template.supportsDatasetGeneration !== true) {
    return false;
  }
  const status = String(template.status ?? 'available').trim();
  if (status && status !== 'available') {
    return false;
  }
  const profile =
    resolveTaskTemplateCapabilities(template.id) ??
    resolveTaskTemplateCapabilities(template.name ?? '');
  return profile?.supportsDatasetGeneration === true;
}

function mergeApiAndLocalGenerateOptions(apiTemplates: TaskTemplateDto[]): string[] {
  const merged = new Set<string>();
  const apiIds = new Set<string>();

  for (const template of apiTemplates.filter(isApiTemplateEligible)) {
    const label = (template.name ?? template.id).trim();
    if (label) merged.add(label);
    if (template.id?.trim()) apiIds.add(template.id.trim());
  }

  for (const label of getStaticGenerateDataTemplateOptions()) {
    const profile = resolveTaskTemplateCapabilities(label);
    const templateId = profile?.templateId;
    if (templateId && apiIds.has(templateId)) continue;
    merged.add(label);
  }

  return [...merged].filter(isDatasetGenerationEnabled).sort((a, b) => a.localeCompare(b, 'zh-CN'));
}

export async function fetchGenerateDataTemplateOptions(): Promise<string[]> {
  try {
    const res = await listTaskTemplates({ limit: 200 });
    return mergeApiAndLocalGenerateOptions(res.taskTemplates);
  } catch {
    return getStaticGenerateDataTemplateOptions();
  }
}
