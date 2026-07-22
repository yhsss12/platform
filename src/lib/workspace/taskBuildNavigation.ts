export function buildTaskBuildTemplateHref(taskTemplateId?: string): string {
  if (!taskTemplateId) return '/workspace/task-build/template';
  return `/workspace/task-build/template?taskTemplateId=${encodeURIComponent(taskTemplateId)}`;
}

export function buildDataGenerateHref(params: {
  taskTemplateId: string;
  templateName?: string;
  taskConfigId?: string;
}): string {
  const search = new URLSearchParams({
    openGenerate: '1',
    taskTemplateId: params.taskTemplateId,
  });
  if (params.templateName) search.set('template', params.templateName);
  if (params.taskConfigId) search.set('taskConfigId', params.taskConfigId);
  return `/workspace/data?${search.toString()}`;
}

export function buildEvaluationCreateHref(params: {
  taskTemplateId: string;
  templateName?: string;
  taskConfigId?: string;
  datasetId?: string;
  modelAssetId?: string;
}): string {
  const search = new URLSearchParams({
    openCreate: '1',
    taskTemplateId: params.taskTemplateId,
  });
  if (params.templateName) search.set('template', params.templateName);
  if (params.taskConfigId) search.set('taskConfigId', params.taskConfigId);
  if (params.datasetId) search.set('dataset', params.datasetId);
  if (params.modelAssetId) search.set('modelAsset', params.modelAssetId);
  return `/workspace/evaluation?${search.toString()}`;
}

export function buildTrainingCreateHref(params: {
  taskTemplateId: string;
  datasetId?: string;
}): string {
  const search = new URLSearchParams({ openCreate: '1', taskTemplateId: params.taskTemplateId });
  if (params.datasetId) search.set('dataset', params.datasetId);
  return `/workspace/training?${search.toString()}`;
}

export function buildRealDataBuildHref(from?: string): string {
  const search = from ? `?from=${encodeURIComponent(from)}` : '';
  return `/workspace/task-build/real-data${search}`;
}
