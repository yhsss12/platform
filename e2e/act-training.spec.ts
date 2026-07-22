/**
 * ACT 训练端到端验收：创建、详情、模型资产、low_dim 拒绝
 */
import { test, expect } from '@playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const API = 'http://127.0.0.1:8000';
const TARGET_DATASET_LABEL = '线缆穿杆';
const LOW_DIM_DATASET_LABEL = '物块堆叠';
const COMPLETED_ACT_JOB = 'train_20260619_163854_b9ab';
const ACT_LOSS_JOB = 'train_20260619_171910_cf85';

async function apiLogin(request: import('playwright').APIRequestContext) {
  const sid = crypto.randomUUID();
  const loginResp = await request.post(`${API}/api/auth/login`, {
    headers: { 'X-Session-Id': sid },
    data: { username: LOGIN_USER, password: LOGIN_PASS },
  });
  expect(loginResp.ok()).toBeTruthy();
  const loginJson = await loginResp.json();
  return {
    token: loginJson.data.access_token as string,
    sid,
    authHeaders: { Authorization: `Bearer ${loginJson.data.access_token}` },
  };
}

async function injectAuth(page: import('@playwright/test').Page, token: string, sid: string) {
  await page.addInitScript(
    ({ token, sessionId }) => {
      localStorage.setItem('epi_locale', 'zh-CN');
      sessionStorage.setItem('auth.access_token', token);
      sessionStorage.setItem('auth.sessionId', sessionId);
    },
    { token, sessionId: sid }
  );
}

test.describe.configure({ mode: 'serial' });

test.describe('ACT training acceptance', () => {
  let token: string;
  let sid: string;
  let authHeaders: Record<string, string>;

  test.beforeAll(async ({ request }) => {
    const caps = await request.get(`${API}/api/workspace/training/capabilities`, {
      headers: (await apiLogin(request)).authHeaders,
    });
    const capJson = await caps.json();
    test.skip(!capJson.supportedTrainingBackends?.includes('act'), 'ACT backend unavailable');

    const auth = await apiLogin(request);
    token = auth.token;
    sid = auth.sid;
    authHeaders = auth.authHeaders;
  });

  test('ACT visible in create modal without 待接入', async ({ page }) => {
    await injectAuth(page, token, sid);
    await page.goto('/workspace/training?openCreate=1');
    const dialog = page.getByRole('dialog', { name: '新建训练任务' });
    await expect(dialog).toBeVisible({ timeout: 20_000 });

    const modelSelect = dialog.locator('select').nth(1);
    await expect
      .poll(async () => {
        const labels = await modelSelect.locator('option').allTextContents();
        return labels.some((label) => /ACT/i.test(label));
      })
      .toBe(true);

    const modelLabels = await modelSelect.locator('option').allTextContents();
    const actLabel = modelLabels.find((label) => /ACT/i.test(label)) ?? '';
    expect(actLabel).not.toMatch(/待接入/);
  });

  test('low_dim dataset + ACT rejected by API without orphan job dir', async ({ request }) => {
    const datasetsRes = await request.get(`${API}/api/workspace/datasets`, { headers: authHeaders });
    expect(datasetsRes.ok()).toBeTruthy();
    const body = (await datasetsRes.json()) as {
      datasets?: Array<{
        id?: string;
        name?: string;
        datasetFile?: string;
        manifestPath?: string;
        taskTemplateId?: string;
        simulatorBackend?: string;
        episodeCount?: number;
        successfulEpisodes?: number;
      }>;
    };
    const datasets = body.datasets ?? [];
    const stack = datasets.find((d) => (d.name || '').includes(LOW_DIM_DATASET_LABEL));
    test.skip(!stack?.id || !stack.datasetFile, 'low_dim stack dataset not found');

    const manifest = {
      datasetId: stack!.id,
      datasetName: stack!.name,
      taskType: 'isaac_block_stacking',
      taskName: LOW_DIM_DATASET_LABEL,
      successfulEpisodes: stack!.successfulEpisodes ?? stack!.episodeCount ?? 1,
      artifacts: {
        hdf5: stack!.datasetFile,
        manifest: stack!.manifestPath,
      },
      quality: {
        status: 'ready',
        hasTrajectory: true,
        hasImage: false,
        hasSuccessfulEpisodes: true,
      },
      taskTemplateId: stack!.taskTemplateId ?? 'isaac_block_stacking',
      simulatorBackend: stack!.simulatorBackend ?? 'isaac_lab',
    };

    const beforeRes = await request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
    const beforeCount = ((await beforeRes.json()) as unknown[]).length;

    const createRes = await request.post(`${API}/api/workspace/training/jobs`, {
      headers: authHeaders,
      data: {
        datasetId: stack!.id,
        datasetManifest: manifest,
        downstreamModelType: 'ACT',
        trainingBackend: 'act',
        epochs: 1,
      },
    });
    expect(createRes.status()).toBe(400);
    const createBody = await createRes.json();
    const detail = String(createBody.detail ?? createBody.message ?? createBody.error ?? '');
    expect(detail).toMatch(/image observations/i);

    const afterRes = await request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
    const afterCount = ((await afterRes.json()) as unknown[]).length;
    expect(afterCount).toBe(beforeCount);
  });

  test('low_dim + ACT shows toast error in UI', async ({ page }) => {
    await injectAuth(page, token, sid);
    await page.goto('/workspace/training?openCreate=1');
    const dialog = page.getByRole('dialog', { name: '新建训练任务' });
    await expect(dialog).toBeVisible({ timeout: 20_000 });

    const datasetSelect = dialog.locator('select').first();
    await expect
      .poll(async () => {
        const labels = await datasetSelect.locator('option').allTextContents();
        return labels.some((label) => label.includes(LOW_DIM_DATASET_LABEL));
      })
      .toBe(true);
    const datasetOptions = await datasetSelect.locator('option').allTextContents();
    const stackIdx = datasetOptions.findIndex((label) => label.includes(LOW_DIM_DATASET_LABEL));
    await datasetSelect.selectOption({ index: stackIdx });

    const modelSelect = dialog.locator('select').nth(1);
    const modelLabels = await modelSelect.locator('option').allTextContents();
    const actIdx = modelLabels.findIndex((label) => /ACT/i.test(label));
    expect(actIdx).toBeGreaterThanOrEqual(0);
    await modelSelect.selectOption({ index: actIdx });

    await dialog.locator('input[type="number"]').first().fill('1');
    await dialog.getByRole('button', { name: '创建训练任务' }).click();

    await expect(page.getByText(/image observations/i)).toBeVisible({ timeout: 20_000 });
    await expect(dialog).toBeVisible();
  });

  test('create ACT job for cable threading image dataset', async ({ page, request }) => {
    test.setTimeout(300_000);

    const adapterCalls: string[] = [];
    await page.route('**/api/adapter/**', (route) => {
      adapterCalls.push(route.request().url());
      route.abort();
    });

    await injectAuth(page, token, sid);

    const beforeRes = await request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
    const beforeJobs = (await beforeRes.json()) as unknown[];

    await page.goto('/workspace/training?openCreate=1');
    const dialog = page.getByRole('dialog', { name: '新建训练任务' });
    await expect(dialog).toBeVisible({ timeout: 20_000 });

    const datasetSelect = dialog.locator('select').first();
    await expect
      .poll(async () => {
        const labels = await datasetSelect.locator('option').allTextContents();
        return labels.some((label) => label.includes(TARGET_DATASET_LABEL));
      })
      .toBe(true);
    const datasetOptions = await datasetSelect.locator('option').allTextContents();
    const cableIdx = datasetOptions.findIndex((label) => label.includes(TARGET_DATASET_LABEL));
    await datasetSelect.selectOption({ index: cableIdx });

    const modelSelect = dialog.locator('select').nth(1);
    const modelLabels = await modelSelect.locator('option').allTextContents();
    const actIdx = modelLabels.findIndex((label) => /ACT/i.test(label));
    await modelSelect.selectOption({ index: actIdx });
    await dialog.locator('input[type="number"]').first().fill('1');
    await dialog.getByRole('button', { name: '创建训练任务' }).click();
    await expect(dialog).toBeHidden({ timeout: 60_000 });
    expect(adapterCalls).toHaveLength(0);

    await page.reload();
    await expect(page.locator('table tbody tr').filter({ hasText: /ACT/i }).first()).toBeVisible({
      timeout: 20_000,
    });

    const afterRes = await request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
    const afterJobs = (await afterRes.json()) as Array<{ trainJobId: string; downstreamModelType?: string }>;
    expect(afterJobs.length).toBeGreaterThan(beforeJobs.length);
  });

  test('completed ACT detail: loss curve, log, Final asset, eval disabled', async ({ page, request }) => {
    test.setTimeout(120_000);
    await injectAuth(page, token, sid);

    const statusRes = await request.get(
      `${API}/api/workspace/training/jobs/${encodeURIComponent(COMPLETED_ACT_JOB)}/status`,
      { headers: authHeaders }
    );
    test.skip(!statusRes.ok(), `completed ACT job ${COMPLETED_ACT_JOB} not available`);

    await page.goto(`/workspace/training?jobId=${encodeURIComponent(COMPLETED_ACT_JOB)}`);
    await expect(page.getByText('训练过程性能')).toBeVisible({ timeout: 20_000 });
    const drawer = page.locator('aside');
    await expect(drawer.getByText('ACT', { exact: true })).toBeVisible();
    await expect(drawer.getByText('Final', { exact: true })).toBeVisible({ timeout: 15_000 });

    const evalDisabled = drawer.locator('span[title="当前评测后端暂不支持 ACT"]');
    await expect(evalDisabled).toBeVisible();
    await expect(evalDisabled).toHaveText('发起评测');

    await drawer.getByRole('button', { name: '日志' }).click();
    await expect(drawer.locator('pre')).toContainText(/train_act\.py|Epoch/i, { timeout: 20_000 });
  });

  test('model assets page lists ACT and batch delete UI', async ({ page, request }) => {
    await injectAuth(page, token, sid);

    const assetsRes = await request.get(`${API}/api/workspace/model-assets`, { headers: authHeaders });
    expect(assetsRes.ok()).toBeTruthy();
    const assetsBody = (await assetsRes.json()) as { modelAssets?: Array<{ modelType?: string }> };
    const assets = assetsBody.modelAssets ?? [];
    const hasAct = assets.some((a) => (a.modelType || '').toLowerCase() === 'act');
    test.skip(!hasAct, 'no ACT model assets in workspace');

    await page.goto('/workspace/resources/model-assets');
    await expect(page.getByRole('heading', { name: '模型资产' })).toBeVisible({ timeout: 20_000 });
    await expect(page.locator('table tbody tr').filter({ hasText: /ACT/i }).first()).toBeVisible({
      timeout: 15_000,
    });

    const batchBtn = page.getByRole('button', { name: '批量删除' });
    await expect(batchBtn).toBeDisabled();
    const checkbox = page.locator('table tbody tr').filter({ hasText: /ACT/i }).first().locator('input[type="checkbox"]');
    if (await checkbox.isVisible().catch(() => false)) {
      await checkbox.check();
      await expect(batchBtn).toBeEnabled();
      await checkbox.uncheck();
    }
  });

  test('ACT running job shows train and valid loss series', async ({ page, request }) => {
    test.setTimeout(120_000);
    const statusRes = await request.get(
      `${API}/api/workspace/training/jobs/${encodeURIComponent(ACT_LOSS_JOB)}/status`,
      { headers: authHeaders }
    );
    test.skip(!statusRes.ok(), `ACT job ${ACT_LOSS_JOB} not available`);

    const detailRes = await request.get(`${API}/api/workspace/jobs/${encodeURIComponent(ACT_LOSS_JOB)}`, {
      headers: authHeaders,
    });
    expect(detailRes.ok()).toBeTruthy();
    const detail = (await detailRes.json()) as {
      metrics?: { lossHistory?: Array<{ epoch?: number; trainLoss?: number; validLoss?: number }> };
    };
    const history = detail.metrics?.lossHistory ?? [];
    expect(history.length).toBeGreaterThanOrEqual(12);
    const epoch12 = history.find((row) => Number(row.epoch) === 12);
    expect(epoch12?.trainLoss).toBeTruthy();
    expect(epoch12?.validLoss).toBeTruthy();

    const statusBody = (await statusRes.json()) as { data?: { status?: string }; status?: string };
    const jobStatus = String(statusBody.data?.status ?? statusBody.status ?? '').toLowerCase();

    await injectAuth(page, token, sid);
    await page.goto(`/workspace/training?jobId=${encodeURIComponent(ACT_LOSS_JOB)}`);
    const drawer = page.getByRole('dialog');
    await expect(drawer.getByText('训练过程性能')).toBeVisible({ timeout: 20_000 });
    await expect(drawer.getByText('Train Loss')).toBeVisible();
    await expect(drawer.getByText('Valid Loss')).toBeVisible();
    await expect(drawer.getByText('Best Loss')).toBeVisible();
    if (jobStatus === 'running' || jobStatus === 'training' || jobStatus === 'pending') {
      await expect(drawer.getByText(/当前 Loss/)).toBeVisible();
    } else if (jobStatus === 'completed' || jobStatus === 'succeeded') {
      await expect(drawer.getByText(/最终 Loss/)).toBeVisible();
    } else {
      await expect(drawer.getByText(/最后 Loss/).first()).toBeVisible();
    }
    await expect(drawer.getByText('Final Loss', { exact: true })).toHaveCount(0);
  });
});
