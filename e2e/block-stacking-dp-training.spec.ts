/**
 * 物块堆叠 + Diffusion Policy 端到端验收
 * - 不调用 /api/adapter/*
 * - 仅 POST /api/workspace/training/jobs
 */
import { test, expect } from '@playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const API = 'http://127.0.0.1:8000';
const TARGET_DATASET_LABEL = '物块堆叠';

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

test.describe('Block stacking + Diffusion Policy training', () => {
  test('frontend can select DP for block stacking dataset and create job', async ({ page, request }) => {
    test.setTimeout(180_000);

    const adapterCalls: string[] = [];
    await page.route('**/api/adapter/**', (route) => {
      adapterCalls.push(route.request().url());
      route.abort();
    });

    const { token, sid, authHeaders } = await apiLogin(request);
    await page.addInitScript(
      ({ token, sessionId }) => {
        localStorage.setItem('epi_locale', 'zh-CN');
        sessionStorage.setItem('auth.access_token', token);
        sessionStorage.setItem('auth.sessionId', sessionId);
      },
      { token, sessionId: sid }
    );

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
    const stackIdx = datasetOptions.findIndex((label) => label.includes(TARGET_DATASET_LABEL));
    expect(stackIdx, `应存在${TARGET_DATASET_LABEL}数据集选项`).toBeGreaterThanOrEqual(0);
    await datasetSelect.selectOption({ index: stackIdx });

    const modelSelect = dialog.locator('select').nth(1);
    await expect
      .poll(async () => {
        const labels = await modelSelect.locator('option').allTextContents();
        return labels.some((label) => /Diffusion Policy/i.test(label));
      })
      .toBe(true);
    const modelLabels = await modelSelect.locator('option').allTextContents();
    const dpIdx = modelLabels.findIndex((label) => /Diffusion Policy/i.test(label));
    expect(dpIdx, 'Diffusion Policy 应出现在模型类型下拉中').toBeGreaterThanOrEqual(0);
    await modelSelect.selectOption({ index: dpIdx });

    await dialog.locator('input[type="number"]').first().fill('1');
    await dialog.getByRole('button', { name: '创建训练任务' }).click();
    await expect(dialog).toBeHidden({ timeout: 60_000 });

    expect(adapterCalls, '前端不应调用 adapter API').toHaveLength(0);

    await page.waitForTimeout(3000);
    const afterRes = await request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
    const afterJobs = (await afterRes.json()) as Array<{ trainJobId: string; downstreamModelType?: string }>;
    expect(afterJobs.length).toBeGreaterThan(beforeJobs.length);

    const created = afterJobs.find(
      (job) =>
        !beforeJobs.some((b) => (b as { trainJobId?: string }).trainJobId === job.trainJobId) &&
        (job.downstreamModelType || '').includes('Diffusion')
    );
    expect(created?.trainJobId).toBeTruthy();

    await page.reload();
    await expect(page.locator('table tbody tr').filter({ hasText: /Diffusion Policy|扩散/i }).first()).toBeVisible({
      timeout: 20_000,
    });
  });
});
