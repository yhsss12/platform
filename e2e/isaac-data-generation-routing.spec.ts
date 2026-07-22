import { expect, test } from '@playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const JOB_ID = 'isaac_gen_20260721_120000_abcd';

test('block stacking exposes separate generation modes and opens the Isaac console', async ({ page }) => {
  await page.route('**/api/workspace/isaac-lab/runtime/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        stackCubeGenerationReady: true,
        scriptedExpertReady: true,
        stackCubeIssueCodes: [],
        scriptedExpertIssueCodes: [],
      }),
    });
  });
  await page.route('**/api/workspace/isaac-lab/generate-dataset', async (route) => {
    expect(route.request().method()).toBe('POST');
    const payload = route.request().postDataJSON() as { generationMode?: string };
    expect(payload.generationMode).toBe('mimic_auto');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ jobId: JOB_ID, status: 'queued', runtimePath: 'runs/isaac_lab/jobs/' + JOB_ID }),
    });
  });
  await page.route(`**/api/workspace/isaac-lab/jobs/${JOB_ID}/**`, async (route) => {
    const url = route.request().url();
    if (url.includes('/log')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ tail: '' }) });
      return;
    }
    if (url.includes('/status')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          jobId: JOB_ID,
          status: 'queued',
          phase: 'queued',
          generationMode: 'mimic_auto',
          datasetName: '物块堆叠测试数据',
        }),
      });
      return;
    }
    await route.fulfill({ status: 404, body: '' });
  });

  await page.goto('/login');
  await page.locator('#login-username').fill(LOGIN_USER);
  await page.locator('#login-password').fill(LOGIN_PASS);
  await page.getByRole('button', { name: /登录|Sign in|Log in/i }).click();
  await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 30_000 });

  await page.goto('/workspace/data');
  await page.getByRole('button', { name: '数据生成' }).click();
  const modal = page.getByRole('dialog', { name: '生成任务数据' });
  await expect(modal).toBeVisible();
  await modal.locator('select').first().selectOption('物块堆叠');

  const generationMode = modal.getByText('生成方式', { exact: true }).locator('..').locator('select');
  await expect(generationMode.locator('option')).toHaveText(['专家策略生成', 'Mimic 示范扩增']);
  await generationMode.selectOption('mimic_auto');
  await modal.getByRole('button', { name: '生成 Isaac 数据' }).click();

  await page.waitForURL(new RegExp(`/workspace/simulation/console.*jobId=${JOB_ID}`));
  expect(page.url()).toContain('taskType=isaac_block_stacking');
  await expect(page.getByText('线缆穿杆 · 数据生成')).toHaveCount(0);
});
