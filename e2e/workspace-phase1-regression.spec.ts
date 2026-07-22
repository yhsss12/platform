import { test, expect } from 'playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const MOCK_FLOW_KEY = 'epi_workspace_mock_flow_v1';

async function apiLogin(request: import('playwright').APIRequestContext) {
  const sid = crypto.randomUUID();
  const loginResp = await request.post('http://127.0.0.1:8000/api/auth/login', {
    headers: { 'X-Session-Id': sid },
    data: { username: LOGIN_USER, password: LOGIN_PASS },
  });
  expect(loginResp.ok()).toBeTruthy();
  const loginJson = await loginResp.json();
  const token = loginJson.data.access_token as string;
  return { token, sid, authHeaders: { Authorization: `Bearer ${token}` } };
}

async function injectBrowserAuth(
  page: import('playwright').Page,
  token: string,
  sessionId: string
) {
  await page.addInitScript(({ token, sessionId }) => {
    localStorage.setItem('epi_locale', 'zh-CN');
    sessionStorage.setItem('auth.access_token', token);
    sessionStorage.setItem('auth.sessionId', sessionId);
  }, { token, sessionId });
}

test.describe('Workspace Phase 1 regression', () => {
  test.setTimeout(180_000);

  test('centers restore real jobs; demo mock hidden; replay/report workspace-first', async ({
    page,
    request,
  }) => {
    const report: Record<string, unknown> = {};

    const { token, sid, authHeaders } = await apiLogin(request);

    const genList = await request.get(
      'http://127.0.0.1:8000/api/workspace/jobs?jobType=generate&source=real&limit=3',
      { headers: authHeaders }
    );
    const genBody = await genList.json();
    const sampleGenId = genBody.jobs?.[0]?.jobId as string | undefined;
    report.sampleGenerateJobId = sampleGenId;

    const evalList = await request.get(
      'http://127.0.0.1:8000/api/workspace/jobs?jobType=evaluation&source=real&limit=3',
      { headers: authHeaders }
    );
    const evalBody = await evalList.json();
    const sampleEvalId =
      (evalBody.jobs as Array<{ jobId: string }> | undefined)?.find((j) =>
        j.jobId.startsWith('eval_')
      )?.jobId ??
      (evalBody.jobs as Array<{ jobId: string }> | undefined)?.find((j) =>
        j.jobId.startsWith('ct_eval_')
      )?.jobId;
    report.sampleEvalJobId = sampleEvalId;

    await injectBrowserAuth(page, token, sid);
    await page.goto('/workspace/data');
    await expect(page.getByRole('heading', { name: /数据中心/i })).toBeVisible({ timeout: 60_000 });

    // Clear session mock flow only (preserve auth tokens)
    await page.evaluate((key) => {
      const accessToken = sessionStorage.getItem('auth.access_token');
      const sessionId = sessionStorage.getItem('auth.sessionId');
      sessionStorage.clear();
      if (accessToken) sessionStorage.setItem('auth.access_token', accessToken);
      if (sessionId) sessionStorage.setItem('auth.sessionId', sessionId);
    }, MOCK_FLOW_KEY);
    if (sampleGenId) {
      await expect(page.getByText(sampleGenId)).toBeVisible({ timeout: 30_000 });
      report.dataCenterShowsRealJob = true;
    }

    // Ensure demo checkbox default: if env true it should be unchecked
    const demoCheckbox = page.getByRole('checkbox', { name: '显示示例数据' });
    await expect(demoCheckbox).toBeVisible();
    const demoChecked = await demoCheckbox.isChecked();
    report.demoCheckboxCheckedByDefault = demoChecked;
    report.demoRealOnlyEnv = process.env.NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY === 'true';

    // Mock seed id from workspaceDataItemsMock first item pattern - look for a known mock-only name
    const mockOnlyVisible = await page.getByText('示范数据集 A').isVisible().catch(() => false);
    report.mockSeedVisibleWithDefaultCheckbox = mockOnlyVisible;

    // Refresh persistence
    await page.reload();
    if (sampleGenId) {
      await expect(page.getByText(sampleGenId)).toBeVisible({ timeout: 30_000 });
      report.dataCenterSurvivesRefresh = true;
    }

    // Evaluation center
    await page.goto('/workspace/evaluation');
    await expect(page.getByRole('heading', { name: /评测中心/i })).toBeVisible();
    await expect(page.getByRole('checkbox', { name: '显示示例数据' })).toHaveCount(0);
    const evalMockVisible = await page.getByText('两次拧螺丝 · ACT 策略评测').isVisible().catch(() => false);
    report.evalMockSeedVisible = evalMockVisible;
    if (sampleEvalId) {
      await expect(page.getByText(/线缆整理|线缆穿杆/).first()).toBeVisible({
        timeout: 30_000,
      });
      await expect(page.locator('thead input[type="checkbox"]')).toHaveCount(1);
      report.evalCenterShowsRealJob = true;
      await page.reload();
      await expect(page.getByText(/线缆整理|线缆穿杆/).first()).toBeVisible({
        timeout: 30_000,
      });
      report.evalCenterSurvivesRefresh = true;
    }

    // Training center
    const trainList = await request.get(
      'http://127.0.0.1:8000/api/workspace/jobs?jobType=training&source=real&limit=1',
      { headers: authHeaders }
    );
    const trainBody = await trainList.json();
    const sampleTrainId = trainBody.jobs?.[0]?.jobId as string | undefined;
    report.sampleTrainJobId = sampleTrainId;
    await page.goto('/workspace/training');
    await expect(page.getByRole('heading', { name: /训练中心/i })).toBeVisible();
    await expect(page.getByRole('checkbox', { name: '显示示例数据' })).toHaveCount(0);
    const trainMockVisible = await page.getByText('示范 Robomimic 训练').isVisible().catch(() => false);
    report.trainingMockSeedVisible = trainMockVisible;
    if (sampleTrainId) {
      const trainJob = trainBody.jobs?.[0] as { taskName?: string; metadata?: { datasetName?: string } } | undefined;
      const trainListLabel =
        trainJob?.metadata?.datasetName || trainJob?.taskName || '训练';
      await expect(page.getByText(new RegExp(trainListLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))).first()).toBeVisible({
        timeout: 30_000,
      });
      await expect(page.locator('thead input[type="checkbox"]')).toHaveCount(1);
      report.trainingCenterShowsRealJob = true;
      report.sampleTrainJobId = sampleTrainId;
      await page.reload();
      await expect(page.getByText(new RegExp(trainListLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))).first()).toBeVisible({
        timeout: 30_000,
      });
      report.trainingCenterSurvivesRefresh = true;
    }

    // Replay workspace-first
    if (sampleGenId) {
      await page.goto(`/workspace/replay?jobId=${encodeURIComponent(sampleGenId)}&taskType=cable_threading`);
      await expect(page.getByText(/回放|Replay|线缆穿杆/i).first()).toBeVisible({ timeout: 30_000 });
      report.replayPageLoadsWithWorkspaceJob = true;
    }

    // Report workspace-first (dual_arm eval if available)
    if (sampleEvalId?.startsWith('eval_')) {
      await page.goto(
        `/workspace/evaluation/report?evalId=${encodeURIComponent(sampleEvalId)}&taskType=dual_arm_cable_manipulation`
      );
      await expect(page.getByRole('heading', { name: '评测报告' }).first()).toBeVisible({
        timeout: 30_000,
      });
      const loading = page.getByText(/正在从后端加载 aggregate_result/i);
      const hasLoading = await loading.isVisible().catch(() => false);
      const hasSections = await page.getByText(/聚合指标|基本信息/i).first().isVisible().catch(() => false);
      report.reportPageUsesBackendOrWorkspace = hasSections || hasLoading;
    }

    test.info().attach('workspace-phase1-browser-report', {
      body: JSON.stringify(report, null, 2),
      contentType: 'application/json',
    });

    expect(genBody.total).toBeGreaterThan(0);
    expect(evalBody.total).toBeGreaterThan(0);
    if (sampleGenId) expect(report.dataCenterSurvivesRefresh).toBe(true);
    if (process.env.NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY === 'true') {
      expect(report.demoCheckboxCheckedByDefault).toBe(false);
      expect(report.mockSeedVisibleWithDefaultCheckbox).toBe(false);
    }
    expect(report.trainingMockSeedVisible).toBe(false);
    expect(report.evalMockSeedVisible).toBe(false);
    if (sampleEvalId) {
      expect(report.evalCenterShowsRealJob).toBe(true);
      expect(report.evalCenterSurvivesRefresh).toBe(true);
    }
  });
});
