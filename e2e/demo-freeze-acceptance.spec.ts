import { test, expect } from 'playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const DRAFT_STORAGE_KEY = 'workspace_real_data_import_drafts';

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

const FORBIDDEN_UI_PATTERNS = [
  /mock/i,
  /demo/i,
  /占位/,
  /示例数据/,
  /未接入/,
  /真实后端/,
  /功能开发中/,
  /敬请期待/,
  /待接入/,
];

test.describe('Phase 6 demo freeze acceptance', () => {
  test.setTimeout(240_000);

  test('route A/B flows, entries, forbidden UI scan', async ({ page, request }) => {
    const report: Record<string, unknown> = {};

    const { token, sid, authHeaders } = await apiLogin(request);

    const templatesRes = await request.get('http://127.0.0.1:8000/api/workspace/task-templates', {
      headers: authHeaders,
    });
    const templatesBody = await templatesRes.json();
    report.taskTemplatesCount = templatesBody.taskTemplates?.length;
    report.taskTemplateIds = (templatesBody.taskTemplates ?? []).map((t: { id: string }) => t.id);

    const datasetsRes = await request.get('http://127.0.0.1:8000/api/workspace/datasets', {
      headers: authHeaders,
    });
    const datasetsBody = await datasetsRes.json();
    report.datasetsCount = datasetsBody.datasets?.length ?? datasetsBody.total;

    const modelsRes = await request.get('http://127.0.0.1:8000/api/workspace/model-assets', {
      headers: authHeaders,
    });
    const modelsBody = await modelsRes.json();
    report.modelAssetsCount = modelsBody.modelAssets?.length ?? modelsBody.total;

    const capsRes = await request.get('http://127.0.0.1:8000/api/workspace/evaluation/capabilities', {
      headers: authHeaders,
    });
    const capsBody = await capsRes.json();
    report.evalCapabilities = capsBody;

    await injectBrowserAuth(page, token, sid);

    // Redirect /workspace/task-generation
    await page.goto('/workspace/task-generation');
    await page.waitForURL(/\/workspace\/task-build/, { timeout: 30_000 });
    report.taskGenerationRedirects = true;

    // Route A: cable_threading_single_arm
    await page.goto('/workspace/task-build');
    await expect(page.getByRole('heading', { name: '创建任务' })).toBeVisible();
    await page.getByRole('button', { name: '开始配置' }).click();
    await page.waitForURL(/task-build\/template/, { timeout: 30_000 });

    await page.getByRole('button', { name: '线缆操作任务族' }).click();
    await page.getByRole('button', { name: /下一步/ }).click();

    await page.getByRole('button', { name: /线缆穿杆/ }).click();
    await page.getByRole('button', { name: /下一步/ }).click();

    await expect(page.getByText('taskFamily')).toBeVisible();
    await expect(page.getByText('simulatorType')).toBeVisible();
    await expect(page.getByText('supportedEvaluationModes')).toBeVisible();
    await expect(page.getByText('可用 Dataset 数量')).toBeVisible();
    await expect(page.getByText('可用 ModelAsset 数量')).toBeVisible();
    report.routeA_step3_detailsVisible = true;

    await page.getByRole('button', { name: /下一步/ }).click();
    await page.getByRole('button', { name: '生成任务配置' }).click();
    await expect(page.getByText('任务配置已生成')).toBeVisible({ timeout: 15_000 });

    const evalLink = page.getByRole('link', { name: '前往评测创建' });
    await expect(evalLink).toBeVisible();
    const evalHref = await evalLink.getAttribute('href');
    report.routeA_evalHref = evalHref;
    expect(evalHref).toContain('taskTemplateId=cable_threading_single_arm');
    expect(evalHref).toContain('openCreate=1');

    await evalLink.click();
    await page.waitForURL(/\/workspace\/evaluation/, { timeout: 30_000 });
    await expect(page.getByText('新建评测任务')).toBeVisible({ timeout: 15_000 });

    const templateSelect = page.locator('label:has-text("任务模板")').locator('..').locator('select');
    await expect(templateSelect).toHaveValue('cable_threading_single_arm');

    const modeSelect = page.locator('label:has-text("评测模式")').locator('..').locator('select');
    const modeOptions = await modeSelect.locator('option').allTextContents();
    report.cableEvalModes = modeOptions;
    expect(modeOptions.some((o) => o.includes('专家策略'))).toBe(true);
    expect(modeOptions.some((o) => o.includes('训练模型'))).toBe(true);
    expect(modeOptions.some((o) => o.includes('未接入'))).toBe(false);

    await page.getByRole('button', { name: '取消' }).click();

    // Route A: dual_arm - preset taskTemplateId skips to step 2 with selection
    await page.goto('/workspace/task-build/template?taskTemplateId=dual_arm_cable_manipulation');
    await page.getByRole('button', { name: /下一步/ }).click();
    await page.getByRole('button', { name: /下一步/ }).click();
    await page.getByRole('button', { name: '生成任务配置' }).click();
    await expect(page.getByText('任务配置已生成')).toBeVisible();

    await page.getByRole('link', { name: '前往评测创建' }).click();
    await expect(page.getByText('新建评测任务')).toBeVisible({ timeout: 15_000 });
    const dualModeSelect = page.locator('label:has-text("评测模式")').locator('..').locator('select');
    const dualModes = await dualModeSelect.locator('option').allTextContents();
    report.dualArmEvalModes = dualModes;
    expect(dualModes.some((o) => o.includes('稳定性') || o.includes('episode'))).toBe(true);
    expect(dualModes.some((o) => o.includes('训练模型'))).toBe(false);
    expect(dualModes.some((o) => o.includes('未接入'))).toBe(false);
    await page.getByRole('button', { name: '取消' }).click();

    // Route B: real-data draft
    await page.goto('/workspace/task-build/real-data');
    await page.getByRole('button', { name: '本地文件' }).click();
    await page.getByRole('button', { name: /下一步/ }).click(); // step 1 -> 2
    await page.fill('input[placeholder="或输入文件名称"]', 'demo_robot_session_001.hdf5');
    await page.getByRole('button', { name: /下一步/ }).click();
    await page.fill(
      'textarea',
      'joint_pos\njoint_vel\ngripper_state'
    );
    await page.getByRole('button', { name: '解析数据结构' }).click();
    await page.getByRole('button', { name: /下一步/ }).click();
    await page.selectOption('select', { index: 1 });
    await page.getByRole('button', { name: /下一步/ }).click();
    await page.fill('input[placeholder="例如 Panda"]', 'Panda');
    await page.getByRole('button', { name: '保存构建草稿' }).click();
    await expect(page.getByText('构建草稿已保存')).toBeVisible({ timeout: 15_000 });

    const draftsRaw = await page.evaluate((key) => localStorage.getItem(key), DRAFT_STORAGE_KEY);
    report.realDataDraftInLocalStorage = Boolean(draftsRaw);
    const drafts = JSON.parse(draftsRaw || '[]') as Array<{ sourceFileName: string }>;
    report.realDataDraftCount = drafts.length;
    expect(drafts.some((d) => d.sourceFileName.includes('demo_robot_session'))).toBe(true);

    // Data center entry
    await page.goto('/workspace/data');
    await page.getByRole('button', { name: '数据构建' }).click();
    await page.waitForURL(/task-build\/real-data/, { timeout: 30_000 });
    report.dataCenterRealDataEntry = true;

    // Training openCreate=1
    await page.goto('/workspace/training?openCreate=1&taskTemplateId=cable_threading_single_arm');
    await expect(page.getByText('新建训练任务')).toBeVisible({ timeout: 15_000 });
    report.trainingOpenCreate = true;

    // Forbidden UI scan on key workspace pages
    const pagesToScan = [
      '/workspace/task-build',
      '/workspace/task-build/template',
      '/workspace/task-build/real-data',
      '/workspace/data',
      '/workspace/evaluation',
      '/workspace/resources/task-templates',
    ];
    const forbiddenHits: string[] = [];
    for (const path of pagesToScan) {
      await page.goto(path);
      await page.waitForTimeout(500);
      const bodyText = await page.locator('body').innerText();
      for (const pattern of FORBIDDEN_UI_PATTERNS) {
        if (pattern.test(bodyText)) {
          forbiddenHits.push(`${path}: ${pattern.source}`);
        }
      }
    }
    report.forbiddenUiHits = forbiddenHits;

    test.info().attach('demo-freeze-report', {
      body: JSON.stringify(report, null, 2),
      contentType: 'application/json',
    });

    expect(templatesBody.taskTemplates?.length).toBe(2);
    expect(datasetsBody.datasets?.length ?? datasetsBody.total).toBeGreaterThanOrEqual(8);
    expect(modelsBody.modelAssets?.length ?? modelsBody.total).toBeGreaterThanOrEqual(2);
    expect(report.taskGenerationRedirects).toBe(true);
    expect(forbiddenHits).toEqual([]);
  });
});
