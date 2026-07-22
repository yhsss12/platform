import { test, expect } from 'playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const MOCK_FLOW_KEY = 'epi_workspace_mock_flow_v1';

test.describe.configure({ mode: 'serial' });
test.setTimeout(20 * 60_000);

test('Dual-Arm Cable UI walkthrough', async ({ page }) => {
  const report: Record<string, unknown> = {};
  let jobId = '';

  page.on('request', (req) => {
    if (req.url().includes('/api/workspace/dual-arm-cable/generate-async') && req.method() === 'POST') {
      report.generateAsyncCalled = true;
    }
  });

  // 1. Login
  await page.addInitScript(() => {
    localStorage.setItem('epi_locale', 'zh-CN');
  });
  await page.goto('/login');
  await page.locator('#login-username').fill(LOGIN_USER);
  await page.locator('#login-password').fill(LOGIN_PASS);
  await page.getByRole('button', { name: /登录|Sign in|Log in/i }).click();
  await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 30_000 });

  // 2. Data center
  await page.goto('/workspace/data');
  await expect(page.getByRole('heading', { name: /数据中心|^Data$/i })).toBeVisible({ timeout: 30_000 });

  // Clear prior mock flow items (keep auth)
  await page.evaluate((key) => {
    const token = sessionStorage.getItem('auth.access_token');
    const sessionId = sessionStorage.getItem('auth.sessionId');
    sessionStorage.clear();
    if (token) sessionStorage.setItem('auth.access_token', token);
    if (sessionId) sessionStorage.setItem('auth.sessionId', sessionId);
    sessionStorage.setItem(key, JSON.stringify({
      simulationRuns: [],
      activeSimulationRunId: null,
      extraDataItems: [],
      extraEvaluationTasks: [],
      lastProcessEvalId: null,
      activeDataGenerationItemId: null,
      activeDataGenerationContext: null,
      cableThreadingGenerateRuns: [],
      activeCableThreadingRunId: null,
      dualArmCableGenerateRuns: [],
      cableThreadingEvaluateRuns: [],
    }));
  }, MOCK_FLOW_KEY);
  await page.reload();

  // 3. Open generate modal
  await page.getByRole('button', { name: '生成数据' }).click();
  await expect(page.getByRole('heading', { name: '生成任务数据' })).toBeVisible();

  // 4. Select dual-arm template (scope to modal dialog)
  const modal = page.getByRole('dialog');
  await modal.locator('select').first().selectOption('线缆整理');
  await expect(modal.locator('select').first()).toHaveValue('线缆整理');

  // 5. Modal copy checks
  const modalArea = (await modal.textContent()) ?? '';
  report.modalHasMujoco = /MuJoCo/.test(modalArea);
  report.modalHasRobot = /Dual Franka FR3/.test(modalArea);
  report.modalHasGripper = /Robotiq 2F-85/.test(modalArea);
  report.modalHasObject = /杂乱柔性线缆/.test(modalArea);
  report.modalNoHdf5 = !/HDF5|Robomimic|ACT/.test(modalArea.replace(/episode_result\.json/g, ''));

  // 6. Start generation (dual-arm default seed=42)
  const generatePromise = page.waitForResponse(
    (res) =>
      res.url().includes('/api/workspace/dual-arm-cable/generate-async') && res.request().method() === 'POST',
    { timeout: 60_000 }
  );
  await modal.getByRole('button', { name: '启动生成' }).click();
  const generateRes = await generatePromise;
  if (generateRes.status() !== 200) {
    const body = await generateRes.text();
    throw new Error(`generate-async failed: ${generateRes.status()} ${generateRes.url()} body=${body.slice(0, 300)}`);
  }
  const generateBody = (await generateRes.json()) as { jobId?: string };
  jobId = generateBody.jobId ?? '';
  report.jobId = jobId;
  expect(jobId).toMatch(/^dac_gen_\d{8}_\d{6}_[a-f0-9]{4}$/);
  expect(jobId).not.toBe('dac_p0_test_seed42');
  expect(jobId).not.toBe('dac_gen_20260611_142635_4d74');

  // 7. Console redirect
  await page.waitForURL(/\/workspace\/simulation\/console.*jobId=/, { timeout: 30_000 });
  expect(page.url()).toContain('taskType=dual_arm_cable_manipulation');
  expect(page.url()).toContain(jobId);

  await expect(page.getByText('线缆整理 · 数据生成')).toBeVisible();
  await expect(page.getByText('Dual Franka FR3')).toBeVisible();
  await expect(page.getByText('Robotiq 2F-85')).toBeVisible();
  await expect(page.getByText('杂乱柔性线缆')).toBeVisible();

  // 8. Wait for completion (frame or completed badge)
  await expect(page.getByText('已完成').or(page.getByText('失败'))).toBeVisible({ timeout: 15 * 60_000 });
  const failed = await page.getByText('失败').isVisible().catch(() => false);
  report.consoleStatus = failed ? 'failed' : 'completed';

  if (!failed) {
    await expect(page.getByText('episodeSuccess', { exact: true })).toBeVisible({ timeout: 5_000 });
    const consoleText = (await page.locator('body').textContent()) ?? '';
    report.consoleHasVideoExists = /videoExists|generate\.mp4|查看回放/.test(consoleText);
    report.consoleHasResultPath = /resultPath|episode_result/.test(consoleText);
    const hasImg = (await page.locator('img[alt*="MuJoCo 运行画面"]').count()) > 0;
    const hasWaiting = /等待|初始化|加载/.test(consoleText);
    report.consoleFrame = hasImg ? 'latest.jpg' : hasWaiting ? 'waiting' : 'unknown';
  }

  // 9. Return to data center
  await page.getByRole('button', { name: '返回数据中心' }).click();
  await page.waitForURL(/\/workspace\/data/, { timeout: 30_000 });

  // 10. sessionStorage check
  const flowState = await page.evaluate((key) => {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  }, MOCK_FLOW_KEY);
  report.sessionStorageHasJob = Boolean(
    flowState?.extraDataItems?.some(
      (item: { id?: string; sourceJobId?: string }) =>
        item.id === jobId || item.sourceJobId === jobId
    )
  );
  report.sessionStorageRunStatus = flowState?.dualArmCableGenerateRuns?.find(
    (r: { jobId?: string }) => r.jobId === jobId
  )?.status;

  // 11. List row
  const row = page.locator('table tbody tr').filter({ hasText: /线缆整理数据|dac_gen_/ }).first();
  await expect(row).toBeVisible({ timeout: 15_000 });
  const rowText = (await row.textContent()) ?? '';
  report.listHasTaskData = /任务数据/.test(rowText);
  report.listHasTemplate = /线缆整理/.test(rowText);
  report.listHasEpisode = /1\s*episode/i.test(rowText);
  report.listNoHdf5 = !/HDF5|Robomimic|可构建|successRate|runs/.test(rowText);
  report.listActions = {
    detail: /详情/.test(rowText),
    replay: /回放/.test(rowText),
    delete: /删除/.test(rowText),
    buildDataset: /构建数据集/.test(rowText),
  };

  // 12. Detail drawer
  await row.getByRole('button', { name: /详情|Detail/i }).click();
  await expect(page.getByRole('heading', { name: '数据详情' })).toBeVisible();
  const drawerText = (await page.locator('aside[role="dialog"]').textContent()) ?? '';
  report.detailHasJobId = drawerText.includes(jobId);
  report.detailHasMetrics = ['left_contact', 'right_contact', 'stretch_reached', 'sag_m', 'span_m'].every((k) =>
    drawerText.includes(k)
  );
  report.detailHasFiles = ['episode_result.json', 'generate.mp4', 'run.log', 'latest_grasp.json'].every((f) =>
    drawerText.includes(f)
  );
  report.detailHasHdf5Note = /暂未生成 HDF5\/Robomimic/.test(drawerText);
  report.detailNoTrainEntry = !/构建数据集/.test(drawerText);

  // 13. Replay
  await page.locator('aside[role="dialog"]').getByRole('link', { name: /回放|Replay/i }).click();
  await page.waitForURL(/\/workspace\/replay.*jobId=/, { timeout: 30_000 });
  await expect(page.getByText('线缆整理 · 过程回放')).toBeVisible();

  const videoReqPromise = page.waitForResponse(
    (res) => res.url().includes(`/api/workspace/dual-arm-cable/jobs/${jobId}/video`),
    { timeout: 60_000 }
  );
  await videoReqPromise;
  report.replayVideoApi = `/api/workspace/dual-arm-cable/jobs/${jobId}/video`;
  await expect(page.locator('video')).toBeVisible({ timeout: 30_000 });

  // 14. Report / metrics tab
  await expect(page.getByText('线缆整理').first()).toBeVisible();
  const reportText = (await page.locator('body').textContent()) ?? '';
  report.reportHasDualArmMetrics = ['episode_success', 'left_contact', 'stretch_reached', 'final_span_m'].every((k) =>
    reportText.includes(k)
  );
  report.reportNoCableThreading = !/successRate|everSuccessRate|endpoint_goal_error|thread_completion|straightness_error/.test(
    reportText
  );

  // 15. Training center boundary
  await page.goto('/workspace/training');
  await page.getByRole('button', { name: /新建训练|创建训练|新建/i }).click().catch(() => undefined);
  const trainModal = page.locator('body');
  const trainText = (await trainModal.textContent()) ?? '';
  report.trainingExcludesDualArm = !trainText.includes(jobId) && !/线缆整理数据/.test(trainText);

  // Write report artifact
  await page.evaluate((data) => {
    (window as unknown as { __DAC_UI_REPORT__: unknown }).__DAC_UI_REPORT__ = data;
  }, report);

  console.log('DAC_UI_WALKTHROUGH_REPORT', JSON.stringify(report, null, 2));
});
