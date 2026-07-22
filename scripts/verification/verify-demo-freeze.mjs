/**
 * 演示冻结前 P0/P1 浏览器刷新验证（Playwright + 真实登录态）。
 */
import { chromium } from 'playwright';

const BASE = process.env.BASE_URL || 'http://127.0.0.1:3001';
const STORAGE_KEY = 'epi_workspace_mock_flow_v1';
const DEMO_DATA_ID = 'ct_gen_e2e_demo';
const DEMO_EVAL_ID = 'ct_eval_e2e_demo';

const mockFlowState = {
  simulationRuns: [],
  activeSimulationRunId: null,
  extraDataItems: [
    {
      id: DEMO_DATA_ID,
      name: '线缆穿杆 E2E 演示数据集',
      taskId: 'cable_threading',
      taskName: '单臂线缆穿杆',
      simulationId: DEMO_DATA_ID,
      dataCategory: '示范数据',
      source: 'MuJoCo 生成',
      targetModelFormat: '—',
      dataVolume: '10 条',
      size: '12 MB',
      status: 'completed',
      generatedAt: '2026-06-09 15:24',
      creator: '当前用户',
      scene: '桌面双杆穿线工位',
      robot: 'Panda',
      simBackend: 'MuJoCo',
      taskType: 'cable_threading',
      cableModel: 'composite_cable',
      difficulty: 'easy',
      successRate: 90,
      successfulEpisodes: 9,
      npzPath: '/runs/cable_threading/jobs/demo/expert_data.npz',
      hdf5Path: '/runs/cable_threading/jobs/demo/expert_data.hdf5',
      manifestPath: '/runs/cable_threading/jobs/demo/manifest.json',
      collectCsvPath: '/runs/cable_threading/jobs/demo/collect.csv',
      failuresPath: '/runs/cable_threading/jobs/demo/failures.json',
    },
  ],
  extraEvaluationTasks: [
    {
      id: DEMO_EVAL_ID,
      name: '单臂线缆穿杆 · scripted 评测',
      evaluationMode: '策略评测',
      relatedTask: '单臂线缆穿杆',
      checkpoint: 'scripted',
      modelType: 'scripted',
      dataVolume: '10 条',
      evalBackend: 'MuJoCo',
      evalRounds: 10,
      status: '已完成',
      successRate: 90,
      createdAt: '2026-06-09 15:21',
      taskType: 'cable_threading',
      cableModel: 'composite_cable',
      difficulty: 'easy',
      policy: 'scripted',
      robot: 'Panda',
      videoExists: true,
      videoPath: '/runs/cable_threading/jobs/demo/replay.mp4',
      videoSizeBytes: 346377,
    },
  ],
  lastProcessEvalId: null,
  activeDataGenerationItemId: null,
  activeDataGenerationContext: null,
};

async function login() {
  const res = await fetch(`${BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'Pibot0001', password: 'jinlian1234' }),
  });
  const json = await res.json();
  if (!json?.data?.access_token) throw new Error(`登录失败: ${JSON.stringify(json)}`);
  return {
    accessToken: json.data.access_token,
    sessionId: json.data.session_id || '00000000-0000-4000-8000-000000000000',
  };
}

async function waitForWorkspace(page, hint) {
  await page.waitForFunction(
    (needle) => {
      const text = document.body?.innerText || '';
      if (text.includes('加载中')) return false;
      return needle ? text.includes(needle) : text.length > 200;
    },
    hint ?? '',
    { timeout: 30000 }
  );
}

async function main() {
  const results = {};
  const { accessToken, sessionId } = await login();
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.addInitScript(
    ({ token, sid, storageKey, flowState }) => {
      window.sessionStorage.setItem('auth.access_token', token);
      window.sessionStorage.setItem('auth.sessionId', sid);
      window.sessionStorage.setItem(storageKey, JSON.stringify(flowState));
    },
    {
      token: accessToken,
      sid: sessionId,
      storageKey: STORAGE_KEY,
      flowState: mockFlowState,
    }
  );

  // A: task-008 唯一 + 后端标识
  await page.goto(`${BASE}/workspace/resources/task-templates`);
  await waitForWorkspace(page, '单臂线缆穿杆');
  const cableCount = await page.locator('table tbody tr').filter({ hasText: '单臂线缆穿杆' }).count();
  const tubeCount = await page.locator('table tbody tr').filter({ hasText: '试管转移任务' }).count();
  results.taskIdUnique = cableCount === 1 && tubeCount === 1;
  if (cableCount === 1) {
    await page.locator('table tbody tr').filter({ hasText: '单臂线缆穿杆' }).getByRole('button', { name: '详情' }).click();
    await page.waitForTimeout(500);
    const drawerText = await page.locator('body').innerText();
    results.backendIdentifier =
      drawerText.includes('task-008') &&
      drawerText.includes('穿线') &&
      drawerText.includes('后端标识') &&
      drawerText.includes('cable_threading');
    await page.keyboard.press('Escape');
  } else {
    results.backendIdentifier = false;
  }

  // C + E(1): 数据中心刷新保留
  await page.goto(`${BASE}/workspace/data`);
  await waitForWorkspace(page, '线缆穿杆 E2E 演示数据集');
  let dataText = await page.locator('body').innerText();
  results.dataSuccessRateVisible =
    dataText.includes('线缆穿杆 E2E 演示数据集') && /成功率\s*90%/.test(dataText);
  await page.reload();
  await waitForWorkspace(page, '线缆穿杆 E2E 演示数据集');
  dataText = await page.locator('body').innerText();
  results.dataRefreshPersist = dataText.includes('线缆穿杆 E2E 演示数据集');
  results.dataDetailFields = false;
  const detailBtn = page.getByRole('button', { name: '详情' }).first();
  if (await detailBtn.count()) {
    await detailBtn.click();
    await page.waitForTimeout(600);
    const drawerText = await page.locator('body').innerText();
    results.dataDetailFields =
      drawerText.includes('MuJoCo') &&
      drawerText.includes('composite_cable') &&
      drawerText.includes('90%') &&
      /expert_data\.npz/.test(drawerText) &&
      /expert_data\.hdf5/.test(drawerText) &&
      /manifest\.json/.test(drawerText);
  }

  // E(2): 评测中心刷新保留
  await page.goto(`${BASE}/workspace/evaluation`);
  await waitForWorkspace(page, '单臂线缆穿杆 · scripted 评测');
  let evalText = await page.locator('body').innerText();
  results.evalVisible = evalText.includes('单臂线缆穿杆 · scripted 评测');
  await page.reload();
  await waitForWorkspace(page, '单臂线缆穿杆 · scripted 评测');
  evalText = await page.locator('body').innerText();
  results.evalRefreshPersist = evalText.includes('单臂线缆穿杆 · scripted 评测');

  // E(3): 报告页 evalId 刷新
  await page.goto(`${BASE}/workspace/evaluation/report?evalId=${DEMO_EVAL_ID}`);
  await waitForWorkspace(page, '评测报告');
  let reportText = await page.locator('body').innerText();
  results.reportOpens = reportText.includes('评测报告') && reportText.includes('单臂线缆穿杆');
  await page.reload();
  await waitForWorkspace(page, '评测报告');
  reportText = await page.locator('body').innerText();
  results.reportRefreshPersist = reportText.includes('评测报告') && reportText.includes('单臂线缆穿杆');

  // E(4): 回放页 videoExists / videoPath
  await page.goto(`${BASE}/workspace/replay?evalId=${DEMO_EVAL_ID}`);
  await waitForWorkspace(page, '回放与视频');
  let replayText = await page.locator('body').innerText();
  results.replayVideo =
    replayText.includes('videoExists') &&
    replayText.includes('true') &&
    replayText.includes('replay.mp4');
  await page.reload();
  await waitForWorkspace(page, '回放与视频');
  replayText = await page.locator('body').innerText();
  results.replayRefreshPersist = replayText.includes('replay.mp4');

  // D: 快速演示配置提示
  await page.goto(`${BASE}/workspace/resources/task-templates`);
  await waitForWorkspace(page, '单臂线缆穿杆');
  await page.locator('table tbody tr').filter({ hasText: '单臂线缆穿杆' }).getByRole('button', { name: '生成数据' }).click();
  await page.waitForTimeout(800);
  const modalText = await page.locator('body').innerText();
  results.quickDemoHint =
    modalText.includes('演示提示') &&
    modalText.includes('完整 HDF5 生成耗时较长') &&
    modalText.includes('快速演示：3 ep，无 HDF5') &&
    modalText.includes('极速演示：1 ep，无 HDF5');

  await browser.close();
  console.log(JSON.stringify(results, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
