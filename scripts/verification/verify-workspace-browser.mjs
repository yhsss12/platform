// Browser-level workspace acceptance check.
import { chromium } from 'playwright';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const BASE = 'http://127.0.0.1:3001';
const API = 'http://127.0.0.1:8000/api';

async function apiLogin(request) {
  const sid = crypto.randomUUID();
  const loginResp = await request.post(`${API}/auth/login`, {
    headers: { 'X-Session-Id': sid },
    data: { username: LOGIN_USER, password: LOGIN_PASS },
  });
  if (!loginResp.ok()) throw new Error(`login failed: ${loginResp.status()}`);
  const loginJson = await loginResp.json();
  const token = loginJson.data.access_token;
  return { token, sid, authHeaders: { Authorization: `Bearer ${token}` } };
}

async function injectBrowserAuth(page, token, sessionId) {
  await page.addInitScript(({ token, sessionId }) => {
    localStorage.setItem('epi_locale', 'zh-CN');
    sessionStorage.setItem('auth.access_token', token);
    sessionStorage.setItem('auth.sessionId', sessionId);
  }, { token, sessionId });
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const report = {};

  const { token, sid, authHeaders } = await apiLogin(page.request);
  await injectBrowserAuth(page, token, sid);

  const evalList = await page.request.get(
    `${API}/workspace/jobs?jobType=evaluation&source=real&limit=10`,
    { headers: authHeaders }
  );
  const evalBody = await evalList.json();
  const sampleEvalId =
    evalBody.jobs?.find((j) => j.jobId.startsWith('eval_'))?.jobId ??
    evalBody.jobs?.find((j) => j.jobId.startsWith('ct_eval_'))?.jobId;
  report.sampleEvalJobId = sampleEvalId;

  await page.goto(`${BASE}/workspace/data`);
  await page.getByRole('heading', { name: /数据中心/i }).waitFor({ timeout: 60000 });
  report.dataCenterHasDacGen = await page.getByText('dac_gen_').first().isVisible().catch(() => false);
  report.dataCenterHasCtGen = await page.getByText('ct_gen_').first().isVisible().catch(() => false);
  await page.reload();
  await page.getByRole('heading', { name: /数据中心/i }).waitFor({ timeout: 60000 });
  report.dataCenterRefreshOk = report.dataCenterHasDacGen || report.dataCenterHasCtGen;

  await page.goto(`${BASE}/workspace/evaluation`);
  await page.getByRole('heading', { name: /评测中心/i }).waitFor({ timeout: 60000 });
  report.evalPageLoaded = true;
  report.evalCenterHasCtEval = await page
    .getByText(/ct_eval_\d/)
    .first()
    .isVisible({ timeout: 15000 })
    .catch(() => false);
  report.evalCenterHasEvalUnderscore = sampleEvalId
    ? await page.getByText(sampleEvalId).first().isVisible({ timeout: 15000 }).catch(() => false)
    : false;
  if (sampleEvalId) {
    await page.reload();
    await page.getByRole('heading', { name: /评测中心/i }).waitFor({ timeout: 60000 });
    report.evalCenterSurvivesRefresh = await page
      .getByText(sampleEvalId)
      .first()
      .isVisible({ timeout: 15000 })
      .catch(() => false);
  }

  await page.goto(`${BASE}/workspace/training`);
  await page.getByRole('heading', { name: /训练中心/i }).waitFor({ timeout: 60000 });
  report.trainingCenterHasTableRow = await page
    .locator('table tbody tr')
    .first()
    .isVisible()
    .catch(() => false);

  const replayJob = evalBody.jobs?.find((j) => j.jobId.startsWith('dac_gen_'))?.jobId
    ?? 'dac_gen_20260611_201433_856b';
  await page.goto(`${BASE}/workspace/replay?jobId=${replayJob}&taskType=dual_arm_cable_manipulation`);
  await page.waitForTimeout(3000);
  report.replayDualArmLoaded = !(await page.getByText(/未找到双臂线缆/i).isVisible().catch(() => false));

  const reportEval = sampleEvalId ?? 'eval_20260611_201433_4e30';
  await page.goto(
    `${BASE}/workspace/evaluation/report?evalId=${reportEval}&taskType=dual_arm_cable_manipulation`
  );
  await page.getByRole('heading', { name: '评测报告' }).first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(3000);
  report.reportHasAggregateSection = await page.getByText('聚合指标').isVisible().catch(() => false);
  report.reportHasBasicSection = await page
    .getByText(/基本信息|任务与资源配置/)
    .first()
    .isVisible()
    .catch(() => false);

  if (sampleEvalId?.startsWith('eval_')) {
    await page.goto(`${BASE}/workspace/evaluation`);
    await page.getByRole('heading', { name: /评测中心/i }).waitFor({ timeout: 60000 });
    const evalRow = page.locator('tr', { hasText: sampleEvalId });
    report.evalRowHasReplayLink = await evalRow
      .getByRole('link', { name: '回放' })
      .isVisible()
      .catch(() => false);
    report.evalRowHasReportLink = await evalRow
      .getByRole('link', { name: '报告' })
      .isVisible()
      .catch(() => false);
  }

  await page.goto(`${BASE}/workspace/data`);
  await page.getByRole('heading', { name: /数据中心/i }).waitFor({ timeout: 60000 });
  report.demoCheckboxCheckedDefault = await page
    .getByRole('checkbox', { name: '显示示例数据' })
    .isChecked();
  report.mockSeedVisible = await page.getByText('示范数据集 A').isVisible().catch(() => false);
  report.NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY = process.env.NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY === 'true';

  console.log(JSON.stringify(report, null, 2));
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
