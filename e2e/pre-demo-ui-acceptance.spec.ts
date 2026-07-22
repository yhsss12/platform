/**
 * 演示前 UI 端到端验收（分步执行，避免单测超时）
 */
import { test, expect, type Page } from 'playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const API = 'http://127.0.0.1:8000';
const REPORT_PATH = path.join(process.cwd(), 'e2e-screenshots', 'acceptance-report.json');
const SCREENSHOT_DIR = path.join(process.cwd(), 'e2e-screenshots');

type CheckResult = { id: string; pass: boolean; detail: string };

const allResults: CheckResult[] = [];
const allFailures: { id: string; screenshot?: string; error: string }[] = [];

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

async function injectBrowserAuth(page: Page, token: string, sessionId: string) {
  await page.addInitScript(
    ({ token, sessionId }) => {
      localStorage.setItem('epi_locale', 'zh-CN');
      sessionStorage.setItem('auth.access_token', token);
      sessionStorage.setItem('auth.sessionId', sessionId);
    },
    { token, sessionId }
  );
}

function record(id: string, pass: boolean, detail: string) {
  allResults.push({ id, pass, detail });
}

function fail(id: string, error: string, screenshot?: string) {
  allFailures.push({ id, error, screenshot });
}

async function shot(page: Page, name: string) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const file = path.join(SCREENSHOT_DIR, `${name}.png`);
  try {
    await page.screenshot({ path: file, fullPage: true });
    return file;
  } catch {
    return undefined;
  }
}

async function openFirstDetailByStatus(page: Page, statusLabel: string) {
  await page.goto('/workspace/training');
  await expect(page.getByRole('heading', { name: '训练中心' })).toBeVisible({ timeout: 20_000 });
  const row = page.locator('tbody tr').filter({ hasText: statusLabel }).first();
  await expect(row).toBeVisible({ timeout: 15_000 });
  await row.getByRole('button', { name: '详情' }).click();
  await expect(page.getByText('训练过程性能')).toBeVisible({ timeout: 20_000 });
}

test.describe.configure({ mode: 'serial' });

test.describe('Pre-demo UI acceptance', () => {
  let token: string;
  let sid: string;
  let authHeaders: Record<string, string>;

  test.beforeAll(async ({ request }) => {
    const auth = await apiLogin(request);
    token = auth.token;
    sid = auth.sid;
    authHeaders = auth.authHeaders;
  });

  test.afterAll(async () => {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
    fs.writeFileSync(
      REPORT_PATH,
      JSON.stringify({ results: allResults, failures: allFailures, timestamp: new Date().toISOString() }, null, 2)
    );
  });

  test('1 training center list', async ({ page }) => {
    test.setTimeout(90_000);
    await injectBrowserAuth(page, token, sid);
    try {
      await page.goto('/workspace/training');
      await expect(page.getByRole('heading', { name: '训练中心' })).toBeVisible({ timeout: 20_000 });
      await expect(page.locator('table tbody tr').first()).toBeVisible({ timeout: 20_000 });
      record('1.1-load', true, '页面加载成功');

      const text = await page.locator('main').innerText();
      record('1.2-no-unknown', !/unknown数据集|unknown 数据/i.test(text), '数据集名检查');
      record(
        '1.3-status',
        ['已完成', '训练中', '等待中', '排队中'].filter((s) => text.includes(s)).length >= 2,
        text.match(/已完成|训练中|等待中|排队中/g)?.join(', ') ?? ''
      );
      record(
        '1.4-model-type',
        /Diffusion Policy|Robomimic BC|ACT/.test(await page.locator('table tbody').innerText()),
        '模型类型列有值'
      );

      const batchBtn = page.getByRole('button', { name: '批量删除' });
      record('1.5-batch-delete', (await batchBtn.isDisabled()) === true, '未选中时禁用');

      const runningRow = page.locator('tbody tr').filter({ hasText: '训练中' }).first();
      if (await runningRow.isVisible().catch(() => false)) {
        const listPct = await runningRow.locator('[role="progressbar"]').getAttribute('aria-valuenow');
        await runningRow.getByRole('button', { name: '详情' }).click();
        await expect(page.getByText('训练过程性能')).toBeVisible({ timeout: 15_000 });
        const detailPct = await page.locator('text=/\\d+%/').first().innerText().catch(() => '');
        record('1.6-progress', Boolean(listPct || detailPct), `列表=${listPct}, 详情=${detailPct}`);
        await page.keyboard.press('Escape');
      }
    } catch (e) {
      fail('1-training-list', String(e), await shot(page, '1-training-list'));
      throw e;
    }
  });

  test('2 create training modal and submit', async ({ page }) => {
    test.setTimeout(120_000);
    await injectBrowserAuth(page, token, sid);
    try {
      const beforeRes = await page.request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
      const beforeJobs = (await beforeRes.json()) as unknown[];

      await page.goto('/workspace/training?openCreate=1');
      const dialog = page.getByRole('dialog', { name: '新建训练任务' });
      await expect(dialog).toBeVisible({ timeout: 15_000 });

      const datasetSelect = dialog.locator('select').first();
      record('2.1-dataset-options', (await datasetSelect.locator('option').count()) > 0, '数据集可选');
      await datasetSelect.selectOption({ index: 0 });

      await expect(dialog.getByText('模型类型', { exact: true })).toBeVisible();
      await expect(dialog.getByText('训练节点', { exact: true })).toBeVisible();
      await expect(dialog.getByText('初始化权重', { exact: true })).toBeVisible();
      await expect(dialog.getByText('模型保存', { exact: true })).toBeVisible();
      record('2.2-fields', true, '表单字段正常');

      const seed = await dialog.locator('input[inputmode="numeric"]').first().inputValue();
      record('2.3-seed', /^\d+$/.test(String(seed)), `Seed=${seed}`);

      const dialogText = await dialog.innerText();
      record('2.4-no-advanced-settings', !dialogText.includes('高级设置'), '无高级结构设置');
      record('2.5-no-hidden-dims', !dialogText.includes('Hidden Dims'), '无 Hidden Dims 字段');
      record('2.6-save-final', dialogText.includes('保存最终模型'), '保留模型保存');

      const epochs = dialog.locator('input[type="number"]').first();
      await epochs.fill('600');
      record('2.7-epochs-600', (await epochs.inputValue()) === '600', 'Epochs 可填 600');

      await epochs.fill('1');
      await dialog.getByRole('button', { name: '创建训练任务' }).click();
      await expect(dialog).toBeHidden({ timeout: 45_000 });

      await page.waitForTimeout(2500);
      const afterRes = await page.request.get(`${API}/api/workspace/training/jobs`, { headers: authHeaders });
      const afterJobs = (await afterRes.json()) as unknown[];
      record('2.8-new-in-list', afterJobs.length > beforeJobs.length, `${beforeJobs.length}→${afterJobs.length}`);

      await page.reload();
      await expect(page.locator('table tbody tr').first()).toBeVisible({ timeout: 20_000 });
      record('2.9-list-visible', true, '创建后列表可见');
    } catch (e) {
      fail('2-create-training', String(e), await shot(page, '2-create-training'));
      throw e;
    }
  });

  test('3 training detail running and completed', async ({ page }) => {
    test.setTimeout(120_000);
    await injectBrowserAuth(page, token, sid);
    try {
      await openFirstDetailByStatus(page, '训练中');
      const t1 = await page.locator('body').innerText();
      await page.waitForTimeout(3500);
      const t2 = await page.locator('body').innerText();
      record('3.1-assets-stable', t2.includes('生成的模型资产'), '资产区域存在');
      record(
        '3.2-no-flicker',
        (t2.match(/加载中…/g)?.length ?? 0) <= 1,
        `loading 次数=${t2.match(/加载中…/g)?.length ?? 0}`
      );
      if (t2.includes('Final')) {
        record('3.3-final-generating', /生成中|等待/.test(t2), 'Final 生成中');
        record(
          '3.4-final-no-eval',
          (await page.getByRole('button', { name: '发起评测' }).count()) === 0,
          '训练中 Final 为禁用 span'
        );
      }
      await page.getByRole('button', { name: /日志/ }).click();
      record('3.5-log', true, '日志可展开');
      await page.locator('aside').getByRole('button', { name: '关闭', exact: true }).last().click();

      await openFirstDetailByStatus(page, '已完成');
      const done = await page.locator('body').innerText();
      record('3.6-progress-100', /100\s*%/.test(done), '100%');
      record('3.7-losses', /Final Loss|Best Loss/.test(done), 'Loss 指标');
      record('3.8-duration', done.includes('训练耗时'), '训练耗时');
      record('3.9-final', done.includes('Final') && !done.includes('暂无模型资产'), 'Final 资产');
      record(
        '3.10-eval-ready',
        await page.getByRole('button', { name: '发起评测' }).first().isEnabled().catch(() => false),
        '可发起评测'
      );
    } catch (e) {
      fail('3-training-detail', String(e), await shot(page, '3-training-detail'));
      throw e;
    }
  });

  test('4 eval from model asset', async ({ page }) => {
    test.setTimeout(90_000);
    await injectBrowserAuth(page, token, sid);
    try {
      await openFirstDetailByStatus(page, '已完成');
      const evalBtn = page.getByRole('button', { name: '发起评测' }).first();
      if (!(await evalBtn.isEnabled().catch(() => false))) {
        record('4.0-skipped', false, '无可评测 Final');
        return;
      }
      await evalBtn.click();
      await page.waitForURL(/\/workspace\/evaluation/, { timeout: 20_000 });
      await expect(page.getByText('新建评测任务')).toBeVisible({ timeout: 15_000 });
      const text = await page.locator('body').innerText();
      record('4.1-navigate', page.url().includes('/workspace/evaluation'), page.url());
      record('4.2-modal', true, '弹窗打开');
      record('4.3-model-mode', /模型评测|已训练模型/.test(text), '模型评测');
      record('4.4-no-expert-only', text.includes('已训练模型') || !text.includes('专家策略'), '未强制专家策略');
      const sel = page.locator('label:has-text("模型资产")').locator('..').locator('select').first();
      if (await sel.isVisible().catch(() => false)) {
        record('4.5-preselect', Boolean(await sel.inputValue()), await sel.inputValue());
      }
      await page.getByRole('button', { name: '取消' }).click();
    } catch (e) {
      fail('4-eval-nav', String(e), await shot(page, '4-eval-nav'));
      throw e;
    }
  });

  test('5 model assets page', async ({ page, request }) => {
    test.setTimeout(90_000);
    await injectBrowserAuth(page, token, sid);
    try {
      const assetsRes = await request.get(`${API}/api/workspace/model-assets`, { headers: authHeaders });
      const assets = ((await assetsRes.json()).modelAssets ?? []) as Array<Record<string, unknown>>;

      await page.goto('/workspace/resources/model-assets');
      await expect(page.getByRole('heading', { name: '模型资产' })).toBeVisible({ timeout: 20_000 });
      await expect(page.locator('table tbody tr').first()).toBeVisible({ timeout: 20_000 });
      record('5.1-load', assets.length > 0, `${assets.length} 条`);

      const text = await page.locator('main').innerText();
      const bestCount = (text.match(/Best/g) ?? []).length;
      record('5.2-best', bestCount <= 6, `Best ${bestCount} 次`);

      await expect(page.getByRole('button', { name: '批量删除' })).toBeVisible();
      await expect(page.getByRole('button', { name: '删除' }).first()).toBeVisible();
      record('5.3-delete-ui', true, '删除按钮可见');

      await page.locator('tbody tr').first().getByRole('checkbox').check();
      const batchBtn = page.getByRole('button', { name: '批量删除' });
      record('5.4-batch-enabled', !(await batchBtn.isDisabled()), '选中后可批量删除');
      await page.locator('tbody tr').first().getByRole('checkbox').uncheck();

      await page.getByRole('button', { name: '删除' }).first().click();
      await expect(page.getByText('删除模型资产')).toBeVisible({ timeout: 8000 });
      record('5.5-delete-dialog', true, '删除确认弹窗');
      await page.getByRole('button', { name: '取消' }).click();
    } catch (e) {
      fail('5-model-assets', String(e), await shot(page, '5-model-assets'));
      throw e;
    }
  });

  test('6 evaluation center', async ({ page }) => {
    test.setTimeout(90_000);
    await injectBrowserAuth(page, token, sid);
    try {
      const t0 = Date.now();
      await page.goto('/workspace/evaluation');
      await expect(page.getByRole('heading', { name: '评测中心' })).toBeVisible({ timeout: 20_000 });
      await expect(page.locator('table tbody tr').first()).toBeVisible({ timeout: 20_000 });
      record('6.1-load-speed', Date.now() - t0 < 12_000, `${Date.now() - t0}ms`);

      await page.waitForTimeout(2500);
      record('6.2-no-spinner', !(await page.getByText('加载评测任务…').isVisible().catch(() => false)), '无长 spinner');

      await page.getByRole('button', { name: '详情' }).first().click();
      await page.waitForTimeout(600);
      record('6.3-detail', true, '详情可开');
      await page.keyboard.press('Escape');

      await page.getByRole('button', { name: '新建任务' }).click();
      await expect(page.getByText('新建评测任务')).toBeVisible({ timeout: 10_000 });
      record('6.4-create-modal', true, '新建弹窗');

      const nameInput = page.locator('label:has-text("任务名称")').locator('..').locator('input').first();
      if (await nameInput.isVisible().catch(() => false)) {
        await nameInput.fill(`UI验收_${Date.now()}`);
      }
      const saveBtn = page.getByRole('button', { name: /保存|创建|提交/ }).first();
      if (await saveBtn.isVisible().catch(() => false)) {
        record('6.5-submit-available', await saveBtn.isEnabled(), '提交按钮可用');
      }
      await page.getByRole('button', { name: '取消' }).click();
    } catch (e) {
      fail('6-evaluation', String(e), await shot(page, '6-evaluation'));
      throw e;
    }
  });
});
