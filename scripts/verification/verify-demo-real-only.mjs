// Verify that demo workflows use real backend records only.
import { chromium } from 'playwright';

const BASE = 'http://127.0.0.1:3001';
const API = 'http://127.0.0.1:8000/api';
const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';

async function apiLogin(request) {
  const sid = crypto.randomUUID();
  const loginResp = await request.post(`${API}/auth/login`, {
    headers: { 'X-Session-Id': sid },
    data: { username: LOGIN_USER, password: LOGIN_PASS },
  });
  if (!loginResp.ok()) throw new Error(`login failed: ${loginResp.status()}`);
  const loginJson = await loginResp.json();
  return { token: loginJson.data.access_token, sid };
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

  const { token, sid } = await apiLogin(page.request);
  await injectBrowserAuth(page, token, sid);

  await page.goto(`${BASE}/workspace/data`);
  await page.getByRole('heading', { name: /数据中心/i }).waitFor({ timeout: 60000 });

  const demoChecked = await page.getByRole('checkbox', { name: '显示示例数据' }).isChecked();
  const mockVisible = await page.getByText('示范数据集 A').isVisible().catch(() => false);
  const realVisible = await page.getByText('dac_gen_').first().isVisible().catch(() => false);

  report.port = 3001;
  report.NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY =
    process.env.NEXT_PUBLIC_WORKSPACE_DEMO_REAL_ONLY === 'true';
  report.demoCheckboxCheckedByDefault = demoChecked;
  report.mockSeedVisible = mockVisible;
  report.realJobVisible = realVisible;

  console.log(JSON.stringify(report, null, 2));
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
