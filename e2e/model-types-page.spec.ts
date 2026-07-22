import { test, expect } from 'playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';

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

test('model types page loads built-in cards without infinite loading', async ({ page, request }) => {
  const { token, sid, authHeaders } = await apiLogin(request);

  const apiResp = await request.get('http://127.0.0.1:8000/api/workspace/model-types', {
    headers: authHeaders,
  });
  expect(apiResp.status()).toBe(200);
  const apiBody = await apiResp.json();
  expect(Array.isArray(apiBody.modelTypes)).toBeTruthy();
  expect(apiBody.total).toBeGreaterThanOrEqual(4);

  await injectBrowserAuth(page, token, sid);
  await page.goto('/workspace/resources/model-types');

  await expect(page.getByText('正在加载模型类型…')).toBeHidden({ timeout: 5_000 });
  await expect(page.getByText('Robomimic BC')).toBeVisible({ timeout: 3_000 });
  await expect(page.getByText('ACT')).toBeVisible();
  await expect(page.getByText('Diffusion Policy')).toBeVisible();
  await expect(page.getByText('pi0')).toBeVisible();
});
