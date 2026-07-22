/**
 * Round-4 i18n acceptance with mocked API (real Chromium, no backend required).
 */
import { chromium } from 'playwright';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const BASE = process.env.BASE_URL || 'http://127.0.0.1:3000';
const OUT_DIR = join(process.cwd(), 'artifacts', 'i18n-verify-output');

const MOCK_USER = {
  id: 'mock-super-1',
  account_id: 'Pibot0001',
  username: 'Pibot',
  role: 'SUPER_ADMIN',
  is_active: true,
  created_at: '2024-01-01T00:00:00Z',
};

const MOCK_BATCH = {
  batchId: 'batch-mock-1',
  taskName: null,
  sourceFormat: null,
  targetFormat: 'HDF5',
  projectId: 'proj-1',
  projectName: 'Demo Project',
  creatorName: 'Pibot',
  totalCount: 10,
  successCount: 10,
  failedCount: 0,
  runningCount: 0,
  pendingCount: 0,
  progressPercent: 100,
  overallStatus: 'SUCCESS',
  createdAt: '2024-06-01T10:00:00Z',
  updatedAt: '2024-06-01T11:00:00Z',
};

const MOCK_CHILDREN = [
  {
    jobId: 'job-1',
    sourceFileName: 'episode_001.mcap',
    outputFileName: 'episode_001.hdf5',
    itemStatus: 'succeeded',
    itemStage: 'validate',
    errorMessage: null,
    createdAt: '2024-06-01T10:00:00Z',
    updatedAt: '2024-06-01T10:30:00Z',
  },
  {
    jobId: 'job-2',
    sourceFileName: 'episode_002.mcap',
    outputFileName: 'episode_002.hdf5',
    itemStatus: 'succeeded',
    itemStage: 'write',
    errorMessage: '',
    createdAt: '2024-06-01T10:00:00Z',
    updatedAt: '2024-06-01T10:35:00Z',
  },
];

const MOCK_USERS_LIST = {
  items: [
    {
      id: 'u-member',
      account_id: 'M001',
      username: 'Member User',
      role: 'USER',
      is_active: true,
      created_at: '2024-01-01T00:00:00Z',
      team_name: 'Platform',
    },
    {
      id: 'u-owner',
      account_id: 'O001',
      username: 'Owner User',
      role: 'OWNER',
      is_active: true,
      created_at: '2024-01-02T00:00:00Z',
      team_name: 'Platform',
    },
    {
      id: 'u-admin',
      account_id: 'A001',
      username: 'Admin User',
      role: 'ADMIN',
      is_active: true,
      created_at: '2024-01-03T00:00:00Z',
      team_name: 'Platform',
    },
  ],
  total: 3,
  page: 1,
  page_size: 20,
};

const MOCK_TASK = {
  id: 'task-mock-1',
  name: 'Collect Task Demo',
  description: 'Demo',
  status: 'PENDING',
  owner: 'Pibot',
  deviceId: 'dev-1',
  deviceName: 'Robot-A',
  projectId: 'proj-1',
  episodeCount: 5,
  durationSec: 30,
  storagePath: '/data/rosbags',
  storageTypes: ['local'],
  createdAt: '2024-06-01T09:00:00Z',
  updatedAt: '2024-06-01T09:00:00Z',
};

const MOCK_JOB = {
  id: 'job-mock-1',
  taskId: 'task-mock-1',
  jobNumber: 'JOB-0001',
  collector: 'Pibot',
  deviceId: 'dev-1',
  deviceName: 'Robot-A',
  collectionQuantity: 5,
  status: 'RUNNING',
  progress: { current: 2, total: 5, percent: 40 },
  createdAt: '2024-06-01T10:00:00Z',
  updatedAt: '2024-06-01T10:15:00Z',
};

const KEY_RE = /(convertRunPage|collectCreatePage|adminUsersPage|collectJobsPage)\.[a-zA-Z0-9_]+/g;
const CJK_RE = /[\u4e00-\u9fff]/;

const EXPECT = {
  'zh-CN': {
    convertTitle: '转换任务详情',
    convertProgress: '执行进度',
    convertTableSource: '源文件',
    convertNone: '无',
    convertDetails: '详情',
    convertBatchName: '批量转换',
    convertUnit: '条',
    collectDevice: '设备',
    collectOwner: '任务负责人',
    collectProject: '所属项目',
    collectCount: '数量',
    collectDuration: '时长',
    collectUnitItems: '条',
    collectUnitSec: '秒',
    collectNotice: '注意事项',
    collectNotice1: '请确保设备连接正常',
    usersRole: '角色',
    usersRoleMember: '成员',
    usersRoleOwner: '负责人',
    usersRoleAdmin: '管理员',
    jobsBack: '返回',
  },
  en: {
    convertTitle: 'Conversion task details',
    convertProgress: 'Execution progress',
    convertTableSource: 'Source file',
    convertNone: 'None',
    convertDetails: 'Details',
    convertBatchName: 'Batch conversion',
    convertUnit: 'items',
    collectDevice: 'Device',
    collectOwner: 'Task owner',
    collectProject: 'Project',
    collectCount: 'Count',
    collectDuration: 'Duration',
    collectUnitItems: 'item',
    collectUnitSec: 'sec',
    collectNotice: 'Notice',
    collectNotice1: 'Ensure the device is connected',
    usersRole: 'Role',
    usersRoleMember: 'Member',
    usersRoleOwner: 'Owner',
    usersRoleAdmin: 'Admin',
    jobsBack: 'Back',
  },
  sv: {
    convertTitle: 'Konverteringsuppgiftsdetaljer',
    convertProgress: 'Körningsframsteg',
    convertTableSource: 'Källfil',
    convertNone: 'Ingen',
    convertDetails: 'Detaljer',
    convertBatchName: 'Batchkonvertering',
    convertUnit: 'poster',
    collectDevice: 'Enhet',
    collectOwner: 'Uppgiftsansvarig',
    collectProject: 'Projekt',
    collectCount: 'Antal',
    collectDuration: 'Varaktighet',
    collectUnitItems: 'st',
    collectUnitSec: 'sek',
    collectNotice: 'Observera',
    collectNotice1: 'Säkerställ att enheten är ansluten',
    usersRole: 'Roll',
    usersRoleMember: 'Medlem',
    usersRoleOwner: 'Ansvarig',
    usersRoleAdmin: 'Administratör',
    jobsBack: 'Tillbaka',
  },
};

function findKeys(text) {
  return [...new Set(text.match(KEY_RE) || [])];
}

function cjkSnippets(text, max = 5) {
  const m = text.match(/[\u4e00-\u9fff][^\n]{0,30}/g);
  return m ? [...new Set(m)].slice(0, max) : [];
}

async function installMocks(page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace(/^\/api/, '');
    const method = route.request().method();

    const json = (body, status = 200) =>
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
      });

    if (path === '/auth/me' && method === 'GET') {
      return json(MOCK_USER);
    }
    if (path === '/auth/login' && method === 'POST') {
      return json({
        access_token: 'mock-access-token',
        refresh_token: 'mock-refresh-token',
        token_type: 'bearer',
        role: 'SUPER_ADMIN',
        account_id: MOCK_USER.account_id,
        username: MOCK_USER.username,
      });
    }
    if (path === '/auth/refresh' && method === 'POST') {
      return json({ access_token: 'mock-access-token', token_type: 'bearer' });
    }
    if (path === '/conversion/batches' && method === 'GET') {
      return json([MOCK_BATCH]);
    }
    if (path.startsWith('/conversion/batches/') && method === 'GET') {
      return json({ batch: MOCK_BATCH, children: MOCK_CHILDREN });
    }
    if (path.startsWith('/users') && method === 'GET') {
      if (path.includes('team-options')) {
        return json({ items: [{ id: 'team-1', name: 'Platform', code: 'PF' }], total: 1 });
      }
      return json(MOCK_USERS_LIST);
    }
    if (path === '/tasks' && method === 'GET') {
      return json({ ok: true, data: [MOCK_TASK] });
    }
    if (path.startsWith('/tasks/') && method === 'GET') {
      return json({ ok: true, data: MOCK_TASK });
    }
    if (path.startsWith('/jobs') && method === 'GET') {
      return json({ ok: true, data: [MOCK_JOB] });
    }
    if (path === '/devices' && method === 'GET') {
      return json({
        ok: true,
        data: [
          {
            id: 'dev-1',
            name: 'Robot-A',
            device_type: 'ROS2',
            status: 'DISCONNECTED',
            hardware_uuid: 'hw-1',
            agent_ip: '',
            agent_port: 0,
          },
        ],
      });
    }
    if (path.startsWith('/projects') && method === 'GET') {
      if (path.includes('permissions-context')) {
        return json({ ok: true, data: { team_admin_team_ids: null } });
      }
      return json({
        ok: true,
        data: {
          items: [
            {
              id: 'proj-1',
              name: 'Demo Project',
              description: '',
              tags: [],
              status: 'active',
              owner_id: 'u-owner',
              created_at: '2024-01-01T00:00:00Z',
              updated_at: '2024-01-01T00:00:00Z',
            },
          ],
          total: 1,
        },
      });
    }
    if (path.includes('/fs/') || path.includes('/online-agents')) {
      return json({ ok: true, data: [] });
    }

    return json({ ok: true, data: null });
  });
}

async function seedAuthAndLocale(page, locale) {
  await page.addInitScript(
    ({ loc, token }) => {
      localStorage.setItem('epi_locale', loc);
      sessionStorage.setItem('auth.access_token', token);
      sessionStorage.setItem('auth.refresh_token', 'mock-refresh');
      sessionStorage.setItem('auth.sessionId', 'mock-session-id-0001');
    },
    { loc: locale, token: 'mock-access-token' }
  );
}

async function switchLocale(page, localeId) {
  const map = {
    'zh-CN': /简体中文/,
    en: /^English$/,
    sv: /^Svenska$/,
  };
  const triggers = page.locator('button').filter({ hasText: /简体中文|English|Svenska/ });
  if ((await triggers.count()) === 0) return false;
  await triggers.first().click();
  await page.waitForTimeout(250);
  const item = page.getByRole('menuitem').filter({ hasText: map[localeId] });
  if ((await item.count()) > 0) {
    await item.first().click();
  } else {
    await page.locator('button').filter({ hasText: map[localeId] }).last().click();
  }
  await page.waitForTimeout(500);
  return true;
}

async function bootstrapSession(page, locale) {
  await seedAuthAndLocale(page, locale);
  await installMocks(page);
  await page.goto(`${BASE}/overview`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2500);
  if (page.url().includes('/login')) {
    throw new Error('Still redirected to login after mock auth');
  }
}

async function verifyRefresh(page, locale, bag) {
  const stored = await page.evaluate(() => localStorage.getItem('epi_locale'));
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2000);
  const stored2 = await page.evaluate(() => localStorage.getItem('epi_locale'));
  const pass = stored === locale && stored2 === locale;
  bag.push({ check: 'refresh-persist', locale, pass, note: pass ? `epi_locale=${stored2}` : `want ${locale}, got ${stored2}` });
}

async function verifyConvert(page, locale, bag, screenshot) {
  const exp = EXPECT[locale];
  await page.goto(`${BASE}/convert`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2000);
  const listRow = page.locator('table tbody tr').first();
  const listText = (await listRow.count()) > 0 ? await listRow.innerText() : '';
  const listIssues = [];
  if (locale !== 'zh-CN') {
    if (CJK_RE.test(listText)) listIssues.push(`task name CJK: ${listText.slice(0, 80)}`);
    if (!listText.includes(exp.convertBatchName) || !listText.includes(exp.convertUnit)) {
      listIssues.push('batch name template not translated in list row');
    }
  } else if (listText && (!listText.includes('批量转换') || !listText.includes('条'))) {
    listIssues.push('zh batch template missing in list row');
  }

  const btn = page.getByRole('button', { name: new RegExp(exp.convertDetails, 'i') }).first();
  if ((await btn.count()) === 0) listIssues.push('Details button missing');
  await btn.click();
  await page.waitForTimeout(1200);

  const drawer = await page.evaluate((title) => {
    const h = [...document.querySelectorAll('h2')].find((el) => el.textContent?.trim() === title);
    if (!h) return '';
    let panel = h.parentElement;
    while (panel && !panel.style?.maxHeight?.includes('vh') && panel.parentElement) {
      panel = panel.parentElement;
    }
    return panel?.innerText || '';
  }, exp.convertTitle);
  await page.screenshot({ path: screenshot, fullPage: true });

  const issues = [...listIssues];
  const keys = findKeys(drawer);
  if (!drawer.includes(exp.convertTitle)) issues.push(`title missing: ${exp.convertTitle}`);
  if (!drawer.includes(exp.convertProgress)) issues.push('progress section missing');
  if (!drawer.includes(exp.convertTableSource)) issues.push('table header missing');
  if (!drawer.includes(exp.convertNone)) issues.push('"none" missing');
  if (locale !== 'zh-CN' && CJK_RE.test(drawer)) issues.push(`drawer CJK: ${cjkSnippets(drawer).join(' | ')}`);
  if (keys.length) issues.push(`keys: ${keys.join(', ')}`);

  await page.keyboard.press('Escape');
  bag.push({
    check: 'convert-detail-drawer',
    locale,
    pass: issues.length === 0,
    note: issues.join('; ') || 'OK',
  });
}

async function verifyCollectCreate(page, locale, bag, screenshot) {
  const exp = EXPECT[locale];
  await page.goto(`${BASE}/collect/tasks`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2000);
  const newBtn = page.getByRole('button', { name: /新建|New|Ny/i }).first();
  await newBtn.click();
  await page.waitForTimeout(1500);
  const previewTitles = {
    'zh-CN': '执行预览',
    en: 'Execution preview',
    sv: 'Körningsförhandsgranskning',
  };
  const text = await page.evaluate((sectionTitle) => {
    const h = [...document.querySelectorAll('h4')].find((el) => el.textContent?.trim() === sectionTitle);
    if (!h) return '';
    let el = h.parentElement;
    while (el && el !== document.body) {
      if (el.style?.width === '320px') return el.innerText;
      el = el.parentElement;
    }
    return h.parentElement?.parentElement?.innerText || '';
  }, previewTitles[locale]);
  await page.screenshot({ path: screenshot, fullPage: true });
  const issues = [];
  for (const token of [
    exp.collectDevice,
    exp.collectOwner,
    exp.collectProject,
    exp.collectCount,
    exp.collectDuration,
    exp.collectUnitItems,
    exp.collectUnitSec,
    exp.collectNotice,
    exp.collectNotice1,
  ]) {
    if (!text.includes(token)) issues.push(`missing "${token}"`);
  }
  if (locale !== 'zh-CN' && CJK_RE.test(text)) issues.push(`CJK: ${cjkSnippets(text).join(' | ')}`);
  const keys = findKeys(text);
  if (keys.length) issues.push(`keys: ${keys.join(', ')}`);
  await page.keyboard.press('Escape');
  bag.push({ check: 'collect-create-preview', locale, pass: issues.length === 0, note: issues.join('; ') || 'OK' });
}

async function verifyUsers(page, locale, bag, screenshot) {
  const exp = EXPECT[locale];
  await page.goto(`${BASE}/admin/users`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2500);
  const text = await page.locator('body').innerText();
  await page.screenshot({ path: screenshot, fullPage: true });
  const issues = [];
  if (!text.includes(exp.usersRole)) issues.push('Role column header missing');
  if (!text.includes(exp.usersRoleMember)) issues.push('Member role missing');
  if (!text.includes(exp.usersRoleOwner)) issues.push('Owner role missing');
  if (!text.includes(exp.usersRoleAdmin)) issues.push('Admin role missing');
  if (locale !== 'zh-CN' && /成员|负责人|管理员/.test(text)) issues.push('Chinese role labels in table');
  const keys = findKeys(text);
  if (keys.length) issues.push(`keys: ${keys.join(', ')}`);

  const createBtn = page.getByRole('button', { name: /新建|New|Ny/i }).first();
  if ((await createBtn.count()) > 0) {
    await createBtn.click();
    await page.waitForTimeout(800);
    const modal = await page.locator('body').innerText();
    if (locale !== 'zh-CN' && /成员|负责人|管理员/.test(modal)) issues.push('Chinese in role dropdown');
    if (locale === 'en' && !/Member|Owner|Admin|Team admin/.test(modal)) issues.push('EN dropdown roles missing');
    await page.keyboard.press('Escape');
  }

  bag.push({ check: 'admin-users-role', locale, pass: issues.length === 0, note: issues.join('; ') || 'OK' });
}

async function verifyJobs(page, locale, bag, screenshot) {
  const exp = EXPECT[locale];
  await page.goto(`${BASE}/collect/jobs?taskId=task-mock-1`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2500);
  const text = await page.locator('body').innerText();
  await page.screenshot({ path: screenshot, fullPage: true });
  const issues = [];
  if (!text.includes(exp.jobsBack)) issues.push(`back button "${exp.jobsBack}" missing`);
  if (locale !== 'zh-CN' && CJK_RE.test(text)) issues.push(`CJK: ${cjkSnippets(text).join(' | ')}`);
  const keys = findKeys(text);
  if (keys.length) issues.push(`keys: ${keys.join(', ')}`);
  bag.push({ check: 'collect-jobs-page', locale, pass: issues.length === 0, note: issues.join('; ') || 'OK' });
}

function summarize(bag) {
  const screens = ['convert-detail-drawer', 'collect-create-preview', 'admin-users-role', 'collect-jobs-page', 'refresh-persist'];
  const locales = ['zh-CN', 'en', 'sv'];
  const out = {};
  for (const s of screens) {
    out[s] = {};
    for (const loc of locales) {
      const rows = bag.filter((r) => r.check === s && r.locale === loc);
      const refresh = bag.find((r) => r.check === 'refresh-persist' && r.locale === loc);
      const pass = rows.every((r) => r.pass);
      const keys = rows.some((r) => /keys:/i.test(r.note));
      out[s][loc] = {
        pass,
        refresh: refresh?.pass ?? false,
        keys,
        note: rows.map((r) => r.note).join('; '),
      };
    }
  }
  return out;
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const bag = [];

  try {
    for (const locale of ['zh-CN', 'en', 'sv']) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
      const page = await context.newPage();
      await bootstrapSession(page, locale);
      if (locale !== 'zh-CN') await switchLocale(page, locale);
      await verifyRefresh(page, locale, bag);
      await verifyConvert(page, locale, bag, join(OUT_DIR, `convert-${locale}.png`));
      await verifyCollectCreate(page, locale, bag, join(OUT_DIR, `collect-create-${locale}.png`));
      await verifyUsers(page, locale, bag, join(OUT_DIR, `users-${locale}.png`));
      await verifyJobs(page, locale, bag, join(OUT_DIR, `jobs-${locale}.png`));
      await context.close();
    }
  } catch (e) {
    bag.push({ check: 'fatal', locale: '-', pass: false, note: String(e?.message || e) });
  } finally {
    await browser.close();
  }

  const summary = summarize(bag);
  writeFileSync(join(OUT_DIR, 'bag.json'), JSON.stringify(bag, null, 2));
  writeFileSync(join(OUT_DIR, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({ bag, summary }, null, 2));
  const failed = bag.filter((r) => !r.pass);
  process.exit(failed.length ? 1 : 0);
}

main();
