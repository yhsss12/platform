import { test, expect } from 'playwright/test';

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem('auth.sessionId', '00000000-0000-4000-8000-000000000000');
    window.sessionStorage.setItem('auth.access_token', 'e2e-token');
  });

  await page.route('**/api/auth/me', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, data: { id: 'u_1', account_id: 'acc_1', username: '测试用户', role: 'ADMIN' } }),
    });
  });

  await page.route('**/api/projects**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        data: {
          items: [
            {
              id: 'p_1',
              name: '项目1',
              description: null,
              tags: [],
              status: '进行中',
              owner_id: 'u_1',
              team_id: null,
              created_at: '2026-01-01T00:00:00.000Z',
              updated_at: '2026-01-02T00:00:00.000Z',
              viewer_is_project_member: true,
              viewer_is_project_owner: true,
              member_count: 1,
            },
          ],
          total: 1,
          stats: {},
        },
      }),
    });
  });
});

test('采集任务列表展示创建人并支持筛选与排序', async ({ page }) => {
  await page.route('**/api/tasks**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        data: [
          {
            id: 't_1',
            name: '任务A',
            status: 'PENDING',
            createdAt: '2026-01-01T00:00:00.000Z',
            updatedAt: '2026-01-02T00:00:00.000Z',
            projectId: 'p_1',
            deviceName: 'dev',
            episodeCount: 50,
            creatorUsername: '张三',
            creatorId: 'u_zs',
          },
          {
            id: 't_2',
            name: '任务B',
            status: 'PENDING',
            createdAt: '2026-01-03T00:00:00.000Z',
            updatedAt: '2026-01-03T00:00:00.000Z',
            projectId: 'p_1',
            deviceName: 'dev',
            episodeCount: 50,
            creatorUsername: '李四',
            creatorId: 'u_ls',
          },
        ],
      }),
    });
  });

  await page.route('**/api/jobs**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, data: [] }),
    });
  });

  await page.goto('/collect/tasks');

  await expect(page.getByRole('columnheader', { name: '创建人' })).toBeVisible();
  await expect(page.getByRole('cell', { name: '张三' })).toBeVisible();
  await expect(page.getByRole('cell', { name: '李四' })).toBeVisible();

  await page.getByRole('combobox', { name: '创建人' }).selectOption({ label: '张三' });
  await expect(page.getByRole('cell', { name: '张三' })).toBeVisible();
  await expect(page.getByRole('cell', { name: '李四' })).toHaveCount(0);

  await page.getByRole('combobox', { name: '排序' }).selectOption('creatorDesc');
  await page.getByRole('combobox', { name: '创建人' }).selectOption('');
  const firstRowCreator = page.locator('table tbody tr').first().locator('td').nth(4);
  await expect(firstRowCreator).toHaveText(/张三|李四/);
});

test('50条任务采集2条后刷新仍保持进度不归零', async ({ page }) => {
  await page.route('**/api/tasks**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        data: [
          {
            id: 't_1',
            name: '任务A',
            status: 'PENDING',
            createdAt: '2026-01-01T00:00:00.000Z',
            updatedAt: '2026-01-02T00:00:00.000Z',
            projectId: 'p_1',
            deviceName: 'dev',
            episodeCount: 50,
            creatorUsername: '张三',
          },
        ],
      }),
    });
  });

  await page.route('**/api/jobs**', async (route) => {
    const url = new URL(route.request().url());
    const taskId = url.searchParams.get('task_id');
    const payload = {
      ok: true,
      data: [
        {
          id: 'j_1',
          type: 'collection',
          status: 'RUNNING',
          taskId: taskId || 't_1',
          target: { taskId: taskId || 't_1' },
          collection_quantity: 50,
          completed_count: 2,
          progress: { percent: 4, current: 2, total: 50 },
          updatedAt: '2026-01-02T00:00:00.000Z',
        },
      ],
    };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) });
  });

  await page.goto('/collect/jobs?taskId=t_1');
  await expect(page.getByRole('cell', { name: /2\s*\/\s*50/ }).first()).toBeVisible();

  await page.reload();
  await expect(page.getByRole('cell', { name: /2\s*\/\s*50/ }).first()).toBeVisible();
});
