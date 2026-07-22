/**
 * Continuation: verify data center / replay / report after a completed browser job.
 * Injects sessionStorage state equivalent to console replaceMockDataItem output.
 */
import { test, expect } from 'playwright/test';

const LOGIN_USER = 'Pibot0001';
const LOGIN_PASS = 'jinlian1234';
const MOCK_FLOW_KEY = 'epi_workspace_mock_flow_v1';
const JOB_ID = process.env.DAC_UI_JOB_ID ?? 'dac_gen_20260611_152506_c83a';

test.setTimeout(120_000);

test('Dual-Arm post-console UI checks', async ({ page, request }) => {
  const loginRes = await request.post('http://127.0.0.1:8000/api/auth/login', {
    data: { username: LOGIN_USER, password: LOGIN_PASS },
  });
  const { data } = (await loginRes.json()) as { data: { access_token: string } };
  const token = data.access_token;

  const statusRes = await request.get(
    `http://127.0.0.1:8000/api/workspace/dual-arm-cable/jobs/${JOB_ID}/status`,
    { headers: { Authorization: `Bearer ${token}` } }
  );
  const status = await statusRes.json();

  await page.addInitScript(() => {
    localStorage.setItem('epi_locale', 'zh-CN');
  });
  await page.goto('/login');
  await page.locator('#login-username').fill(LOGIN_USER);
  await page.locator('#login-password').fill(LOGIN_PASS);
  await page.getByRole('button', { name: /登录/i }).click();
  await page.waitForURL((url) => !url.pathname.includes('/login'));

  await page.evaluate(
    ({ key, jobId, st }) => {
      const item = {
        id: jobId,
        name: `线缆整理数据_ui_${jobId.slice(-4)}`,
        taskId: 'dual_arm_cable_manipulation',
        taskName: '线缆整理',
        simulationId: jobId,
        sourceJobId: jobId,
        dataCategory: '示范数据',
        source: 'MuJoCo 生成',
        targetModelFormat: '—',
        dataVolume: '1 episode',
        size: '含 MP4',
        status: 'completed',
        generatedAt: new Date().toLocaleString('zh-CN'),
        creator: '当前用户',
        scene: '双臂桌面线缆整理工位',
        robot: 'Dual Franka FR3',
        simBackend: 'MuJoCo',
        taskType: 'dual_arm_cable_manipulation',
        dualArmMaxCables: st.maxCables ?? 1,
        dualArmSeed: 42,
        dualArmStretchMode: 'fixed_distance',
        dualArmReleaseMode: 'three_phase',
        dualArmEpisodeSuccess: st.episodeSuccess,
        dualArmSucceededCables: st.succeededCables,
        dualArmLeftContact: st.metrics?.left_contact,
        dualArmRightContact: st.metrics?.right_contact,
        dualArmStretchReached: st.metrics?.stretch_reached,
        dualArmSagM: st.metrics?.sag_m,
        dualArmSpanM: st.metrics?.span_m,
        dualArmFinalSagM: st.metrics?.final_sag_m,
        dualArmFinalSpanM: st.metrics?.final_span_m,
        episodeResultPath: st.resultPath,
        generateVideoPath: st.videoPath,
        generateVideoExists: st.videoExists,
        qualityStatus: '不可构建',
        contents: ['过程视频', '运行结果', '运行日志', '感知结果'],
      };
      sessionStorage.setItem(
        key,
        JSON.stringify({
          simulationRuns: [],
          activeSimulationRunId: null,
          extraDataItems: [item],
          extraEvaluationTasks: [],
          lastProcessEvalId: null,
          activeDataGenerationItemId: null,
          activeDataGenerationContext: null,
          cableThreadingGenerateRuns: [],
          activeCableThreadingRunId: null,
          dualArmCableGenerateRuns: [
            {
              jobId,
              dataItemId: `dac-pending-${jobId}`,
              status: 'completed',
              payload: { template: '线缆整理', seed: 42 },
              startedAt: new Date().toISOString(),
            },
          ],
          cableThreadingEvaluateRuns: [],
        })
      );
    },
    { key: MOCK_FLOW_KEY, jobId: JOB_ID, st: status }
  );

  await page.goto('/workspace/data');
  const row = page.locator('table tbody tr').filter({ hasText: /线缆整理数据_ui/ }).first();
  await expect(row).toBeVisible();
  const rowText = (await row.textContent()) ?? '';
  expect(rowText).toMatch(/任务数据/);
  expect(rowText).not.toMatch(/构建数据集|HDF5|successRate|runs/);

  await row.getByRole('button', { name: '详情' }).click();
  const drawer = page.locator('aside[role="dialog"]');
  await expect(drawer.getByRole('heading', { name: '数据详情' })).toBeVisible();
  const drawerText = (await drawer.textContent()) ?? '';
  expect(drawerText).toContain(JOB_ID);
  expect(drawerText).toContain('left_contact');
  expect(drawerText).toContain('episode_result.json');
  expect(drawerText).toContain('暂未生成 HDF5/Robomimic');
  expect(drawerText).not.toContain('构建数据集');

  await drawer.getByRole('link', { name: '回放' }).click();
  await page.waitForURL(new RegExp(`/workspace/replay.*jobId=${JOB_ID}`));
  await expect(page.getByText('线缆整理 · 过程回放')).toBeVisible();
  await page.waitForResponse((r) => r.url().includes(`/jobs/${JOB_ID}/video`) && r.status() === 200, {
    timeout: 60_000,
  });
  await expect(page.locator('video')).toBeVisible();
  await expect(page.getByText('线缆整理').first()).toBeVisible();
  const reportText = (await page.textContent('body')) ?? '';
  expect(reportText).toMatch(/episode_success|left_contact|final_span_m/);
  expect(reportText).not.toMatch(/successRate|endpoint_goal_error|thread_completion/);

  await page.goto('/workspace/training');
  const trainText = (await page.textContent('body')) ?? '';
  expect(trainText).not.toContain(JOB_ID);
});
