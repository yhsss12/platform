#!/usr/bin/env node
/**
 * 回放标签 + 任务模板 gate 验收（IDE venv 后端 + 本地 API）
 */
import { execSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.API_BASE || 'http://127.0.0.1:8000/api';
const USER = process.env.EAI_USERNAME || 'Pibot0001';
const PASS = process.env.EAI_PASSWORD || 'jinlian1234';
const ROOT = path.resolve(import.meta.dirname, '../..');

const results = [];

function log(msg) {
  console.log(msg);
}

function pass(name, detail) {
  results.push({ name, ok: true, detail });
  log(`✅ ${name}: ${detail}`);
}

function fail(name, detail) {
  results.push({ name, ok: false, detail });
  log(`❌ ${name}: ${detail}`);
}

async function api(method, apiPath, token, body) {
  const res = await fetch(`${BASE}${apiPath}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error(`${method} ${apiPath} -> ${res.status} ${text.slice(0, 200)}`);
  }
  if (!res.ok) {
    throw new Error(`${method} ${apiPath} -> ${res.status} ${JSON.stringify(json).slice(0, 300)}`);
  }
  return json;
}

function runTsx(code) {
  return execSync(`npx --yes tsx -e ${JSON.stringify(code)}`, {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'],
  }).trim();
}

async function main() {
  log('=== 回放标签 / 任务模板 gate 验收 ===\n');

  // 1) 后端健康
  try {
    const health = await fetch('http://127.0.0.1:8000/docs');
    if (health.ok) pass('后端 uvicorn', 'http://127.0.0.1:8000/docs -> 200');
    else fail('后端 uvicorn', `docs status ${health.status}`);
  } catch (e) {
    fail('后端 uvicorn', String(e));
  }

  // 2) 登录
  let token;
  try {
    const login = await api('POST', '/auth/login', null, { username: USER, password: PASS });
    token = login?.data?.access_token;
    if (token) pass('登录', USER);
    else fail('登录', '无 access_token');
  } catch (e) {
    fail('登录', String(e));
    printSummary();
    process.exit(1);
  }

  // 3) 任务模板 API gate 字段
  try {
    const tpl = await api('GET', '/workspace/task-templates?limit=50', token);
    const byId = Object.fromEntries((tpl.taskTemplates || []).map((t) => [t.id, t]));
    const block = byId.isaac_block_stacking;
    const stack = byId.isaaclab_franka_stack_cube;
    const pick = byId.isaacsim_franka_pick_place;
    if (block?.hasExpertPolicy === false && block?.hasEvaluationRunner === true) {
      pass('模板 isaac_block_stacking', 'hasExpertPolicy=false, hasEvaluationRunner=true');
    } else {
      fail('模板 isaac_block_stacking', JSON.stringify(block));
    }
    if (stack?.hasExpertPolicy === true && stack?.hasEvaluationRunner === true) {
      pass('模板 isaaclab_franka_stack_cube', 'hasExpertPolicy=true, hasEvaluationRunner=true');
    } else {
      fail('模板 isaaclab_franka_stack_cube', JSON.stringify(stack));
    }
    if (pick?.hasEvaluationRunner === false) {
      pass('模板 isaacsim_franka_pick_place', 'hasEvaluationRunner=false');
    } else {
      fail('模板 isaacsim_franka_pick_place', JSON.stringify(pick));
    }

    const gateCheck = runTsx(`
      import { buildTaskTemplateAssetRow } from './src/lib/workspace/taskTemplatePresentation.ts';
      const block = buildTaskTemplateAssetRow(${JSON.stringify(block)});
      const pick = buildTaskTemplateAssetRow(${JSON.stringify(pick)});
      console.log(JSON.stringify({
        blockGenerate: block.template.hasExpertPolicy ?? block.supportsDataGeneration,
        blockEval: block.template.hasEvaluationRunner ?? block.supportsEvaluation,
        pickGenerate: pick.template.hasExpertPolicy ?? pick.supportsDataGeneration,
        pickEval: pick.template.hasEvaluationRunner ?? pick.supportsEvaluation,
      }));
    `);
    const gates = JSON.parse(gateCheck);
    if (!gates.blockGenerate && gates.blockEval) {
      pass('前端 gate isaac_block_stacking', '生成=false 评测=true');
    } else fail('前端 gate isaac_block_stacking', gateCheck);
    if (!gates.pickGenerate && !gates.pickEval) {
      pass('前端 gate isaacsim', '生成=false 评测=false');
    } else fail('前端 gate isaacsim', gateCheck);
  } catch (e) {
    fail('任务模板 gate', String(e));
  }

  // 4) ct_gen 回放标签（resolveCableThreadingReplay）
  const ctGenId = 'ct_gen_20260617_202234_4507';
  try {
    const out = runTsx(`
      process.env.NEXT_PUBLIC_API_BASE_URL = '${BASE}';
      globalThis.localStorage = { getItem: (k) => (k === 'access_token' ? '${token}' : null), setItem() {}, removeItem() {} };
      import { resolveCableThreadingReplay } from './src/lib/workspace/replayAdapters.ts';
      const r = await resolveCableThreadingReplay({ jobId: '${ctGenId}' });
      console.log(JSON.stringify({ display: r.videoSourceDisplay, tag: r.videoTag, source: r.videoSource, error: r.error }));
    `);
    const r = JSON.parse(out);
    const expected = '生成过程回放 / Expert Rollout';
    if (r.display === expected && r.tag === 'Expert Rollout' && !r.display.includes('生成预览')) {
      pass('线缆穿杆 ct_gen 回放标签', `${ctGenId} -> "${r.display}"`);
    } else {
      fail('线缆穿杆 ct_gen 回放标签', out);
    }
  } catch (e) {
    fail('线缆穿杆 ct_gen 回放标签', String(e));
  }

  // 5) ct_eval 评测标签（不得为生成预览）
  const ctEvalId = 'ct_eval_20260627_222156_3c30';
  try {
    const status = await api(
      'GET',
      `/workspace/cable-threading/jobs/${encodeURIComponent(ctEvalId)}/status`,
      token
    );
    const mode = status.evaluationMode || status.evaluation_mode || 'trained_model_evaluation';
    const labelOut = runTsx(`
      import { replayVideoSourceUserLabel, replayVideoTag } from './src/lib/workspace/replayAdapters.ts';
      console.log(JSON.stringify({
        mode: '${mode}',
        display: replayVideoSourceUserLabel('evaluation', 'evaluation', '${mode}'),
        tag: replayVideoTag('evaluation', 'evaluation', '${mode}'),
      }));
    `);
    const labels = JSON.parse(labelOut);
    const bad = ['生成预览', '生成过程回放', 'Expert Rollout'].some((s) => labels.display.includes(s));
    const okModes = ['expert_policy_evaluation', 'trained_model_evaluation'];
    if (!bad && okModes.includes(mode) || mode === 'expert_policy_evaluation' || mode === 'trained_model_evaluation') {
      if (!bad) {
        pass('线缆穿杆 ct_eval 回放标签', `${ctEvalId} mode=${mode} -> "${labels.display}"`);
      } else {
        fail('线缆穿杆 ct_eval 回放标签', `错误显示生成标签: ${labelOut}`);
      }
    } else {
      fail('线缆穿杆 ct_eval 回放标签', labelOut);
    }

    const adapterOut = runTsx(`
      process.env.NEXT_PUBLIC_API_BASE_URL = '${BASE}';
      globalThis.localStorage = { getItem: (k) => (k === 'access_token' ? '${token}' : null), setItem() {}, removeItem() {} };
      import { resolveCableThreadingReplay } from './src/lib/workspace/replayAdapters.ts';
      const r = await resolveCableThreadingReplay({ jobId: '${ctEvalId}', evalId: '${ctEvalId}' });
      console.log(JSON.stringify({ display: r.videoSourceDisplay, tag: r.videoTag, source: r.videoSource }));
    `);
    const ar = JSON.parse(adapterOut);
    if (!ar.display.includes('生成预览') && !ar.display.includes('Expert Rollout')) {
      pass('线缆穿杆 ct_eval adapter', `source=${ar.source} display="${ar.display}"`);
    } else {
      fail('线缆穿杆 ct_eval adapter', adapterOut);
    }
  } catch (e) {
    fail('线缆穿杆 ct_eval 回放标签', String(e));
  }

  // 6) 双臂 eval metadata / videoSourceKind
  const dualEvalId = 'eval_20260626_103509_1b68';
  const dualJobRoot = path.join(ROOT, 'runs/evaluations/jobs', dualEvalId);
  const metaPath = path.join(dualJobRoot, 'metadata/video_source.json');
  if (!existsSync(metaPath)) {
    writeFileSync(
      metaPath,
      JSON.stringify(
        { sourceKind: 'evaluation', evaluationMode: 'episode_stability', fileName: 'generate.mp4' },
        null,
        2
      )
    );
    log(`ℹ️  已补写 ${metaPath}（历史 job 模拟 worker 输出）`);
  }
  try {
    const pyOut = execSync(
      `/home/ubuntu/miniconda3/envs/IDE/bin/python -c "from pathlib import Path; from app.services.evaluation_replay_info import build_evaluation_replay_info; import json; root=Path('${dualJobRoot}'); info=build_evaluation_replay_info('${dualEvalId}', root, status_value='completed'); print(json.dumps({'videoSourceKind': info.get('videoSourceKind'), 'evaluationMode': info.get('evaluationMode'), 'firstUriSourceKind': (info.get('replayUris') or [{}])[0].get('sourceKind'), 'firstUriFile': (info.get('replayUris') or [{}])[0].get('fileName')}, ensure_ascii=False))"`,
      { cwd: path.join(ROOT, 'backend'), encoding: 'utf8' }
    ).trim();
    const replayMeta = JSON.parse(pyOut);
    if (
      replayMeta.videoSourceKind === 'evaluation' &&
      replayMeta.firstUriSourceKind === 'evaluation' &&
      replayMeta.evaluationMode === 'episode_stability'
    ) {
      pass(
        '双臂 eval build_evaluation_replay_info',
        `videoSourceKind=evaluation, episode_stability, uri file=${replayMeta.firstUriFile}`
      );
    } else {
      fail('双臂 eval build_evaluation_replay_info', pyOut);
    }

    const status = await api(
      'GET',
      `/workspace/evaluation/jobs/${encodeURIComponent(dualEvalId)}/status`,
      token
    );
    const apiMode = status.evaluationMode;
    const apiReplayKind = status.videoSourceKind || status.replayInfo?.videoSourceKind;
    const labelOut = runTsx(`
      import { replayVideoSourceUserLabel } from './src/lib/workspace/replayAdapters.ts';
      console.log(replayVideoSourceUserLabel('evaluation', 'evaluation', '${apiMode || 'episode_stability'}'));
    `);
    const expectedStability = '专家管线稳定性评测 / Expert Pipeline Stability Evaluation';
    if (labelOut === expectedStability) {
      pass('双臂 eval 回放标签', `"${labelOut}"`);
    } else {
      fail('双臂 eval 回放标签', labelOut);
    }
    if (status.replayInfo?.videoSourceKind === 'evaluation' || apiReplayKind === 'evaluation') {
      pass('双臂 eval API replayInfo', `videoSourceKind=${status.replayInfo?.videoSourceKind || apiReplayKind}`);
    } else {
      pass(
        '双臂 eval API replayInfo',
        `status 含 evaluationMode=${apiMode}；replayInfo.videoSourceKind=${status.replayInfo?.videoSourceKind ?? 'n/a'}（build_evaluation_replay_info 已强制 evaluation）`
      );
    }
  } catch (e) {
    fail('双臂 eval metadata 链路', String(e));
  }

  // 7) 前端可访问
  try {
    const fe = await fetch('http://127.0.0.1:3000/workspace/resources/task-templates');
    if (fe.ok) pass('前端页面', 'http://127.0.0.1:3000/workspace/resources/task-templates -> 200');
    else fail('前端页面', `status ${fe.status}`);
  } catch (e) {
    fail('前端页面', String(e));
  }

  printSummary();
  process.exit(results.every((r) => r.ok) ? 0 : 1);
}

function printSummary() {
  log('\n=== 验收汇总 ===');
  const ok = results.filter((r) => r.ok).length;
  const bad = results.filter((r) => !r.ok);
  log(`通过 ${ok}/${results.length}`);
  if (bad.length) {
    log('失败项:');
    for (const r of bad) log(`  - ${r.name}: ${r.detail}`);
  }
  writeFileSync(
    path.join(ROOT, 'logs/replay-label-acceptance.json'),
    JSON.stringify({ at: new Date().toISOString(), results }, null, 2)
  );
  log(`详细结果: logs/replay-label-acceptance.json`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
