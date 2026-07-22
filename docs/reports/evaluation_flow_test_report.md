# 评测流程全链路测试报告

> 项目：`/home/ubuntu/project/eai-idev2.1`  
> 测试日期：2026-06-24  
> 测试账号：`Pibot0001` / `jinlian1234`（**实际运行**：登录成功）  
> 说明：文中标注 **【实测】** 为命令/API/DB/runtime 实际结果；**【代码审查】** 为静态阅读推断，未在浏览器中逐像素验证。

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 操作系统 | Linux 5.15.0-179-generic |
| 前端 | Next.js 15.5.3 + React 19，端口 **3001** |
| 后端 | FastAPI + uvicorn，端口 **8000**，Python **IDE** conda 环境 |
| 数据库 | **PostgreSQL** `postgresql+asyncpg://admin:admin123@127.0.0.1:5432/eai_ide` |
| Alembic 版本 | **024_resource_definition_catalog (head)** 【实测】 |
| 服务状态 | `scripts/restart-all.sh` 已拉起前后端 【实测】 |

---

## 2. 启动方式与依赖

| # | 项 | 路径 / 命令 |
|---|---|---|
| 1 | 前端启动 | `npm run dev` → `next dev -p 3001`；或 `yarn dev`（`restart-all.sh` 自动检测） |
| 2 | 后端启动 | `cd backend && $EAI_PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` |
| 3 | 一键重启 | `bash scripts/restart-all.sh` |
| 4 | 数据库 | Docker compose `docker-compose.postgres-minio.yml` + `.env` DATABASE_URL |
| 5 | runtime_outputs 根目录 | `/home/ubuntu/project/eai-idev2.1/runtime_outputs/`（子目录：`cable_threading/`、`evaluations/`、`dual_arm_cable/`、`isaac_lab/`、`training/`） |
| 6 | resource_registry 根目录 | `/home/ubuntu/project/eai-idev2.1/configs/resources/` |
| 7 | 任务模板配置 | `configs/resources/tasks/*.yaml`（4 个） |
| 8 | 指标配置 | `configs/resources/metrics/*.yaml`（7 个） |
| 9 | 登录账号 | `Pibot0001` / `jinlian1234`（`admin/admin123` **无效**） |

### 可用模型资产（【实测】DB 抽样）

| model_asset_id | model_name | model_type | status |
|---|---|---|---|
| model__180319_7ad0_final | 线缆穿杆数据_20260623_1030 · Final | pi0 | ready |
| model__180207_5242_final | 线缆穿杆数据_20260623_1002 · Final | pi0 | ready |
| model__180013_16d2_final | 线缆穿杆数据_20260623_1000 · Final | pi0 | ready |
| model__132914_c503_33be3945a2 | （Isaac 历史任务引用） | — | — |

专家策略：线缆穿杆支持 `expert_policy_evaluation`（无需 checkpoint）；双臂/Isaac 训练模型评测需 ready 的 checkpoint。

### 注册表任务模板一览

| registry asset_id | 前端 taskTemplateId | 是否在评测弹窗 |
|---|---|---|
| task_cable_threading_v1 | `cable_threading_single_arm` | 是 |
| task_dual_arm_cable_manipulation_v1 | `dual_arm_cable_manipulation` | 是 |
| task_isaaclab_franka_stack_cube_v1 | `isaac_block_stacking` | 是 |
| task_isaacsim_franka_pick_place_v1 | — | **否**（评测 UI 未引用） |

---

## 3. 测试矩阵

| 测试编号 | 任务模板 | taskType | taskTemplateId | taskConfigId | 仿真平台 | 机器人 | 线路/环境 | Episodes | Horizon | Seed | 默认指标 | 评测对象 | job_id 前缀 | runtime 目录 | 需真实仿真 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TC-01 | 线缆穿杆 | `cable_threading` | `cable_threading_single_arm` | `task_cable_threading_v1` | MuJoCo | Panda | composite_cable | 10（registry） | 600 | 42 | success_rate | 专家/训练模型 | `ct_eval_*` | `runtime_outputs/cable_threading/jobs/{id}/` | 是 |
| TC-02 | 线缆整理 | `dual_arm_cable_manipulation` | `dual_arm_cable_manipulation` | `task_dual_arm_cable_manipulation_v1` | MuJoCo | Dual FR3 | 杂乱柔性线缆 | 1 | 2000 | 42 | episode_stability | episode 稳定性 / torch_bc | `eval_*` | `runtime_outputs/evaluations/jobs/{id}/` | 是 |
| TC-03 | Isaac Lab 堆块 | `block_stacking` / `isaaclab_franka_stack_cube` | `isaac_block_stacking` | `task_isaaclab_franka_stack_cube_v1` | Isaac Lab | Franka Panda | Isaac-Stack-Cube-Franka-IK-Rel-v0 | 1 | 400 | 0 | success_rate, mean_reward, … | Robomimic BC | `isaac_eval_*` | `runtime_outputs/evaluations/jobs/{id}/` | 是（需 Isaac） |
| TC-04 | Isaac Sim Pick&Place | `isaacsim_franka_pick_place` | — | `task_isaacsim_franka_pick_place_v1` | Isaac Sim | Franka | pick_place 场景 | — | — | — | — | — | — | **未接入评测中心** |

### 本轮 FLOW_TEST 任务（【实测】）

| job_id | 类型 | 状态 | 说明 |
|---|---|---|---|
| `ct_eval_20260624_085422_61f0` | TC-01 专家策略 | running（测试时） | `taskName=FLOW_TEST_cable_expert_01`，episodes=2，horizon=300，seed=77 |

> TC-02 / TC-03 未新建 `FLOW_TEST_*` 任务（避免长时间占用仿真）；以下结论来自历史 job + 代码审查 + API 抽样。

---

## 4. 前端弹窗同步测试

**审查文件**：`CreateEvaluationModal.tsx`、`evaluationTaskDerivation.ts`

### 测试 1：初次打开（【代码审查】+ registry 默认值）

| 字段 | cable_threading_single_arm | dual_arm_cable_manipulation | isaac_block_stacking |
|---|---|---|---|
| 默认关联任务 | URL/`cable_threading_single_arm` | 切换后 | 切换后 |
| 任务名称 | 空或自动生成 `线缆穿杆评测_*` | `线缆整理 · episode 稳定性评测_*` | `物块堆叠评测_*` |
| 仿真平台 | MuJoCo（binding） | MuJoCo | Isaac Lab |
| 机器人 | Panda | Dual FR3 | Franka Panda |
| 线路/环境 | composite_cable | 双臂线缆场景 | Stack Cube env |
| Episodes | **10**（registry `episode_count`） | **1** | **1** |
| Horizon | **600**（registry `max_steps`） | **2000** | **400** |
| Seed | **42** | **42** | **0** |
| 默认指标 | success_rate | episode_stability | success_rate + mean_reward 等 |

### 测试 2：切换关联任务（【代码审查】）

`useEffect`（依赖 `taskTemplateId`, `evaluationConfig`, `metricDefinitions.defaultSelectedMetricKeys`）在模板变更时重置 Episodes/Horizon/Seed/指标勾选，逻辑 **存在且依赖较完整**。

### 测试 3：异步加载（【代码审查】→ **P1 风险**）

- `registryTasksById` / `registryMetrics` 在 `open` 后异步 `fetch`（约 400ms 级）。
- 当 `registryTaskResource` 晚于首次 render 到达时，`evaluationConfig` / `metricDefinitions` 会变化，**再次触发 reset effect**，可能覆盖用户已手动修改的 Episodes/Horizon/指标。
- 依赖项已包含 `selectedTemplate`、`registryTaskResource`、`registryMetrics`，**不存在「taskId 已变但永不刷新」的明显遗漏**；主要风险是 **二次刷新覆盖用户输入**。

> **【未实测】** 浏览器内切换任务 UI 动画与 Network 时序；建议后续用 Playwright 补测。

---

## 5. Payload 测试

### 5.1 前端构建逻辑（【代码审查】）

`buildPayload` → `page.tsx` `handleStart` → `createEvaluationJob` → `POST /workspace/evaluation/evaluate-async`

关键字段映射（`page.tsx:307-356`）：

- `metrics` ← `payload.selectedMetricKeys`
- `config` ← `payload.evaluationConfig`（含 simulationPlatform、robotType、cableModel、episodes、horizon、seed、recordVideo、taskConfigId）
- `taskConfigId`、`cableThreading` / `dualArmCable` 分任务注入

**已知问题**：Isaac 分支错误复用 `cableThreading` 传 `modelName`（`page.tsx:349-356`）→ **P1**。

### 5.2 真实 Payload 示例

#### TC-01 【实测】`POST /workspace/evaluation/evaluate-async`（FLOW_TEST）

```json
{
  "taskName": "FLOW_TEST_cable_expert_01",
  "taskType": "cable_threading_single_arm",
  "taskTemplateId": "cable_threading_single_arm",
  "taskConfigId": "task_cable_threading_v1",
  "evaluationMode": "expert_policy_evaluation",
  "modelAssetId": null,
  "modelAssetName": null,
  "metrics": ["success_rate"],
  "config": {
    "simulationPlatform": "MuJoCo",
    "robotType": "Panda",
    "cableModel": "composite_cable",
    "episodes": 2,
    "horizon": 300,
    "seed": 77,
    "recordVideo": true,
    "taskConfigId": "task_cable_threading_v1"
  },
  "cableThreading": {
    "robot": "Panda",
    "cableModel": "composite_cable",
    "difficulty": "easy",
    "horizon": 300,
    "modelName": "FLOW_TEST_cable_expert_01"
  }
}
```

#### TC-02 【实测】历史 job `eval_20260617_132145_df53` 的 DB `evaluationRequest`

```json
{
  "taskName": null,
  "taskType": "dual_arm_cable_manipulation",
  "taskTemplateId": "dual_arm_cable_manipulation",
  "taskConfigId": null,
  "evaluationMode": "episode_stability",
  "metrics": [],
  "config": {},
  "numEpisodes": 1,
  "seed": 0,
  "dualArmCable": { "stretchMode": "fixed_distance", "releaseMode": "three_phase" }
}
```

> 历史任务 **无 metrics/config**；改造后字段 schema 已支持，需新建任务验证写入 `evaluation_request.json`。

#### TC-03 【实测】历史 job `isaac_eval_20260617_193339_c9cf` 的 DB `evaluationRequest`（节选）

```json
{
  "taskName": "线缆穿杆评测_20260617_268",
  "taskType": "block_stacking",
  "taskTemplateId": "isaac_block_stacking",
  "evaluationMode": "trained_model_evaluation",
  "modelAssetId": "model__132914_c503_33be3945a2",
  "horizon": 400,
  "numEpisodes": 1,
  "cableThreading": { "taskName": "线缆穿杆评测_20260617_268", "modelName": "线缆穿杆评测_20260617_268" }
}
```

> Isaac 任务名误用「线缆穿杆」前缀；`metrics`/`config` 仍为空。

---

## 6. 后端接收与数据库保存测试

### 6.1 Schema（【代码审查】+ 【实测】）

`EvaluateAsyncRequest` 已包含 `metrics`、`config`、`taskConfigId`、`taskTemplateId`、`cableThreading`、`modelAssetId` 等。Python 构造验证通过。

### 6.2 workspace_jobs 保存（【实测】SQL）

```sql
SELECT job_id, task_name, task_type, status,
       metadata_json->'evaluationRequest' AS evaluation_request,
       metadata_json
FROM workspace_jobs
WHERE job_type = 'evaluation'
ORDER BY created_at DESC
LIMIT 10;
```

| job 前缀 | evaluationRequest | metadata 结构 |
|---|---|---|
| `ct_eval_*` | **全部为 NULL/NONE** | `build_job_resource_metadata`（robot、episodes、metricIds…） |
| `eval_*` / `isaac_eval_*` | **有**（来自 runtime sync 写入 `evaluation_request.json`） | 含 evaluationRequest + resource 字段 |

**FLOW_TEST ct job 【实测】**：

- `task_name` = `线缆穿杆专家策略评测_20260624_0854`（**非**用户输入 `FLOW_TEST_cable_expert_01`）
- `metadata_json.evaluationRequest` = **无**

### 6.3 缺失列（【代码审查】）

`workspace_jobs` **无**独立列：`simulator_platform`、`task_template_id`、`model_asset_name`；信息分散在 `metadata_json` / `metrics_json`。

---

## 7. 后端真实执行链路测试

### TC-01 线缆穿杆 【实测】

```
POST /workspace/evaluation/evaluate-async
  → CableThreadingTaskAdapter.start_evaluation
  → cable_threading_service.start_evaluate_async(episodes=2, horizon=300, seed=77, robot, cable_model, …)
  → subprocess run.py
  → runtime_outputs/cable_threading/jobs/ct_eval_20260624_085422_61f0/
```

| 参数 | 是否传入执行 | 证据 |
|---|---|---|
| episodes | ✅ | `live/status.json` → `"episodes": 2` |
| horizon | ✅ | `"horizon": 300` |
| seed | ✅ | evaluation_context + CLI |
| robot / cableModel | ✅ | evaluation_context.json |
| taskConfigId | ✅ | evaluation_context + DB metadata |
| metrics / config | ✅ 写入 context | evaluation_context.json 含完整 metrics/config |
| metrics 进入报告 | ⚠️ 部分 | aggregate 含仿真指标，但 **未**按用户勾选 metrics 过滤 |

### TC-02 / TC-03 【代码审查】+ 历史 runtime

- 双臂：`dual_arm_cable_adapter` 将 **完整 request** 写入 `metadata/evaluation_request.json`，再 `start_async`；历史文件 **未含** metrics/config。
- Isaac：经 `isaac_eval_*` adapter；历史 `evaluation_request.json` 含 horizon/checkpoint，**cableThreading 字段污染**。

### P0：统一 API 与 ct_eval 断链 【实测】

| 接口 | ct_eval | eval_* | isaac_eval_* |
|---|---|---|---|
| `GET /workspace/evaluation/jobs/{id}/status` | **400 Invalid eval job ID format** | 200 | 200 |
| `GET /workspace/cable-threading/jobs/{id}/status` | **200** | N/A | N/A |

根因：`job_paths.validate_eval_job_id` 仅接受 `eval_*` / `isaac_eval_*`（`evaluation_service.get_evaluation_status` 首行即校验），**adapter 分支永远走不到 ct_eval**。

创建响应 `statusUrl` 指向统一 API → **前端若用 statusUrl 轮询 ct 任务会失败**。

---

## 8. runtime_outputs 检查

### 样本汇总

| job_id | runtime 目录 | status | request/config | result | video | report |
|---|---|---|---|---|---|---|
| `ct_eval_20260624_085422_61f0` 【实测】 | `cable_threading/jobs/...` | ✅ live/status.json | ✅ evaluation_context（含 metrics/config） | 进行中 | 待定 | 待定 |
| `ct_eval_20260623_190221_6aba` 【实测】 | 同上 | ✅ completed | ⚠️ 旧 context **无** metrics/config | ✅ aggregate_result.json | ✅ eval.mp4 | 前端报告页走 cable API |
| `eval_20260617_132145_df53` 【实测】 | `evaluations/jobs/...` | ⚠️ queued 无 aggregate | ✅ evaluation_request.json（旧 schema） | ❌ | ❌ | ❌ |
| `isaac_eval_20260617_193339_c9cf` 【实测】 | `evaluations/jobs/...` | ✅ completed | ✅ evaluation_request.json | ✅ aggregate | ❌ list 显示 false | report_uri 在 summary |

**参数一致性（FLOW_TEST）**：evaluation_context 中 episodes=2、horizon=300、seed=77 与请求一致 【实测】。

---

## 9. 状态同步测试

| 数据源 | ct_eval 样本 | eval 样本 | isaac 样本 |
|---|---|---|---|
| workspace_jobs.status | completed / running | queued | completed |
| runtime live/status | completed（190221） | 无进展 | completed |
| eval_metric_summary | **0 条 ct_eval** 【实测】 | 有 | 有 |
| 列表 videoAvailable | **false**（虽有 mp4） | false | false |
| 列表 evaluationMode | **null** | episode_stability | trained_model_evaluation |

根因：

1. `_is_eval_job_id` **不含** `ct_eval_` → `sync_eval_job_from_runtime` 对 ct 任务 **no-op**。
2. `videoAvailable` 仅看 `eval_metric_summary.replay_uri`，ct 无 summary → 永远 false。

---

## 10. 报告页测试

**文件**：`report/page.tsx`、`evaluationReport.ts`、`workspaceEvaluationRecordsMock.ts`

| 检查项 | TC-01 ct | TC-02 eval | TC-03 isaac |
|---|---|---|---|
| 按 evalId 打开 | ✅ 走 `getCableThreadingEvalResult` | ✅ `getEvaluationJobResult` | ✅ |
| 任务名称 | ⚠️ 优先 workspace job，DB 为 displayName 非用户命名 | ⚠️ mock 列表 fallback 可能介入 | ⚠️ 历史名「线缆穿杆评测_*」 |
| 指标来源 | aggregate_result + cable API | result API / mock | result API |
| 硬编码 | 【代码审查】`resolveEvaluationReportCardTitle` 有 fallback 链 | dualArm 无 aggregate 时可能空 | Isaac aggregate 来自 API |
| GET /report API | **404** 【实测】 | 不存在 | 不存在 |

演示 mock：`shouldShowWorkspaceDemo()` 默认 **false**（需 `NEXT_PUBLIC_WORKSPACE_SHOW_DEMO=true`），正常环境不合并 demo 记录。

历史 runtime 存在 `线缆穿杆评测demo8` 等 **modelName**（旧任务数据，非当前代码硬编码）。

---

## 11. 回放页测试

**【代码审查】** 以下组件 **硬编码任务名称**，不读 job/taskName：

| 文件 | 硬编码值 |
|---|---|
| `CableEvalReplayPanel.tsx` | `CABLE_THREADING_TASK_DISPLAY_NAME`（「线缆穿杆」） |
| `DualArmEvalReplayPanel.tsx` | `DUAL_ARM_CABLE_TASK_NAME`（「线缆整理」） |
| `IsaacEvalReplayPanel.tsx` | `ISAAC_BLOCK_STACKING_DISPLAY_NAME`（「物块堆叠」） |
| `ReplayRunInfoPanel.tsx` | 同上 |
| `replayAdapters.ts` | 同上 |

视频来源：ct 走 cable-threading API + runtime `videos/`；【实测】`ct_eval_20260623_190221_6aba` 有 `eval.mp4`。

evalId 在 report ↔ replay 间通过 query 传递（【代码审查】`buildCableThreadingReplayHref` 等保留 evalId）。

---

## 12. API 返回字段检查

### `GET /api/workspace/evaluation/jobs` 【实测】

ct 样本字段：`evalJobId`, `taskType`, `evaluationMode`, `status`, `taskName`, `runner`, `runtimePath`, `metrics`, `videoAvailable`, `createdAt`, …  
**缺失/空**：`evaluationMode=null`（ct），`reportUri/replayUri=null`，`taskTemplateId/taskConfigId` 不在 list item schema。

### `GET /api/workspace/evaluation/jobs/{id}/status`

| job | HTTP | 备注 |
|---|---|---|
| ct_eval_* | **400** | P0 |
| eval_* | 200 | phase/progress |
| isaac_eval_* | 200 | metrics 完整 |

### `GET /api/workspace/evaluation/jobs/{id}/result`

| job | HTTP |
|---|---|
| ct_eval_* | **400** |
| eval_*（queued） | 200，仅 status/message |
| isaac_eval_*（completed） | 200，含 successRate、meanReward 等 |

### `GET /api/workspace/jobs/{jobId}` 【实测】

ct job 返回 `taskName`、`metadata`（无 evaluationRequest），可供报告页基本信息。

### 报告专用 API

**不存在** `GET /api/workspace/evaluation/report`（【实测】404）。

---

## 13. 自动化检查命令结果

| 命令 | 结果 |
|---|---|
| `npx tsc --noEmit` | **失败**（6 处既有 TS 错误，含 `workspaceJobMapper.ts` 的 `message` 字段；**非本次评测专属**） |
| `python -m compileall app`（backend） | **通过** |
| `alembic current` | `024_resource_definition_catalog (head)` |
| `alembic check` | **有 drift**（检测到已删除表 conversations 等，与当前 models 不一致） |
| `npm run lint` | **不存在**（package.json 无此 script） |
| `npm run build` | **未执行**（耗时长；本轮以链路/API 为主） |
| `pytest tests/test_eval_job_db_list.py tests/test_evaluation_adapter_registry.py` | **16 passed** |

---

## 14. 问题清单

| 编号 | 优先级 | 问题描述 | 复现步骤 | 影响范围 | 相关文件 | 初步原因 | 建议修复 |
|---|---|---|---|---|---|---|---|
| ISS-01 | **P0** | 统一评测 API 拒绝 `ct_eval_*` status/result | 创建 ct 任务 → GET `/workspace/evaluation/jobs/{ct_eval}/status` | 所有线缆穿杆评测轮询、statusUrl | `job_paths.py`, `evaluation_service.py` | `validate_eval_job_id` 不含 ct 前缀 | 扩展校验或在 service 层先 resolve adapter |
| ISS-02 | **P0** | `ct_eval_*` 不同步 `eval_metric_summary` | 完成 ct 评测 → 查 summary 表 | 列表 metrics、reportUri、videoAvailable | `training_job_sync_service.py:429-431` | `_is_eval_job_id` 排除 ct | 纳入 ct 或独立 sync 路径读 cable runtime |
| ISS-03 | **P0** | ct 任务 DB 无 `metadata_json.evaluationRequest` | 任意 ct 评测 → SQL 查 metadata | 报告基本信息、审计、回放元数据 | `cable_threading_service.py`, `cable_threading_adapter.py` | 仅用 `build_job_resource_metadata` | 创建时 merge evaluationRequest 或 sync 读 evaluation_context |
| ISS-04 | **P0** | 用户 taskName 被 displayName 覆盖 | FLOW_TEST 创建 → 查 task_name | 列表/报告/回放任务名不一致 | `cable_threading_service.py:1163-1172`, dual_arm/isaac 同类 | `build_evaluation_display_name` 强制覆盖 | 优先 `request.taskName`，displayName 作副标题 |
| ISS-05 | **P1** | ct 有视频但 `videoAvailable=false` | 完成 ct + 有 eval.mp4 → 列表 API | 评测中心、报告「观看回放」 | `eval_job_db_service.py:49` | 仅看 summary.replay_uri | ct 回退查 runtime videos 或先 sync summary |
| ISS-06 | **P1** | 回放页任务名硬编码 | 打开任意 ct/eval/isaac 回放 | 回放信息面板 | `*EvalReplayPanel.tsx`, `replayAdapters.ts` | 使用常量 DISPLAY_NAME | 改为 `workspaceJob.taskName` 或 metadata.originalName |
| ISS-07 | **P1** | Isaac 创建 payload 误用 `cableThreading` | 代码审查 page.tsx Isaac 分支 | Isaac 请求语义混乱 | `evaluation/page.tsx:349-356` | 复制粘贴 | 改用 isaac 专用字段或顶层 taskName |
| ISS-08 | **P1** | registry 异步加载可能覆盖用户已改参数 | 打开弹窗 → 快速改 Episodes → 等 registry 返回 | 新建评测参数偶发回退 | `CreateEvaluationModal.tsx:474-517` | effect 依赖 evaluationConfig | registry 就绪前禁用编辑或仅首次 bind 重置 |
| ISS-09 | **P1** | 列表 ct 任务 `evaluationMode=null` | list API 看 ct 行 | 筛选/展示 | `eval_job_db_service.py` | metadata 结构不含 evaluationMode 顶层 | 从 metadata.evaluationMode 或 context 补全 |
| ISS-10 | **P1** | 双臂/Isaac 历史 request 无 metrics/config | 旧 eval job runtime 文件 | 指标勾选无法追溯 | dual_arm adapter 写 request | 改造前未写入 | 确认 `_write_evaluation_request` 使用含 metrics 的完整 request |
| ISS-11 | **P2** | 报告页仍依赖 mock fallback 链 | eval 无 aggregate 时 | 非 ct 报告可能空/mock | `report/page.tsx`, `workspaceEvaluationRecordsMock.ts` | 多层 fallback | 以 API + workspace job 为唯一源 |
| ISS-12 | **P2** | `EvaluationJobListItem` 缺 taskTemplateId 等 | list API | 前端需额外请求 | `schemas/evaluation.py` | schema 未扩展 | 列表 DTO 补字段 |
| ISS-13 | **P2** | alembic check drift | 运行 alembic check | 迁移维护 | models vs DB | 历史表未清理 | 单独迁移或 baseline |
| ISS-14 | **P2** | tsc 既有错误 | tsc --noEmit | 全项目类型检查 | 多处 | 历史债务 | 分批修复 |

---

## 15. 结论

### 三类任务能否完整跑通

| 任务类型 | 创建 | 执行 | 状态/结果 API | DB/summary 一致 | 报告 | 回放 | 总体 |
|---|---|---|---|---|---|---|---|
| **线缆穿杆 (ct_eval)** | ✅ 【实测 FLOW_TEST】 | ✅ episodes/horizon 传入 | ❌ 统一 status/result 400 | ❌ 无 summary / evaluationRequest | ⚠️ 可用 cable 专用 API，名称不准 | ⚠️ 视频有，名称硬编码 | **部分跑通** |
| **双臂/线缆整理 (eval_)** | ⚠️ 需仿真/环境 | ⚠️ 历史 job 长期 queued | ✅ 统一 API | ⚠️ 依赖 sync | ⚠️ 缺 result 时为空 | ⚠️ 名称硬编码 | **链路通，实测未完成** |
| **Isaac (isaac_eval_)** | ⚠️ 需 Isaac + 模型 | ✅ 历史 completed 样本 | ✅ | ✅ summary 有 | ✅ result API 有指标 | ⚠️ 名称硬编码 | **部分跑通（依赖 Isaac 环境）** |

### 当前最需优先修复的 3 个问题

1. **ISS-01**：`ct_eval_*` 无法使用统一 `/workspace/evaluation/jobs/{id}/status|result`（创建响应 statusUrl 亦失效）。
2. **ISS-02 + ISS-05**：`ct_eval_*` 不同步 `eval_metric_summary`，导致列表 metrics/video/reportUri 系统性缺失。
3. **ISS-04 + ISS-06**：用户任务名被 displayName 覆盖 + 回放页硬编码，造成 **列表 / 报告 / 回放展示不一致**。

### 改造有效性（弹窗 metrics/config）

- **【实测】** 2026-06-24 FLOW_TEST 任务已在 `evaluation_context.json` 写入 `metrics` + `config`，且 runtime `live/status.json` 使用 episodes=2、horizon=300，说明 **新一轮前端→后端→adapter→执行 的参数链路在线缆穿杆上已生效**。
- **【实测】** 但该参数 **尚未** 进入 `workspace_jobs.evaluationRequest`，也 **未** 进入 `eval_metric_summary`。

---

*报告生成方式：API/DB/runtime 实测 + 关键路径代码审查；浏览器 UI 切换任务测试未在本轮完整执行。*
