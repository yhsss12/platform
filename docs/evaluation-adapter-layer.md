# Platform Evaluation Adapter Layer

## 1. 背景

平台当前有两个真实任务，评测能力差异明显：

| 任务 | 评测方式 | 产物 |
|------|----------|------|
| `cable_threading` | `run.py eval`；scripted / random / robomimic + checkpoint | eval.csv, eval.results.json, eval.mp4 |
| `dual_arm_cable_manipulation` | 无 eval.py；重复完整 episode | episode_result.json, generate.mp4 |

**目标**：评测中心只调用统一 `/api/workspace/evaluation/*`，底层由任务 Adapter 负责真实脚本调用，不伪造结果，不强行套 checkpoint 评测。

## 2. 架构

```
Frontend (评测中心)
        │
        ▼
POST /api/workspace/evaluation/evaluate-async
        │
        ▼
evaluation_service.py  ──► registry.get_adapter(taskType)
        │
        ├── CableThreadingEvaluationAdapter  ──► run.py eval (现有 cable_threading_service)
        └── DualArmCableEvaluationAdapter    ──► platform_runner.py × N seeds (Phase 2)
```

## 3. 评测模式

| evaluationMode | 适用任务 | 含义 |
|----------------|----------|------|
| `policy_evaluation` | cable_threading | 加载 policy / checkpoint 执行仿真评测 |
| `episode_stability` | dual_arm_cable_manipulation | 多 seed 重复完整 episode，聚合稳定性指标 |

未来可扩展：`data_generation_quality`, `sim2real_transfer`, `safety_robustness`, `perturbation_robustness`。

## 4. 统一输出目录

```
runtime_outputs/evaluations/{evalJobId}/
  status.json
  logs/eval.log
  results/aggregate_result.json
  results/per_episode_results.json
  videos/
  frames/
  artifacts/
  metadata/
    evaluation_request.json
    source_jobs.json          # 指向任务私有 job 目录（迁移期）
```

迁移期 `cable_threading` 仍写入 `runtime_outputs/cable_threading/jobs/ct_eval_*`，统一层通过 `metadata/source_jobs.json` 关联。

## 5. 统一 status.json

```json
{
  "evalJobId": "eval_20260611_120000_a1b2",
  "taskType": "cable_threading",
  "evaluationMode": "policy_evaluation",
  "status": "running",
  "phase": "evaluating",
  "progress": 0.4,
  "currentEpisode": 4,
  "totalEpisodes": 10,
  "message": "正在执行策略评测",
  "metrics": {},
  "artifacts": {},
  "updatedAt": "2026-06-11T12:00:00Z"
}
```

## 6. 统一 aggregate_result.json

```json
{
  "evalJobId": "...",
  "taskType": "...",
  "evaluationMode": "...",
  "completedAt": "...",
  "summary": {
    "totalEpisodes": 10,
    "successEpisodes": 7,
    "successRate": 0.7
  },
  "taskMetrics": {},
  "perEpisode": [],
  "artifacts": {}
}
```

- `cable_threading`：`taskMetrics` 来自 eval.results.json（successRate, everSuccessRate 等）
- `dual_arm_cable`：`taskMetrics` 含 meanFinalSag, contactSuccessRate, stretchReachedRate 等聚合值

## 7. 能力边界（前端须展示）

**cable_threading**：支持策略评测 / checkpoint 评测。

**dual_arm_cable_manipulation**：支持 episode 稳定性评测；**不支持** checkpoint / Robomimic / ACT 模型评测。

## 8. 迁移顺序

1. **Phase 1（当前）**：Adapter 接口 + 统一 API 骨架；cable_threading 委托现有服务；dual_arm 仅校验 + 501。
2. **Phase 2**：DualArm episode_stability 多 episode 编排 + 聚合。
3. **Phase 3**：评测中心前端改调统一 API；保留旧 cable-threading evaluate-async 作兼容别名。
4. **Phase 4**：cable_threading 输出迁入 evaluations/ 目录（可选）；废弃旧私有 eval 路由。

## 9. Phase 1 不实现项

- dual_arm 多 episode 实际运行与聚合
- 统一 API 替换前端现有 cable_threading 直连
- 删除 `/api/workspace/cable-threading/evaluate-async`
- checkpoint 资产自动解析（仍走现有 cable_threading 参数）
