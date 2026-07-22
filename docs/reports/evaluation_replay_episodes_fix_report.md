# 评测回放 Episodes 与视频数量不一致问题修复报告

## 1. 原因定位

**属于：情况 B — 实际执行 2 轮，但只录制 1 段代表性视频**

### 证据（【实测】任务 `ct_eval_20260624_085422_61f0`）

| 检查项 | 结果 |
|---|---|
| `live/status.json` → `episodes` | **2** |
| `live/status.json` → `completedEpisodes` | 未写入（null） |
| `results/aggregate_result.json` → `total_episodes` | **2** |
| `results/aggregate_result.json` → `success_episodes` | **2** |
| `results/per_episode_results.json` | **2 条**（episode 0、1，均 success） |
| `videos/` | 仅 `eval.mp4` + `eval.browser.mp4`（同内容浏览器兼容副本，**计为 1 段回放**） |
| 视频时长 | 约 1 秒级（live 合成视频，非每轮独立 mp4） |

### 根因说明

1. **执行层**：`run.py eval` 通过 `--episodes 2` 确实跑完 2 轮，`per_episode_results.json` 与 aggregate 统计一致。
2. **录制层**：runner 设计为 `--live-video-out videos/eval.mp4`，在评测过程中写入 **一段 live 合成视频**（通常为最后一轮或短片段），**不会**为每个 episode 生成 `episode_0.mp4` / `episode_1.mp4`。
3. **展示层（修复前）**：`CableEvalReplayPanel` 右侧只显示 `Episodes: 2` + `进度: 100%`，左侧播放唯一 `eval.mp4`，**未区分「计划轮数」与「回放视频段数」**，造成用户误以为应有 2 段视频。

**不是**情况 A（只执行 1 轮），也**不是**情况 C（多 mp4 被前端忽略）。

---

## 2. 修改文件

### 后端

| 文件 | 变更 |
|---|---|
| `backend/app/services/evaluation_replay_info.py` | **新增** 回放元数据聚合（计划/完成/成功/失败/视频数/replayUris） |
| `backend/app/services/cable_threading_service.py` | status/result 接口合并 replay 字段；`resolve_job_video_path` 支持 `episode` 参数 |
| `backend/app/schemas/cable_threading.py` | `CableThreadingJobStatusResponse` 增加 replay 相关字段 |
| `backend/app/api/routes_workspace_cable_threading.py` | `GET .../video?episode=` 支持按 episode 取独立 mp4（向前兼容） |
| `backend/tests/test_evaluation_replay_info.py` | **新增** 单元测试 |

### 前端

| 文件 | 变更 |
|---|---|
| `src/lib/workspace/evaluationReplayInfo.ts` | **新增** 类型与解析/提示文案工具 |
| `src/components/workspace/replay/EvaluationReplayEpisodeStats.tsx` | **新增** 右侧「计划轮数 / 实际完成 / …」面板 |
| `src/components/workspace/replay/CableEvalReplayPanel.tsx` | 使用 replay 字段；视频下方提示；多视频选择器 |
| `src/components/workspace/replay/CableThreadingVideoPlayer.tsx` | 支持自定义 `videoApiPath`（多 episode 切换） |
| `src/lib/api/cableThreadingClient.ts` | TS 类型补齐 replay 字段 |

**未改动**：双臂 / Isaac 回放页（`DualArmEvalReplayPanel`、`IsaacEvalReplayPanel`）。

---

## 3. 新增或修正字段

线缆穿杆 status / result API 现返回（顶层）：

```json
{
  "requestedEpisodes": 2,
  "completedEpisodes": 2,
  "successfulEpisodes": 2,
  "failedEpisodes": 0,
  "recordedVideoCount": 1,
  "replayUri": "/api/workspace/cable-threading/jobs/{job_id}/video",
  "replayUris": [
    {
      "episodeIndex": null,
      "uri": "/api/workspace/cable-threading/jobs/{job_id}/video",
      "label": "代表性回放",
      "fileName": "eval.mp4"
    }
  ],
  "videoAvailable": true,
  "isRepresentativeVideo": true
}
```

字段来源优先级见 `evaluation_replay_info.py`（status / evaluation_context / aggregate / per_episode / videos 目录扫描）。

---

## 4. 前端展示变化

右侧「运行进度」由：

```text
Episodes 2
Horizon 300
进度 100%
```

改为：

```text
计划轮数       2
实际完成       2
成功轮数       2
失败轮数       0
回放视频       1 段（代表性）
Horizon        300
Seed           …
进度           100%
```

当 `requestedEpisodes > recordedVideoCount` 时，面板内显示橙色提示：

```text
当前仅生成 1 段代表性视频，评测实际执行 2 轮。
```

视频播放器下方同步显示：

```text
当前视频为代表性回放，评测共执行 2 个 episode。
```

---

## 5. 多视频处理逻辑

| 场景 | 行为 |
|---|---|
| 仅 `eval.mp4`（当前 runner 默认） | 不显示 Episode 选择器；标签「代表性回放」+ 上述提示 |
| `videos/` 含 `episode_0.mp4`、`episode_1.mp4` 等 | `recordedVideoCount=2+`，显示 **Episode 1 \| Episode 2** 选择器，切换 `?episode=` 加载对应 mp4 |
| 旧任务缺 `completedEpisodes` | 从 aggregate / per_episode 推断，不伪造 |
| 旧任务无 `replayUris` | 回退 `replayUri` / 默认 video API |

---

## 6. 验证结果

### 当前问题任务 `ct_eval_20260624_085422_61f0`（【实测】）

| 项 | 值 |
|---|---|
| requestedEpisodes | 2 |
| completedEpisodes | 2 |
| successfulEpisodes | 2 |
| failedEpisodes | 0 |
| videos 目录 mp4（去重后） | **1**（eval.mp4；不含 eval.browser.mp4 重复计数） |
| API status/result | 字段正确返回 |
| 多视频选择器 | **不显示**（仅 1 段） |
| 代表性视频提示 | **显示** |

### 新建测试任务 `FLOW_TEST_episode_video_check_01`

本轮复用已存在的 `FLOW_TEST_cable_expert_01`（即 `ct_eval_20260624_085422_61f0`，episodes=2, horizon=300）验证，**未再启动新仿真**。

单元测试覆盖：

- 2 轮完成 + 1 段代表性视频 → `recordedVideoCount=1`, `isRepresentativeVideo=true`
- 2 个 episode_*.mp4 → `recordedVideoCount=2`, 选择器逻辑可用

---

## 7. 未处理事项

1. **Runner 设计**：当前 `CableThreadingMVP/run.py` live 视频仍为 **单文件代表性录制**；若产品需要「每 episode 独立 mp4」，需改 runner（例如每轮 `--live-video-out videos/episode_{i}.mp4`），前端已预留多视频选择能力。
2. **统一评测 API**：`GET /workspace/evaluation/jobs/{ct_eval}/status` 仍因 job_id 校验不支持 ct（全链路测试报告 ISS-01），回放页走 cable-threading 专用 API，不受影响。
3. **视频时长偏短**：1 秒视频是 live 合成策略/帧率问题，非本次 UI 修复范围；UI 已明确说明「代表性回放 ≠ 全部轮次」。

---

*修复完成时间：2026-06-24*
