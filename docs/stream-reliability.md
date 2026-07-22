# 实时预览黑屏：架构自愈与验证

本项目实时预览链路分为两条：

- WebRTC：浏览器 ↔ 平台 ↔ 采集端（隧道转发 Offer/Answer）
- MJPEG：浏览器 → 平台 `/api/stream/{camera_id}` → 隧道帧重组缓存 → 采集端 ROS2 订阅

黑屏/卡顿通常不是“没有连接”，而是链路某一环进入“无帧/无候选/阻塞”状态，且缺少自愈机制导致长期不可恢复。

## 1. 常见根因清单（按优先级）

- **彩色话题选择**：对同一 `cameraN`，若 ROS 上同时有 `.../color/image_raw` 与 `.../compressed` 时，**采集端 `refresh_topics` 优先订阅原始 `sensor_msgs/msg/Image`（raw）** 作为 MJPEG/WebRTC 预览，避免个别驱动/编码器下压缩流异常发「黑图」；仅当**无** raw 时才用 `CompressedImage`（与 `agent/ros2_camera_stream.py` 实现一致）。

- 占位帧为空字节：前端表现为黑屏/破图；原因是服务端依赖 OpenCV 生成占位 JPEG，环境缺少 cv2 时返回空字节。
- 无帧时不输出：MJPEG generator 在无帧时不 yield，连接长时间无数据会被中间层/浏览器视为断流。
- ROS2 订阅卡死或 topic 重映射：节点仍在但订阅不再收到数据；无 watchdog 时只能重启进程恢复。
- WebRTC Trickle ICE 未实现：SDP 发送过早导致 candidate 不完整，连接偶发失败或恢复后不自动重协商。
- 网络抖动/隧道重连：平台与采集端的隧道断开再恢复后，前端没有强制拆旧连接，导致“看似在线但无帧”。
- 资源耗尽：解码错误积累、内存增长、线程死锁（锁顺序不一致）、GPU/CPU 竞争导致帧延迟无限增大。

## 2. 已实现的自愈策略（代码层）

### 2.1 多级缓存（避免黑屏）

- 采集端：相机订阅持有 `latest_frame`，无帧时回退输出占位 JPEG，保证连接不断流。
- 平台：隧道侧保存每路 last_good frame；当隧道 2s 未到新帧时复用 last_good，避免“空占位/黑屏”。
- 占位 JPEG：即使 OpenCV 不可用也会返回内置 1×1 JPEG，杜绝空字节。

### 2.2 卡顿检测 + 自动恢复

- 采集端 watchdog：当某路 `last_update` 超过阈值（默认 4s）会触发 refresh + 重建订阅。
- 平台侧：`/api/stream/status` 可拉取采集端 stream_status（帧年龄、解码错误、重启次数），用于监控告警。

## 3. 监控与告警建议

建议把以下指标接入监控系统（Prometheus/日志平台均可）：

- 采集端（/api/agent/stream-status 或 tunnel CMD STREAM_STATUS）
  - `age_sec`：单路距离最后一帧的秒数（> 4s 警告，> 10s 严重）
  - `decode_errors`：解码错误累计（突增说明数据格式/解码器异常）
  - `restart_count`：watchdog 自动恢复次数（频繁恢复说明上游不稳定）
- 平台端（agent_tunnel_manager.get_metrics）
  - `mjpeg_frames_timeout / mjpeg_frames_stale`：隧道重组/等待超时
  - `connected_agents`：隧道连接数

## 4. 验证与报告生成

### 4.1 性能测试（单路/多路）

使用脚本探测 MJPEG 稳定性：

```bash
python scripts/stream_tests/stream_probe.py \
  --url "http://127.0.0.1:8000/api/stream/camera1?device_id=1" \
  --duration-sec 60 \
  --stall-timeout-sec 3 \
  --auth "$TOKEN"
```

建议记录：

- FPS（目标：持续稳定，且波动小于 30%）
- stalls/max_stall（目标：0）
- bytes（用于估算带宽）

### 4.2 压力测试（并发观看）

多进程/多机并发执行 `stream_probe.py`，统计 stalls 与 max_stall。

### 4.3 故障模拟测试

建议至少覆盖：

- 隧道断开 30s 再恢复（应自动恢复出画面，且不会持续黑屏）
- ROS2 topic 短暂消失/重映射（watchdog 应自动重订阅）
- 服务端缺少 OpenCV（仍应返回可渲染占位 JPEG，不应黑屏）

## 5. 目标与验收口径（建议）

- 7×24 运行中：连续 24h `stalls==0`、`max_stall_sec < 0.5`（按项目实际网络可调整）
- 可用性：99.9%（以“可出画面”定义成功；按每路每小时统计）

