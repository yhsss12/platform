# SAM3 前后端交互审计

审计对象：`/home/ubuntu/project/asset3Drecons/sam3/scripts/run_sam3_box_select.py`

## 1. argparse 参数

| 参数 | 说明 |
|------|------|
| `--image` | 输入图片路径（必填） |
| `--out` | 输出目录（必填，**不是** `--output-dir`） |
| `--prompt` | 文本提示 |
| `--box` | 单个正向框（兼容旧用法，`nargs=4`） |
| `--pos-box` | 正向 xyxy 框，可重复多次（`action=append`, `nargs=4`） |
| `--neg-box` | 负向 xyxy 框，可重复多次 |
| `--text-only` | 仅文本提示，不进入框选界面 |
| `--no-neg-interactive` | 交互模式下不询问负框 |
| `--checkpoint` / `--bpe` / `--device` / `--confidence-threshold` | 模型与阈值 |

**不存在**：`--output-dir`、`--no-interactive`、`--skip-interactive`

## 2. 正框 / 负框传参

```bash
--pos-box 430 80 1320 900 --pos-box 100 100 200 200
--neg-box 10 10 50 50
```

每个框 4 个独立 float 参数（xyxy 像素坐标）。

## 3. 交互行为

脚本在以下情况进入 **matplotlib 交互**：

1. `--prompt` 未传（`None`）→ `input()` 等待 stdin
2. 无正框且非 `--text-only` → `interactive_select_positive_boxes()`
3. 未指定 `--no-neg-interactive` → `interactive_select_negative_boxes()`

## 4. 平台非交互策略（不改上游）

`integrations/Sam3dAssetPipeline/segment_cli.py`：

- 始终传 `--prompt`（空字符串也可，避免 stdin）
- 始终传 `--no-neg-interactive`
- 有正框或 `--text-only` + prompt 时才运行
- `subprocess.run(..., stdin=DEVNULL)` 防止阻塞
- 输出目录：`{job_dir}/sam3`，参数 `--out`

## 5. 输出文件

| 路径 | 说明 |
|------|------|
| `overlay.png` | 可视化叠加 |
| `combined_mask.png` | 合并 mask |
| `detections.json` | 检测结果元数据 |
| `masks/mask_{i:03d}.png` | 单 mask（i 从 0 起） |
| `cutouts/cutout_{i:03d}.png` | 抠图 |

## 6. detections.json 结构

```json
{
  "num_masks": 12,
  "detections": [
    {
      "index": 0,
      "score": 0.91,
      "box_xyxy": [430, 80, 1320, 900],
      "mask": "/abs/path/mask_000.png",
      "cutout": "/abs/path/cutout_000.png"
    }
  ]
}
```

## 7. manifest 统一

分割完成后由 `sam3_output_normalizer.py` 生成 `sam3/manifest.json`，`maskIndex` 与 SAM3 `index` 字段一致（0-based，与 `mask_009` 对应 index=9）。

## 8. 结论

- **无需修改上游**；现有脚本在正确传参下可非交互运行
- 前端框选坐标必须为 **原图像素 xyxy**
- 后端通过 manifest 消除 masks/cutouts/detections 命名不一致
