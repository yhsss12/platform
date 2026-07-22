# RelMan：MCAP 数据转换工具

将 ROS2 MCAP 录制数据转换为 HDF5 或 LeRobot 格式，用于机器人学习与 VLA 模型训练。

## 功能

- **MCAP → HDF5**：通过 `flexible_mcap_to_hdf5.py` 将 MCAP 转为 HDF5
- **MCAP → LeRobot**：通过 `mcap_to_lerobot.py` 直接转 LeRobot 数据集（data/videos/meta）

---

## 依赖

### 环境
- Python 3.9+（推荐 3.10/3.11）
- NumPy 1.x 或 2.x（若遇 cv2 导入错误，可尝试 `pip install "numpy<2"`）

### 基础依赖（MCAP → HDF5）
```bash
pip install numpy h5py pyyaml tqdm opencv-python scipy mcap mcap-ros2-support
```

### MCAP → LeRobot 额外依赖
```bash
pip install lerobot torch
```

### Ubuntu 可选（OpenCV 显示）
```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

---

## 配置

### 1. config_aloha.yaml（MCAP 话题与对齐）

定义 MCAP 话题到 HDF5 路径的映射及时间对齐策略。

```yaml
topics:
  - topic_name: "/left/joint_states"
    message_type: "sensor_msgs/msg/JointState"
    hdf5_path: "/observations/arm_joint_left_state"
    custom_processor: "joint_state"
    custom_params: { joint_count: -1 }

  - topic_name: "/camera1/camera1/color/image_raw"
    message_type: "sensor_msgs/msg/Image"
    hdf5_path: "/images/camera1/color/origin"
    custom_processor: "image"

alignment:
  strategy: "backfill_on_grid"
  grid_fps: 20.0
  target_duration: null
```

**常用 processor**：`joint_state`、`image`、`float32`（夹爪）、`compressed_image`

### 2. config_lerobot.yaml（LeRobot 映射）

定义 state、action、相机的数据来源及 LeRobot 输出格式。

```yaml
lerobot:
  repo_id: "my_org/my_lerobot_dataset"
  robot_type: "aloha"
  fps: 20.0
  use_videos: true
  mode: "video"

  state_sources:
    - topic: "/left/joint_states"
      field: "data"
    - topic: "/right/joint_states"
      field: "data"
    - topic: "/left_gripper_state"
      field: "data"
    - topic: "/right_gripper_state"
      field: "data"

  action_mode: "next_state"

  camera_mapping:
    cam_extra_1: "/camera1/camera1/color/image_raw"
    cam_extra_2: "/camera2/camera2/color/image_raw"
    cam_extra_3: "/camera3/camera3/color/image_raw"

  default_instruction: "pick and place the object"
  instructions_path: "instruction.json"
```

**说明**：
- `state_sources`：按顺序拼接为 `observation.state`
- `action_mode: "next_state"`：使用下一帧 state 作为 action
- `camera_mapping`：LeRobot 相机名 → MCAP 话题
- `instructions_path`：任务指令 JSON（可选，与 MCAP 同目录）

---

## 使用方式

### MCAP → LeRobot（推荐）

```bash
python mcap_to_lerobot.py \
  --config config_aloha.yaml \
  --lerobot-config config_lerobot.yaml \
  --input <MCAP文件或目录> \
  --output-repo my_org/my_dataset \
  [--output-dir lerobot_output] \
  [--verbose]
```

**参数**：

| 参数 | 必需 | 说明 |
|------|------|------|
| `--config` | 是 | MCAP 话题与对齐配置 |
| `--lerobot-config` | 是* | LeRobot 配置（若 `--config` 含 `lerobot` 节可省略） |
| `--input` | 是 | 单个 .mcap 文件或包含 .mcap 的目录 |
| `--output-repo` | 是 | LeRobot repo_id，如 `org/dataset` |
| `--output-dir` | 否 | 输出目录，默认 `lerobot_output` |
| `--verbose` | 否 | 详细日志 |

### MCAP → HDF5

```bash
python flexible_mcap_to_hdf5.py \
  --config config_aloha.yaml \
  --input <MCAP文件或目录> \
  --output <输出HDF5或目录>
```

---

## 示例

### 1. 单 MCAP 转 LeRobot

```bash
python mcap_to_lerobot.py \
  --config config_aloha.yaml \
  --lerobot-config config_lerobot.yaml \
  --input 11/episode_11.mcap \
  --output-repo my_org/episode11 \
  --output-dir lerobot_output
```

输出目录结构：

```
lerobot_output/
├── data/chunk-000/episode_000000.parquet
├── meta/info.json
├── meta/episodes.jsonl
├── meta/stats.json
├── meta/tasks.jsonl
└── videos/chunk-000/observation.images.cam_extra_1/episode_000000.mp4
    observation.images.cam_extra_2/...
    observation.images.cam_extra_3/...
```

### 2. 目录内多 MCAP 批量转换

```bash
python mcap_to_lerobot.py \
  -c config_aloha.yaml \
  -l config_lerobot.yaml \
  -i /path/to/mcap_folder \
  -o my_org/batch_dataset \
  -d ./lerobot_output
```

### 3. 使用合并配置（config 中含 lerobot 节）

若 `config_aloha.yaml` 已包含 `lerobot:` 节，可省略 `--lerobot-config`：

```bash
python mcap_to_lerobot.py -c config_full.yaml -i 11/ -o my_org/ds
```

### 4. 单 MCAP 转 HDF5

```bash
python flexible_mcap_to_hdf5.py \
  -c config_aloha.yaml \
  -i 11/episode_11.mcap \
  -o output/episode_0.hdf5
```

---

## 数据格式与适配

- **11 数据**：`config_aloha.yaml` 与 `config_lerobot.yaml` 已适配 `11/` 目录下的 MCAP（`/left/joint_states`、`/right/joint_states`、`/camera1/camera1/color/image_raw` 等）
- **其他机器人**：按实际 MCAP 话题修改 `topics`、`state_sources`、`camera_mapping`
- **视频质量**：LeRobot 默认 CRF=23，可在 `lerobot/datasets/video_utils.py` 中调整

---

## 常见问题

**Q: `numpy.core.multiarray` 或 cv2 导入失败**  
A: 尝试 `pip install "numpy<2"` 或升级 opencv-python。

**Q: `'float' object has no attribute 'numerator'`**  
A: 需在 `lerobot/datasets/video_utils.py` 中将 fps 转为 `Fraction`，或在 `config_lerobot.yaml` 中设置 `use_videos: false` 使用 image 模式。

**Q: 视频文件较小**  
A: 可在 `video_utils.py` 的 `encode_video_frames` 中把默认 `crf` 从 30 调为 23 或 20。
