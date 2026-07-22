# HDF5 数据格式说明

本文档描述 **MCAP → HDF5** 转换产物（`FlexibleHdf5Converter`）及数据资产 **导出 ZIP** 的目录约定。实现主要见 `backend/app/services/mcap_converter.py`、`backend/app/api/routes_data_assets.py`。

---

## 1. 数据资产导出 ZIP（平台侧）

批量导出 HDF5 资产时，ZIP 根目录名为 `export_yyyyMMdd_HHmmss`，结构如下：

```text
export_yyyyMMdd_HHmmss/
├── hdf5/                    # 各条 episode：{资产编号}_{原文件名}.hdf5 或 .h5
├── annotations/             # 可选：{资产编号}_{标注文件名}
└── asset_list.xlsx         # 资产清单
```

---

## 2. 单个 `.hdf5` 文件内部结构

### 2.1 默认顶层：`observations/` 与 `actions/`

每个 ROS Topic 在配置中有 `hdf5_path`。若配置**未**显式以 `observations/` 或 `actions/` 开头，转换器会按话题语义重写根路径（函数 `_rewrite_hdf5_root_path`）：

| 消息大类 | 默认 HDF5 根路径模式 |
|---------|---------------------|
| 图像（`image` / `compressed_image`） | `{observations\|actions}/images/{camera_key}` |
| 非图像 | `{observations\|actions}/{topic_name 去前导 /}` |

- `observations` / `actions` 由 `_infer_semantic_root` 根据 topic 名称启发式判定（图像类多归入观测；含 `cmd`、`action` 等关键词多归入 `actions`）。
- `camera_key` 从 topic 名解析（如 `camera_0`）；解析失败时有回退名。

因此**默认可读布局**为：几乎所有有效数据挂在 **`observations/...`** 或 **`actions/...`** 之下，而不是散落在 HDF5 根下。

若 YAML/配置里已写死 `observations/...` 或 `actions/...`，则**不会**被覆盖。

下文用 `{hdf5_path}` 表示**重写后的完整前缀**（通常形如 `observations/images/camera_0` 或 `actions/your/topic/path`）。

### 2.2 按消息类型：在 `{hdf5_path}` 下的数据集

#### 图像：`compressed_image` / `image`

```text
{hdf5_path}/
├── color/
│   ├── origin      # Dataset，uint8，形状 (T, H, W, C)。原始图像写此处；纯压缩图话题时可为空占位
│   └── compress    # Dataset，uint8。压缩图写此处；原始图话题时可为空占位
├── depth           # Dataset，float32
└── pointcloud      # Dataset，float32
```

#### 关节状态：`joint_state`

```text
{hdf5_path}/
├── qpos
├── effort
└── vel
```

#### 空间速度：`twist_stamped`

```text
{hdf5_path}/
├── qpos      # 整段 twist 展平后的主数组
├── linear
└── angular
```

#### 位姿：`pose_stamped`

```text
{hdf5_path}/
└── data
```

#### 其它类型（夹爪、`Float32MultiArray` 等）

```text
{hdf5_path}/
└── data
```

无对齐数据或未匹配到 topic 时，仍会按类型创建上述路径的**空 Dataset**，以保持结构一致。

### 2.3 后处理与合并（可选路径）

配置 **数据合并**（`data_merging_config`）时，可能在 HDF5 中**新增或覆盖**路径，例如合并后的目标 `target_path`；若存在 `{某 topic 的 hdf5_path}_timestamps`，可能写入 `{target_path}_timestamps`。具体以后处理配置为准。

### 2.4 文件根属性与元数据组

**根 Group 属性（attrs）**（`_save_config_info`）示例：

- `alignment_window`、`target_fps`、`sample_drop`、`relative_start`、`delta_action`
- `topic_configs`：YAML 字符串，记录各 topic 的 `message_type`、`hdf5_path`、`data_type`、描述等

**可选数据集：**

```text
/metadata/warning_stats    # Dataset，UTF-8 JSON 字节；对齐/校验报警统计（有则写入）
```

---

## 3. 与读服务的一致性

`backend/app/services/hdf5_service.py` 查找图像组时，会优先尝试 `observations/images`、`images`、`observations` 等路径，与本文 **默认 `observations/.../images/...`** 的布局一致。

---

## 4. 简例（概念树）

某场景经默认重写后可能类似：

```text
/                                    # HDF5 根
├── observations/
│   ├── images/
│   │   └── camera_0/
│   │       ├── color/origin
│   │       ├── color/compress
│   │       ├── depth
│   │       └── pointcloud
│   └── …/…/data 或其他 qpos 等路径
├── actions/
│   └── …                            # 被判为动作的话题
└── metadata/
    └── warning_stats                # 可选
```

实际深度与命名完全由 topic 配置与 `_rewrite_hdf5_root_path` 结果决定；上树仅作导航参考。
