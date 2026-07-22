# Universal RP-RF PINN 使用说明

本目录提供一个适配 12 个 MimicGen 任务的共享 RP-RF PINN。实际使用时只运行 `universal_rprf_inference.py`，另外两个文件会被自动加载。

## 1. 文件说明

```text
universal_rprf_pinn.py
universal_rprf_inference.py
universal_rprf_pinn.pt
```

| 文件 | 用途 | 是否直接运行 |
|---|---|---:|
| `universal_rprf_pinn.py` | 定义共享 PINN、加载权重、计算候选分数 | 否 |
| `universal_rprf_inference.py` | 读取测试数据、筛选修复参数并执行真实 rollout | 是 |
| `universal_rprf_pinn.pt` | 使用 12 个任务反馈数据联合训练的共享权重 | 否 |

三个文件必须放在同一个目录中，不需要为不同任务更换 PT 文件。

## 2. 环境要求

运行前需要准备：

- MimicGen 项目；
- robosuite 和 MuJoCo；
- `mimicgen` Python 环境；
- 当前任务的测试数据和 prepared source HDF5。

当前服务器上的项目和 Python 路径为：

```text
/home/zyf/mimicgen/MimicGen_physics_refine
/home/zyf/anaconda3/envs/mimicgen/bin/python
```

## 3. 放置文件

建议将三个文件放在项目的同一个工具目录：

```text
MimicGen_physics_refine/
`-- tools/
    `-- universal_rprf/
        |-- universal_rprf_pinn.py
        |-- universal_rprf_inference.py
        `-- universal_rprf_pinn.pt
```

当前服务器已经放置在：

```text
/home/zyf/mimicgen/MimicGen_physics_refine/tools/universal_2py_20260714/
```

## 4. 准备测试数据

测试数据目录应包含：

```text
<dataset>/
`-- demo/
    |-- mg_config.json
    |-- demo.hdf5
    `-- demo_failed.hdf5
```

- `demo.hdf5`：MimicGen 生成成功的 Demo。
- `demo_failed.hdf5`：MimicGen 生成失败、需要修复的 Demo。
- `mg_config.json`：生成该批 Demo 时使用的 MimicGen 配置。
- prepared source HDF5：MimicGen 重新生成轨迹时使用的源成功演示，通过 `--source-hdf5` 指定。

测试集可以是模型未见过的新数据。推理时不会使用测试结果重新训练模型。

## 5. 设置运行环境

登录服务器并进入项目目录：

```bash
cd /home/zyf/mimicgen/MimicGen_physics_refine

export ROBOSUITE_ROOT=/home/zyf/robosuite
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/zyf/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=$PWD/tools/universal_2py_20260714:$PWD/tools:$PWD:${PYTHONPATH:-}
```

## 6. 运行测试

下面以模型未见过的 Kitchen D1 test50 为例：

```bash
/home/zyf/anaconda3/envs/mimicgen/bin/python \
  tools/universal_2py_20260714/universal_rprf_inference.py \
  --task kitchen \
  --dataset datasets/generated/kitchen_d1_test50_seed71351_20260713_pinn \
  --source-hdf5 datasets/source/kitchen_mg_prepared_n10.hdf5 \
  --checkpoint tools/universal_2py_20260714/universal_rprf_pinn.pt \
  --output-dir outputs/kitchen_d1_universal_rprf_test \
  --candidate-mode safe \
  --pool-size 128 \
  --budget 5 \
  --seed 12345 \
  --rollout-seed 22345
```

不需要单独运行 `universal_rprf_pinn.py`，也不需要对 `universal_rprf_pinn.pt` 再训练。

## 7. 主要参数

| 参数 | 含义 | 建议值 |
|---|---|---:|
| `--task` | 当前 MimicGen 任务名称 | 按实际任务填写 |
| `--dataset` | 测试数据集根目录 | 必填 |
| `--source-hdf5` | prepared source HDF5 路径 | 必填或由配置提供 |
| `--checkpoint` | 共享 PT 权重路径 | 必填 |
| `--output-dir` | 本次实验输出目录 | 必填 |
| `--candidate-mode` | 候选参数模式 | `safe` |
| `--pool-size` | 每个失败 Demo 生成的候选数量 | `128` |
| `--budget` | 每个失败 Demo 最多执行的真实 rollout 数量 | `5` |
| `--seed` | PINN 候选生成与排序随机种子 | `12345` |
| `--rollout-seed` | MuJoCo rollout 随机种子 | `22345` |

默认启用 success-stop：同一失败 Demo 只要有一个候选 rollout 成功，就立即停止测试该 Demo 的其他候选。因此 `--budget 5` 是最大预算，不代表每条失败 Demo 都必须运行 5 次。

## 8. 自动执行流程

运行上述命令后，程序会自动完成：

1. 读取 `demo.hdf5` 和 `demo_failed.hdf5`；
2. 对每条失败 Demo 提取状态上下文 `c`；
3. 生成 128 组修复参数 `theta=(d,z)`；
4. 使用共享 PINN 预测候选的 `V`、`q` 和 `p`；
5. 使用 selector 选择最多 5 个候选；
6. 使用 MimicGen 和 MuJoCo 执行真实 rollout；
7. 某条 Demo 修复成功后立即停止继续尝试；
8. 保存每次候选测试和最终成功率。

整个测试过程不需要人工修改每条 Demo 的参数。

## 9. 查看输出

输出目录中主要包含：

```text
<output-dir>/
|-- candidate_plan.jsonl
|-- feedback_candidates.jsonl
`-- summary.json
```

- `candidate_plan.jsonl`：PINN 为每条失败 Demo 选择的候选修复参数。
- `feedback_candidates.jsonl`：每次真实 rollout 的参数、成功标记和物理指标。
- `summary.json`：原始成功率、修复数量、最终成功率和总 rollout 数量。

重点查看 `summary.json` 中的字段：

| 字段 | 含义 |
|---|---|
| `raw_success` | 原始成功 Demo 数量 |
| `raw_failed` | 原始失败 Demo 数量 |
| `raw_success_rate` | 原始成功率 |
| `repaired_count` | 成功修复的失败 Demo 数量 |
| `final_success` | 原始成功数加成功修复数 |
| `final_success_rate` | 修复后的总成功率 |
| `num_candidates` | 实际执行的真实 rollout 数量 |
| `num_problematic` | 出现仿真异常的候选数量 |

## 10. 更换任务

测试其他任务时保持共享 PT 不变，只修改以下参数：

```text
--task
--dataset
--source-hdf5
--output-dir
```

支持的任务名称为：

```text
square
nut_assembly
coffee
coffee_preparation
hammer_cleanup
kitchen
mug_cleanup
pick_place
stack
stack_three
threading
three_piece_assembly
```

## 11. 已验证结果

使用固定串行种子在模型未见过的 Kitchen D1 test50 上验证：

| 指标 | 结果 |
|---|---:|
| 原始成功率 | 20/50 = 40% |
| 原始失败 Demo | 30 |
| 成功修复 | 21/30 |
| 修复后总成功率 | 41/50 = 82% |
| 真实 rollout 数量 | 94 |
| Problematic | 0 |

为了保证结果可复现，建议保留命令中的 `--seed 12345`、`--rollout-seed 22345` 并使用串行运行。
