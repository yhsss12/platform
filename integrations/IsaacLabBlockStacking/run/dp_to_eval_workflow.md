# Diffusion Policy 接入到 Eval 流程说明

本文说明后续如何把 Isaac Lab Mimic / SkillGen 生成的专家数据接到 Diffusion Policy，并最终回到 Isaac 环境里做 online eval。

核心链路：

```text
Isaac Lab Mimic / SkillGen
  -> stack_cube_generated.hdf5
  -> convert_isaac_hdf5_to_zarr.py
  -> isaac_stack_cube_dp.zarr
  -> Diffusion Policy 训练
  -> dp_checkpoint.ckpt
  -> Isaac Lab online eval
  -> success_rate / reward / episode_length
```

## 1. 生成 Isaac 专家数据

先在 Isaac Lab 中用 Mimic 或 SkillGen 生成专家 demonstration。

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --device cpu \
  --num_envs 10 \
  --generation_num_trials 100 \
  --input_file ./datasets/stack_cube_seed_annotated.hdf5 \
  --output_file ./datasets/stack_cube_generated.hdf5
```

如果使用 SkillGen：

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --device cpu \
  --num_envs 10 \
  --generation_num_trials 100 \
  --use_skillgen \
  --input_file ./datasets/stack_cube_seed_annotated.hdf5 \
  --output_file ./datasets/stack_cube_skillgen_generated.hdf5
```

输出数据格式是 HDF5，大致结构为：

```text
stack_cube_generated.hdf5
  data/
    demo_0/
      obs/
      actions
    demo_1/
      obs/
      actions
```

## 2. 检查 HDF5 里的 obs key

不同 Isaac Lab 任务的 observation key 可能不同，所以转数据前要先看一眼 HDF5 结构。

示例检查脚本：

```python
import h5py

with h5py.File("stack_cube_generated.hdf5", "r") as f:
    demo = f["data"]["demo_0"]
    print("obs keys:", list(demo["obs"].keys()))
    print("actions shape:", demo["actions"].shape)
```

如果发现 obs key 和 `convert_isaac_hdf5_to_zarr.py` 里的字段不一致，需要修改 `flatten_obs()` 中的 `candidate_keys`。

第一版建议只做 low-dimensional DP，也就是先用关节状态、末端位姿、物体位姿，不接 RGB 图像。

## 3. 转成 Diffusion Policy 的 Zarr ReplayBuffer

运行：

```bash
python run/convert_isaac_hdf5_to_zarr.py
```

目标输出：

```text
isaac_stack_cube_dp.zarr/
  data/
    state
    action
  meta/
    episode_ends
```

其中：

```text
state: 每一帧 observation 拼出来的状态向量
action: Isaac 专家每一步执行的动作
episode_ends: 每条 demo 在拼接数组中的结束位置
```

`episode_ends` 很关键，因为 DP 训练时会从连续大数组里切时间窗口，它必须知道每条轨迹在哪里结束，避免把两条 demo 错误接在一起。

## 4. 训练 Diffusion Policy

训练时，DP 学的是：

```text
过去几帧 observation -> 未来一段 action sequence
```

例如：

```text
obs_horizon = 2
action_horizon = 8
pred_horizon = 16
```

训练样本可以理解为：

```text
输入：
state[t-1], state[t]

输出：
action[t], action[t+1], ..., action[t+7]
```

需要在 Diffusion Policy 工程中新增一个 Isaac Stack Cube dataset loader，读取：

```text
data/state
data/action
meta/episode_ends
```

然后按 DP 原本的数据采样方式生成训练 batch。

训练完成后得到：

```text
dp_checkpoint.ckpt
```

## 5. 回到 Isaac 做 online eval

Eval 不是只看训练 loss，而是把训练好的 checkpoint 加载回 Isaac 环境，让策略自己控制机器人完成任务。

流程：

```text
1. 启动 Isaac Stack Cube 环境
2. env.reset()
3. 读取当前 obs
4. 把 obs 转成训练时一致的 state
5. DP policy 预测 action sequence
6. 执行前几个 action：env.step(action)
7. 重新读取 obs，再预测下一段 action
8. 直到 success 或 timeout
9. 统计 success_rate / reward / episode_length
```

伪代码：

```python
env = make_isaac_stack_cube_env()
policy = load_diffusion_policy("dp_checkpoint.ckpt")

success_count = 0
num_eval_episodes = 50

for episode_id in range(num_eval_episodes):
    obs, info = env.reset()
    obs_history = []

    for step in range(max_episode_steps):
        state = isaac_obs_to_dp_state(obs)
        obs_history.append(state)

        if len(obs_history) < obs_horizon:
            action = np.zeros(action_dim)
        else:
            action_sequence = policy.predict(obs_history[-obs_horizon:])

            for action in action_sequence[:action_horizon]:
                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("success", False):
                    success_count += 1
                    break

            if info.get("success", False) or terminated or truncated:
                break

success_rate = success_count / num_eval_episodes
print("success_rate:", success_rate)
```

实际实现时要保证两件事一致：

```text
训练时 state 怎么拼，eval 时 obs 也必须怎么拼。
训练时 action 是什么控制空间，eval 时 env.step(action) 也必须用同一个控制空间。
```

## 6. Eval 指标

建议最小记录这些指标：

```text
num_eval_episodes: 50 或 100
success_rate: 成功次数 / 总次数
average_reward: 平均奖励
average_episode_length: 平均步数
timeout_rate: 超时比例
```

如果需要可视化，还可以保存：

```text
rollout video
eval log
failed episode hdf5
```

## 7. 给师兄的简短解释

可以这样说：

```text
后续接 DP 的流程是：先用 Isaac Lab Mimic / SkillGen 生成 HDF5 专家 demonstration，再把 HDF5 中的 obs/actions 转成 DP 常用的 Zarr ReplayBuffer。DP 在 Zarr 上训练得到 checkpoint。Eval 时不是离线看 loss，而是把 checkpoint 加载回 Isaac Stack Cube 环境中 online rollout，让策略根据当前 obs 输出 action，用 env.step(action) 执行，并统计 success rate、reward 和 episode length。
```

