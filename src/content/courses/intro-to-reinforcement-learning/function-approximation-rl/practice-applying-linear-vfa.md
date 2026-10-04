---
course: "intro-to-reinforcement-learning"
chapter: "function-approximation-rl"
lesson: "practice-applying-linear-vfa"
sourceId: 1657
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-6-function-approximation-rl/practice-applying-linear-vfa"
title: "实践：应用线性价值函数近似"
description: "在一个合适的问题上，使用线性方法实现价值函数近似。"
order: 8
plots: ["plots/1657-0.json"]
sourceHash: "c27f0acd2128fd423ebb4f765e631a0e173cb5defdd018de9c5827af58739f44"
sourceCorrections: []
---

函数近似在大状态空间中是必需的，线性方法结合半梯度下降 (gradient descent)提供了一种可行的途径。我们将对一个经典的强化学习 (reinforcement learning)控制问题——登山车（Mountain Car）——应用线性方法实现价值函数近似。

在登山车问题（来自Gymnasium库）中，一辆动力不足的汽车位于山谷中，必须爬上右侧的山才能达到目标。由于重力大于汽车引擎的动力，它无法直接向上行驶。它必须通过在山丘之间来回行驶来累积动量。

状态 $s$ 是连续的，由两个变量定义：

1. 位置 $p \in [-1.2, 0.6]$
2. 速度 $v \in [-0.07, 0.07]$

由于状态空间是连续的，我们不能使用简单的表格来存储每个可能状态的值。这使其成为函数近似的理想选择。我们的目标是使用半梯度TD(0)方法，在固定的简单策略下，*估算状态价值函数* $V(s)$。我们将用线性函数近似 $V(s)$：


$$
\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i x_i(s)
$$


其中 $\mathbf{x}(s)$ 是从状态 $s$ 中得到的特征向量 (vector)，而 $\mathbf{w}$ 是我们需要学习的权重 (weight)向量。

### 特征工程：瓦片编码

在连续状态空间中，为线性函数近似创建特征的一种常用且有效的方法是**瓦片编码**。想象一下在状态空间上叠加多个网格（瓦片组），每个网格都相对于其他网格略微偏移。对于给定状态，我们确定它落在每个网格中的哪个瓦片里。特征向量 (vector) $\mathbf{x}(s)$ 变成一个大型二进制向量，其中每个分量对应一个瓦片组中的一个瓦片。如果状态落入对应的瓦片，则分量为1，否则为0。

这种方法以分布式方式离散化空间。每个状态会激活多个特征（每个瓦片组一个），相似的状态会激活许多相同的特征，从而实现泛化能力。相邻的状态会比距离较远的状态共享更多活跃瓦片。

对于我们的登山车例子，我们可以定义瓦片组的数量和每个网格的分辨率（每维度瓦片数量）。一个状态 $(p, v)$ 将在每个瓦片组中激活一个瓦片。

```python
# 瓦片编码的示例配置
num_tilings = 8
num_tiles_per_dim = 8 # 为每个瓦片组创建8x8的网格

# 状态空间维度用于归一化/缩放
pos_min, pos_max = -1.2, 0.6
vel_min, vel_max = -0.07, 0.07

# 特征总数 = 瓦片组数量 * (每维度瓦片数量 * 每维度瓦片数量)
total_features = num_tilings * (num_tiles_per_dim ** 2)

# 获取状态的活跃特征（瓦片索引）的函数
# 注意：实际实现通常使用像 `TileCoder` 这样的库
# 这是一个函数占位符
def get_feature_vector(state, num_tilings, tiles_per_dim, total_features):
    position, velocity = state
    feature_indices = [] # 活跃瓦片（特征）的索引

    # --- 实际瓦片编码逻辑的占位符 ---
    # 此逻辑将计算状态落入哪个瓦片
    # 对于每个'num_tilings'，并考虑偏移量。
    # 为简单起见，我们假设它返回一个活跃索引列表。
    # 示例：feature_indices = compute_active_tiles(state, config)
    # -------------------------------------------------

    # 创建二进制特征向量
    x = np.zeros(total_features)
    # 假设 feature_indices 包含 '1' 位的索引
    # 基于上面计算出的活跃瓦片。
    # 在实际实现中，瓦片编码会直接给出这些索引。
    # 对于此占位符，我们只模拟激活几个特征。
    # 请将其替换为使用状态、瓦片组数量等进行实际瓦片计算的代码。
    for i in range(num_tilings):
        # 用于演示的简化哈希/索引计算
        idx = hash((round(position*10), round(velocity*100), i)) % total_features
        if idx >= 0 and idx < total_features:
             feature_indices.append(idx)

    if feature_indices:
      x[np.array(feature_indices, dtype=int)] = 1.0

    return x

# --- 更具体、简化的网格瓦片编码示例 ---
# （理解上述占位符的替代方案）
def get_simple_grid_features(state, num_tilings, tiles_per_dim, total_features):
    position, velocity = state
    pos_scale = tiles_per_dim / (pos_max - pos_min)
    vel_scale = tiles_per_dim / (vel_max - vel_min)
    features = np.zeros(total_features)

    for i in range(num_tilings):
        # 为每个瓦片组应用一个简单的偏移
        offset_factor = i / num_tilings
        pos_offset = offset_factor * (pos_max - pos_min) / tiles_per_dim
        vel_offset = offset_factor * (vel_max - vel_min) / tiles_per_dim

        pos_shifted = position + pos_offset
        vel_shifted = velocity + vel_offset

        # 查找此瓦片组的瓦片索引
        pos_tile = int((pos_shifted - pos_min) * pos_scale)
        vel_tile = int((vel_shifted - vel_min) * vel_scale)

        # 将索引限制在边界内
        pos_tile = max(0, min(tiles_per_dim - 1, pos_tile))
        vel_tile = max(0, min(tiles_per_dim - 1, vel_tile))

        # 计算此瓦片组中此瓦片的平坦索引
        base_index = i * (tiles_per_dim ** 2)
        tile_index = base_index + vel_tile * tiles_per_dim + pos_tile

        if 0 <= tile_index < total_features:
            features[tile_index] = 1.0 # 激活此特征

    return features
```

*注意：`get_simple_grid_features` 函数提供了一个基本的网格瓦片编码实现。适当的瓦片编码通常涉及更复杂的哈希和偏移策略以获得更好的泛化能力，但这显示了核心思路。*

### 预测：实现半梯度TD(0)

现在，我们将使用半梯度TD(0)算法来学习我们的线性价值函数近似器 $\hat{v}(s, \mathbf{w})$ 的权重 (weight) $\mathbf{w}$。我们将评估一个简单固定的策略：始终沿动作索引2对应的方向加速（向右加速）。

给定从状态 $S$ 到状态 $S'$ 的转移及奖励 $R$，权重 $\mathbf{w}$ 在每一步的更新规则是：

$\mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})] \nabla \hat{v}(S, \mathbf{w})$

由于 $\hat{v}(S, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(S)$，关于 $\mathbf{w}$ 的梯度就是特征向量 (vector) $\mathbf{x}(S)$。更新规则变为：


$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \mathbf{w}^T \mathbf{x}(S') - \mathbf{w}^T \mathbf{x}(S)] \mathbf{x}(S)
$$


我们来实现这个学习循环。

```python
import numpy as np
import gymnasium as gym
# import matplotlib.pyplot as plt # 用于绘图 (可选)

# --- 参数 ---
alpha = 0.1 / num_tilings # 学习率，通常按活跃特征数量缩放
gamma = 1.0             # 折扣因子（登山车是分幕式任务，gamma=1很常见）
num_episodes = 5000

# 使用简化的网格瓦片编码实现
feature_func = get_simple_grid_features

# --- 初始化 ---
weights = np.zeros(total_features)
env = gym.make('MountainCar-v0')

# 预测的辅助函数
def predict_value(state, w):
    features = feature_func(state, num_tilings, num_tiles_per_dim, total_features)
    return np.dot(w, features)

# --- 学习循环 ---
episode_rewards = [] # 记录每幕奖励

print("开始训练...")
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # 固定策略：始终选择动作2（向右加速）
        action = 2

        # 获取当前状态 S 的特征
        current_features = feature_func(state, num_tilings, num_tiles_per_dim, total_features)
        current_value = np.dot(weights, current_features)

        # 执行动作，观察下一个状态 S' 和奖励 R
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # 计算TD目标
        next_value = predict_value(next_state, weights) if not terminated else 0.0
        td_target = reward + gamma * next_value

        # 计算TD误差差值
        td_error = td_target - current_value

        # 使用半梯度TD(0)更新权重
        # 梯度就是特征向量 current_features
        weights += alpha * td_error * current_features

        # 移动到下一个状态
        state = next_state
        step_count += 1

        # 如果需要，为超长回合设置安全中断
        if step_count > 10000:
            print(f"警告：回合 {episode + 1} 超过10000步。强制中断。")
            done = True # 强制中断

    episode_rewards.append(total_reward)
    if (episode + 1) % 500 == 0:
        print(f"回合 {episode + 1}/{num_episodes} 完成。总奖励：{total_reward}")

print("训练完成。")
env.close()

# --- 可选：分析结果 ---
# 你可以绘制 episode_rewards 以观察固定策略是否有所改进（可能不会太多，
# 因为这里的目标是价值预测，而不是策略改进）。
# 更有意思的是，将学习到的价值函数可视化。
```

### 学习到的价值函数可视化

训练结束后，`weights` 向量 (vector)包含学习到的参数 (parameter)。我们现在可以估算任意状态 $s$ 的值 $\hat{v}(s, \mathbf{w})$。了解智能体学习内容的有效方法是绘制价值函数的负值（因为成本是负奖励，更高的价值意味着“更接近目标”或“更好的状态”）。我们期望接近目标位置（p > 0.5）或以高速度向目标移动的状态具有更高的价值（负值较小）。

我们创建一个状态网格（位置、速度），并使用我们学习到的 `weights` 计算每个点的预测值。然后我们可以将其可视化为热力图或等高线图。

````python
# --- 可视化代码（使用 Plotly） ---
import plotly.graph_objects as go
import numpy as np # 确保导入了 numpy

# 生成用于绘图的网格点
positions = np.linspace(pos_min, pos_max, 30) # 增加密度以获得更平滑的图
velocities = np.linspace(vel_min, vel_max, 30)
value_grid = np.zeros((len(velocities), len(positions)))

for i, vel in enumerate(velocities):
    for j, pos in enumerate(positions):
        state_eval = (pos, vel)
        # 使用训练好的权重预测价值
        value_grid[i, j] = predict_value(state_eval, weights)

# 创建 Plotly 等高线图（热力图样式）
plotly_fig = go.Figure(data=go.Contour(
    z=value_grid,
    x=positions,
    y=velocities,
    colorscale='Viridis', # 或 'Blues', 'RdBu' 等
    contours=dict(
        coloring='heatmap', # 像热力图一样填充等高线
        showlabels=False # 可选：在等高线上显示值标签
    ),
    colorbar=dict(title='状态价值 (V)')
))

plotly_fig.update_layout(
    title='登山车（固定策略）的学习到的状态价值函数 (V)',
    xaxis_title="位置",
    yaxis_title="速度",
    width=700,
    height=550,
    margin=dict(l=60, r=60, b=60, t=90)
)

# 显示图表（如果交互式运行）
# plotly_fig.show()

# 获取用于嵌入的 JSON 表示：
plotly_json_string = plotly_fig.to_json(pretty=False)
# 包装在 markdown 代码块中（删除换行符以获得单行）
plotly_json_single_line = ''.join(plotly_json_string.splitlines())

print("\n用于价值函数可视化的 Plotly JSON：")
# print(f"```plotly\n{plotly_json_single_line}\n```") # 在最终的 markdown 中打印此内容
````



![登山车（固定策略）的学习到的状态价值函数 (V)](plots/1657-0.json)



> 在始终向右加速的策略下，登山车环境的估算状态价值函数 $\hat{v}(s, \mathbf{w})$。更高的值（负值较小）表示学习到的近似认为更好的状态。

### 讨论

本次练习展示了线性函数近似在连续状态空间环境中进行价值预测的应用。要点包括：

1. **必要性：** 表格方法在此处不可行。函数近似使得我们能够通过学习参数 (parameter)化函数来处理连续或非常大的状态空间。
2. **特征的重要性：** 特征的选择（此处为瓦片编码）很重要。好的特征有助于线性近似器捕捉真实价值函数中的重要变化。瓦片编码在局部化和泛化之间提供了良好平衡。
3. **半梯度：** 我们使用半梯度更新，因为我们在自举（使用 $\hat{v}(S', \mathbf{w})$ 更新 $\hat{v}(S, \mathbf{w})$），并且对目标价值进行求导会很复杂，并经常对稳定性造成损害。梯度 $\nabla \hat{v}(S, \mathbf{w}) = \mathbf{x}(S)$ 使更新计算高效。
4. **泛化：** 学习到的权重 (weight)使我们能够通过组合特征的激活来估算*任何*状态的价值，包括训练期间从未明确访问过的状态。可视化结果显示了一个平滑的价值图，表明了泛化能力。

本例侧重于*预测*（估算固定策略下的 $V(s)$）。下一步的自然发展，是在此基础上进行*控制*：使用类似的技术（例如，带函数近似的半梯度SARSA或Q学习）近似动作价值函数 $Q(s, a)$，从而学习最优策略。你通常会根据状态和动作来定义特征 $x(s, a)$。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapters 9-10
  强化学习的经典教材，详细介绍了线性函数逼近、半梯度方法和瓦片编码。
- [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) — Gymnasium Developers (2023)
  Publisher: Farama Foundation
  Gymnasium库中山地车环境的官方文档，说明了其状态空间、动作空间、奖励和动力学。
