---
course: "advanced-diffusion-architectures"
chapter: "consistency-models"
lesson: "practice-consistency-distillation"
sourceId: 5462
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-5-consistency-models/practice-consistency-distillation"
title: "实践操作：基本一致性蒸馏"
description: "实现一个从玩具扩散模型开始的简化一致性蒸馏过程。"
order: 8
plots: []
sourceHash: "55a13cb92ba64ecdd85a72b8363190c81940e54f67a1104c45b60c053eebf280"
sourceCorrections: []
---

我们将一致性蒸馏的理论付诸实践。本节将详细介绍如何实现一个基本的一致性蒸馏过程。我们将使用一个预先存在的扩散模型（我们的“教师模型”），并训练一个“学生”一致性模型，以快速近似其输出，目标是实现单步生成。

一致性特性的强制执行是其主要思想：$f(x_t, t) \approx x_0$ 适用于由概率流ODE定义的轨迹上的所有 $t$。蒸馏通过训练学生网络 $f_\theta(x, t)$ 来匹配目标网络 $f_{\theta^-}(x', t')$ 的输出，其中 $x$ 和 $x'$ 是同一轨迹上的相邻点，而 $\theta^-$ 表示缓慢更新的权重 (weight)（$\theta$ 的EMA）以保证稳定性。

### 准备工作与设置

我们假定您已具备以下条件：

1. 一个预训练 (pre-training)的扩散模型（教师模型）。为简单起见，我们假设该模型预测 $\epsilon$（噪声），但我们可以轻松调整它来预测 $x_0$。我们将教师模型的预测函数表示为 `teacher_model(xt, t)`。
2. 一个与教师模型兼容的数据集（例如MNIST、CIFAR-10，或者一个更简单的二维数据集以便快速实验）。
3. 一个标准的深度学习 (deep learning)框架，例如PyTorch或TensorFlow。示例将使用类PyTorch伪代码。

我们的目标是训练一个学生一致性模型 `student_model(xt, t)`，它通常使用与教师模型相同的架构进行初始化。我们还需要一个学生模型的指数移动平均（EMA）版本 `target_model(xt, t)`，其权重 (weight) $\theta^-$ 会根据学生模型的权重 $\theta$ 缓慢更新。

```python
# 示例设置（类PyTorch）
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm # 用于进度显示

# 假定teacher_model已预加载（预测epsilon）
# teacher_model.eval() # 将教师模型设置为评估模式

# 初始化学生模型（与教师模型架构相同）
student_model = deepcopy(teacher_model)
student_model.train() # 将学生模型设置为训练模式

# 使用学生模型的初始权重初始化目标模型
target_model = deepcopy(student_model)
target_model.eval() # 目标模型仅用于推断

# 学生模型的优化器
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

# 超参数
num_training_steps = 100000
batch_size = 64
ema_decay = 0.99 # 典型的EMA衰减率
N = 100 # 训练时的离散化步数
T = teacher_model.num_timesteps # 教师模型的最大时间步

# 损失函数（例如，L2距离）
def consistency_loss_fn(online_pred, target_pred):
    return F.mse_loss(online_pred, target_pred)

# 从epsilon预测获取x0预测的函数
def get_x0_from_epsilon(xt, epsilon, t, alphas_cumprod):
    alpha_t_cumprod = alphas_cumprod[t].view(-1, 1, 1, 1) # 确保正确的形状
    return (xt - torch.sqrt(1.0 - alpha_t_cumprod) * epsilon) / torch.sqrt(alpha_t_cumprod)

# 假定'get_dataloader()'提供x0批次数据
dataloader = get_dataloader(batch_size)

# 假定'alphas_cumprod'包含教师模型噪声调度中的(1 - beta_t)的累积乘积
alphas_cumprod = teacher_model.alphas_cumprod.to(device)
```

### 蒸馏训练循环

一致性蒸馏的核心在于迭代采样相邻时间步对，计算相应的含噪样本，并最小化学生模型在较早时间点的预测与目标模型在较晚时间点的预测之间的差异。

以下是单个训练步骤的分解：

1. **采样数据：** 获取一批干净数据点 $x_0$。
2. **采样时间步：** 从 $\{1, ..., N-1\}$ 中随机采样一个时间步索引 $n$，其中 $N$ 是我们为训练选择的离散化步数（例如100）。这定义了我们的相邻时间步 $t_i = (i/N)T$ 和 $t_{i+1} = ((i+1)/N)T$。
3. **生成含噪样本：** 通过向 $x_0$ 添加适量的高斯噪声来创建 $x_{t_i}$ 和 $x_{t_{i+1}}$，这些噪声量对应于教师模型噪声调度中 $t_i$ 和 $t_{i+1}$ 的噪声水平。
4. **获取目标预测：** 使用目标网络 $f_{\theta^-}$（此步骤中权重 (weight)冻结）来预测给定较嘈杂样本 $x_{t_{i+1}}$ 和时间 $t_{i+1}$ 时的原始数据 $x_0$。
   - 由于我们的基础模型预测 $\epsilon$，我们首先获取 $\epsilon_{\theta^-}(x_{t_{i+1}}, t_{i+1})$，然后将其转换为 $x_0$ 预测：\hat{x}\_0^{\text{目标}} = \text{get\_x0\_from\_epsilon}(x\_{t\_{i+1}}, \epsilon\_{\theta^-}(x\_{t\_{i+1}}, t\_{i+1}), t\_{i+1}, \alpha\_{\text{cumprod}})。
5. **获取学生预测：** 使用学生网络 $f_\theta$ 来预测给定较不嘈杂样本 $x_{t_i}$ 和时间 $t_i$ 时的 $x_0$。
   - 类似地，获取 $\epsilon_\theta(x_{t_i}, t_i)$ 并转换：\hat{x}\_0^{\text{学生}} = \text{get\_x0\_from\_epsilon}(x\_{t\_i}, \epsilon\_\theta(x\_{t\_i}, t\_i), t\_i, \alpha\_{\text{cumprod}})。
6. **计算损失：** 计算学生预测与目标预测之间的距离（例如MSE或L1损失）。重要的是，我们阻止梯度反向流回目标网络。 L = d(\hat{x}\_0^{\text{学生}}, \text{stop\_grad}(\hat{x}\_0^{\text{目标}}))。
7. **更新学生模型：** 执行反向传播 (backpropagation)并使用优化器更新学生模型的权重 $\theta$。
8. **更新目标网络：** 使用EMA更新目标网络权重 $\theta^-$：\theta^- \leftarrow \text{ema\_decay} \times \theta^- + (1 - \text{ema\_decay}) \times \theta。

```python
# 训练循环代码片段（简化版）
for step in tqdm(range(num_training_steps)):
    optimizer.zero_grad()

    x0 = next(iter(dataloader)).to(device) # 1. 采样数据

    # 2. 采样时间步（索引n从1到N-1）
    n = torch.randint(1, N, (batch_size,), device=device)
    t_i = (n / N) * T
    t_i_plus_1 = ((n + 1) / N) * T

    # 如果模型需要离散步，确保时间步为整数
    t_idx_i = n.long() # 或者将连续的t映射到离散索引
    t_idx_i_plus_1 = (n + 1).long()

    # 3. 生成含噪样本（使用教师模型的调度逻辑）
    noise_i = torch.randn_like(x0)
    noise_i_plus_1 = torch.randn_like(x0) # 通常重新使用噪声以减少方差
    xt_i = get_noisy_version(x0, t_idx_i, noise_i) # 基于DDPM前向过程的函数
    xt_i_plus_1 = get_noisy_version(x0, t_idx_i_plus_1, noise_i_plus_1) # 同上

    # 4. 获取目标预测（使用target_model并转换epsilon->x0）
    with torch.no_grad():
        target_epsilon = target_model(xt_i_plus_1, t_idx_i_plus_1)
        target_x0_pred = get_x0_from_epsilon(xt_i_plus_1, target_epsilon, t_idx_i_plus_1, alphas_cumprod)

    # 5. 获取学生预测（使用student_model并转换epsilon->x0）
    student_epsilon = student_model(xt_i, t_idx_i)
    student_x0_pred = get_x0_from_epsilon(xt_i, student_epsilon, t_idx_i, alphas_cumprod)

    # 6. 计算损失
    loss = consistency_loss_fn(student_x0_pred, target_x0_pred)

    # 7. 更新学生模型
    loss.backward()
    optimizer.step()

    # 8. 更新目标网络（EMA）
    for param, target_param in zip(student_model.parameters(), target_model.parameters()):
        target_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

    if step % 1000 == 0:
        print(f"Step: {step}, Loss: {loss.item()}")
        # 可选：保存检查点，生成样本图像
```

下图显示了单个训练步骤的数据流：

> 图中显示了一致性蒸馏训练单步的数据流。输入（数据、时间、噪声）用于生成相邻的含噪样本，这些样本被送入学生模型和目标模型。损失函数 (loss function)比较它们对 $x_0$ 的估计，从而驱动学生权重（$\theta$）和EMA目标权重（$\theta^-$）的更新。

### 使用训练好的一致性模型进行采样

一旦 `student_model`（或者从技术上讲，通常用于推断的是包含EMA权重 (weight)的最终 `target_model`）训练完成，采样过程将变得非常简单：

1. **采样噪声：** 从标准高斯分布 $\mathcal{N}(0, I)$ 中抽取样本 $z$。这表示 $x_T$。
2. **单步生成：** 将噪声 $z$ 和最大时间步 $T$（或相应的索引）通过训练好的一致性模型 $f_{\theta^-}$（或 $f_\theta$）。输出即为估计的 $x_0$。

```python
# 单步采样
consistency_model = target_model # 使用EMA模型进行推断
consistency_model.eval()

with torch.no_grad():
    z = torch.randn(num_samples, *data_shape).to(device) # 采样噪声 (x_T)
    t_max = torch.full((num_samples,), T-1, dtype=torch.long, device=device) # 最大时间步索引

    # 获取T时的epsilon预测
    pred_epsilon = consistency_model(z, t_max)

    # 转换为x0预测
    generated_x0 = get_x0_from_epsilon(z, pred_epsilon, t_max, alphas_cumprod)

# 'generated_x0' 包含最终的样本。
```

就是这样！单次前向传播即可生成样本。为了在略微增加计算成本的情况下获得更高质量，可以使用多步采样，其中涉及类似于DDIM的中间步骤，但使用的是一致性函数 $f_{\theta^-}$。然而，其主要吸引力在于单步生成所带来的显著计算量减少。

### 实践考量

- **$N$ 的选择：** 训练过程中离散化步数 $N$ 的多少影响着生成质量。理论上 $N$ 越大越好，但每个epoch的计算成本也越高。100-200这样的值是常见的起始点。
- **距离度量 $d$：** 尽管L2（MSE）很常见，但L1损失或伪Huber损失有时能产生更好的结果，或者对异常值更精确。
- **EMA衰减：** `ema_decay` 控制目标网络适应的速度。接近1的值（例如0.99、0.999）提供稳定性。
- **教师模型质量：** 蒸馏后的一致性模型的性能与其学习来源的教师扩散模型的质量密切相关。
- **速度与质量：** 正如所讨论的，此方法高度优先考虑速度。尽管结果可能令人满意，但它们可能无法始终与运行多步的原始教师模型保持相同的保真度。多步一致性采样可以在一定程度上弥补这一差距。

本实践练习为理解一致性蒸馏的工作原理奠定了基础。通过实现这个基本版本，您可以更好地了解训练模型以进行快速、少步生成的机制，这是使扩散模型在实时应用中更具实用性的重要进展。

## 参考资料

- [Consistency Models](https://arxiv.org/abs/2303.01469) — Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023)
  Journal: arXiv preprint arXiv:2303.01469; DOI: [10.48550/arXiv.2303.01469](https://doi.org/10.48550/arXiv.2303.01469)
  介绍了一致性模型概念以及用于从扩散模型进行高效单步生成的一致性蒸馏训练方法。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: arXiv; Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  去噪扩散概率模型的奠基性论文，这类模型在一致性蒸馏中作为教师模型使用。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2020)
  Journal: International Conference on Learning Representations (ICLR 2021); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  通过随机微分方程建立了基于分数的模型与扩散模型之间的理论联系，为概率流常微分方程提供了背景信息。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  PyTorch 深度学习框架的官方文档，用于实现一致性蒸馏的实践操作。
