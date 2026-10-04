# Adam 优化器

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization/adam-optimizer)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Adam (自适应矩估计) 优化器结合了动量（Momentum）在相关方向上加速梯度下降 (gradient descent)的优势，以及 RMSprop 为每个参数 (parameter)调整学习率的特点。这种强大的算法为每个参数计算自适应学习率，同时融入动量。Adam 是目前深度学习 (deep learning)中最常用且有效的优化算法之一，通常是一个不错的默认选择。

### 结合动量与自适应缩放

Adam 保留了过去梯度的两个指数衰减移动平均值：

1. **一阶矩估计 (类似于动量):** 这跟踪梯度的*均值*。它类似于我们之前看到的动量项，有助于在一致的梯度方向上加速进展并抑制振荡。我们称之为 $m_t$。

   
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   

   其中，$g_t$ 是当前时间步 $t$ 的梯度，而 $\beta_1$ 是此一阶矩估计的指数衰减率（通常接近 1，例如 0.9）。
2. **二阶矩估计 (类似于 RMSprop):** 这跟踪梯度的*未中心化方差*。它类似于 RMSprop 中的机制，根据每个参数 (parameter)近期梯度的大小反向调整学习率。我们称之为 $v_t$。

   
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   

   其中，$g_t^2$ 表示梯度的逐元素平方，而 $\beta_2$ 是此二阶矩估计的指数衰减率（通常也接近 1，例如 0.999）。

### 偏差校正

这些移动平均值可能存在一个问题，特别是在训练初期（当 $t$ 较小时），它们被初始化为零。这种初始化会使矩估计偏向零。Adam 通过计算偏差校正后的估计值来抵消这种影响：


$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$


$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$


随着时间步 $t$ 的增加，项 $\beta_1^t$ 和 $\beta_2^t$ 趋近于零，使得偏差校正的重要性降低。然而，在训练初期，这种校正能提供更好的矩估计。

### Adam 更新规则

最后，Adam 更新规则使用这些偏差校正后的估计值来更新模型参数 (parameter) $\theta$：


$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$


我们来分解一下：

- $\alpha$: 学习率（或步长）。
- $\hat{m}_t$: 偏差校正后的一阶矩估计，确定更新方向（类似于动量）。
- $\sqrt{\hat{v}_t}$: 偏差校正后的二阶矩估计的平方根，用于自适应地为每个参数调整学习率（类似于 RMSprop）。近期梯度较大（方差较高）的参数，其有效学习率将降低。
- $\epsilon$: 一个很小的常数（例如 $10^{-8}$），用于数值稳定性，防止在 $\hat{v}_t$ 可能非常接近零时发生除零错误。

实际上，Adam 使用梯度方差估计为每个参数计算一个单独的自适应学习率，并沿梯度均值估计平滑的方向进行更新。

### 超参数 (parameter) (hyperparameter)与使用

Adam 有几个超参数：

- **学习率 ($\alpha$):** 控制整体步长。常见值为 0.001 或 0.0001。
- **$\beta_1$:** 一阶矩估计的衰减率。默认值通常是 0.9。
- **$\beta_2$:** 二阶矩估计的衰减率。默认值通常是 0.999。
- **$\epsilon$:** 用于数值稳定的小常数。默认值通常是 $10^{-7}$ 或 $10^{-8}$。

Adam 的一个显著优点是其默认超参数值通常在各种问题上表现良好，与带有动量的 SGD 相比，需要较少的手动调整。

```python
# 在 PyTorch 中使用 Adam 的例子

import torch
import torch.optim as optim
import torch.nn as nn

# 假设 'model' 是你定义的神经网络 (nn.Module)
# 假设 'loss_fn' 是你的损失函数
# 假设 'dataloader' 提供批量数据

# model = YourModelDefinition(...)
# loss_fn = nn.CrossEntropyLoss() # 分类例子
# dataloader = YourDataLoader(...)

# 初始化 Adam 优化器
# 常见做法：学习率 = 0.001, betas=(0.9, 0.999), eps=1e-8
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# --- 在你的训练循环中 ---
# for data, labels in dataloader:
#     optimizer.zero_grad()      # 清除之前的梯度
#     outputs = model(data)      # 前向传播
#     loss = loss_fn(outputs, labels) # 计算损失
#     loss.backward()            # 反向传播（计算梯度）
#     optimizer.step()           # 使用 Adam 更新权重
# --- 训练循环代码片段结束 ---

print("示例：优化器已初始化。")
# 注意：以上代码需要定义 'model' 和 'dataloader' 才能完整运行。
```

### 优点与考量

- **结合优点：** 整合了动量（克服局部最小值、加速收敛）和 RMSprop（每个参数 (parameter)的自适应学习率）两者的优势。
- **高效性：** 计算高效，且所需内存相对较少。
- **良好默认值：** 采用建议的默认超参数 (hyperparameter)值时，通常表现良好。
- **适用场景：** 适用于大数据集、高维参数空间以及有噪声或稀疏梯度的问题。

尽管 Adam 是一个强大且广泛使用的优化器，但值得注意的是，对于某些特定任务，经过精细调整的带有动量的 SGD 偶尔可能会实现略好的泛化性能。然而，Adam 在大多数深度学习 (deep learning)应用中仍然是一个极佳的起点和表现出色的算法。

通过将自适应学习率与动量结合，并引入偏差校正，Adam 提供了一种有效的方式来应对训练深度神经网络 (neural network)时遇到的损失曲面。它直接建立在通过反向传播 (backpropagation)进行梯度计算以及动量和 RMSprop 中迭代改进的基础之上。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化器的原始研究论文，其中详细说明了其数学细节、偏差校正和实证表现。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  一本广受认可的教材中的章节，提供了Adam在深度学习优化中的全面理论解释。
- [Adam - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) — PyTorch Development Team (2024)
  Publisher: PyTorch Foundation
  PyTorch官方文档中关于Adam优化器的说明，包含参数描述和实际使用示例。

---

[上一节](06-RMSprop%20%E4%BC%98%E5%8C%96%E5%99%A8.md) · [下一节](08-%E9%80%89%E6%8B%A9%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md)
