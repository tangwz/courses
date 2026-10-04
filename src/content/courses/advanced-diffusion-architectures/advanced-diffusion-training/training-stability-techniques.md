---
course: "advanced-diffusion-architectures"
chapter: "advanced-diffusion-training"
lesson: "training-stability-techniques"
sourceId: 5437
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/training-stability-techniques"
title: "训练稳定性的方法 (梯度裁剪, 指数移动平均)"
description: "诸如梯度裁剪和权重指数移动平均 (EMA) 等方法，以确保稳定收敛。"
order: 6
plots: []
sourceHash: "d6263fd81fb8e7e232a5488fffaceb503bfce2e071432da6fc80b39d936e79ac"
sourceCorrections: []
---

训练深度生成模型，例如扩散模型，经常面临长时间复杂的优化挑战。虽然精巧的网络结构和损失函数 (loss function)是主要组成部分，但要获得稳定的收敛和最佳性能，通常需要额外技术来调节训练动态。梯度裁剪和模型权重 (weight)的指数移动平均 (EMA) 是提高训练稳定性的两种主要方法。

## 梯度裁剪

深度网络，包括扩散模型中使用的U-Net或Transformer，有时会遇到梯度爆炸问题。在反向传播 (backpropagation)过程中，梯度可能累积并变得非常大，导致模型权重 (weight)发生剧烈更新。这些大幅更新会使训练不稳定，导致损失发散或剧烈震荡，从而阻止模型收敛到一个好的解。

梯度裁剪直接处理此问题，通过限制梯度的幅度，在它们被用来更新模型参数 (parameter)之前。最常用的方法是按范数裁剪，通常是L2范数 (欧几里得范数)。

设 $\mathbf{g} = \nabla_{\theta} L$ 是损失 $L$ 关于模型参数 $\theta$ 的梯度。我们计算整个梯度向量 (vector)（所有参数）的L2范数：$\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$。如果此范数超过预设的阈值 $c$，梯度向量将被重新缩放，使其范数等于 $c$。


$$
\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{如果 } \|\mathbf{g}\| \le c \\
\frac{c}{\|\mathbf{g}\|} \mathbf{g} & \text{如果 } \|\mathbf{g}\| > c
\end{cases}
$$


这确保了权重更新步长的幅度受到限制，避免了由异常梯度引起的极端变化。

### 实际考量

- **阈值 $c$ 的选择：** 这是一个重要的超参数 (parameter) (hyperparameter)。过高的值可能很少触发裁剪，从而无法阻止不稳定。过低的值可能会过度缩小梯度，减缓收敛速度或阻止模型达到最佳性能。典型值通常在1.0到10.0之间，但最佳值取决于模型结构、数据集、批量大小和学习率。它通常需要一些实验。
- **实现：** 大多数深度学习 (deep learning)框架都提供了梯度裁剪的内置函数。例如，在PyTorch中，通常会在调用 `loss.backward()` 之后、调用 `optimizer.step()` 之前使用 `torch.nn.utils.clip_grad_norm_`。

```python
# 示例代码 (PyTorch)
optimizer.zero_grad()
loss = compute_loss(model, batch)
loss.backward()
# 在优化器步骤前裁剪梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm 即是 'c'
optimizer.step()
```

梯度裁剪作为一种安全机制，在训练的初始阶段或使用较高学习率时尤其有用。

## 权重 (weight)的指数移动平均 (EMA)

随机梯度下降 (gradient descent) (SGD) 及其变体由于小批量更新的特性，会给训练过程引入噪声。尽管这种随机性有助于摆脱不好的局部最小值，但也意味着模型参数 (parameter)即使在训练后期也可能在最优解附近大幅波动。使用最后一步训练的参数可能无法带来最佳模型性能，因为它可能代表一个瞬时、有噪声的状态。

指数移动平均 (EMA) 提供了一种方法，通过对近期训练历史中的参数进行平均，来获得更稳定且通常性能更好的模型权重集。它维护模型参数的“影子”副本，这些副本根据当前训练参数缓慢更新。

设 $\theta_t$ 是第 $t$ 步更新后的模型参数，设 $\theta'_t$ 是对应的 EMA 参数。EMA 参数使用衰减因子 $\beta$ 进行更新：


$$
\theta'_{t} = \beta \theta'_{t-1} + (1 - \beta) \theta_t
$$


这里，$\beta$ 是一个超参数 (hyperparameter)，通常设为接近1的值（例如0.99、0.999，甚至0.9999）。较高的 $\beta$ 意味着EMA权重变化更慢，并包含了更长的参数历史（平滑效果更好）。较低的 $\beta$ 使EMA权重更紧密地跟随当前训练权重。

> 图表说明了通过优化步骤更新的常规训练权重 ($\theta_t$，蓝色节点) 与作为前一个 EMA 权重和当前训练权重的加权平均值更新的 EMA 权重 ($\theta'_t$，绿色节点) 之间的关系。虚线表示当前训练权重对下一个 EMA 权重的影响。

### 实际考量

- **何时使用 EMA 权重 (weight)：** EMA 参数 (parameter) $\theta'$ 通常 *不* 用于训练循环本身（即它们不影响梯度计算或优化器对 $\theta$ 的更新）。相反，EMA 权重是单独维护的，并在训练完成后使用，或在训练期间定期用于评估，尤其是用于从扩散模型生成样本。使用 EMA 权重通常会带来明显更好的样本质量，优于使用最终原始权重 $\theta_T$。
- **衰减 $\beta$ 的选择：** 最佳衰减率 $\beta$ 取决于训练时长和学习率调度。0.999 或 0.9999 等较高的值常用于长时间训练，提供显著的平滑效果。对于较短的训练或学习率快速变化的情况，可能更偏好稍小的 $\beta$ 值。
- **实现：** 许多训练库或代码库都实现了 EMA 回调或工具。这通常包括在开始时创建模型参数的副本，并在每个优化器步骤后根据 EMA 公式更新此副本。

```python
# 代码片段（通常是回调或实用程序类的一部分）
ema_decay = 0.999
ema_model_params = [p.clone().detach() for p in model.parameters()]

# 在训练循环中，optimizer.step() 之后
with torch.no_grad():
    for ema_p, current_p in zip(ema_model_params, model.parameters()):
        ema_p.mul_(ema_decay).add_(current_p, alpha=1 - ema_decay)

# 用于推断/采样时，将 ema_model_params 加载到模型中
# ... 加载 EMA 参数 ...
# model.eval()
# generate_samples(model, ...)
```

## 组合使用

梯度裁剪和 EMA 经常一起使用。裁剪可以防止因孤立的大梯度引起的灾难性发散，而 EMA 则平滑了优化过程中的整体随机性，从而获得一个稳定的最终模型状态。通过应用这些方法，您可以明显提高训练高级扩散模型的可靠性，并常能获得卓越的生成性能。

## 参考资料

- [On the difficulty of training recurrent neural networks](https://doi.org/10.48550/arXiv.1211.5063) — Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Pages: 1310-1318; DOI: [10.48550/arXiv.1211.5063](https://doi.org/10.48550/arXiv.1211.5063)
  这篇基础性论文介绍了梯度裁剪，以解决梯度爆炸问题，该技术现已广泛用于稳定深度神经网络的训练。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇里程碑式的论文介绍了去噪扩散概率模型（DDPM），并强调了模型权重指数移动平均（EMA）在实现高质量样本生成方面的实际好处。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习基础概念的权威教科书，包括对梯度爆炸等优化挑战以及稳定训练通用策略的详细解释。
- [\`torch.nn.utils.clip_grad_norm_\`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — PyTorch Contributors (2024)
  PyTorch官方关于梯度范数裁剪的文档，提供了实现这种训练稳定技术的实际使用方法和参数。
