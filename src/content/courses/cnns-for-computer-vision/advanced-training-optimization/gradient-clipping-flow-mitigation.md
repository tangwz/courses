---
course: "cnns-for-computer-vision"
chapter: "advanced-training-optimization"
lesson: "gradient-clipping-flow-mitigation"
sourceId: 2544
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-2-advanced-training-optimization/gradient-clipping-flow-mitigation"
title: "梯度裁剪与梯度流动缓解"
description: "了解并应用如梯度裁剪等技术，以防止梯度爆炸并管理梯度流动。"
order: 6
plots: ["plots/2544-0.json"]
sourceHash: "e5d1bd82fe5d166d38d88d20a79ae8f7c57f2fea347f02dc24402e4350f9151e"
sourceCorrections: []
---

训练非常深层网络会带来独特的稳定性问题。反向传播 (backpropagation)过程中一个常见问题是梯度可能会变得非常大，这种现象称为**梯度爆炸**。当梯度过大时，参数 (parameter)更新量会非常大，有效地“越过”损失函数 (loss function)中的最佳点，并可能导致数值溢出（NaN值）或训练行为不稳定，使损失发散而非收敛。这种不稳定性会阻碍模型有效学习。

反之，梯度也可能变得极小，特别是在反向传播过程中更深的层（靠近输入端）中。这种**梯度消失**问题阻碍学习，因为这些层中权重 (weight)的更新变得微不足道，导致它们学习非常缓慢或根本不学习。尽管像残差连接（第一章）和归一化 (normalization)技术（本章前面已介绍）这样的架构选择是缓解梯度消失的主要方法，但对于梯度爆炸通常需要直接干预。

梯度裁剪是一种直接技术，专门用于对抗梯度爆炸问题，通过在训练更新期间限制梯度的幅度。

### 梯度裁剪

梯度裁剪的核心思想很直接：如果训练步骤中，梯度的整体大小（范数）或单个值超过预设阈值，它们将被重新缩放或限制在一个可管理的范围内。这能防止单个批次或少数不稳定步骤大幅改变模型权重 (weight)并使训练过程偏离轨道。主要有两种方法：

1. **按值裁剪：** 这种方法涉及为梯度设置逐元素边界。对于梯度向量 (vector) $g = \nabla_{\theta} L$ 的每个分量 $g_i$，它被裁剪到特定范围 $[-c, c]$ 内。
   $g_i = \max(-c, \min(c, g_i))$
   尽管实现简单，但如果不同分量被不同裁剪，按值裁剪可能会改变整体梯度向量的*方向*。这可能会稍微改变预期的更新方向。
2. **按范数裁剪：** 这通常是更推荐的方法，因为它保留了梯度更新的方向，只在幅度超过阈值时重新缩放其大小。它作用于整个梯度向量 $g$（包含所有可训练参数 (parameter) $\theta$ 的梯度），而非单个分量。

   首先，计算梯度向量的整体范数，通常是L2范数（欧几里得范数）：
   $\|g\| = \sqrt{\sum_{i} g_i^2}$
   令 $T$ 为裁剪阈值（一个超参数 (hyperparameter)）。如果梯度向量 $g$ 的范数 $\|g\|$ 超过 $T$，则它被重新缩放：
   $g \leftarrow \begin{cases} g \times \frac{T}{\|g\|} & \text{如果 } \|g\| > T \\ g & \text{如果 } \|g\| \le T \end{cases}$
   这确保了最终梯度向量的L2范数永远不会超过 $T$。如果原始范数已低于或等于阈值，梯度保持不变。

#### 实现示例（按范数裁剪）

大多数深度学习 (deep learning)框架都提供梯度裁剪的内置支持。例如，在PyTorch中，通常会像往常一样计算梯度（`loss.backward()`），然后在优化器步骤（`optimizer.step()`）*之前*应用裁剪：

```python
# 假设模型参数和损失已计算
loss.backward() # 计算梯度

# 按范数裁剪梯度
threshold = 1.0 # 示例阈值
torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)

# 执行优化器步骤
optimizer.step()
```

在TensorFlow中，裁剪通常直接集成到优化器中，或使用 `tf.clip_by_global_norm` 来应用：

```python
# 假设已定义损失、可训练变量和优化器
with tf.GradientTape() as tape:
  predictions = model(inputs)
  loss = compute_loss(labels, predictions)

# 计算梯度
gradients = tape.gradient(loss, model.trainable_variables)

# 按全局范数裁剪梯度
threshold = 1.0 # 示例阈值
clipped_gradients, _ = tf.clip_by_global_norm(gradients, threshold)

# 应用裁剪后的梯度
optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
```

#### 阈值的选择

裁剪阈值 $T$ 是一个超参数，通常需要一些调整。设置过低可能会不必要地减慢收敛速度，因为它限制了可能有用的较大更新。设置过高可能无法有效防止不稳定性。一个常见做法是在稳定训练的初始阶段（不进行裁剪或使用非常高的阈值）监测梯度范数的典型范围，然后将阈值设置在观察到的平均或中位数范数之上一点。1.0、5.0 或 10.0 等值通常是好的起点，但最佳值取决于模型架构、数据和损失比例。

### 梯度流动缓解与监测

尽管裁剪直接解决梯度爆炸问题，但确保网络中健康的*梯度流动*对于解决梯度消失问题和整体训练效果都很重要。如前所述，残差连接、仔细的权重 (weight)初始化以及归一化 (normalization)层（批归一化、层归一化、组归一化）是实现此目的的主要机制。它们有助于在梯度通过深层网络反向传播 (backpropagation)时保持梯度信号。

然而，即使有这些技术，监测梯度的流动也是一种有用的诊断做法。逐层观察梯度的幅度（范数）可以提供对训练动态的了解。

- **梯度消失：** 如果靠近输入端的层的平均梯度范数始终比靠近输出端的层小几个数量级，这表明存在梯度消失问题。这些早期层没有接收到强的学习信号。
- **梯度爆炸：** 如果梯度范数意外飙升至非常大的值，这表明可能存在不稳定性，可能需要更强的裁剪或对学习率或架构进行调整。
- **健康流动：** 在一个表现良好的训练过程中，梯度范数在各层之间可能会有所不同，但通常应保持在合理的非零范围内，这表明网络的所有部分都在为学习做贡献。

像TensorBoard或Weights & Biases这样的工具允许您记录和可视化统计数据，例如每个层的梯度L2范数或整个模型在训练步骤中的梯度L2范数。这种可视化对于调试与梯度流动相关的训练问题非常有帮助。



![跨层的梯度范数](plots/2544-0.json)



> 不同训练场景下每层平均梯度L2范数的示意图。梯度消失表现为朝向输入端范数迅速衰减，而梯度爆炸则表现为范数迅速增加。健康梯度在各层保持合理的幅度。（注意：可视化使用对数坐标）。

总结而言，管理梯度幅度对于成功训练深层复杂CNN模型必不可少。梯度裁剪提供了一种直接机制来防止梯度爆炸并稳定训练。监测梯度范数可作为一种诊断工具，以了解梯度流动并找出梯度消失等潜在问题，从而指导对架构、归一化、初始化或超参数 (parameter) (hyperparameter)的调整。这些技术，与本章中讨论的高级优化器和正则化 (regularization)方法结合使用，构成了一个工具集，可有效训练先进的深度学习 (deep learning)模型。

## 参考资料

- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf) — Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio (2013)
  Journal: JMLR Workshop and Conference Proceedings, ICML; Publisher: JMLR.org; Volume: 28; Pages: 1310-1318; DOI: [10.55982/pascanu13.3](https://doi.org/10.55982/pascanu13.3)
  讨论了梯度消失/爆炸问题，并介绍了用于训练深度网络的梯度裁剪方法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了深度学习的全面概述，包括梯度不稳定性及裁剪。
- [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch梯度范数裁剪函数的官方文档。
- [tf.clip_by_global_norm](https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm) — TensorFlow Developers (2024)
  TensorFlow全局梯度范数裁剪函数的官方文档。
