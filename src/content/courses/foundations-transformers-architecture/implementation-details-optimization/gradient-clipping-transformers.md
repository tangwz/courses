---
course: "foundations-transformers-architecture"
chapter: "implementation-details-optimization"
lesson: "gradient-clipping-transformers"
sourceId: 2065
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/gradient-clipping-transformers"
title: "梯度裁剪"
description: "使用梯度裁剪防止训练期间梯度爆炸。"
order: 6
plots: []
sourceHash: "58a9698f8b519b8501d361132db78671d395ba01ae6fafd3ac0781b025e321ab"
sourceCorrections: []
---

虽然像 AdamW 这样的优化器和学习率调度等技术对 Transformer 模型训练很有帮助，但这些模型自身的层数多以及其计算的特点有时会带来另一个数值问题：梯度爆炸。与循环网络中常提及的梯度消失不同，梯度爆炸指的是在反向传播 (backpropagation)过程中，梯度幅度变得过大的情况。这些过大的梯度可能导致在优化器步进期间模型权重 (weight)产生相应的大幅更新，从而可能引起数值溢出、训练不稳定（例如，损失震荡或发散），甚至丢失先前学到的信息。

梯度爆炸可能在深层网络中出现，因为在反向传播过程中，梯度是逐层相乘的。如果梯度范数持续大于1，它们的乘积会随网络深度呈指数级增长。虽然层归一化 (normalization)等组件有助于缓解此问题，但仍可能突然出现大梯度，这可能是由于特定输入序列或注意力机制 (attention mechanism)或前馈网络内部的交互所致，尤其是在训练初期模型参数 (parameter)尚未稳定时。

梯度裁剪是一种直接用于解决此问题的技术。它的作用是在优化器更新模型权重之前，对梯度的幅度（范数）设定上限。如果所有模型参数梯度的总范数超过预设阈值，梯度将按比例缩小到与该阈值相等。这能防止单个大梯度事件破坏训练的稳定性。

### 按全局范数裁剪

最常用的方法是**按全局范数裁剪**。这包括计算L2范数（欧几里得范数），该范数针对将模型中*所有*可训练参数 (parameter)的*所有*梯度连接起来形成的梯度向量 (vector)。令 $G$ 表示这个包含所有参数 $\theta_i$ 的梯度 $\frac{\partial L}{\partial \theta_i}$ 的全局梯度向量。L2范数计算方式如下：


$$
\text{全局范数} = \sqrt{\sum_i \left( \frac{\partial L}{\partial \theta_i} \right)^2}
$$


然后我们将这个 `全局范数` 与预设的 `max_norm` 阈值进行比较。如果 `全局范数` 超过 `max_norm`，则整个梯度向量 $G$ 将按比例缩放：


$$
G_{\text{裁剪后}} = G \times \frac{\text{max\_norm}}{\text{全局范数}}
$$


如果 `全局范数` 小于或等于 `max_norm`，则梯度保持不变：$G_{\text{裁剪后}} = G$。

这种方法的一个重要特点是它统一缩放所有梯度。这意味着参数空间中总梯度向量的*方向*得以保留；仅其幅度受到限制。这有助于防止步长过大，而不会大幅改变梯度所指示的优化路径。

> 流程图，说明了按全局范数裁剪梯度的过程。

### 实现与阈值选择

深度学习 (deep learning)框架提供了内置函数用于梯度裁剪，使其易于集成到训练循环中。它通常在 `loss.backward()` 调用（计算梯度）*之后*、`optimizer.step()` 调用（根据梯度更新权重 (weight)）*之前*应用。

```python
# PyTorch 示例
# 假设已定义模型、优化器和损失

# 前向传播
outputs = model(inputs)
loss = compute_loss(outputs, targets)

# 反向传播计算梯度
loss.backward()

# 梯度裁剪（在反向传播*之后*，步进*之前*应用）
gradient_threshold = 1.0 
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_threshold)

# 优化器步进（使用可能已裁剪的梯度）
optimizer.step()

# 为下一次迭代清零梯度
optimizer.zero_grad()
```

在 TensorFlow 中，等效操作可能涉及使用 `tf.clip_by_global_norm`，通常在自定义训练循环中，或者在指定时由更高级别的API隐式处理。

`max_norm` 值是一个超参数 (parameter) (hyperparameter)，通常需要进行一些调整。常见值通常在 [0.5, 5.0] 范围内，1.0 是一个常用起始点。

- **过低：** 可能过度减慢训练速度，因为它过度限制梯度幅度，尤其是在训练初期大梯度确实包含有用信息时。
- **过高：** 可能无法防止由真正有问题的梯度爆炸引起的不稳定。

在训练期间监测实际梯度范数（如果你的框架或监测工具允许）有助于确定阈值的选择。如果你观察到频繁的裁剪事件或裁剪前范数非常大，这证实了裁剪的必要性，并有助于调整阈值。否则，观察训练损失曲线以判断稳定性是主要的经验指导。

梯度裁剪是训练大型 Transformer 模型中的标准做法。它起到保护作用，补充了仔细的初始化、归一化 (normalization)层和学习率调度等其他技术，以促进稳定有效的收敛，特别是对于梯度动态可能很复杂的深层架构。

## 参考资料

- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html) — Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning; Publisher: PMLR; Volume: 28; Pages: 1310-1318
  阐述深度网络中梯度爆炸问题并提出梯度裁剪作为稳定训练方法的开创性论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍深度学习理论基础的教科书，详细解释了梯度问题和梯度裁剪等正则化技术。
- [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — PyTorch Team (2023)
  PyTorch官方梯度裁剪工具文档，提供了实用的实现细节和使用示例。
