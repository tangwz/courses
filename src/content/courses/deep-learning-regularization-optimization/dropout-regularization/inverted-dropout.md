---
course: "deep-learning-regularization-optimization"
chapter: "dropout-regularization"
lesson: "inverted-dropout"
sourceId: 4931
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/inverted-dropout"
title: "反转Dropout实现"
description: "描述实践中广为采用的‘反转Dropout’技术。"
order: 4
plots: []
sourceHash: "f53fae46c0b6da19b220c407562b8b789f51fa7e2d6e0bb1f972cbb2692aff90"
sourceCorrections: []
---

标准的 Dropout 过程涉及在训练期间随机丢弃单元，然后在测试期间按保留概率 $p$ 缩小激活值。这种方法在训练和测试之间造成了轻微的不对称，因为网络的正向传播计算会根据其是在训练还是评估模式而变化。

一种普遍且巧妙的实现方法，称作 **反转Dropout**，解决了这个问题。反转Dropout不是在测试时缩小激活值，而是在 *训练期间* 执行缩放。

它的工作方式如下：

1. **生成Dropout掩码：** 就像标准Dropout一样，在训练正向传播期间，会创建一个二进制掩码 $m$，其中每个元素对应层中的一个神经元。元素以概率 $p$ (保留概率) 设置为 1，以概率 $1-p$ (丢弃概率) 设置为 0。
2. **应用掩码：** 该掩码应用于层的激活值 $a$：$a_{masked} = a * m$。
3. **训练期间向上缩放：** 这是主要区别。结果激活值 ($a_{masked}$) 立即通过除以保留概率 $p$ 来 *向上缩放*：
   $a_{dropout} = \frac{a_{masked}}{p}$

为什么这样做有效？通过在训练期间向上缩放 *保留* 神经元的激活值，我们确保层输出的期望值与没有 Dropout 时相同。如果一个神经元被保留 (概率为 $p$)，其输出会乘以 $1/p$ 进行缩放。如果它被丢弃 (概率为 $1-p$)，其输出为 0。因此，期望输出为 $p \times (\frac{a}{p}) + (1-p) \times 0 = a$。

### 反转Dropout的优点

反转Dropout的主要优点是，**测试时** 的正向传播保持不变。您无需记住按 $p$ 缩放激活值。您只需按原样使用网络，从而有效地关闭了 Dropout 机制 (即使用保留概率为 1)。这大大简化了推理 (inference)和部署所需的代码。

大多数现代深度学习 (deep learning)框架，包括 PyTorch 和 TensorFlow，在您使用其内置的 Dropout 层时，默认情况下都会使用反转Dropout技术来实现 Dropout。

### 实现示例

在 PyTorch 中，您通常在网络定义中将 Dropout 作为一层添加。框架会在训练期间自动处理反向缩放。

```python
import torch
import torch.nn as nn

# 定义保留概率 (通常为 1 - 丢弃概率)
p_keep = 0.8 # 相当于 0.2 的 Dropout 概率

# 定义一个 Dropout 层
dropout_layer = nn.Dropout(p=1.0 - p_keep) # nn.Dropout 接受丢弃概率

# 示例激活张量 (例如，来自前一层)
# 批量大小为 2，特征数量为 10
activations = torch.randn(2, 10)

# --- 训练期间 ---
# 将模型设置为训练模式
dropout_layer.train()
output_train = dropout_layer(activations)

# 观察输出：
# - 某些元素将为零。
# - 非零元素将按 1/p_keep (1 / 0.8 = 1.25) 向上缩放
print("激活值:\n", activations)
print("\n训练期间的输出 (反转Dropout):\n", output_train)
# 非零元素的期望幅度大约是原始幅度的 1.25 倍

# --- 测试/推理期间 ---
# 将模型设置为评估模式
dropout_layer.eval()
output_test = dropout_layer(activations)

# 观察输出：
# - 没有元素被置为零。
# - 没有应用缩放。输出等于输入。
print("\n测试期间的输出 (Dropout 未激活):\n", output_test)

# 检查测试输出是否与激活值相同
print("\n测试输出是否与输入相同？", torch.allclose(output_test, activations))
```

从示例中可以看出，调用 `model.train()` 会启用 Dropout (带反向缩放)，而 `model.eval()` 会禁用它，使该层原样通过输入。这种顺畅的处理方式是反转Dropout成为标准实现的原因。您只需定义一次 Dropout 层，框架就会根据模型的模式 (训练或评估) 管理其行为。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  介绍了奠基性的Dropout正则化技术，反向Dropout是其高效的实现方式。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 326-328 (Chapter 7.12)
  提供了关于Dropout的详尽解释，探讨了其理论基础和常见的实现策略，包括缩放。
- [torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — PyTorch Developers (2024)
  PyTorch `nn.Dropout`层的官方文档，详细说明了其参数并确认使用了反向Dropout方法。
- [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow Keras `Dropout`层的官方文档，概述了其用法并间接展示了反向Dropout的行为。
