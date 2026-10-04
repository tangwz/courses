---
course: "introduction-to-deep-learning"
chapter: "activation-functions-architecture"
lesson: "tanh-activation"
sourceId: 5027
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/tanh-activation"
title: "双曲正切（Tanh）激活函数"
description: "了解 Tanh 激活函数，作为 Sigmoid 的替代。"
order: 3
plots: ["plots/5027-0.json"]
sourceHash: "70ef2b2f957d49f4d37662d64ceaffc53a107ffea334edb817cb70c49715c8c9"
sourceCorrections: []
---

双曲正切，通常称为 Tanh，是一种常见的 S 形激活函数 (activation function)。与 Sigmoid 等其他 S 形函数类似，Tanh 引入非线性，使网络能从简单的线性关系中学到复杂模式。然而，由于其输出范围，Tanh 在特定情境下具有独特的优点。

从数学上看，Tanh 函数的定义如下：


$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$


有趣的是，Tanh 与 Sigmoid 函数关系密切。它可以表示为 Sigmoid 的缩放和偏移形式：


$$
tanh(x) = 2 \cdot sigmoid(2x) - 1
$$


### Tanh 的属性

我们来看一下 Tanh 函数的一些属性：

1. **输出范围：** 与 Sigmoid 不同，Sigmoid 输出值在 0 到 1 之间，而 Tanh 输出值在 $(-1, 1)$ 范围内。这意味着输出可以是负值、零或正值。
2. **零中心：** Tanh 的输出以零为中心。这通常被认为是 Sigmoid 的一个优点。当激活值以零为中心时，反向传播 (backpropagation)期间的梯度更为均衡，这可能导致训练期间更快地收敛，因为权重 (weight)更新不太可能始终朝一个方向推动。
3. **形状：** Tanh 也具有 S 形，与 Sigmoid 相似。与 Sigmoid 相比，它在 $x=0$ 附近更陡峭。
4. **非线性：** 它是一个非线性函数，有助于多层网络近似复杂函数。

以下是 Tanh 函数及其导数的可视化图示：



![Tanh 激活函数及其导数](plots/5027-0.json)



> Tanh 函数将输入映射到 (-1, 1) 范围，并以零为中心。当输入为 0 时，其导数达到峰值 1，并且随着输入大小的增加而趋近于 0。

### 优点与不足

**优点：**

- **零中心输出：** 如前所述，(-1, 1) 的输出范围意味着平均激活值更接近零。这有助于避免反向传播 (backpropagation)期间梯度更新中的强烈偏差，与 Sigmoid 相比，这可能加快训练过程。
- **更陡峭的梯度：** 在零附近，Tanh 的梯度比 Sigmoid ($0.25$) 更陡峭 ($1$)。这有时会带来更强的初始梯度流。

**不足：**

- **梯度消失：** 与 Sigmoid 类似，Tanh 也存在梯度消失问题。对于较大的正或负输入值，函数会饱和（变平），导数变得非常接近零。这会使较深网络的学习停滞，因为梯度在通过包含饱和 Tanh 单元的层时难以回传。
- **计算成本：** 尽管在大多数现代应用中这不是一个主要因素，但涉及指数的计算使得 Tanh 比 ReLU 等函数在计算上略微耗时。

### 实际使用

过去，Tanh 因其零中心输出，常被用于隐藏层，优于 Sigmoid。它普遍用于前馈网络，尤其是在某些类型的循环神经网络 (neural network)（RNN）中。

然而，随着 ReLU 及其变体（我们接下来会谈到）的兴起，Tanh 作为标准前馈网络和卷积神经网络（CNN）隐藏层的默认选择已不再那么常见。它仍然在特定情境中发挥作用，尤其是在 LSTM 和 GRU（某种类型的 RNN）的门控机制中，其 (-1, 1) 范围是有益的。

以下是如何在 PyTorch 层中使用 Tanh 的示例：

```python
import torch
import torch.nn as nn

# 示例输入张量
input_tensor = torch.randn(5, 10) # 批大小为 5，每个样本有 10 个特征

# 定义一个使用 Tanh 激活的层
# 选项 1：使用 nn.Tanh() 作为模块
tanh_activation = nn.Tanh()
output_tensor = tanh_activation(input_tensor)

# 选项 2：直接使用 torch.tanh（函数式方法）
output_tensor_functional = torch.tanh(input_tensor)

# 定义一个线性层，后接 Tanh
linear_layer = nn.Linear(in_features=10, out_features=20)
activated_output = torch.tanh(linear_layer(input_tensor))

print("输入形状：", input_tensor.shape)
print("输出形状（模块方式）：", output_tensor.shape)
print("输出形状（函数方式）：", output_tensor_functional.shape)
print("输出形状（线性层 + Tanh）：", activated_output.shape)
print("\n输出示例（第一个元素，前 5 个特征）：\n", activated_output[0, :5])
```

虽然 Tanh 解决了 Sigmoid 非零中心的问题，但它没有解决饱和函数本身具有的梯度消失问题。这一局限性促成了 ReLU 及其变体的发展和广泛使用，我们将在下节中介绍这些内容。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/contents/mlp.html) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6: Deep Feedforward Networks
  一本基础教科书，提供了关于激活函数（包括 Tanh）的全面理论背景、其特性以及深度神经网络中的梯度消失问题。
- [Tanh - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch 中 Tanh 激活函数的官方文档，展示了其作为神经网络模块和函数式 API 的实际实现。
- [Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/#actfuncs) — Andrej Karpathy, Justin Johnson, Fei-Fei Li (2024)
  Publisher: Stanford University
  来自一门备受推崇的深度学习课程的讲义，清晰地解释了激活函数（包括 Tanh）的特性，并与 Sigmoid 进行了比较。
