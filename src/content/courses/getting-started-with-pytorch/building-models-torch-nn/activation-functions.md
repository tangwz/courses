---
course: "getting-started-with-pytorch"
chapter: "building-models-torch-nn"
lesson: "activation-functions"
sourceId: 2158
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/activation-functions"
title: "激活函数 (ReLU, Sigmoid, Tanh)"
description: "使用 `torch.nn` 中常见的激活函数，在神经网络中引入非线性。"
order: 4
plots: ["plots/2158-0.json", "plots/2158-1.json", "plots/2158-2.json"]
sourceHash: "e82f36bf5aa4de6a95d5156f73bd005b4dbcfc0dffa60c3eb9704797c2dc0f00"
sourceCorrections: []
---

神经网络 (neural network)的表示能力很大程度上得益于在层之间引入非线性。如果只是简单地堆叠线性变换（如 `nn.Linear` 层）而没有任何介入函数，整个网络将简化为一个单一的等效线性变换。无论网络有多少层，它都只能学习输入与输出之间的线性关系。

激活函数 (activation function)是引入这些重要非线性的组成部分。它们逐元素应用于层的输出（常被称为预激活值或logit），在将值传递给下一层之前对其进行转换。PyTorch 在 `torch.nn` 模块中提供了各种各样的激活函数，通常通过将它们实例化为层在模型定义中使用。我们来看看其中最常见的三种：ReLU、Sigmoid 和 Tanh。

## ReLU (修正线性单元)

修正线性单元，简称ReLU，可以说是现代深度学习 (deep learning)中最受欢迎的激活函数 (activation function)，尤其是在卷积神经网络 (neural network) (CNN)中。它的定义非常简单：如果输入为正，它直接输出输入值，否则输出零。

其数学定义为：


$$
\text{ReLU}(x) = \max(0, x)
$$


在 PyTorch 中，可以使用 `nn.ReLU`：

```python
import torch
import torch.nn as nn

# 示例用法
relu_activation = nn.ReLU()
input_tensor = torch.randn(4) # 示例输入张量
output_tensor = relu_activation(input_tensor)

print(f"输入: {input_tensor}")
print(f"ReLU 输出: {output_tensor}")

# 在简单模型中的示例
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x) # 应用 ReLU
        x = self.layer2(x)
        return x

model = SimpleNet()
```



![ReLU 激活函数](plots/2158-0.json)



> ReLU 函数对负输入为零，对正输入为线性。

**优点：**

- **计算效率高：** 计算非常简单（$\max(0, x)$）。
- **减少梯度消失：** 对于正输入，梯度为1，这有助于在训练期间梯度反向传播 (backpropagation)，相比于 Sigmoid 或 Tanh 等饱和函数。
- **引入稀疏性：** 由于负输入被映射到零，这可以导致网络中出现稀疏激活，有时可能是有益的。

**缺点：**

- **ReLU 死亡问题：** 输入始终落在负区间的神经元将输出零。因此，流经它们的梯度也将为零，这意味着它们的权重 (weight)在反向传播期间不会被更新。这些神经元实际上“死亡”了，不再对学习有贡献。Leaky ReLU 或 Parametric ReLU (PReLU) 等变体试图解决此问题。
- **非零中心：** 输出始终为非负值。

## Sigmoid

Sigmoid 函数，有时也称为逻辑函数，将其输入压缩到 0 到 1 的范围内。它在历史上很受欢迎，尤其是在二元分类模型的输出层，其中输出代表一个概率。

其数学形式为：


$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$


在 PyTorch 中，可以使用 `nn.Sigmoid`：

```python
import torch
import torch.nn as nn

# 示例用法
sigmoid_activation = nn.Sigmoid()
input_tensor = torch.randn(4) # 示例输入张量
output_tensor = sigmoid_activation(input_tensor)

print(f"输入: {input_tensor}")
print(f"Sigmoid 输出: {output_tensor}")
```



![Sigmoid 激活函数](plots/2158-1.json)



> Sigmoid 函数将任意实数平滑地映射到 (0, 1) 的范围内。

**优点：**

- **输出易于理解：** (0, 1) 的范围便于表示概率。
- **梯度平滑：** 函数处处可微，提供平滑的梯度。

**缺点：**

- **梯度消失：** 对于非常大或非常小的输入，函数会饱和（输出接近 1 或 0），梯度变得非常接近零。这会严重减缓或停止深度网络的学习，因为梯度难以通过多层反向传播 (backpropagation)。
- **非零中心：** 输出始终为正，这有时会减缓收敛速度，相比于零中心激活函数 (activation function)。
- **计算成本更高：** 指数函数比 ReLU 的简单比较成本更高。

由于梯度消失问题，Sigmoid 在今天的深度网络隐藏层中不如 ReLU 常用，但它在特定任务（例如二元分类或多标签分类）的输出层中仍然适用。

## Tanh (双曲正切)

双曲正切函数，即 Tanh 函数，在数学上与 Sigmoid 相关，但将其输入压缩到 (-1, 1) 的范围内。

其定义为：


$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = 2 \sigma(2x) - 1
$$


在 PyTorch 中，可以使用 `nn.Tanh`：

```python
import torch
import torch.nn as nn

# 示例用法
tanh_activation = nn.Tanh()
input_tensor = torch.randn(4) # 示例输入张量
output_tensor = tanh_activation(input_tensor)

print(f"输入: {input_tensor}")
print(f"Tanh 输出: {output_tensor}")
```



![Tanh 激活函数](plots/2158-2.json)



> Tanh 函数将任意实数平滑地映射到 (-1, 1) 的范围内。

**优点：**

- **零中心输出：** 与 Sigmoid 不同，Tanh 的输出以零为中心，这通常有助于模型在训练期间的收敛。零中心数据通常与基于梯度的优化方法配合得更好。
- **梯度平滑：** 与 Sigmoid 类似，它处处可微。

**缺点：**

- **梯度消失：** 与 Sigmoid 类似，Tanh 也会在很大正值或负值输入时出现饱和，导致深度网络中梯度消失。虽然由于其零中心性质，在隐藏层中它通常比 Sigmoid 更受青睐，但它仍然容易受到此问题的影响。
- **计算成本更高：** 涉及指数函数，使其比 ReLU 成本更高。

在 ReLU 兴起之前，Tanh 在隐藏层中通常比 Sigmoid 更受青睐，主要因为其零中心输出范围。它仍然常见于循环神经网络 (neural network) (RNN) 和 LSTM 中。

## 选择激活函数 (activation function)

没有一个“最佳”激活函数适用于所有情况。然而，有一些通用指导原则：

- **ReLU** 通常是前馈网络和 CNN 中隐藏层的默认选择，因为它高效且能有效缓解正输入时的梯度消失问题。从 ReLU 开始，如果遇到诸如死亡神经元之类的问题，再考虑其他替代方案。
- 如果怀疑存在“ReLU 死亡”问题，**Leaky ReLU** 或 **Parametric ReLU (PReLU)** 是不错的替代方案。它们为负输入引入了一个小的非零斜率。
- **Tanh** 在隐藏层中可能很有效，尤其是在 RNN 中，因为它有零中心输出。
- **Sigmoid** 通常保留用于 *输出层*，当你需要用于二元或多标签分类的概率时。因为梯度消失问题，避免在深度隐藏层中大量使用它。

通常需要进行实验，以找到适用于特定架构和数据集的最佳激活函数。在 PyTorch 中，更换激活函数很简单，通常只需更改一行代码，即激活模块实例化或在 `nn.Module` 的 `forward` 方法中被调用的位置。

## 参考资料

- [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf) — Xavier Glorot, Antoine Bordes, Yoshua Bengio (2011)
  Journal: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 15; Pages: 315-323; DOI: [10.5555/3104322.3104360](https://doi.org/10.5555/3104322.3104360)
  这篇开创性论文引入了整流线性单元（ReLU）作为激活函数，展示了其在深度学习模型中的优势并解决了梯度消失问题。
- [\`torch.nn\` - PyTorch documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch (2024)
  PyTorch神经网络模块的官方文档，详细介绍了ReLU、Sigmoid和Tanh等激活函数的实现和使用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，提供了神经网络的理论基础和实践见解，包括对激活函数及其属性的详细讨论。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes](https://cs231n.github.io/) — Fei-Fei Li, Andrej Karpathy, Justin Johnson (2024)
  Publisher: Stanford University
  一门备受推崇的大学课程的讲义，提供了对激活函数、其作用以及神经网络设计中实际考虑因素的清晰解释。
