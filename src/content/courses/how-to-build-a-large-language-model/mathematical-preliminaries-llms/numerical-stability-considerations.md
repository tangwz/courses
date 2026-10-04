---
course: "how-to-build-a-large-language-model"
chapter: "mathematical-preliminaries-llms"
lesson: "numerical-stability-considerations"
sourceId: 5955
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-2-mathematical-preliminaries-llms/numerical-stability-considerations"
title: "数值稳定性考量"
description: "阐述梯度消失/爆炸等问题及归一化的作用。"
order: 4
plots: ["plots/5955-0.json"]
sourceHash: "de2c377fd83aa60e750909e7d8b73f3d5a2a4b9de9c985e122539cadfc37c58a"
sourceCorrections: []
---

虽然矩阵乘法或微分等运算的数学定义是精确的，但它们在计算机上的实现涉及有限精度的浮点数（如32位`float`或16位`half`）。这种有限精度可能在深度神经网络 (neural network)训练过程中引出一些不易察觉，有时却又十分显著的问题，尤其是在大型语言模型中常见的那些非常深的架构。若不妥善处理这些数值稳定性问题，它们可能极大地妨碍甚至完全停止训练过程。

## 梯度消失

在反向传播 (backpropagation)过程中，梯度利用链式法则计算，从输出层反向传播通过网络。每一步都涉及到乘以该层操作的局部梯度（包括激活函数 (activation function)导数）和该层的权重 (weight)。设想一个具有许多层的深层网络。如果这些梯度（特别是激活函数导数或权重矩阵）的模值持续小于1，那么梯度信号在反向传播时会呈指数级缩小。


$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \text{out}_N} \dots \frac{\partial \text{out}_3}{\partial \text{in}_3} \frac{\partial \text{in}_3}{\partial \text{out}_2} \frac{\partial \text{out}_2}{\partial \text{in}_2} \frac{\partial \text{in}_2}{\partial W_1}
$$


如果这个链式乘积中的许多项的模值都小于1，那么对于早期层（如$W_1$）的最终梯度 $\frac{\partial L}{\partial W_1}$ 会变得非常小，接近于零。

这种现象被称为**梯度消失问题**。当梯度消失时，网络早期层的权重几乎得不到更新，网络也就无法有效地从这些层的数据中学习到有意义的表示。这在训练早期深层网络时是一个很大的难题，特别是对于使用Sigmoid或Tanh等激活函数的网络，这些函数的导数在输入值较大或较小时会饱和（接近于零）。

```python
import torch
import matplotlib.pyplot as plt

# Sigmoid导数饱和示例
x = torch.linspace(-10, 10, 200)
sigmoid_x = torch.sigmoid(x)
sigmoid_grad = sigmoid_x * (1 - sigmoid_x) # Sigmoid函数的导数

# 绘图
fig, ax1 = plt.subplots()

color = '#4263eb' # 蓝色
ax1.set_xlabel('输入值 (x)')
ax1.set_ylabel('Sigmoid激活', color=color)
ax1.plot(x, sigmoid_x, color=color, label='Sigmoid(x)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':')

ax2 = ax1.twinx() # 创建一个共享x轴的第二个坐标轴

color = '#f03e3e' # 红色
ax2.set_ylabel('Sigmoid导数', color=color)
ax2.plot(
    x,
    sigmoid_grad,
    color=color,
    linestyle='--',
    label='d(Sigmoid)/dx'
)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0) # 导数非负

fig.tight_layout() # 否则右侧y轴标签可能略有裁剪
#plt.title("Sigmoid Activation and its Derivative") # 为直接显示而移除标题
#plt.show() # 为直接JSON输出而注释掉

# 创建Plotly JSON表示
plotly_json = {
  "data": [
    {
      "x": x.tolist()[::10], # 采样点以保持JSON小巧
      "y": sigmoid_x.tolist()[::10],
      "type": "scatter",
      "mode": "lines",
      "name": "Sigmoid(x)",
      "line": {"color": "#4263eb"}
    },
    {
      "x": x.tolist()[::10],
      "y": sigmoid_grad.tolist()[::10],
      "type": "scatter",
      "mode": "lines",
      "name": "d(Sigmoid)/dx",
      "yaxis": "y2",
      "line": {"color": "#f03e3e", "dash": "dash"}
    }
  ],
  "layout": {
    "xaxis": {"title": "输入值 (x)"},
    "yaxis": {
        "title": "Sigmoid激活",
        "titlefont": {"color": "#4263eb"},
        "tickfont": {"color": "#4263eb"},
        "gridcolor": "#e9ecef"
    },
    "yaxis2": {
        "title": "Sigmoid导数",
        "titlefont": {"color": "#f03e3e"},
        "tickfont": {"color": "#f03e3e"},
        "overlaying": "y",
        "side": "right",
        "range": [0, 0.3],
        "gridcolor": "#e9ecef"
    },
    "legend": {"x": 0.1, "y": 0.9},
    "margin": {"l": 50, "r": 50, "t": 20, "b": 40}
  }
}
```



![Interactive chart](plots/5955-0.json)



> Sigmoid函数的导数很小（最大0.25），并且在输入值很大（正或负）时接近于零。在反向传播过程中将许多小数值相乘会导致梯度消失。

## 梯度爆炸

反之，如果链式法则乘积中的项（梯度、权重 (weight)）的模值持续大于1，那么梯度信号在反向传播 (backpropagation)时会呈指数级增长。这会引发**梯度爆炸问题**。

梯度爆炸会导致模型权重（$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} J(\theta)$）的更新值过大。这些过大的更新可能使优化过程变得不稳定，导致剧烈震荡或完全发散。在实践中，这通常表现为在训练过程中损失函数 (loss function)突然飙升至`NaN`（非数字）或`Inf`（无穷大），因为数值超过了浮点数的表示范围。这对于循环连接或重复多次乘以相同权重矩阵的非常深层网络来说，尤为突出。

## 浮点运算限制

标准的深度学习 (deep learning)通常使用32位浮点数（FP32或`float`）。然而，训练大型语言模型常采用低精度格式，如16位浮点数（FP16或`half`）或BFloat16（BF16），以减少内存消耗并加速计算，尤其是在配备NVIDIA Tensor Cores等专用硬件的设备上。

这些低精度格式的表示范围明显小于FP32，并且精度也较低。

- **FP16**: 表示范围非常有限，容易出现溢出（值变为`Inf`）或下溢（值变为零）。在梯度消失的情况下常见的小梯度，在FP16中很容易变为零，从而停止学习。大梯度或中间激活值可能超出最大可表示值，导致`Inf`或`NaN`。
- **BF16**: 牺牲了精度，但保持了与FP32相似的表示范围。与FP16相比，这使其较不易受到溢出/下溢的影响，但仍可能因精度较低而引入噪声。

使用这些格式需要谨慎处理以保持数值稳定性，我们将在第20章中对此进行更详细的阐述。

## 缓解策略概述

幸运的是，已经发展出多种技术来应对这些稳定性问题，它们构成了训练深度模型的一套标准方法：

1. **细致的初始化：** 适当初始化权重 (weight)有助于从一开始就避免梯度消失或爆炸。Xavier/Glorot和Kaiming初始化（第12章）等技术根据层维度设定初始权重比例。
2. **归一化 (normalization)层：** 诸如批归一化（Batch Normalization）或在Transformer中更为常见的层归一化（Layer Normalization）（第4章）等层会将层内的激活值重新缩放，使其具有零均值和单位方差。这有助于将激活值和梯度保持在合理范围内，从而稳定训练。
3. **梯度裁剪：** 这种技术通过在权重更新步骤之前，限制梯度的最大模值或范数来直接解决梯度爆炸问题。如果梯度范数超过设定的阈值，它会被向下重新缩放。（第17章）。

   ```python
   import torch
   import torch.nn as nn

   # 示例参数和梯度（请替换为您的模型参数）
   param1 = torch.randn(100, 100, requires_grad=True)
   param2 = torch.randn(50, 100, requires_grad=True)
   parameters = [param1, param2]

   # 模拟梯度（例如，在loss.backward()之后）
   if param1.grad is None: # 如果梯度不存在，则创建虚拟梯度
       param1.grad = torch.randn_like(param1) * 100
   if param2.grad is None:
       param2.grad = torch.randn_like(param2) * 50

   # 计算总梯度范数
   total_norm = 0
   for p in parameters:
       if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           total_norm += param_norm.item() ** 2
   total_norm = total_norm ** 0.5
   print(f"Original Gradient Norm: {total_norm:.2f}")

   # 应用梯度裁剪（使用PyTorch工具）
   max_norm = 1.0
   nn.utils.clip_grad_norm_(parameters, max_norm)

   # 裁剪后计算范数
   clipped_total_norm = 0
   for p in parameters:
        if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           clipped_total_norm += param_norm.item() ** 2
   clipped_total_norm = clipped_total_norm ** 0.5
   print(f"Clipped Gradient Norm: {clipped_total_norm:.2f}")

   # 预期输出将显示原始范数可能 > 1.0
   # 裁剪后的范数非常接近 1.0
   ```
4. **激活函数 (activation function)：** 使用非饱和激活函数，如ReLU（修正线性单元）或其变体（GeLU、SwiGLU），与Sigmoid或Tanh相比，有助于缓解梯度消失问题（第11章）。
5. **混合精度训练技术：** 诸如损失缩放（loss scaling）等方法专用于FP16，以动态调整损失函数 (loss function)的规模，从而在反向传播 (backpropagation)过程中有效放大梯度以防止下溢，然后在权重更新前将梯度重新缩放回来（第20章）。

了解这些潜在的数值难题以及应对它们的策略，对于处理现代大型语言模型的规模和深度特点来说是必不可少的。若不仔细考量数值稳定性，训练这些强大的模型在实际中将无法实现。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习基础概念的综合性教材，包括反向传播、梯度问题、激活函数和初始化技术。
- [On the difficulty of training Recurrent Neural Networks](https://proceedings.mlr.press/v28/pascanu13.html) — Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning; Publisher: PMLR; Volume: 28; Pages: 1310-1318
  这篇论文分析了深度网络中的梯度消失和梯度爆炸问题，并提出了梯度裁剪作为一种解决方案。
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  这项开创性工作引入了深度神经网络的混合精度训练，详细介绍了利用损失缩放等技术高效使用FP16。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton (2016)
  Journal: arXiv preprint arXiv:1607.06450; DOI: [10.48550/arXiv.1607.06450](https://doi.org/10.48550/arXiv.1607.06450)
  引入了层归一化作为批归一化的替代方案，因其不依赖于批大小，对循环神经网络和Transformer尤其有效。
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot and Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: JMLR Workshop and Conference Proceedings; Volume: 9; Pages: 249-256; DOI: [10.5555/3104322.3104327](https://doi.org/10.5555/3104322.3104327)
  这篇论文分析了深度网络中激活和梯度的初始分布，提出了Xavier/Glorot初始化方法以缓解梯度消失/爆炸问题。
