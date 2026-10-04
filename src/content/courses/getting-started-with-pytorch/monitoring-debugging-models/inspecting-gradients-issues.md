---
course: "getting-started-with-pytorch"
chapter: "monitoring-debugging-models"
lesson: "inspecting-gradients-issues"
sourceId: 2227
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/inspecting-gradients-issues"
title: "检查梯度问题（消失/爆炸）"
description: "检查训练期间梯度大小的方法，以诊断梯度消失或梯度爆炸问题。"
order: 4
plots: ["plots/2227-0.json"]
sourceHash: "f150fa9a6a19e9703bfaa6b8f8b697cf4e19c29c06d1347f63e880c7e67bd0ca"
sourceCorrections: []
---

有效的训练依赖于反向传播 (backpropagation)期间计算的梯度。这些梯度引导优化器更新模型参数 (parameter)以最小化损失函数 (loss function)。然而，这些梯度的大小有时会成为问题，导致训练不稳定或停滞。两个常见问题是梯度消失和梯度爆炸。了解如何检查梯度是诊断训练中出现的问题的一项重要技能。

### 理解梯度问题

在反向传播 (backpropagation)过程中，梯度使用链式法则逐层计算。在深度网络中，这涉及将许多小数字（导数）相乘。

- **梯度消失：** 当梯度从输出层向初始层反向传播时变得极其小，就会发生这种情况。结果是，初始层的权重 (weight)和偏置 (bias)更新得非常缓慢，甚至完全不更新。网络基本上停止从早期层的数据中学习有意义的特征。这在深度网络中尤其常见，当使用 sigmoid 或 tanh 等激活函数 (activation function)时，这些函数在大多数区域的导数都小于 1。
- **梯度爆炸：** 这是相反的问题，即在反向传播过程中梯度变得过大。大梯度会导致模型权重发生显著更新。这可能导致优化过程变得不稳定，损失剧烈波动甚至变成 `NaN`（非数字），从而有效地停止训练。梯度爆炸可能由于不佳的权重初始化、过高的学习率或某些网络结构引起，尤其是在循环神经网络 (neural network) (RNN)中。

### 在 PyTorch 中检测梯度问题

PyTorch 的 Autograd 系统计算梯度，并将其存储在 `requires_grad=True` 的张量的 `.grad` 属性中。这些梯度在 `loss.backward()` 调用后即可访问，并在 `optimizer.step()` 更新模型参数 (parameter)或 `optimizer.zero_grad()` 清除梯度之前保持可用。

#### 监控整体梯度范数

一个常用做法是监控模型中所有可训练参数的梯度整体大小（范数）。L2 范数（欧几里得范数）是常用的一种。非常小的范数表明梯度消失，而非常大或 `NaN` 的范数则表明梯度爆炸。

以下是在训练循环中计算并记录总梯度范数的方法：

```python
# 在训练循环中，在 loss.backward() 之后

total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.detach().data.norm(2) # 计算此参数梯度的 L2 范数
        total_norm += param_norm.item() ** 2      # 平方和
total_norm = total_norm ** 0.5                   # 平方和的平方根

print(f"总梯度范数: {total_norm}")
# 通常，你会使用 TensorBoard 或其他日志框架来记录此值
```

随时间监控此值可以提供信息：



![训练期间的总梯度范数](plots/2227-0.json)



> 模型梯度总 L2 范数随训练步数变化的趋势，以对数尺度显示。稳定的训练显示相对一致的范数，梯度爆炸显示快速增加（常导致 NaN），梯度消失则显示趋近于零的下降。

#### 逐层检查梯度

有时，梯度问题可能局限于特定层。你可以直接检查单个参数的梯度。

```python
# 在训练循环中，在 loss.backward() 之后

# 示例：检查第一个卷积层的权重梯度
if hasattr(model, 'conv1') and model.conv1.weight.grad is not None:
    conv1_grad_mean = model.conv1.weight.grad.abs().mean().item()
    conv1_grad_max = model.conv1.weight.grad.abs().max().item()
    print(f"层 conv1 - 平均绝对梯度: {conv1_grad_mean:.6f}, 最大绝对梯度: {conv1_grad_max:.6f}")

# 示例：检查特定线性层的偏置梯度
if hasattr(model, 'fc2') and model.fc2.bias.grad is not None:
    fc2_bias_grad_norm = model.fc2.bias.grad.norm(2).item()
    print(f"层 fc2 (偏置) - L2 范数: {fc2_bias_grad_norm:.6f}")
```

查看平均或最大绝对梯度值，或特定层的范数，可以帮助确定梯度是在减小还是在不受控制地增长。使用直方图（例如，使用 Matplotlib 或通过 TensorBoard 记录）来可视化某一层的梯度值分布也很有用。

#### 使用钩子进行更细致的检查

为了进行更详细的调试，PyTorch 提供了钩子。可以在任何 `nn.Module` 上注册一个反向钩子（`register_full_backward_hook`）。当为该模块计算了梯度时，此钩子会执行，允许你检查甚至修改通过它的梯度（`grad_input`，`grad_output`）。尽管功能强大，但钩子会增加复杂性，通常在简单检查方法不足时使用。

#### 观察损失行为

间接来看，训练损失本身就是一个强有力的指示器。

- **损失变为 `NaN`：** 几乎总是梯度爆炸或数学上无效操作（如 `log(0)`）的迹象。
- **损失下降极其缓慢或过早停滞：** 可能是梯度消失的症状，特别是如果涉及初始层。
- **损失剧烈波动：** 可能表示梯度爆炸或学习率过高。

### 缓解的初步措施

检测梯度问题是第一步。解决这些问题通常涉及其他地方更详细介绍的技术，但常见策略包括：

- **梯度裁剪：** 对于梯度爆炸，在优化器步骤之前限制梯度的最大范数或值。`torch.nn.utils.clip_grad_norm_` 或 `torch.nn.utils.clip_grad_value_` 是标准实用工具。
- **激活函数 (activation function)：** 在深度网络中用 ReLU 或其变体（Leaky ReLU, PReLU, ELU）替换 sigmoid/tanh，这些函数通常具有问题较少的导数特性。
- **权重 (weight)初始化：** 使用旨在维持各层方差的初始化方案，例如 Xavier/Glorot 或 He 初始化。
- **批量归一化 (normalization)：** 通过归一化层输入，有助于稳定学习并可以缓解梯度消失/爆炸问题。
- **网络架构：** 使用跳跃连接或残差连接（如 ResNets 中）为梯度流动提供替代路径，从而在非常深的神经网络 (neural network)中对抗梯度消失。
- **学习率调整：** 降低学习率有时可以帮助解决梯度爆炸问题，尽管它可能无法解决根本原因。

模型工作后，你不一定需要在每次训练运行时都检查梯度，但当训练不稳定或无效时，它是一个必不可少的诊断工具。通过监控梯度范数和检查单个层的梯度，你可以获得关于训练动态的有价值信息，并发现潜在的梯度消失或梯度爆炸问题。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，提供了深度学习的理论背景，包括反向传播、梯度消失/爆炸以及各种缓解策略的详细解释。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Contributors (2023)
  Publisher: PyTorch
  PyTorch官方文档，详细说明了如何使用Autograd系统计算、存储和访问梯度，这对于检测和调试梯度问题至关重要。
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.pdf) — Sergey Ioffe and Christian Szegedy (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Volume: 37; Pages: 448-456; DOI: [10.48550/arXiv.1502.03167](https://doi.org/10.48550/arXiv.1502.03167)
  介绍了批标准化，这是一种通过归一化层输入来稳定和加速深度神经网络训练的常用技术，从而有助于缓解梯度消失/爆炸问题。
- [Deep Residual Learning for Image Recognition](https://ieeexplore.ieee.org/document/7780459/) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  提出了带有跳跃连接的残差网络（ResNets），这是一种通过提供替代梯度路径来有效解决超深网络中梯度消失问题的基础架构。
