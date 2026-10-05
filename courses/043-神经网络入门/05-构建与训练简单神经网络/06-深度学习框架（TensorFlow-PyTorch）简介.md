# 深度学习框架（TensorFlow/PyTorch）简介

来源：[原文](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/intro-dl-frameworks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

从零开始构建神经网络 (neural network)，能很好地了解其底层工作方式。这个过程涉及定义层、执行前向传播、计算损失、通过反向传播 (backpropagation)计算梯度以及使用梯度下降 (gradient descent)更新参数 (parameter)。但是，对于规模更大、更复杂的网络，手动实现所有这些步骤会变得繁琐、易出错且计算效率低下。

这就是 TensorFlow 和 PyTorch 等深度学习 (deep learning)框架派上用场的地方。它们是专门的库，旨在优化神经网络及其他机器学习 (machine learning)模型的开发、训练和部署过程。您可以把它们看作强大的工具集，它们能处理许多底层实现细节，让您可以专注于模型架构和训练策略。

### 为何使用深度学习 (deep learning)框架？

与手动实现相比，框架提供了一些重要优势：

1. **抽象与便利：** 它们提供高级 API，包含用于常见任务的预构建组件。您通常只需几行代码就能定义复杂的网络层（全连接层、卷积层、循环层）、选择激活函数 (activation function)（ReLU、Sigmoid、Tanh）、选择损失函数 (loss function)（MSE、交叉熵）并应用优化器（SGD、Adam、RMSprop）。这大大加快了开发时间。
2. **自动微分：** 这可能是最重要的功能。框架无需您手动推导和实现反向传播 (backpropagation)的复杂链式法则计算，而是自动计算损失函数相对于网络参数 (parameter)的梯度（例如，$\frac{\partial L}{\partial W}$ 和 $\frac{\partial L}{\partial B}$）。您只需定义前向传播（即网络架构和数据流向），框架就会在后台构建计算图，以便在反向传播过程中高效地计算梯度。
3. **计算效率：** 框架基于高度优化的 C++ 或 CUDA（针对 NVIDIA GPU）后端构建。数学运算，特别是对神经网络 (neural network)很重要的矩阵乘法，执行速度比单独使用 NumPy 等标准 Python 实现快得多。
4. **GPU 加速：** 训练深度神经网络可能计算量很大。框架与图形处理器（GPU）提供了良好的集成，GPU 可以比 CPU 更快地执行训练所需的并行计算。在框架内使用 GPU 加速通常只需很少的代码修改。
5. **社区与生态：** 流行的框架拥有庞大且活跃的社区、全面的文档、大量教程以及可用于各种任务的预训练 (pre-training)模型。这种生态使得查找解决方案、学习新技术和在现有工作基础上进行构建变得更容易。

### 主要框架：TensorFlow 和 PyTorch

当今最广泛使用的两个深度学习 (deep learning)框架是 TensorFlow（由 Google 开发）和 PyTorch（主要由 Meta AI 开发）。

- **TensorFlow：** 最初以其静态计算图方法（先定义图，再执行图）而闻名，TensorFlow（特别是其高级 API Keras）现已变得非常灵活。Keras 提供了一个用户友好的界面，用于构建和训练可在 TensorFlow（或其他后端）上运行的模型。
- **PyTorch：** 由于其动态计算图（图在计算运行时即时定义），常受研究界青睐，对某些用户来说，这感觉更具“Python风格”且更易于调试。其界面在构建和训练模型方面也很直观。

尽管它们在图执行模型和 API 风格上存在历史差异，但两个框架的现代版本提供了相似的功能和灵活度。两者都支持自动微分、GPU 加速、分布式训练，并拥有丰富的生态系统。选择哪个框架通常取决于个人偏好、项目需求或团队习惯。

### 框架如何简化流程

让我们对比一下我们学过的步骤与它们如何对应于框架的使用：

**手动实现：**

1. 使用数学/NumPy 定义网络结构（层、激活函数 (activation function)）。
2. 初始化权重 (weight) $W$ 和偏置 (bias) $b$。
3. **开始训练循环：**
   a. 获取一批数据。
   b. **前向传播：** 手动逐层计算预测。
   c. **计算损失：** 使用选定的损失函数 (loss function)公式。
   d. **反向传播 (backpropagation)：** 使用链式法则手动计算梯度 $\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial B}$。
   e. **更新参数 (parameter)：** 应用梯度下降 (gradient descent)更新规则：$W = W - \eta \frac{\partial L}{\partial W}$。
4. 重复循环，进行多个 epoch/批次。
5. 手动监控损失/准确率。

**框架实现：**

1. 使用框架的层 API 定义网络结构（例如，`tf.keras.Sequential` 或 `torch.nn.Sequential`）。
2. 框架处理参数初始化（可选择自定义）。
3. **配置训练：**
   a. 选择优化器（例如，'adam'，`torch.optim.Adam`）。
   b. 选择损失函数（例如，'mse'，`torch.nn.MSELoss`）。
   c. 指定要监控的指标（例如，'accuracy'）。
4. **开始训练：** 调用诸如 `model.fit(data, epochs=...)` 的函数（TensorFlow/Keras），或编写在 `loss.backward()` 后调用 `optimizer.step()` 的循环（PyTorch）。
   - **前向传播：** 数据传递给模型时在内部处理。
   - **损失计算：** 根据配置在内部处理。
   - **反向传播（自动微分）：** 由 `loss.backward()` 或在 `fit` 函数内自动处理。
   - **参数更新：** 由 `optimizer.step()` 或在 `fit` 函数内自动处理。
5. 框架通常提供用于监控和记录进度的内置工具。

以下是工作流程对比的简化视图：

> 这是一个对比，说明了使用深度学习 (deep learning)框架与从基本操作开始实现所有内容相比，显式手动步骤的减少。框架封装了核心的前向、反向和更新逻辑。

虽然本课程侧重于了解基本构成要素，并通常使用 NumPy 以便清晰，但要高效构建实用的大规模神经网络 (neural network)，转向 TensorFlow 或 PyTorch 等框架是必要的。它们提供了实现、训练和评估模型的必要工具，使您无需纠缠于重复的底层代码，从而能够更快地进行实验并处理更复杂的问题。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  学习和使用 PyTorch 的官方指南，涵盖其 API、模块和训练工作流。
- [TensorFlow Core APIs Documentation](https://www.tensorflow.org/api_docs/python/tf) — TensorFlow Authors (2024)
  TensorFlow 核心 API 的官方参考资料，包括 Keras，提供了构建和训练模型的详细用法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖了深度学习的数学和概念基础，包括框架所自动化的自动微分和计算图。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  一本使用 Keras 和 TensorFlow 构建和训练深度学习模型的实践指南，通过示例展示了框架的优势。

---

[上一节](05-%E7%9B%91%E6%8E%A7%E8%AE%AD%E7%BB%83%E8%BF%9B%E7%A8%8B.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%88%86%E7%B1%BB%E5%99%A8.md)
