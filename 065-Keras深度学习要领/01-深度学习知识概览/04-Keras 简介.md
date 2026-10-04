# Keras 简介

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-1-intro-deep-learning-concepts/introduction-to-keras)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了成功地构建和训练深度学习 (deep learning)网络，需要对人工神经元、网络结构以及它们如何通过监督方法学习的原理有所了解。对于任何非简单的网络来说，手动实现正向传播和反向传播 (backpropagation)的数学运算会迅速变得复杂且容易出错。这时，深度学习框架就派上用场了，它们提供高层构建模块和优化过的操作。Keras 便是其中一个最受欢迎且用户友好的框架。

“Keras 是一个用于神经网络 (neural network)的高级 API (应用程序编程接口)。可以将其视为一个规范，或者一个用户友好的控制面板，用于定义、训练和评估深度学习模型。它注重开发者体验，让您只需相对较少的代码行数即可构建复杂的模型。Keras 的设计遵循指导原则，使其特别适合于学习深度学习知识和构建应用程序：”

- **易用性：** Keras API 的设计一致、直观，在常见使用场景下所需精力极少。这显著降低了认知负担并加快了开发速度。
- **模块化：** Keras 模型通常通过连接可配置的构建块（如层、优化器、激活函数 (activation function)）来构建。这使得尝试不同架构变得容易。
- **易于扩展：** 您可以轻松创建自定义构建块（层、损失函数 (loss function)等），以支持新的研究思路或特定的项目需求。
- **原生 Python：** Keras 模型使用标准 Python 代码定义，使其易于集成到现有 Python 工作流程中，并利用丰富的 Python 数据科学生态系统（NumPy、Scikit-learn、Pandas 等）。

### Keras 及其计算后端

需要了解的一个重要方面是，Keras 本身不直接执行底层张量操作（如矩阵乘法和梯度计算）。相反，它作为一个接口，依赖于一个底层**后端引擎**来完成这些优化的计算。

从历史上看，Keras 与 TensorFlow 密切相关。然而，从 Keras 3 开始，该框架真正实现了多后端支持。这意味着您只需编写一次 Keras 代码，即可选择在不同的优化张量库上运行它：

1. **TensorFlow：** 由 Google 开发的一个成熟、广泛使用的框架，以其生产部署能力（TensorFlow Serving、TensorFlow Lite）和广泛的生态系统而闻名。
2. **PyTorch：** 由 Meta AI 开发的一个流行框架，因其动态计算图和更“Pythonic”的风格而在研究社区中备受青睐。
3. **JAX：** Google 用于数值计算和机器学习 (machine learning)研究的高性能库，以其函数变换（如自动微分和向量 (vector)化）而闻名。

> Keras 作为一个高级接口，让您可以选择一个底层后端（PyTorch、TensorFlow 或 JAX）进行数值计算。

在本课程中，**我们将主要使用 Keras 3 和 PyTorch 后端。** 这一选择反映了 PyTorch 日益增长的使用率，尤其是在研究中，及其备受赞誉的开发者体验。然而，Keras 3 的优点在于，您编写的大部分代码都可以在不同后端之间移植，只需少量或无需修改，为您提供了宝贵的灵活性。

### 为何选择 Keras？

Keras 提供多个优势，尤其是在您学习和构建标准深度学习 (deep learning)模型时：

- **快速原型开发：** 其简单、一致的 API 让您可以快速定义和训练模型，从而实现想法的快速迭代。构建标准网络通常只需几行代码，而与低层框架中可能需要数百行代码相比，这大大简化了流程。
- **广泛兼容性：** 借助 Keras 3，您的技能和代码在不同计算后端之间更具可移植性。
- **内置支持常见架构：** Keras 提供预置层和实用工具，用于构建卷积神经网络 (neural network) (CNN)、循环神经网络 (RNN) 和其他标准架构，这些我们将在后续章节中讲解。
- **庞大社区和生态系统：** 广泛使用意味着有大量的文档、教程、社区支持，以及丰富的预训练 (pre-training)模型可供使用。

### Keras 在您的工作流程中

通常，使用 Keras 涉及以下步骤：

1. **定义模型：** 您可以使用 Sequential 或 Functional API 将层堆叠在一起，创建所需的网络架构（在第 2 章中讲解）。
2. **编译模型：** 您通过指定优化器、损失函数 (loss function)和评估指标来配置学习过程（在第 3 章中讲解）。
3. **训练模型：** 您使用 `fit()` 方法将模型拟合到您的训练数据上（在第 3 章中讲解）。
4. **评估模型：** 您评估模型在未见过数据上的表现（在第 3 章和第 6 章中讲解）。
5. **进行预测：** 您使用训练好的模型在新数据上生成预测。

Keras 为这些步骤的每一步都提供了清晰、高级的函数，抽象化了大部分底层复杂性。

有了这些了解，知道了 Keras 是什么以及它为何是深度学习 (deep learning)的一个有用工具，接下来要确保您已安装并准备好使用它。下一节将指导您设置 Python 环境，使其与 TensorFlow（默认包含 Keras 3）或 PyTorch 后端一起使用 Keras。

## 参考资料

- [Keras Documentation](https://keras.io/) — Keras team (2024)
  Keras的官方资源，提供构建和训练神经网络的全面指南和API参考。
- [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras的创建者撰写，这本书使用Keras提供了深度学习的实践和概念介绍，涵盖了其设计理念和应用。

---

[上一节](03-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0.md) · [下一节](05-%E9%85%8D%E7%BD%AEKeras%E7%8E%AF%E5%A2%83.md)
