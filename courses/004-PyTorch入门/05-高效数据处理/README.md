# 第 5 章：高效数据处理

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling)

[返回课程目录](../README.md)

既然你已经了解如何使用 `torch.nn` 构建模型以及如何用 Autograd 计算梯度，下一步就是有效地为这些模型提供数据。处理大型数据集、进行必要的预处理以及在不耗尽内存的情况下分批加载数据，是深度学习工作中常见的问题。

本章将讲解 PyTorch 管理数据流程的方案：即 `torch.utils.data` 模块。你将学习如何：

*   使用 `Dataset` 类来组织数据。
*   使用预设数据集，例如 `torchvision` 中提供的。
*   使用 `torchvision.transforms` 进行数据转换和增强。
*   使用 `DataLoader` 类高效地分批加载数据、打乱数据，并可能并行加载。

学完本章后，你将能够为你的 PyTorch 项目构建高效的数据流程。

## 小节

- 1. [对专用数据加载器的需求](01-%E5%AF%B9%E4%B8%93%E7%94%A8%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%99%A8%E7%9A%84%E9%9C%80%E6%B1%82.md)
- 2. [使用 \`torch.utils.data.Dataset\`](02-%E4%BD%BF%E7%94%A8%20%60torch.utils.data.Dataset%60.md)
- 3. [内置数据集（例如：TorchVision）](03-%E5%86%85%E7%BD%AE%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%88%E4%BE%8B%E5%A6%82%EF%BC%9ATorchVision%EF%BC%89.md)
- 4. [数据变换 (\`torchvision.transforms\`)](04-%E6%95%B0%E6%8D%AE%E5%8F%98%E6%8D%A2%20%28%60torchvision.transforms%60%29.md)
- 5. [使用 \`torch.utils.data.DataLoader\`](05-%E4%BD%BF%E7%94%A8%20%60torch.utils.data.DataLoader%60.md)
- 6. [自定义DataLoader用法](06-%E8%87%AA%E5%AE%9A%E4%B9%89DataLoader%E7%94%A8%E6%B3%95.md)
- 7. [动手实践：构建数据处理流程](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/quiz)
