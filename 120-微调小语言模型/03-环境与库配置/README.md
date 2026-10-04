# 第 3 章：环境与库配置

来源：[原章节](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-3-environment-and-library-setup)

[返回课程目录](../README.md)

在训练小语言模型之前，需要建立稳定的软件环境。训练算法高度依赖特定的库来管理硬件资源、处理文本并有效更新神经网络权重。即便小语言模型比超大规模模型占用的显存更少，但配置不当仍会迅速导致内存不足报错或运行速度极慢。

本章将指导你配置本地模型训练所需的工具。首先，我们要安装支持 CUDA 的 PyTorch，从而开启 GPU 硬件加速。这些框架屏蔽了神经网络底层繁杂的数学运算。你无需手动编写像 $Y = WX + b$ 这样的矩阵乘法，也不必为损失计算编写如下自定义函数：

$$L = -\sum y_i \log(\hat{y}_i)$$

而是直接调用成熟的库，由它们在后台原生处理这些计算。

随后，我们将了解 Hugging Face 体系。你将通过 Transformers 库加载基座模型，使用 Datasets 库处理训练数据。我们还会引入 Accelerate 库，以便高效管理内存并分配计算任务。最后，你将编写一个核心 Python 脚本，整合上述组件，并为正式的微调流程做好系统准备。

## 小节

- 1. [配置 PyTorch 和 CUDA](01-%E9%85%8D%E7%BD%AE%20PyTorch%20%E5%92%8C%20CUDA.md)
- 2. [Hugging Face Transformers 库简介](02-Hugging%20Face%20Transformers%20%E5%BA%93%E7%AE%80%E4%BB%8B.md)
- 3. [使用 Hugging Face Datasets 管理数据集](03-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20Datasets%20%E7%AE%A1%E7%90%86%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 4. [使用 Accelerate 优化显存](04-%E4%BD%BF%E7%94%A8%20Accelerate%20%E4%BC%98%E5%8C%96%E6%98%BE%E5%AD%98.md)
- 5. [动手实践：配置训练脚本](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%85%8D%E7%BD%AE%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)
