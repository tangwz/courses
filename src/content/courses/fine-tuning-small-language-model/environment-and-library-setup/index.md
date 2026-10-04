---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-3-environment-and-library-setup"
sourceId: 1503
chapter: "environment-and-library-setup"
title: "环境与库配置"
order: 3
description: "配置 PyTorch、CUDA 和 Hugging Face 相关工具，为本地模型训练做好准备。"
hasQuiz: false
---

在训练小语言模型之前，需要建立稳定的软件环境。训练算法高度依赖特定的库来管理硬件资源、处理文本并有效更新神经网络权重。即便小语言模型比超大规模模型占用的显存更少，但配置不当仍会迅速导致内存不足报错或运行速度极慢。

本章将指导你配置本地模型训练所需的工具。首先，我们要安装支持 CUDA 的 PyTorch，从而开启 GPU 硬件加速。这些框架屏蔽了神经网络底层繁杂的数学运算。你无需手动编写像 $Y = WX + b$ 这样的矩阵乘法，也不必为损失计算编写如下自定义函数：

$$L = -\sum y_i \log(\hat{y}_i)$$

而是直接调用成熟的库，由它们在后台原生处理这些计算。

随后，我们将了解 Hugging Face 体系。你将通过 Transformers 库加载基座模型，使用 Datasets 库处理训练数据。我们还会引入 Accelerate 库，以便高效管理内存并分配计算任务。最后，你将编写一个核心 Python 脚本，整合上述组件，并为正式的微调流程做好系统准备。
