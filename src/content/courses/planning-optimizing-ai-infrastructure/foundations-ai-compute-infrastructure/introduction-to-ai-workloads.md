---
course: "planning-optimizing-ai-infrastructure"
chapter: "foundations-ai-compute-infrastructure"
lesson: "introduction-to-ai-workloads"
sourceId: 6961
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/introduction-to-ai-workloads"
title: "人工智能工作负载概览"
description: "了解人工智能训练与推理工作负载之间的区别及其各自的计算需求。"
order: 1
plots: []
sourceHash: "ca285e3f9357c5d22fcce19a1372b022d0b430a98c3291516d83ab595dfb5e6e"
sourceCorrections: []
---

在选择硬件或配置云实例之前，您必须首先了解您希望它完成的工作性质。在人工智能领域，这项工作分为两种主要且本质上不同的工作负载类型：**训练**和**推理 (inference)**。尽管它们都涉及神经网络 (neural network)和数据，但它们的计算模式、资源需求和性能目标却有所不同。掌握这种区别是构建高效且经济的基础设施的第一步。

### 训练阶段：塑造模型

训练是教授机器学习 (machine learning)模型的过程。就像学生学习教科书一样，模型通过处理大量数据集并调整其内部参数 (parameter)来最大程度地减少预测误差。这是一个迭代的、计算量大的且通常耗时的过程。

大多数深度学习 (deep learning)训练的核心是一系列矩阵运算。神经网络 (neural network)的运作涉及一个正向传播过程，其中输入数据通过网络生成预测；以及一个反向传播 (backpropagation)过程（反向传播），其中模型计算其预测中的误差并使用该误差更新其参数或权重 (weight)。这个循环会重复进行，通常针对数百万或数十亿个样本，经历许多次迭代，称为“纪元”（epochs）。

训练的计算特点如下：

- **高并行度：** 单个训练步骤涉及数千或数百万次独立的计算，最显著的是矩阵乘法 ($C = A \cdot B$)。这种工作负载非常适合图形处理单元（GPU）的架构，GPU包含数千个核心，旨在同时对不同数据执行相同的指令。
- **大数据量和模型大小：** 训练需要将模型的参数、反向传播过程中计算的梯度以及批量的训练数据保存在内存中。对于现代语言模型或高分辨率计算机视觉模型等大型模型，这可能需要数十甚至数百GB的高速内存（GPU上的显存 (VRAM)）。
- **迭代性与长时间运行：** 单次训练运行可能持续数小时、数天甚至数周。目标是实现模型尽可能高的准确率，因此总训练时间是一个重要指标。基础设施必须稳定并具备持久性。
- **高吞吐量 (throughput)：** 在训练过程中，目标是在最短的时间内处理尽可能多的数据。这意味着需要优化整个流程，从快速数据存储和加载到处理器上的高效计算，以最大化数据吞吐量。

### 推理 (inference)阶段：模型投入使用

推理是使用经过充分训练的模型对新的、未见过的数据进行预测的过程。一旦模型训练完成，其参数 (parameter)就被固定。它不再学习，而是运用所学知识。这相当于学生完成学业后参加考试。

推理工作负载包含一次通过网络的正向传播。一个输入，例如图像或一行文本，提供给模型，模型随后执行计算并输出结果，例如对象分类或语言翻译。

推理的计算特点如下：

- **低延迟：** 对于许多应用，特别是面向用户的应用，预测必须几乎即时返回。用户等待产品推荐或自动驾驶汽车识别行人时，无法容忍长时间的延迟。获得单个预测所需的时间（延迟）通常是最重要的性能指标。
- **高吞吐量 (throughput)（并发请求）：** 一个已部署的模型可能需要同时为数千名用户提供预测。系统必须能够处理大量并发、独立的请求，这是一个称为吞吐量的指标。
- **计算效率：** 与训练不同，推理不涉及反向传播 (backpropagation)。它是一个单一的、计算量较小的传播。尽管GPU仍然非常有效，但功能强大的CPU或专门的、更小的加速器通常是更具成本效益的选择，特别是当延迟要求不在个位数毫秒范围时。
- **部署灵活性：** 推理可以在任何地方发生：在大型数据中心、本地服务器，甚至在智能手机或传感器等小型边缘设备上。这要求模型针对不同的硬件和功耗限制进行优化。

### 两种工作负载的区别

下图展示了训练和推理 (inference)各自不同的流程和优先事项。训练是一个循环的、高负荷的过程，侧重于模型优化；而推理是一个线性的、轻量级的过程，侧重于速度和效率。

> 两种不同的人工智能工作负载。训练是一个迭代循环，侧重于产生高质量模型。推理是从新数据到使用已训练模型进行预测的直接路径。

这些区别直接决定了您的基础设施选择。专为快速训练实验而构建的系统将优先考虑配备高速互连的强大多GPU服务器。相反，为大规模经济高效推理而设计的基础设施可能使用一组小型CPU实例或专用推理芯片。理解您正在为哪种工作负载进行优化，是所有其他基础设施决策的依据。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本综合性教材提供了深度学习的数学和概念基础，包括对神经网络架构、训练算法（如反向传播）以及模型训练和推理之间区别的详细解释。
- [MLOps Engineering at Scale](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFV2wqVaUe0u2_3wo560qN1-1_h4qx1IABwZa2_A8h4_44dv9xWzMD7gZBGvh8AxQz-ygFqXyhgG7YrGQEwR6rf6Abt8-Ug80fRDJCRhiQ4xBvcChsEdQNllp_L8mdpFhC8-wJ_n_wGs9X9dbKGuW37z-GEGG4KDGkCFlKwMwzj8G2l) — Carl Osipov (2022)
  Publisher: O'Reilly Media
  为设计和管理生产中的机器学习系统提供实用指导，包括构建优化AI工作负载训练和推理阶段基础设施的架构考量。
- [CS230: Deep Learning](https://cs230.stanford.edu/) — Andrew Ng, Kian Katanforoosh (2024)
  Publisher: Stanford University
  这门备受推崇的斯坦福大学课程提供了全面的深度学习材料，包括区分AI模型训练和推理计算需求及硬件要求的讲座和资源。
