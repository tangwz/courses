---
course: "cnns-for-computer-vision"
chapter: "attention-transformers-vision"
lesson: "non-local-neural-networks"
sourceId: 2587
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/non-local-neural-networks"
title: "非局部神经网络"
description: "了解在图像或视频数据中捕获长距离关联的非局部块。"
order: 2
plots: []
sourceHash: "3fabece25cf1e22d699119b87d3dbf7afb79bdcd73b08890342281fe76d349f7"
sourceCorrections: []
---

标准卷积层进行局部操作，在卷积核大小定义的受限区域内处理信息。虽然堆叠这些层可以增加有效感受野，但有效地捕获空间上相距较远特征之间的关联（长距离关联）仍然有难度。比如，理解图像中一个人手持一个物体以及物体本身的关联，即使它们在图像中相隔较远，也需要对非相邻的相互关联进行建模。

非局部神经网络 (neural network)通过将某个位置的响应计算为输入特征图中*所有*位置特征的加权和，直接解决这一局限。这使得网络能够获取全局信息，并模拟任意两个位置之间的关联，不论它们的空间距离如何。可以将其视为应用于空间或时空数据的自注意力 (self-attention)机制 (attention mechanism)的推广。

### 非局部操作

标准卷积层只在局部区域内处理信息，这使得捕捉空间上远距离的特征依赖关系变得困难。非局部操作旨在解决这一挑战，通过计算图像中所有位置之间的相互作用来捕获长距离依赖。非局部操作有一个一般公式来描述其计算方式。给定一个输入特征图 $\mathbf{x}$（可以是图像，也可以是CNN中间层的特征图），在位置 $i$ 处的输出特征图 $\mathbf{y}$ 计算为：


$$
\mathbf{y}_i = \frac{1}{\mathcal{C}(\mathbf{x})} \sum_{\forall j} f(\mathbf{x}_i, \mathbf{x}_j) g(\mathbf{x}_j)
$$


我们来分解一下各个组成部分：

- $i$：正在计算响应的输出位置的索引。
- $j$：遍历输入特征图中*所有*可能位置的索引。
- $\mathbf{x}_i$：输入中位置 $i$ 处的特征向量 (vector)。
- $\mathbf{x}_j$：输入中位置 $j$ 处的特征向量。
- $f(\mathbf{x}_i, \mathbf{x}_j)$：一个成对函数，计算一个标量，表示位置 $i$ 和位置 $j$ 之间的关联（例如，亲和度或相似度）。
- $g(\mathbf{x}_j)$：一个函数，计算输入特征向量在位置 $j$ 处的表示或转换。这通常是一个线性嵌入 (embedding)。
- $\mathcal{C}(\mathbf{x})$：一个归一化 (normalization)因子，计算方式为 $\sum_{\forall j} f(\mathbf{x}_i, \mathbf{x}_j)$，确保权重 (weight)和适当（通常为1，类似于softmax）。

本质上，位置 $i$ 处的响应（$\mathbf{y}_i$）是来自所有位置 $j$ 的转换后特征（$g(\mathbf{x}_j)$）的加权平均。权重由位置 $i$ 与每个位置 $j$ 之间的相似性或关联 ($f$) 确定。

### 成对函数和转换函数的具体实例

通用的非局部操作公式允许根据函数 $f$ 和 $g$ 的选择进行不同的具体实现。

**转换函数 $g$：**
对于 $g$，常见的选择是通过 $1 \times 1$ 卷积学习的简单线性嵌入 (embedding)：
$g(\mathbf{x}_j) = W_g \mathbf{x}_j$
这里，$W_g$ 代表 $1 \times 1$ 卷积层的权重 (weight)。

**成对函数 $f$：**
成对函数 $f$ 有多种选项，用于衡量 $i$ 和 $j$ 之间的关联：

1. **嵌入高斯：** 这可能是最常见的选择，与自注意力 (self-attention)直接关联。首先，将线性嵌入 $\theta$ 和 $\phi$（同样，通常是带有权重矩阵 $W_{\theta}$ 和 $W_{\phi}$ 的 $1 \times 1$ 卷积）应用于输入特征。该函数为：
   $f(\mathbf{x}_i, \mathbf{x}_j) = e^{\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)} = e^{(W_{\theta}\mathbf{x}_i)^T (W_{\phi}\mathbf{x}_j)}$
   在这种情况下，归一化 (normalization)因子 $\mathcal{C}(\mathbf{x})$ 使权重 $\frac{f(\mathbf{x}_i, \mathbf{x}_j)}{\mathcal{C}(\mathbf{x})}$ 等同于应用于所有位置 $j$ 的 softmax 函数。
2. **点积：** 嵌入高斯的一个更简单的版本，省略了指数部分：
   $f(\mathbf{x}_i, \mathbf{x}_j) = \theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$
   这里，归一化 $\mathcal{C}(\mathbf{x})$ 可能是位置的数量 $N$。
3. **拼接：** 嵌入的特征被拼接，通过一个线性层（权重向量 (vector)为 $w_f$），并进行激活（例如，使用ReLU）：
   $f(\mathbf{x}_i, \mathbf{x}_j) = \text{ReLU}(w_f^T [\theta(\mathbf{x}_i), \phi(\mathbf{x}_j)])$
   这使得建模更复杂的关联成为可能。

嵌入高斯方法因其与注意力机制 (attention mechanism)的联系和实践中的成功而被广泛使用。

### 实现非局部块

非局部操作通常作为“非局部块”整合到现有深度学习 (deep learning)架构中。这些块通常采用残差连接来辅助训练，类似于ResNet块。

常见非局部块（使用嵌入 (embedding)高斯版本）的结构如下所示：

1. **输入**：特征图 $\mathbf{x}$（例如，维度为 $H \times W \times C$）。
2. **嵌入**：使用独立的 $1 \times 1$ 卷积计算 $\theta(\mathbf{x})$、$\phi(\mathbf{x})$ 和 $g(\mathbf{x})$。$\theta$ 和 $\phi$ 通常会减少通道维度（例如，到 $C/2$）以节省计算，而 $g$ 可能会保持或减少它。设输出为 $\mathbf{\Theta}$、$\mathbf{\Phi}$、$\mathbf{G}$。
3. **亲和度计算**：将 $\mathbf{\Theta}$ 和 $\mathbf{\Phi}$ 重塑为矩阵，其中一个维度代表空间位置 ($H \times W$)，另一个代表通道 ($C/2$)。计算点积 $\mathbf{\Theta}^T \mathbf{\Phi}$（经过适当的转置后）。这将得到一个大小为 $(HW) \times (HW)$ 的矩阵，表示所有位置对之间的亲和度。
4. **归一化 (normalization)**：对亲和度矩阵进行逐行（或逐列，取决于矩阵设置）softmax 操作，得到归一化权重 (weight) $\mathbf{A}$。
5. **加权和**：将 $\mathbf{G}$ 重塑为与 $\mathbf{\Theta}$ 和 $\mathbf{\Phi}$ 类似的形式（维度为 $HW \times C'$）。通过将归一化亲和度 $\mathbf{A}$ 与 $\mathbf{G}$ 进行矩阵乘法来计算加权和。设结果为 $\mathbf{Y}'$。
6. **重塑与投影**：将 $\mathbf{Y}'$ 重塑回空间维度 $H \times W \times C'$。应用一个最终的 $1 \times 1$ 卷积（权重为 $W_z$），将其投影回原始通道维度 $C$。设结果为 $\mathbf{Y}$。权重 $W_z$ 通常初始化为零，因此该块最初表现为恒等函数。
7. **残差连接**：将输出 $\mathbf{Y}$ 添加回原始输入 $\mathbf{x}$：$\mathbf{z} = \mathbf{Y} + \mathbf{x}$。

该结构使得网络能够学习长距离关联，同时通过残差连接保留原始信息流。

> 非局部块内的数据流，使用嵌入高斯亲和度及残差连接。$\theta$、$\phi$ 和 $g$ 转换通常通过 $1 \times 1$ 卷积实现。

### 与自注意力 (self-attention)的关联

如果您熟悉Transformer架构，您会认出嵌入 (embedding)高斯非局部操作等同于缩放点积自注意力机制 (attention mechanism)。

- $\theta(\mathbf{x}_i)$ 对应于“查询”（Query）。
- $\phi(\mathbf{x}_j)$ 对应于“键”（Key）。
- $g(\mathbf{x}_j)$ 对应于“值”（Value）。

非局部块计算每个查询位置 $i$ 与所有位置 $j$ 之间的注意力权重 (weight)，然后使用这些权重来形成值的加权和。非局部网络将自注意力引入了计算机视觉方面，直接应用于特征图，而非词嵌入序列。

### 应用与考量

非局部块在长距离关联很重要的任务中显示出效果：

- **视频分析：** 捕获帧间的时间关联，用于动作识别。
- **目标检测与分割：** 建立上下文 (context)模型，例如对象之间或对象部分与整体对象之间的关联。例如，Mask R-CNN 从整合上下文信息中获益。
- **图像生成：** 提高生成图像的全局一致性和结构。

然而，一个重要的考量是计算成本。计算成对亲和度涉及比较每个位置与所有其他位置。如果输入特征图有 $N = H \times W$ 个空间位置，复杂度为 $O(N^2)$，这对高分辨率特征图来说要求较高。

**缓解策略：**
为了管理此成本，实现方式通常会：

- 通过 $\theta$、$\phi$ 和 $g$ 中的 $1 \times 1$ 卷积减少通道维度。
- 在成对计算之前，对 ($\phi$) 和值 ($g$) 输入应用空间降采样（例如，最大池化）。这会减少参与求和的位置 $j$ 的数量。
- 仅在网络中特征图较小的较深层插入非局部块，或在整个架构中少量使用。

非局部网络提供了一种强大而通用的机制，用于在视觉领域的深度学习 (deep learning)模型中纳入非局部关联。它们代表了向能够理解全局图像结构和上下文的模型迈出的重要一步，补充了标准卷积的局部特征提取能力，并促成了像Vision Transformers这样的架构。

## 参考资料

- [Non-local Neural Networks](https://openaccess.thecvf.com/content_CVPR_2018/papers/Wang_Non-local_Neural_Networks_CVPR_2018_paper.pdf) — Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE Computer Society; Pages: 7794-7803; DOI: [10.1109/CVPR.2018.00813](https://doi.org/10.1109/CVPR.2018.00813)
  介绍了非局部神经网络架构、其通用公式和具体实现，是本节内容的原始来源。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出了Transformer架构和自注意力机制，非局部操作对此进行了推广，并与它紧密相关。
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了Vision Transformer (ViT) 模型，展示了如何将Transformer架构和自注意力机制有效地应用于图像分类，这建立在全局上下文建模的思想之上。
