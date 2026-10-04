# CNN中的自注意力机制

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/self-attention-cnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准卷积神经网络 (neural network)（CNN）主要通过卷积层和池化层构建视觉数据的层次化表示。虽然在学习局部模式和纹理方面表现出色，但卷积操作固定的局部感受野天生限制了网络直接模拟长距离依赖和获取全局信息的能力。例如，对于标准CNN来说，理解场景中远距离对象的关系，或将微小细节与整体图像结构联系起来，如果不使用非常深的网络或激进的池化（这可能会丢失细致信息），将面临困难。

自注意力 (self-attention)机制 (attention mechanism)提供了一种有效方法来解决此局限，它让网络能够根据输入本身，动态地衡量不同空间位置或通道中特征的重要性。注意力机制不依赖固定的感受野，而是使模型能够根据当前任务选择性地关注特征图中信息量最多的部分，从而有效建立动态的、内容相关的连接。

### 通道注意力：关注“是什么”

CNN中自注意力 (self-attention)机制 (attention mechanism)的一个突出应用是通道注意力，其目的是模拟特征通道间的相互依赖。其核心思想是，特征图中的不同通道通常对应不同的语义属性或对象检测器，并且并非所有通道对于后续层都同等重要。通道注意力机制学习为每个通道明确分配权重 (weight)，增强有益特征并抑制相关性较低的特征。

#### 挤压-激励（SE）网络

挤压-激励（SE）块是一种计算开销小且有效的通道注意力实现。它可以方便地集成到现有CNN架构中。SE块分三个步骤运行：

1. **挤压**：这个步骤将全局空间信息汇集到通道描述符中。对于输入特征图 $U \in \mathbb{R}^{H \times W \times C}$（高、宽、通道），通常使用全局平均池化（GAP）来生成一个向量 (vector) $z \in \mathbb{R}^{1 \times 1 \times C}$。$z$ 的第 $c$ 个元素计算方式如下：

   
   $$
   z_c = F_{sq}(u_c) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_c(i, j)
   $$
   

   这里，$u_c$ 表示输入特征图 $U$ 的第 $c$ 个通道。这种挤压后的表示 $z$ 包含通道维度的统计信息，有效概括了每个通道的全局信息。
2. **激励**：这个步骤利用挤压后的信息学习一个非线性、通道特定的激活函数 (activation function)。它的目标是获取通道间的复杂依赖。一种常见方法是使用两个带瓶颈结构的全连接（FC）层：

   
   $$
   s = F_{ex}(z, W) = \sigma(W_2 \delta(W_1 z))
   $$
   

   这里，$\delta$ 是ReLU激活函数，$\sigma$ 是Sigmoid激活函数。$W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ 和 $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ 是两个FC层的权重。第一个FC层将通道维度缩小 $r$ 倍（$r$ 为缩减比，一个超参数 (parameter) (hyperparameter)），形成一个瓶颈，这会限制模型复杂度并辅助泛化。第二个FC层将通道维度恢复到 $C$。最终的Sigmoid激活函数确保输出权重 $s \in \mathbb{R}^{1 \times 1 \times C}$ 被归一化 (normalization)到0和1之间。这些权重表示每个通道的学习到的重要性或“激励”程度。
3. **缩放（重新缩放）**：最后一步是将学习到的通道注意力权重 $s$ 应用到原始输入特征图 $U$ 上。输出特征图 $\tilde{X} \in \mathbb{R}^{H \times W \times C}$ 通过通道维度乘法获得：

   
   $$
   \tilde{x}_c = F_{scale}(u_c, s_c) = s_c \cdot u_c
   $$
   

   输入特征图的每个通道 $u_c$ 都由其对应的注意力权重 $s_c$ 进行缩放。这会自适应地逐通道重新校准特征响应，强化有益通道并减弱不太有用的通道。

> 挤压-激励（SE）块内的数据流。输入特征图通过注意力路径（挤压和激励）处理以计算通道权重，然后这些权重用于重新缩放原始输入图。

SE块显著提升了CNN的表示能力，而计算开销和参数增加微小。它们可以轻松插入到各种现有架构中，通常放置在残差块内的卷积层之后（例如，构建SE-ResNet）。

### 空间注意力和非局部网络：关注“在哪里”和长距离依赖

通道注意力侧重于*哪些*特征重要，而空间注意力旨在判断特征图中*何处*包含最相关信息。一些机制基于通道间关系计算空间注意力图。然而，一种更直接获取长距离空间依赖的方法，其灵感源于自然语言处理中的自注意力 (self-attention)，可以在非局部网络中看到。

#### 非局部神经网络 (neural network)

非局部网络引入的模块，将某个位置的响应计算为输入特征图中*所有*位置特征的加权和。这使得网络能够直接获取远距离空间位置间的依赖，从而克服局部感受野的局限。

通用的非局部操作可定义为：


$$
y_i = \frac{1}{\mathcal{C}(x)} \sum_{\forall j} f(x_i, x_j) g(x_j)
$$


我们来分析这个公式：

- $x$ 是输入特征图。
- $i$ 是输出位置的索引（例如，特征图中的一个空间位置）。
- $j$ 列举输入特征图中的所有可能位置。
- $f(x_i, x_j)$ 是一个成对函数，它计算一个标量，表示位置 $i$ 和位置 $j$ 之间的关系（例如，亲和度或相似度）。
- $g(x_j)$ 是一个一元函数，它计算位置 $j$ 处输入信号的表示。通常，这是一个简单的线性嵌入 (embedding)：$g(x_j) = W_g x_j$，其中 $W_g$ 是一个学习到的权重 (weight)矩阵。
- $y_i$ 是位置 $i$ 处的输出响应。
- $\mathcal{C}(x)$ 是一个归一化 (normalization)因子，通常是位置 $i$ 的所有成对亲和度的总和：$\mathcal{C}(x) = \sum_{\forall j} f(x_i, x_j)$。

成对函数 $f$ 的不同选择会产生非局部块的不同变体。一个常见且有效的选择是**嵌入式高斯**函数：


$$
f(x_i, x_j) = e^{\theta(x_i)^T \phi(x_j)}
$$


这里，$\theta(x_i) = W_\theta x_i$ 和 $\phi(x_j) = W_\phi x_j$ 分别是位置 $i$ 和 $j$ 处输入特征的线性嵌入（学习到的变换）。这个公式与Transformer中使用的点积注意力很相似。指数函数根据嵌入表示之间的点积计算亲和度。整个操作有效计算了变换后的输入特征 $g(x_j)$ 的加权平均值，其中权重由目标位置 $i$ 与所有其他位置 $j$ 之间的相似度确定。

非局部块可以插入到CNN的多种深度位置。当放置在更深层时，它们可以在语义更丰富的特征上运行，并获取复杂的空间关系。然而，计算所有空间位置（$H \times W$ 个位置）的成对交互可能会耗费大量计算资源，其计算复杂度与空间位置的数量呈平方关系。这种开销通常限制了它们在已进行空间下采样的特征图上的应用。

### 将注意力机制 (attention mechanism)集成到CNN中

SE块和非局部块都作为标准卷积层的补充增强。

- **SE块**：通常添加在卷积块（如残差块）之内或之后，以根据全局信息重新校准通道特征。它们的低开销使其得到广泛应用。
- **非局部块**：策略性地插入，通常在网络的较后期阶段或穿插在标准卷积块之间，以明确模拟长距离空间依赖。它们的计算开销需要仔细权衡放置位置和输入特征图大小。

通过引入自注意力 (self-attention)机制，CNN能够根据输入图像的全局信息动态调整其特征表示。通道注意力帮助网络关注最相关的特征类型，而空间注意力和非局部操作使其能够明确模拟图像远距离部分之间的关系。这些技术使CNN能够构建更强大的、具备情境感知的表示，从而在各种计算机视觉任务上取得性能提升，特别是那些需要理解更广阔场景结构或对象之间关系的任务。

## 参考资料

- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) — Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 7132-7141; DOI: [10.48550/arXiv.1709.01507](https://doi.org/10.48550/arXiv.1709.01507)
  介绍了Squeeze-and-Excitation (SE) 模块，这是一种本节详细讨论的通道注意力机制。
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) — Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 7794-7803; DOI: [10.1109/CVPR.2018.00749](https://doi.org/10.1109/CVPR.2018.00749)
  提出了非局部神经网络，这是一种利用自注意力机制捕获长距离空间依赖的方法，是本节的核心主题。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和自注意力机制的开创性论文，启发了包括非局部网络在内的许多后续视觉领域基于注意力的方法。

---

[上一节](../04-%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E6%8A%80%E6%9C%AF/08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E6%A8%A1%E5%9E%8B.md) · [下一节](02-%E9%9D%9E%E5%B1%80%E9%83%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
