---
course: "advanced-diffusion-architectures"
chapter: "transformer-diffusion-models"
lesson: "dit-conditioning"
sourceId: 5408
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/dit-conditioning"
title: "扩散Transformer模型中的条件作用"
description: "将条件信息（类别标签、文本）引入DiT架构的方法。"
order: 4
plots: []
sourceHash: "ffef21e831ed652648f4eca05805239e698448014fed64cef78288b1551db064"
sourceCorrections: []
---

整合条件信息对于引导扩散模型的生成过程是必不可少的，它使我们能够根据类别标签、文本描述或其他引导信号等特定输入来控制输出。与U-Net架构中使用的条件作用方法不同，扩散Transformer模型（DiTs）提供了适应Transformer结构的不同方法，主要通过操作令牌嵌入 (embedding)和修改Transformer块本身来实现。

与U-Net不同，U-Net中条件作用通常通过输入层的拼接或穿插在卷积块中的交叉注意力机制 (attention mechanism)注入，DiTs则更内在地将其整合到其核心构建块中。由于DiTs在图像块嵌入序列上运行，条件信号通常会经过处理，然后用于影响每个Transformer块内部的计算，并均匀地应用于所有图像块令牌以进行全局引导。

### DiT中的条件作用方法

有几种策略已被证实能有效将条件作用结合到DiT架构中。选择通常取决于条件信号的类型和期望的控制程度。

#### 1. 自适应归一化 (normalization) (AdaLN-Zero)

Peebles和Xie在原始DiT论文（“基于Transformer的可扩展扩散模型”）中提出了一种高效方法，它涉及使用条件信息调节Transformer块的内部激活。这种方式通常被称为AdaLN-Zero。

它工作方式如下：

1. **嵌入 (embedding)条件：** 离散时间步 $t$ 和条件信号 $c$（例如类别标签）首先分别映射到嵌入向量 (vector) $\text{embed}(t)$ 和 $\text{embed}(c)$。
2. **组合嵌入：** 这些嵌入通常简单地相加：$\text{cond\_embedding} = \text{embed}(t) + \text{embed}(c)$。
3. **计算调制参数 (parameter)：** 这个组合嵌入通过一个多层感知机（MLP）来生成调制参数。特别是在AdaLN-Zero中，这些参数通常表示为 $(\gamma, \beta, \alpha)$，用于缩放和偏移Transformer块内层归一化或特定线性层的输出。原始DiT论文使用这些参数来调节层归一化之后、主MLP或注意力层之前的状态，并用于残差连接的缩放。
4. **应用调制：** 令 $h$ 为Transformer块中的隐藏状态。对于前面是层归一化的特定层（例如MLP或注意力层），调制可能看起来像这样：
   
   $$
   h' = \alpha \cdot \text{层}(\text{层归一化}(h)) + \beta
   $$
   
   这里，$\alpha$ 和 $\beta$ 是通过MLP从条件嵌入中获取的。参数根据 $t$ 和 $c$ 每个块计算一次，并逐元素应用。

这种方法使条件信号能够动态调整整个网络中的特征尺度和偏移，有效地将去噪过程引导至期望的条件。

> 图示AdaLN-Zero在扩散Transformer (DiT) 块内的条件作用机制。时间和条件嵌入由MLP组合并处理，以生成调制参数 ($\alpha, \beta$)，随后影响归一化层之后的主要数据路径。

#### 2. 输入注入

一种更简单的替代方法是将条件作用直接注入到馈送给Transformer的输入序列中。

- **附加令牌：** 条件信号（例如，为某个类别学习到的嵌入，或一系列文本嵌入）可以被视为一个或多个附加令牌，添加到图像块令牌序列的前面或后面。自注意力 (self-attention)机制 (attention mechanism)在处理图像令牌时，可以自然地整合来自这些条件令牌的信息。
- **嵌入加法：** 类似于位置嵌入，条件嵌入可以直接添加到图像块嵌入中，然后再进入第一个Transformer块。这要求条件嵌入与图像块嵌入具有相同的维度。

尽管更简单，但这些方法可能无法像自适应归一化那样有效地在网络深度上传播条件信号，而自适应归一化会在每个块中重复强化条件。

#### 3. 交叉注意力

尽管典型的DiT架构主要依赖于自注意力，但引入交叉注意力层是另一种可行的方法，特别适用于文本描述等复杂的条件作用。

在这种设置中，Transformer块内的特定层将采用交叉注意力，其中：

- **查询：** 源自图像块令牌（类似于自注意力）。
- **键和值：** 源自条件信号嵌入（例如，来自CLIP等文本编码器的输出令牌）。

这使得每个图像块表示能够直接关注条件信息的关联部分。这与Stable Diffusion的U-Net中使用的机制非常相似，但已适配Transformer骨干。实现这一点需要修改标准DiT块以包含这些交叉注意力层，可能增加计算成本，但能提供图像和条件之间细致的对齐 (alignment)。

### 实现上的考虑

- **嵌入 (embedding)维度：** 确保条件嵌入（类别、文本等）被投影到合适的维度，无论是用于添加到时间嵌入、直接添加到图像块嵌入，还是用于交叉注意力。
- **参数 (parameter)数量：** 自适应归一化 (normalization)增加的参数相对较少（主要在生成 $\gamma, \beta, \alpha$ 的MLP中），而增加交叉注意力层可以显著增加模型大小和计算量。
- **位置：** 使用AdaLN-Zero时，在每个Transformer块内应用它可确保条件影响多个表示层面的处理。对于交叉注意力，决定在哪些块中加入它涉及控制保真度和效率之间的权衡。

通过采用这些技术，特别是像AdaLN-Zero这样的自适应归一化方案，扩散Transformer模型可以有效整合条件信息，使得生成由类别标签或可能更复杂模态等各种输入引导的高保真图像成为可能。这种适应性是其可扩展性和在大规模图像生成任务上表现优异的重要因素。

## 参考资料

- [Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) — William Peebles, Saining Xie (2023)
  Journal: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 4195-4205; DOI: [10.1109/ICCV51070.2023.01168](https://doi.org/10.1109/ICCV51070.2023.01168)
  介绍了扩散变换器（DiTs）和AdaLN-Zero条件机制，这是本节讨论的核心。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 10684-10695; DOI: [10.1109/CVPR52688.2022.00196](https://doi.org/10.1109/CVPR52688.2022.00196)
  详细介绍了扩散模型中用于文本条件化的交叉注意力机制，为DiTs和U-Net架构中交叉注意力的比较提供了参考。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  介绍了Transformer架构的基础论文，该架构是扩散变换器的基础。
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://www.aaai.org/ojs/index.php/AAAI/article/view/11671) — Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville (2018)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 32; DOI: [10.1609/aaai.v32i1.11671](https://doi.org/10.1609/aaai.v32i1.11671)
  介绍了特征级线性调制（FiLM），该技术为DiTs中使用的AdaLN-Zero等自适应归一化方法提供了更广泛的背景。
