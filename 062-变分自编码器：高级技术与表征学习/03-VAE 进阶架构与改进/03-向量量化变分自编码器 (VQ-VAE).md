# 向量量化变分自编码器 (VQ-VAE)

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/vector-quantized-vaes-vq-vaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准VAE凭借其连续潜在空间提供了一个扎实的基础，但有时会生成缺乏清晰度的样本，这通常归因于连续表示的特性和KL散度项。当您的目标是生成更清晰的输出或处理本质上离散的数据结构时，向量 (vector)量化 (quantization)变分自编码器（VQ-VAE）提供了一种值得考虑的架构替代方案。VQ-VAE通过引入离散潜在空间来实现这一点，该空间通过有限的嵌入 (embedding)向量码本学习得到。这种设计选择通常能明显提高样本质量，并提供了一种不同的机制来形成信息瓶颈，从而摆脱了传统VAE中显式的KL散度项。

### VQ-VAE的架构

VQ-VAE的核心创新点在于对编码器输出的量化 (quantization)。它不直接使用连续潜在向量 (vector)，而是将编码器的输出映射到学习到的有限嵌入 (embedding)向量集中最接近的向量，这个集合被称为码本。

VQ-VAE通常由以下部分组成：

1. **编码器（Encoder）**: 这个神经网络 (neural network)$f_{enc}$处理输入$x$并产生一个连续表示$z_e(x)$。如果输入是图像，$z_e(x)$可能是一个形状为$H' \times W' \times D$的特征图；对于其他数据类型，它可以是一个单一向量。这个$z_e(x)$是一个中间输出，而不是最终的潜在变量。
2. **码本（嵌入空间）**: 这是一个可学习字典$E = \{e_1, e_2, ..., e_K\}$，包含$K$个嵌入向量，其中每个$e_i \in \mathbb{R}^D$。您可以将这个码本视为模型学习到的代表性特征向量调色板。
3. **量化器（Quantizer）**: 对于编码器输出$z_e(x)$中的每个向量（或者如果$z_e(x)$本身是一个单一向量，则直接对它），量化器在码本$E$中执行最近邻查找。它识别出在欧几里得距离上最接近的嵌入向量$e_k$：
   $k = \text{argmin}_j ||z_e(x) - e_j||_2$
   量化后的潜在表示，$z_q(x)$，就是这个被选中的码本向量：$z_q(x) = e_k$。
4. **解码器（Decoder）**: 这个网络$f_{dec}$以量化表示$z_q(x)$为输入，旨在重构原始输入$x$，得到$\hat{x}$。

> VQ-VAE架构的概览。编码器产生$z_e(x)$，通过在学习到的码本$E$中找到最接近的向量，将其量化为$z_q(x)$。解码器从$z_q(x)$重构输入。

### 训练动态与不可微分挑战

训练VQ-VAE的一大挑战是量化 (quantization)步骤中的$\text{argmin}$操作不可微分。这阻止了梯度从解码器直接反向传播 (backpropagation)到编码器。VQ-VAE通过采用直通估计器（STE）巧妙地解决了这个问题。

在反向传播过程中，来自解码器输入的梯度$\nabla_{z_q} L$直接传递到编码器的输出$z_e(x)$。本质上，解码器的梯度信号绕过了不可微分的量化操作。
$\frac{\partial L}{\partial z_e(x)} \approx \frac{\partial L}{\partial z_q(x)}$
这使得编码器能够学习产生在量化后能得到良好重构的输出$z_e(x)$。编码器学习生成接近有用码本条目的连续向量 (vector)。

### VQ-VAE的损失函数 (loss function)

训练VQ-VAE的总损失函数通常包含三个不同的部分：

1. **重构损失（$L_{recon}$）**: 这一项促使解码器从其量化 (quantization)潜在表示$z_q(x)$精确重构输入$x$。对于图像等连续数据，这通常是均方误差（MSE）：
   $L_{recon} = ||x - \text{dec}(z_q(x))||^2_2$
   对于其他数据类型，如二值数据，二元交叉熵（BCE）可能更合适。
2. **码本损失（或嵌入 (embedding)损失，$L_{codebook}$）**: 这种损失负责更新码本$E$中的嵌入向量 (vector)$e_i$。它促使被选中的码本向量$e_k$（最接近$z_e(x)$的那个）向编码器的输出$z_e(x)$移动。对于此更新，编码器的输出$z_e(x)$被视为常数（使用停止梯度操作符$\text{sg}$进行分离）。
   $L_{codebook} = ||\text{sg}[z_e(x)] - e_k||^2_2$
   这个部分类似于k-means聚类中的质心更新规则。它将选定的码本向量$e_k$拉向映射到它的编码器输出集群。
3. **承诺损失（$L_{commit}$）**: 这一项对编码器进行正则化 (regularization)，促使其输出$z_e(x)$保持“归属于”所选码本向量$e_k$。它有助于防止编码器输出过度波动或变得过大，确保它们保持接近离散表示集。对于这种损失，码本向量$e_k$被视为常数（分离）。超参数 (parameter) (hyperparameter)$\beta$控制此项的影响。
   $L_{commit} = \beta ||z_e(x) - \text{sg}[e_k]||^2_2$
   如果没有这种损失，编码器输出$z_e(x)$可能会严重偏离它们所映射的实际嵌入$e_k$，可能使码本更新的稳定性和效果下降。

要最小化的组合损失函数为：
$L_{VQVAE} = L_{recon} + L_{codebook} + L_{commit}$
需要注意的是，标准VAE中显式的KL散度项在这里缺失，该项将编码器的潜在分布$q(z|x)$正则化到先验$p(z)$。在VQ-VAE中，正则化作用主要通过有限码本施加的信息瓶颈和承诺损失来实现。

### 使用VQ-VAE的优点

通过学习码本引入离散潜在空间带来了几个显著的优点：

- **更清晰的生成样本**：具有连续潜在空间的标准VAE有时会因为潜在空间中的平均效应而产生模糊或过于平滑的样本。VQ-VAE中$z_q(x)$的离散性通常会迫使解码器从更明确的“原型”特征集中选择，从而产生更清晰、更详细、更高保真度的输出，特别是对于图像和音频等复杂数据。
- **缓解后验坍缩**：训练标准VAE时的一个常见难题是“后验坍缩”，即如果ELBO中的KL散度项严重惩罚与先验的偏差，潜在变量就会变得不包含信息。VQ-VAE避开了这个特定问题，因为它们不对编码器输出使用相同的KL正则化 (regularization)机制。信息瓶颈转而由量化 (quantization)过程本身强制实现。
- **学习到的离散表示**：VQ-VAE学习到的离散码（索引$k$）本身就很有价值。例如，在语音建模中，这些码可能捕获类似音素的单元。此外，这些离散序列可以作为强大自回归 (autoregressive)模型的输入，实现我们将在稍后讨论的两阶段生成过程。
- **受控的潜在容量**：潜在空间的信息容量由码本大小$K$和嵌入 (embedding)的维度$D$决定。与调整标准VAE中KL散度的权重 (weight)相比，这提供了一种更直接的方式来控制表示瓶颈。

### 考量与潜在挑战

虽然VQ-VAE功能强大，但仍有一些实际方面和挑战需要注意：

- **码本大小（$K$）**：选择嵌入 (embedding)向量 (vector)的数量$K$是一个重要的设计决策。小的$K$可能会限制模型捕获数据完整多样性的能力，导致重构效果不佳。相反，非常大的$K$会增加计算成本和内存，并可能导致许多码未被充分利用。
- **码本坍缩（死码）**：训练期间可能只有码本向量的一个子集被主动使用，而许多嵌入（“死码”）很少或从未被编码器选中。承诺损失在一定程度上有所帮助，但有时会采用针对未使用的码的特定初始化或定期重置策略。
- **训练稳定性**：编码器学习产生与现有码本条目匹配的输出，以及码本条目向编码器输出移动，两者之间的关系可能会引入与标准VAE不同的训练动态。承诺损失的超参数 (parameter) (hyperparameter)$\beta$通常需要仔细调整。
- **量化 (quantization)的计算成本**：对于非常大的码本，量化步骤中的最近邻搜索可能会变得计算密集。然而，对于典型的码本大小（例如，$K$从几百到几千），通过高效的搜索算法通常可以很好地管理。

### VQ-VAE与离散潜在变量上的自回归 (autoregressive)先验

VQ-VAE最具有影响力的用途之一是它们与自回归模型的联系，用于学习离散潜在空间上的先验。与标准VAE中假设一个简单、因子化的先验$p(z)$不同，您可以训练一个独立的、强大的自回归模型（例如用于图像的PixelCNN或用于序列的Transformer）来模拟VQ-VAE编码器生成的离散潜在码$k$的分布。

这通常涉及一个两阶段过程：

1. **训练VQ-VAE**：VQ-VAE按照描述进行训练，学习一个编码器、一个解码器和码本$E$。一旦训练完成，编码器可以将输入$x$映射到离散码本索引序列$k_1, k_2, ..., k_M$。
2. **训练自回归先验**：随后，在这些索引序列$k_i$上训练一个独立的自回归模型，以学习先验分布$p(k) = \prod_i p(k_i | k_{<i})$。

对于生成：

1. 从训练好的自回归先验$p(k)$中采样一个潜在码序列$k_1, ..., k_M$。
2. 对于每个采样的索引$k_i$，从VQ-VAE学习到的码本$E$中检索相应的嵌入 (embedding)向量 (vector)$e_{k_i}$。
3. 将这个嵌入序列通过VQ-VAE的解码器，以生成新的数据样本$\hat{x}$。

> 使用VQ-VAE的两阶段生成过程。阶段1训练VQ-VAE。阶段2在学习到的离散码上训练一个自回归先验。对于生成，从该先验中采样码，转换为嵌入，然后进行解码。

这种两阶段方法在VQ-VAE-2等模型中得到了著名的展示，它有效地分离了关注点。VQ-VAE擅长学习紧凑、高质量的局部特征词汇（即“是什么”），而自回归先验则专注于模拟这些特征如何组合的远距离依赖关系和全局结构（即“如何”）。这种组合使得生成高度真实和连贯的图像和音频成为可能，展现了离散表示与富有表现力的序列模型结合时的能力。随着您学习高级架构，理解VQ-VAE将为您提供一个有效的工具，用于高保真生成和学习结构化离散表示。

## 参考资料

- [Neural Discrete Representation Learning](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) — Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 6306–6315; DOI: [10.5555/3295222.3295240](https://doi.org/10.5555/3295222.3295240)
  介绍了矢量量化变分自编码器（VQ-VAE）架构，详细阐述了其组件、训练方法以及如何利用离散潜在空间实现高保真生成。
- [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/pdf/1906.00446.pdf) — Ali Razavi, Aaron van den Oord, Oriol Vinyals (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 32; Pages: 5979-5990; DOI: [10.48550/arXiv.1906.00446](https://doi.org/10.48550/arXiv.1906.00446)
  介绍了VQ-VAE-2，通过结合分层VQ-VAE和强大的自回归先验，显著提高了图像生成质量，对两阶段生成过程至关重要。
- [Zero-Shot Text-to-Image Generation](https://proceedings.mlr.press/v139/ramesh21a.html) — Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever (2021)
  Journal: Proceedings of the 38th International Conference on Machine Learning; Publisher: PMLR; Volume: 139; Pages: 8821-8831; DOI: [10.32473/pmlr.v139.ramesh21a](https://doi.org/10.32473/pmlr.v139.ramesh21a)
  展示了VQ-VAE在DALL-E中的重要应用，强调其在学习离散视觉标记方面的作用，这些标记随后由Transformer模型用于文本到图像生成。

---

[上一节](02-%E7%94%A8%E4%BA%8E%E5%A4%8D%E6%9D%82%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E7%9A%84%E5%B1%82%E6%AC%A1%E5%8C%96%20VAE.md) · [下一节](04-%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E4%B8%AD%E7%9A%84%E8%87%AA%E5%9B%9E%E5%BD%92%E8%A7%A3%E7%A0%81%E5%99%A8.md)
