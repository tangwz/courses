---
course: "speech-recognition-synthesis-asr-tts"
chapter: "advanced-acoustic-modeling-asr"
lesson: "hybrid-hmm-dnn"
sourceId: 3118
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-2-advanced-acoustic-modeling-asr/hybrid-hmm-dnn"
title: "混合HMM-DNN系统"
description: "介绍混合ASR系统的架构、训练流程及解码机制。"
order: 1
plots: []
sourceHash: "8194ace5e8d47ec9134bdd160f8690759a6734d50c55ae7c1f5db4c42c8ed0ff"
sourceCorrections: []
---

尽管端到端模型代表了当前自动语音识别（ASR）的最新发展，但了解混合HMM-DNN系统仍然有必要。它们在性能上比传统GMM-HMM系统有了很大提升，并构成了许多当时性能优异的识别器的核心组成部分，持续了数年。这些系统巧妙地将隐马尔可夫模型（HMM）在序列建模方面的特长与深度神经网络 (neural network)（DNN）有效的判别能力结合起来。

主要思路直接且有效：用DNN替代过去常用于计算给定HMM状态下观测到声学特征概率的高斯混合模型（GMMs）。DNN不再直接使用GMMs对特征分布进行建模，而是根据输入声学特征预测HMM状态的*后验概率*。

### 架构和组成

混合HMM-DNN系统保持HMM在语音时序变化建模方面的整体结构，但使用DNN作为其发射概率计算器。

1. **输入特征：** 常用声学特征，如梅尔频率倒谱系数（MFCCs）或滤波器组能量，通常会加入其差分特征（delta、delta-delta），有时会跨多个帧进行拼接，以向DNN提供时序信息。
2. **深度神经网络 (neural network)（DNN）：** 一般是一个具有多层隐层的全连接神经网络。它将给定时间帧$t$的声学特征向量 (vector)$\mathbf{o}_t$作为输入。输出层通常包含对应于上下文 (context)相关HMM状态（常被称为senones）的神经元，并使用softmax激活函数 (activation function)。DNN输出每个HMM状态$q_i$的后验概率$P(q_i | \mathbf{o}_t)$。
3. **HMM框架：** 规定了语音单元（如音素）的状态序列结构，并明确了这些状态之间允许的转移及其对应的概率（$P(q_j | q_i)$）。它根据发音词典规定了音素如何连接形成词语。
4. **解码图（WFST）：** HMM拓扑结构、发音词典和语言模型常被编译成一个加权有限状态转换器（WFST）图。这个图代表了所有可能且有效的HMM状态和词语序列。

> 常用混合HMM-DNN ASR系统中的信息流向。

### 混合系统训练

训练包括两个主要步骤：生成对齐 (alignment)和DNN训练。

1. **对齐生成：** 训练DNN的一个必要条件是帧级别的对齐，它将每个声学特征向量 (vector)$\mathbf{o}_t$映射到其对应的HMM状态$q_i$。这种对齐一般通过在相同训练数据上使用预训练 (pre-training)好的GMM-HMM系统来生成。维特比算法以“强制对齐”模式运行，即提供已知的文本标注，算法会找到与音频和标注对应的最可能的隐藏状态序列。
2. **DNN训练：** 一旦帧级别的对齐准备好，DNN便以标准的监督学习 (supervised learning)方式进行训练。
   - **输入：** 训练集中的声学特征向量（$\mathbf{o}_t$）。
   - **目标：** 该帧对应的对齐HMM状态标签（常表示为独热向量）。
   - **损失函数 (loss function)：** 一般而言，是DNN预测的后验分布与目标独热向量之间的分类交叉熵。
   - **优化：** 标准的反向传播 (backpropagation)和梯度下降 (gradient descent)算法（例如Adam、带动量的SGD）。

这个过程有时可以重复：训练好的DNN可以用于生成更好的对齐，然后这些对齐又用于再次训练DNN。

### 从后验概率到似然概率用于解码

HMM解码器，如维特比算法，利用*似然概率* $P(\mathbf{o}_t | q_i)$进行操作，它代表了给定HMM状态下观测到声学特征的概率。然而，DNN直接输出的是*后验概率* $P(q_i | \mathbf{o}_t)$。我们需要使用贝叶斯定理将这些后验概率转换为似然概率：


$$
P(\mathbf{o}_t | q_i) = \frac{P(q_i | \mathbf{o}_t) P(\mathbf{o}_t)}{P(q_i)}
$$


在解码过程中，我们比较的是针对*相同*观测序列$\mathbf{O} = (\mathbf{o}_1, ..., \mathbf{o}_T)$的不同状态序列。因此，$P(\mathbf{o}_t)$作为所有状态在时间$t$上的一个常数缩放因子，在寻找最可能的状态序列时可以不予考虑。$P(q_i)$项是状态$q_i$的先验概率。这个先验概率可以根据状态在训练对齐 (alignment)中的相对频率来计算。

因此，HMM解码器中使用的量常是*缩放似然概率*：


$$
P'(\mathbf{o}_t | q_i) = \frac{P(q_i | \mathbf{o}_t)}{P(q_i)}
$$


这种除以状态先验概率$P(q_i)$的做法补偿了训练数据中状态频率的不平衡，避免DNN偏向于为频繁出现的状态输出高概率，仅因为它们频繁。这些缩放似然概率随后在维特比搜索算法中使用，并与HMM转移概率和语言模型分数（常通过WFSTs集成）结合，以寻找最可能的词语序列。

### 上下文 (context)相关建模：Senones

为了捕捉协同发音效应（即一个音素的发音如何受其相邻音素影响），混合系统几乎总是对上下文相关的语音单元进行建模，一般是三音素（一个音素加上其左右上下文）。由于可能的三音素数量庞大，它们会根据声学相似性聚类成数量较少的绑定状态，即senones。因此，混合系统中DNN的输出层预测的是这些senones的后验概率，而不是上下文无关音素的后验概率。

### 优点与缺点

**优点：**

- **准确性提升：** DNN在判别声学状态方面远优于GMM，使得词错误率（WER）相比GMM-HMM系统有显著下降。
- **特征表示：** DNN能自动地从输入声学信息中学习有用的特征表示。
- **兼容现有设施：** 可以方便地与成熟的HMM解码工具和技术（如基于WFST的解码器）结合。

**缺点：**

- **训练流程复杂：** 要求多阶段训练流程，常需要一个预训练 (pre-training)的GMM-HMM系统来生成初始对齐 (alignment)。
- **条件独立性假设：** HMM框架仍然假设在给定状态下观测是独立的，这对于语音来说并非完全符合实际。DNN仅对发射概率进行建模，不直接处理序列动态性。
- **对齐质量敏感：** 性能受初始对齐质量的影响。
- **非完全端到端：** 声学建模（DNN）与序列建模（HMM）的分离使得无法一次性联合优化整个系统。

尽管主要被CTC、RNN-T和注意力模型等端到端架构（我们接下来会介绍）所取代，但混合HMM-DNN系统是ASR演进中的重要一步。它们展现了深度学习 (deep learning)在声学建模方面的巨大潜力，并为后来的进展打下了基础。了解其架构有助于理解现代端到端系统的设计选择和优势。

## 参考资料

- [Deep Neural Networks for Acoustic Modeling in Speech Recognition: An Overview](https://ieeexplore.ieee.org/document/6296520) — G. Hinton, L. Deng, D. Yu, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury (2012)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 29; Pages: 82-97; DOI: [10.1109/MSP.2012.2205597](https://doi.org/10.1109/MSP.2012.2205597)
  这篇开创性论文总结了使用深度神经网络进行声学建模的突破，显著提升了自动语音识别性能，超越了传统的GMM-HMM系统，并为后续的深度学习进展奠定了基础。
- [Automatic Speech Recognition: A Deep Learning Approach](https://books.google.com/books/about/Automatic_Speech_Recognition_A_Deep_Learn.html?id=R2s_BQAAQBAJ) — Dong Yu, Li Deng (2014)
  Publisher: Springer; DOI: [10.1007/978-1-4614-9454-8](https://doi.org/10.1007/978-1-4614-9454-8)
  这是一本基础性书籍，广泛涵盖了深度学习在自动语音识别中的应用，包括详细介绍混合HMM-DNN系统的架构、训练以及从GMM-HMM到HMM-DNN的演变。
