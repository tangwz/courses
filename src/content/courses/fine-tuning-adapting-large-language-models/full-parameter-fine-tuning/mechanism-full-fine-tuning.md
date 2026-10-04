---
course: "fine-tuning-adapting-large-language-models"
chapter: "full-parameter-fine-tuning"
lesson: "mechanism-full-fine-tuning"
sourceId: 2996
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-3-full-parameter-fine-tuning/mechanism-full-fine-tuning"
title: "全参数微调的机制"
description: "了解大语言模型在全参数微调过程中所有参数的更新方式。"
order: 1
plots: []
sourceHash: "98b22bdb67c9dcadda2da750c71a63cb616de1e8cd1e5ba7f64a694951242618"
sourceCorrections: []
---

全参数 (parameter)微调 (fine-tuning)，通常简称为“微调”，是调整预训练 (pre-training)大语言模型（LLM）以适应特定下游任务或某一方面的最直接方法。顾名思义，此方法涉及使用新的、针对任务的数据来更新模型中*每一个*可训练参数。这与您后面会遇到的其他技术形成对比，例如参数高效微调（PEFT），后者只修改一小部分参数，或引入新的小参数集。

其根本原则是迁移学习 (transfer learning)。我们从一个模型开始，它已通过在庞大数据集上进行的广泛预训练阶段，学习了语言、语法的通用模式以及相当多的知识。我们将预训练模型表示为由其权重 (weight) $\theta_{pre}$ 参数化的函数 $f$。此模型接受输入 $\mathbf{x}$ 并产生输出，因此我们有 $f(\mathbf{x}; \theta_{pre})$。这些预训练权重 $\theta_{pre}$ 作为学习新任务的一个高效起始点。

全参数微调的目标是调整这些参数 $\theta_{pre}$，使其变为一套新的参数 $\theta_{tuned}$，从而在我们的特定目标任务上表现良好。此目标任务的特点是有一个新数据集 $D_{task} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_ {i=1}^N$，其中 $\mathbf{x}_i$ 是一个输入示例（例如，提示、问题），而 $\mathbf{y}_i$ 是期望的输出（例如，分类标签、生成的回复）。我们通过最小化此数据集上的任务特定损失函数 (loss function) $\mathcal{L}$ 来实现这一目标。从数学上看，我们旨在找到：


$$
\theta_{tuned} = \arg \min_{\theta} \sum_{(\mathbf{x}_i, \mathbf{y}_i) \in D_{task}} \mathcal{L}(f(\mathbf{x}_i; \theta), \mathbf{y}_i)
$$


优化过程始于用预训练权重初始化模型权重：$\theta \leftarrow \theta_{pre}$。然后，我们使用标准随机梯度下降 (gradient descent)（SGD）或其自适应变体之一，例如Adam或AdamW（带有权重衰减的Adam，通常是Transformer模型的首选），来迭代更新权重。

主要的更新循环对于从 $D_{task}$ 中抽取的每个数据批次 $(\mathbf{X}_{batch}, \mathbf{Y}_{batch})$ 如下进行：

1. **前向传播：** 使用当前参数 $\theta$ 计算模型对该批次的预测：
   
   $$
   \hat{\mathbf{Y}}_{batch} = f(\mathbf{X}_{batch}; \theta)
   $$
   
2. **损失计算：** 计算预测值 $\hat{\mathbf{Y}}_{batch}$ 与真实目标值 $\mathbf{Y}_{batch}$ 之间的损失：
   
   $$
   \mathcal{L}_{batch} = \mathcal{L}(\hat{\mathbf{Y}}_{batch}, \mathbf{Y}_{batch})
   $$
   
   特定的损失函数 $\mathcal{L}$ 取决于任务（例如，分类或序列生成的交叉熵损失）。
3. **反向传播 (backpropagation)：** 计算批次损失相对于*所有*模型参数 $\theta$ 的梯度：
   
   $$
   \nabla_{\theta} \mathcal{L}_{batch}
   $$
   
   此步骤计算每个单独参数对批次预测误差的贡献程度。
4. **参数更新：** 沿损失最小化方向调整参数 $\theta$，并按学习率 $\eta$ 进行缩放。以基本的SGD更新规则为例：
   
   $$
   \theta \leftarrow \theta - \eta \cdot \nabla_{\theta} \mathcal{L}_{batch}
   $$
   
   像AdamW这样的优化器采用更复杂的更新规则，涉及动量和每个参数的自适应学习率，但使用梯度更新所有权重的核心原则保持不变。

此过程会重复进行多个批次和周期，直到模型在验证集上的性能停止提升，或达到设定的步数为止。

> 从预训练状态（$\theta_{pre}$）开始的参数权重，使用来自任务特定损失（$\nabla \mathcal{L}_{task}$）的梯度进行迭代更新，以达到针对新任务优化后的微调状态（$\theta_{tuned}$）。

全参数微调的一个显著特点是梯度会回传到*整个*网络架构。调整不仅限于最终输出层，还可能涉及所有Transformer块、注意力机制 (attention mechanism)、前馈网络，甚至初始嵌入 (embedding)层。这使得模型能够调整其所有级别的内部表示，以更好地适应目标任务的细节。

全参数微调的有效性源于其借助预训练阶段学到的强大、通用表示。我们不是从随机权重开始学习过程（对于这种规模的模型来说，这在计算上是不可行的），而是从一个已掌握语言结构和语义的状态开始。这通常会带来：

- 与从头开始训练相比，在目标任务上收敛更快。
- 最终性能更佳，尤其当目标数据集相对较小时，因为模型会从其预训练知识中进行泛化。

然而，这种全面的更新机制伴随着显著的计算需求。更新数十亿参数需要大量的GPU内存（用于存储参数、梯度和优化器状态）和计算时间。此外，在可能更小、更专业的数据集上进行微调会带来*过拟合 (overfitting)*的风险，即模型记忆了微调数据，但失去了一些通用能力，或在未见过的任务示例上表现不佳。这些难题促使我们必须仔细调整超参数 (hyperparameter)、采用正则化 (regularization)技术和资源管理策略，我们将在本章后续部分讨论这些内容，这也为在本课程后期研究更参数高效的方法铺平了道路。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  一篇重要论文，介绍了BERT模型并推广了基于Transformer的语言模型的预训练和微调范式，直接阐释了全参数微调方法。
- [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=rTeoPmkrb) — Ilya Loshchilov, Frank Hutter (2018)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  本文介绍了AdamW优化器，阐述了其机制和优势，尤其适用于Transformer模型，与章节内容提及相符。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习基本原理的综合教材，包括迁移学习、SGD和Adam等优化算法以及反向传播机制。
