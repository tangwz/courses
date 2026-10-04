---
course: "fine-tuning-small-language-model"
chapter: "principles-of-small-language-models"
lesson: "supervised-fine-tuning-mechanics"
sourceId: 8499
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-1-principles-of-small-language-models/supervised-fine-tuning-mechanics"
title: "有监督微调机制"
description: "分析通过有监督学习更新权重的数学和结构机制。"
order: 3
plots: []
sourceHash: "3b4ba6b67769194e08d093b6e926c8671bde05bc1db2a383bbd6facb15ad68f3"
sourceCorrections: []
---

要理解有监督学习 (supervised learning)如何更新模型的内部结构，需要观察其逐步训练的循环过程。有监督微调 (fine-tuning)通过使用特定的“指令-响应”对数据集，对已经掌握语法和常识的预训练 (pre-training)模型应用权重 (weight)更新规则。

在启动微调过程时，模型通过一个包含三个主要阶段的迭代循环进行学习：前向传播、损失计算和反向传播 (backpropagation)。

### 前向传播与损失计算

在训练期间，模型接收输入序列并尝试预测下一个标记（token）。这就是前向传播。原始文本被转换为数值标记，这些标记穿过模型的 Transformer 块，其中应用了自注意力 (self-attention)机制 (attention mechanism)和前馈神经网络 (neural network)。输出是针对整个词汇表 (vocabulary)的概率分布，通常被称为 logits。

由于这是有监督学习 (supervised learning)，训练脚本可以访问正确答案。模型的预测结果会与自定义数据集中的实际目标标记进行对比。这种对比通过一个被称为损失函数 (loss function)的数学函数来量化 (quantization)。对于语言模型，通常使用交叉熵损失（Cross-Entropy Loss）。

$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$

在该方程中，$C$ 代表类别数量，即模型的词汇表大小。变量 $y_i$ 是真实概率分布，正确标记为 1，其他均为 0。变量 $\hat{y}_i$ 是对标记 $i$ 的预测概率。计算出的损失越低，表示模型的预测与目标数据越吻合。

### 反向传播 (backpropagation)与梯度

计算出损失后，模型通过反向传播执行反向过程。这一步计算损失相对于模型中每一个权重 (weight)的梯度。我们在之前的权重更新方程中将此梯度表示为 $\nabla L(w_t)$。梯度指出了每个参数 (parameter)为了降低总损失所需要的调整方向和幅度。

此计算依赖于微积分中的链式法则。数学误差从最终输出层一直传导回初始输入嵌入 (embedding)层。在此阶段，系统会计算每个特定权重对最终误差的影响程度。

> 有监督微调 (fine-tuning)的循环过程，从产生预测的前向传播到更新模型权重的优化器步骤。

### 优化器与权重 (weight)更新

计算出梯度后，优化器负责实际更新模型的权重。虽然标准梯度下降 (gradient descent)使用基础的 $w_{t+1} = w_t - \alpha \nabla L(w_t)$ 公式，但现代微调 (fine-tuning)通常使用更先进的优化器，如 AdamW。

优化器应用学习率（用 $\alpha$ 表示），它控制模型在更新权重时迈出的步长。如果学习率过高，模型可能会跳过最优权重而无法收敛；如果学习率过低，训练过程会异常缓慢。AdamW 在基础公式上进行了改进，它根据历史梯度动态调整每个参数 (parameter)的学习率，同时加入权重衰减以防止模型死记硬背训练数据。

由于反向传播 (backpropagation)需要存储前向传播的中间激活值，因此有监督微调期间的内存开销明显高于标准文本生成。对于模型中的每个参数，系统必须存储权重本身、计算出的梯度以及优化器状态。

这正是训练小语言模型需要合理管理资源的原因。一个在生成文本时占用 4 GB 显存 (VRAM)的模型，在使用标准有监督微调进行训练时可能需要 16 GB 或更多显存。通过仅修改这些权重的一小部分，我们可以大幅降低内存占用。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 5998–6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  Transformer 架构的开创性论文，详细解释了前向传播章节中提到的自注意力机制。
- [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR)
  介绍 AdamW 优化器的原始论文，该优化器在现代微调中用于处理权重衰减和学习率自适应。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 200-220
  标准的学术教科书，提供了反向传播、梯度和交叉熵损失的数学基础。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Christopher Manning (2023)
  Publisher: Stanford University
  一门全面的大学课程，涵盖了语言模型训练的机制，包括分词和损失计算。
