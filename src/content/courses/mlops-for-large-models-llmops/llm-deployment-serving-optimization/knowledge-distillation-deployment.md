---
course: "mlops-for-large-models-llmops"
chapter: "llm-deployment-serving-optimization"
lesson: "knowledge-distillation-deployment"
sourceId: 3175
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/knowledge-distillation-deployment"
title: "知识蒸馏在部署中的应用"
description: "采用知识蒸馏创建更小、更快的学生模型以供部署。"
order: 5
plots: []
sourceHash: "b43497bff471ceb76da711e6db7806249a84ec114e4f93ec00906692d82ab605"
sourceCorrections: []
---

直接部署具有数十亿参数 (parameter)的语言模型通常会给基础设施预算带来压力，并且难以满足延迟要求。虽然量化 (quantization)等技术可以减小模型占用空间，但\*\*知识蒸馏 (knowledge distillation)（KD）\*\*提供了一种补充方法，通过训练一个显著更小、更快的模型（“学生模型”）来复现原始大型模型（“教师模型”）的行为。这让您能够在一个更易于管理的部署包中，获取一个强大LLM的多数能力。

核心思想是在学生模型的训练阶段，将大型教师模型学到的“知识”传递给小型学生模型。我们不仅使用真实标签（硬目标）训练学生模型，还使用教师模型输出概率（logits）中包含的更丰富信息来引导它。

### 蒸馏过程

1. **教师模型：** 您从一个预训练 (pre-training)的、高性能的大型语言模型开始。该模型作为知识的来源，但由于过于缓慢或昂贵，不适合在您的目标场景中直接部署。
2. **学生模型：** 您定义一个更小、计算成本更低的模型架构。这可能包括更少的层、更小的隐藏维度，或者完全不同的架构类型（例如，将大型Transformer蒸馏为小型Transformer，甚至LSTM，尽管现在较不常见）。目标是获得一个满足部署限制（大小、延迟）的模型。
3. **迁移数据集：** 您需要一个数据集（通常是原始训练集、一个子集，甚至是与任务相关的未标记 (token)数据），以便在学生模型训练期间为两个模型提供输入。
4. **训练目标：** 学生模型通过最小化一个组合损失函数 (loss function)进行训练。这通常包含：
   - **标准损失 ($L_{CE}$)：** 使用学生模型的预测以及迁移数据集中真实的地面真值标签（如果可用）计算。这确保学生模型学会任务本身。这通常是交叉熵损失。
   - **蒸馏损失 ($L_{Distill}$)：** 这促使学生模型的输出分布与教师模型对相同输入的输出分布匹配。实现此目的的常见方法是比较两个模型的logits（softmax前输出）。

### 软标签与温度

大型模型通常会产生高度确定的预测，这意味着词汇上的概率分布在选定词元 (token)处急剧集中。这对于单次最佳预测提供的信息不多。为了获取教师模型在不同可能输出之间学习到的关系的更详细知识，我们使用由**温度**缩放参数 (parameter) ($T$) 生成的“软标签”。

类 $i$ 的概率 $p_i$ 可以通过 logits $z_i$ 使用标准 softmax 计算得到：


$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$


通过温度缩放 ($T > 1$)，计算变为：


$$
p_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$


更高的温度 ($T$) 会使概率分布更平滑，使概率不那么尖锐集中，并将更高的概率分配给不太可能的类别。这促使学生模型不仅学习教师模型预测了*什么*，还学习教师模型在不同选项之间*如何*分配概率，从而捕获“暗知识”。

蒸馏损失通常使用教师模型和学生模型的软化输出之间的 Kullback-Leibler (KL) 散度：


$$
L_{Distill} = KL(p_{\text{student}}^T || p_{\text{teacher}}^T)
$$


最终的损失函数 (loss function)通过加权因子 $\alpha$ 平衡了标准任务损失和蒸馏损失：


$$
L_{\text{Total}} = \alpha L_{CE} + (1 - \alpha) L_{Distill}
$$


请注意，蒸馏损失项 $L_{Distill}$ 通常对学生模型和教师模型的 logits 使用相同的温度 $T$ 进行计算，并且梯度计算通常将此项乘以 $T^2$。标准损失 $L_{CE}$ 使用学生模型的标准 (T=1) 输出进行计算。

> 知识蒸馏 (knowledge distillation)的流程：在训练期间，输入数据同时提供给大型教师模型和小型学生模型。学生模型通过最小化一个组合损失函数进行训练，该函数结合了标准交叉熵（使用硬标签）和蒸馏损失（比较学生模型logits与软化后的教师模型logits）。

### 知识蒸馏 (knowledge distillation)的实际操作

将知识蒸馏集成到您的LLMOps管线中涉及以下步骤：

1. **选择教师/学生模型：** 选择一个合适的预训练 (pre-training)教师模型，并设计一个满足您部署限制（例如，目标延迟、内存占用）的学生模型架构。
2. **准备迁移数据集：** 整理或选择用于知识传递的数据。
3. **实现训练：** 设置一个训练任务，加载两个模型（教师模型处于评估模式），计算标准损失和蒸馏损失，并更新学生模型。这通常需要定制标准训练脚本。
4. **超参数 (parameter) (hyperparameter)调整：** 试验温度 $T$ 和损失加权因子 $\alpha$。更高的 $T$ 值会强调更软的目标，而 $\alpha$ 则平衡任务准确性与模仿教师模型的能力。
5. **评估学生模型：** 严格评估蒸馏后的学生模型，不仅要看标准准确性指标，还要看部署指标，如与教师模型相比的推理 (inference)速度、吞吐量 (throughput)和模型大小。
6. **部署学生模型：** 使用前面讨论的部署策略（容器化、优化服务器等）打包并部署更小、更快的学生模型。

### 权衡与考量

- **性能上限：** 学生模型的性能通常受到教师模型的上限限制。与大型教师模型相比，原始准确性预计会有所下降，但考虑到效率上的显著提升，这通常是可以接受的。
- **训练复杂度：** 相比于标准微调 (fine-tuning)，设置和调整蒸馏过程会增加复杂性。
- **架构选择：** 学生模型架构的选择很重要。它需要有足够能力吸取知识，但也要小到足以达到效率目标。
- **任务特异性：** 知识蒸馏 (knowledge distillation)在为教师模型表现出色的特定下游任务进行蒸馏时效果最好。

知识蒸馏提供了一种有效技术，能够将大型、难以处理的语言模型的能力压缩成更小、更利于部署的格式。通过精心传递知识，您可以显著降低推理 (inference)成本和延迟，使先进的LLM能力在更多资源受限的生产环境中可用。它补充了量化 (quantization)和剪枝等其他优化方法，为LLMOps工具库增添了一个工具，以实现高效部署。

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, and Jeff Dean (2015)
  Journal: arXiv preprint arXiv:1503.02531; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  这篇原始论文介绍了知识蒸馏、软目标和温度缩放技术，用于模型压缩。
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) — Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (2019)
  Journal: 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS 2019; DOI: [10.48550/arXiv.1910.01108](https://doi.org/10.48550/arXiv.1910.01108)
  本文介绍了DistilBERT，这是将知识蒸馏应用于基于Transformer的语言模型以提高效率的经典示例。
