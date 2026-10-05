# 标准基准：GLUE 和 SuperGLUE

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-22-extrinsic-evaluation-downstream-tasks/standard-benchmarks-glue-superglue)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估大型语言模型（LLM）在实际任务中的表现，外部评估至关重要。在预训练 (pre-training)期间常用的一些指标，例如困惑度，虽然能提供有价值的指示，但无法保证LLM能有效地分类情感、回答问题或判断两句话是否为释义。为了衡量这些应用能力，研究界开发了包含多种自然语言处理（NLP）任务的基准测试集。影响力较大的包括GLUE及其后续版本SuperGLUE。

这些基准测试作为比较不同模型的共同标准。研究人员无需在分散、可能特殊化的任务上进行评估，而可以使用这些已有测试集来衡量在一般语言理解方面的进展。GLUE和SuperGLUE都汇总了多个数据集上的表现，旨在提供一个单一、全面的分数，以反映模型的通用性。

### GLUE：通用语言理解评估

GLUE基准测试于2018年推出，是迈向标准化NLP评估的重要一步。它捆绑了九项不同的英语语言理解任务，旨在涵盖多种语言现象。其目的是鼓励开发能学习通用表示、可跨不同任务迁移的模型，而不是只针对单一能力进行特殊化训练的模型。

GLUE中的任务大致可以分为以下几类：

1. **单句任务：** 评估对单个句子内部属性的理解。

   - **CoLA（语言可接受性语料库）：** 判断一个句子在语法上是否可接受。（二元分类）
   - **SST-2（斯坦福情感树库）：** 对电影评论的情感进行分类。（二元分类）
2. **相似性和释义任务：** 评估判断句子对之间语义关系的能力。

   - **MRPC（微软研究释义语料库）：** 判断两个句子是否互为释义。（二元分类）
   - **STS-B（语义文本相似度基准）：** 预测两个句子之间的相似度分数（回归，分数1-5）。
   - **QQP（Quora问题对）：** 判断Quora上发布的两个问题是否语义等效。（二元分类）
3. **自然语言推理 (inference)（NLI）任务：** 评估模型推理前提和假设句之间关系的能力。

   - **MNLI（多类型自然语言推理）：** 给定一个前提，判断一个假设是蕴含、矛盾还是中立。在匹配（同领域）和不匹配（跨领域）测试集上进行评估。（三元分类）
   - **QNLI（问题自然语言推理）：** 判断一个句子是否包含问题的答案。源自SQuAD。（二元分类）
   - **RTE（识别文本蕴含）：** 一个较小的数据集，结合了来自多个来源的文本蕴含数据。（二元分类）
   - **WNLI（Winograd NLI）：** 一个小型但非常难的数据集，专注于代词消解。（二元分类）

> GLUE基准测试中任务的大致分类。

模型通常会在每个任务的训练数据上分别进行微调 (fine-tuning)。性能使用任务专用指标衡量（例如，分类的准确率、F1分数、CoLA的马修斯相关系数、STS-B的皮尔逊/斯皮尔曼相关系数）。最终的GLUE分数是各个任务分数的未加权平均值。尽管影响力很大，但GLUE分数在许多任务上很快达到了接近人类的表现，这表明该基准测试可能不足以有效区分最先进的模型。

### SuperGLUE：提高标准

为解决GLUE出现的饱和问题并进一步拓展语言理解的边界，SuperGLUE于2019年推出。它遵循相似的结构，但包含更困难的任务，这些任务需要更复杂的推理 (inference)、常识知识和处理歧义的能力。它还包括更多样的任务格式，并更注重有挑战性的示例。

SuperGLUE包含一套新任务，尽管有些与GLUE任务重叠或相关：

- **BoolQ（布尔问题）：** 基于给定段落的是/否问答。需要推理和阅读理解能力。
- **CB（承诺库）：** 专注于识别文本中表达信念和承诺的蕴含关系的NLI任务。（三元分类：蕴含、矛盾、中立）
- **COPA（合理替代选项）：** 因果推理任务，模型必须为给定前提选择更合理的原因或结果。（二元分类）
- **MultiRC（多句阅读理解）：** 问答任务，答案可能来自段落中的多句话，且问题可以有多个正确答案。（每个答案候选的二元分类）
- **ReCoRD（常识推理阅读理解数据集）：** 完形填空式阅读理解，需要常识推理来填充新闻文章中的遮蔽实体。（抽取式问答）
- **RTE（识别文本蕴含）：** 使用来自不同来源的数据，类似于GLUE的RTE但数据经过聚合。（二元分类）
- **WiC（上下文 (context)中的词语）：** 判断一个目标词在两个不同句子中是否具有相同含义。多义词任务。（二元分类）
- **WSC（Winograd图式挑战）：** 指代消解任务，需要常识来解决精心构建句子（Winograd图式）中的歧义代词。（二元分类）

SuperGLUE也使用了适合每个任务的多种指标，最终分数是平均值。它为模型设定了更高的标准，与GLUE相比，在SuperGLUE上的进展通常被认为是模型具有更强高级语言理解能力的指标。

> SuperGLUE被设计为GLUE的更具挑战性的后续版本。

### 使用GLUE和SuperGLUE进行评估

在这些基准上评估预训练 (pre-training)的LLM通常涉及每个任务的标准化微调 (fine-tuning)过程：

1. **选择预训练模型：** 从您的基础LLM（例如BERT、RoBERTa或自定义Transformer）开始。
2. **为任务调整：** 对于大多数GLUE/SuperGLUE任务（通常是分类或回归），在基础LLM之上添加一个小的任务专用“头部”层。这个头部获取LLM的最终隐藏状态（通常是对应特殊`[CLS]`标记 (token)的表示），并将其映射到任务的输出空间（例如，类别的logits，单一回归值）。
3. **微调：** 在特定任务的训练数据集上训练这个组合模型（LLM + 头部）。通常，LLM的预训练权重 (weight)在此过程中会更新，尽管通常学习率低于随机初始化的头部。此步骤对基准测试集中的每个任务都 *独立* 进行。
4. **评估：** 使用官方指标衡量微调模型在任务的开发集或测试集上的表现。
5. **汇总：** 计算所有任务的平均分数，以获得最终的GLUE或SuperGLUE分数。

Hugging Face的`transformers`等库极大地简化了这个过程。它们提供了对基准数据集的便捷访问以及用于微调各种预训练模型的标准化脚本。

以下是一个PyTorch代码片段，展示了如何向预训练模型添加一个简单的分类头部，以在类似SST-2（情感分类）的任务上进行微调：

```python
import torch
import torch.nn as nn
from transformers import AutoModel
# 使用Hugging Face transformers的示例

# 加载预训练基础模型
base_model_name = "bert-base-uncased" # 或任何其他兼容的模型
base_model = AutoModel.from_pretrained(base_model_name)

# 定义一个简单的分类头部
class SentimentClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        # 用于正则化的Dropout层
        self.dropout = nn.Dropout(0.1)
        # 将池化输出映射到
        # 类别数量的线性层
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output):
        # 通常，使用对应[CLS]标记的输出
        # sequence_output形状：(batch_size,
        # sequence_length, hidden_size)
        # 我们取第一个标记（[CLS]）的输出
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 组合基础模型和头部
class FullSentimentClassifier(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 获取最后的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        # 将其通过分类头部
        logits = self.head(last_hidden_state)
        return logits

# 实例化完整模型
# 假设是SST-2（二元分类），类别数量为2
config = base_model.config # 从基础模型获取配置
classification_head = SentimentClassifierHead(
    config.hidden_size,
    num_labels=2
)
model = FullSentimentClassifier(base_model, classification_head)

# 现在，'model'可以在SST-2数据集上进行微调
# 示例输入（需要先进行分词）
# input_ids = ... # 经过分词的输入句子
# attention_mask = ... # 对应的注意力掩码
# logits = model(input_ids, attention_mask)
# 使用logits和标签计算损失，
# 然后进行反向传播
```

> PyTorch代码展示了如何向预训练的Transformer模型添加分类头部，以在类似GLUE的任务上进行微调。

### 理解基准测试分数

GLUE和SuperGLUE分数提供了一种有价值的标准化衡量标准，用于评估模型的通用语言能力。更高的分数通常表明模型学习了更具迁移性的表示。然而，解释这些分数需要谨慎：

- **相关性与因果关系：** 在GLUE/SuperGLUE上表现优异并不自动保证在基准测试中未包含的特定、新颖的下游任务上表现出色。基准任务可能无法完全捕捉您目标应用的细微特点。
- **基准伪影：** 模型可能学会利用基准数据集中存在的虚假相关性或伪影，而非实现真正的理解。这对测试集中较小的数据集尤其重要。
- **饱和：** 如前所述，GLUE很快达到饱和，这意味着顶尖模型取得了接近估计人类表现的分数，使得难以区分它们。SuperGLUE更具抵抗力，但随着时间推移可能面临类似问题。
- **数据污染：** 存在一种风险，即基准数据集的部分可能无意中被包含在用于现代LLM的大规模预训练 (pre-training)语料库中，从而可能虚高分数。

尽管有这些局限性，GLUE和SuperGLUE仍然是重要的工具。它们提供了一个共同的衡量标准，用于比较不同团队开发的模型，并追踪此方面的进展。它们迫使模型展现应对多种语言挑战的能力。

### 局限性与考量

尽管有用，但仅依赖GLUE/SuperGLUE存在不足：

- **计算成本：** 在测试集中对每个任务进行大型模型的微调 (fine-tuning)和评估可能计算开销大且耗时。
- **任务侧重：** 它们主要侧重于基于对输入文本理解的分类和回归任务。它们不直接有效评估连贯性、创造性或长文本 (long context)生成等生成能力。
- **静态性质：** 数据集是固定的。它们不一定反映不断变化的语言使用或新兴的NLP挑战。

因此，尽管GLUE和SuperGLUE是外部评估的非常重要组成部分，但它们应辅以对与您的特定目标直接相关的任务进行评估，以及在适当时潜在地使用更新、更具动态性的基准测试或人工评估。它们提供了对通用能力的基线理解，但通常需要特定方面或特定任务的评估才能获得完整的情况。

## 参考资料

- [GLUE: A Multi-Task Benchmark for Natural Language Understanding](https://arxiv.org/abs/1804.07461) — Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman (2018)
  Journal: International Conference on Learning Representations (ICLR) 2019 (published 2018); DOI: [10.48550/arXiv.1804.07461](https://doi.org/10.48550/arXiv.1804.07461)
  介绍通用语言理解评估（GLUE）基准的原始论文，详细说明其任务和方法。
- [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/abs/1905.00537) — Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 2019; DOI: [10.48550/arXiv.1905.00537](https://doi.org/10.48550/arXiv.1905.00537)
  介绍了SuperGLUE，作为GLUE更具挑战性的后续基准，用于评估高级语言理解能力。
- [Fine-tuning a pretrained model](https://huggingface.co/learn/nlp-course/chapter3/3) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face NLP课程中的一章，解释了微调预训练Transformer模型以完成特定NLP任务的实际过程，包含与GLUE/SuperGLUE评估相关的代码示例。

---

[上一节](03-%E8%AF%84%E4%BC%B0%E6%97%B6%E7%9A%84%E5%BE%AE%E8%B0%83%E6%AD%A5%E9%AA%A4.md) · [下一节](05-%E5%B0%91%E9%87%8F%E7%A4%BA%E4%BE%8B%E5%92%8C%E9%9B%B6%E7%A4%BA%E4%BE%8B%E8%AF%84%E4%BC%B0.md)
