---
course: "how-to-build-a-large-language-model"
chapter: "extrinsic-evaluation-downstream-tasks"
lesson: "common-downstream-nlp-tasks"
sourceId: 6061
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-22-extrinsic-evaluation-downstream-tasks/common-downstream-nlp-tasks"
title: "常见下游自然语言处理任务"
description: "考察文本分类、问答、摘要、翻译等任务。"
order: 2
plots: []
sourceHash: "f4b62640d735269add60616cac129df484b0e946f66c1762264698ebe5f1db14"
sourceCorrections: []
---

内在衡量标准，例如困惑度，能提供关于语言模型核心能力的信息，但往往无法准确预测模型在特定、实际任务上的表现。为了更全面地了解大型语言模型（LLM）的实用性，外在评估衡量其在一系列成熟的下游自然语言处理（NLP）任务上的表现。这种方法将评估与实际应用相结合，有助于了解模型在不同场景下的优点和不足。

对下游任务进行评估通常涉及调整预训练 (pre-training)的LLM，这常通过微调 (fine-tuning)或提示技术完成，然后使用任务特定的衡量标准来评估其表现。接下来，我们来看一些为此目的最常用的任务。

### 文本分类

文本分类是一项核心NLP任务，其目标是为给定的文本输入分配预设的类别或标签。示例包括：

- **情感分析：** 判断评论、推文或句子的情感基调（例如，积极、消极、中立）。
- **主题分类：** 将文档归类到一个或多个主题（例如，体育、科技、政治）。
- **垃圾邮件检测：** 识别电子邮件或消息是否为垃圾邮件。

**相关性：** 此任务测试模型理解文本段落整体含义、语气和主旨的能力。

**评估：** 性能通常使用准确率、精确率、召回率和F1分数来衡量，尤其是在类别不平衡的情况下（如垃圾邮件检测）。

**LLM应用：** 对于微调 (fine-tuning)，一种常见做法是在特殊标记 (token)（如BERT风格模型中的`[CLS]`）的最终隐藏状态或序列的池化输出之上添加一个线性分类层。然后，模型会在针对分类任务的标注数据集上进行微调。

```python
import torch.nn as nn
from transformers import AutoModel

# 示例：向预训练模型添加分类头部
class SimpleClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        # 使用模型的配置获取隐藏层大小
        self.classifier_head = nn.Linear(
            self.transformer.config.hidden_size, num_labels
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        # 从基础Transformer模型获取输出
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 通常使用第一个标记（[CLS]）的隐藏状态
        # 用于分类
        pooled_output = outputs.last_hidden_state[:, 0]
        # 通过分类层
        logits = self.classifier_head(pooled_output)
        return logits

# 用法
# model = SimpleClassifier("bert-base-uncased", num_labels=3)
# # 例如，用于积极/消极/中立情感
# 假设input_ids和attention_mask是已准备好的张量
# logits = model(input_ids, attention_mask)
# loss = loss_function(
#     logits.view(-1, self.num_labels),
#     labels.view(-1)
# )
```

另一种方法是，在少样本或零样本设定下，可以通过指令和示例来提示LLM执行分类，而无需显式微调。

### 问答 (QA)

问答系统旨在为以自然语言提出的问题提供答案。常见变体有：

- **抽取式问答：** 答案是从提供的上下文 (context)文档中直接抽取的一段文本。像SQuAD（斯坦福问答数据集）这样的数据集是此任务的基准。
- **生成式问答：** 模型以自己的话语生成答案，可能综合了上下文信息或其内部知识。
- **开放域问答：** 模型必须从大量文档语料库或其内部知识库中找到答案，而无需在问题旁提供特定上下文。

**相关性：** 问答任务严格测试阅读理解、信息检索，有时还测试综合并生成连贯文本的能力。

**评估：** 抽取式问答通常使用精确匹配（EM）和F1分数来评估预测的答案段与真实答案的比较。生成式和开放域问答常使用像ROUGE（面向召回的摘要评估替补）或BLEU（双语评估替补）这样的指标，尽管人工评估也同样重要。

**LLM应用：** 对于抽取式问答，模型常被微调 (fine-tuning)以预测上下文内答案段落的起始和结束标记 (token)索引。对于生成式问答，序列到序列模型或仅解码器模型会被微调或提示以直接生成答案。

> 抽取式问答任务的输入组成部分（上下文、问题）和结果答案。

### 摘要

文本摘要的目标是在保留源文档最主要信息的同时，生成一个更短的版本。

- **抽取式摘要：** 直接从原文中选择显著的句子或短语。
- **生成式摘要：** 生成新的摘要，对源文档进行转述和浓缩，可能使用新颖的词语或措辞。现代LLM在此方面表现出色。

**相关性：** 摘要测试模型识别重要信息、理解上下文 (context)以及生成流畅、简洁文本的能力。

**评估：** ROUGE分数（特别是ROUGE-1、ROUGE-2、ROUGE-L）是标准衡量指标，通过N-gram重叠将生成的摘要与人工编写的参考摘要进行比较。对连贯性、流畅性和信息覆盖度进行人工评估也普遍采用。

**LLM应用：** 序列到序列架构曾是传统做法，但大型仅解码器模型现在非常有效。它们通常在文档-摘要对上进行微调 (fine-tuning)，或通过“总结以下文本：”等指令进行提示。

### 机器翻译

机器翻译涉及将文本从源语言翻译成目标语言。

**相关性：** 测试模型对多种语言的语法、句法、语义和文化背景的理解，以及其生成能力。

**评估：** BLEU分数是广泛使用的衡量指标，衡量生成的译文与参考译文相比的N-gram准确性。METEOR和chrF等其他指标也常被使用。人工评估对于评估翻译质量和流畅性仍然很重要。

**LLM应用：** 序列到序列模型曾是标准做法。大型多语言LLM通常可以在零样本或少样本设定下执行翻译，通过在提示中包含示例或指令（例如，“将以下英文文本翻译成法文：...”）。在并行语料库上进行微调 (fine-tuning)仍是实现特定语言对高性能的常见做法。

### 自然语言推理 (inference) (NLI) / 文本蕴含

NLI任务要求模型判断一对句子：一个“前提”和一个“假设”之间的逻辑关系。关系通常是以下之一：

- **蕴含：** 假设逻辑上来源于前提。
- **矛盾：** 假设与前提相矛盾。
- **中立：** 假设既不蕴含于前提，也不与前提相矛盾。

**相关性：** NLI被认为是衡量一般语言理解和推理能力的良好替代，因为它要求掌握句子的含义和推论。

**评估：** 准确率是主要衡量指标，衡量正确分类的关系的百分比。

**LLM应用：** 类似于文本分类，模型通常通过连接前提和假设（带分隔符标记 (token)），将其输入LLM，并在池化输出上添加分类头部来预测三个标签之一进行微调 (fine-tuning)。

> NLI任务的输入组成部分（前提、假设）和结果标签。

这些仅仅是用于LLM外在评估的一些常见下游任务。其他包括命名实体识别（NER）、共指消解、跨不同场景的情感分析，以及更多在GLUE和SuperGLUE等基准测试中发现的专业任务，我们将在后面讨论。通过评估这些多样化的任务，可以对LLM的整体能力及其对多种用途的适应性进行评估。

## 参考资料

- [GLUE: A Multi-Task Benchmark for Natural Language Understanding](https://doi.org/10.48550/arXiv.1804.07461) — Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1804.07461](https://doi.org/10.48550/arXiv.1804.07461)
  介绍了通用语言理解评估（GLUE）基准，这是一个包含多种NLP任务的集合，用于在下游应用中评估通用语言理解模型。
- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本广受认可的教材，全面涵盖了NLP任务、评估指标和基础模型，对于理解下游应用及其评估非常相关。
