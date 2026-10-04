---
course: "how-to-build-a-large-language-model"
chapter: "analyzing-model-behavior"
lesson: "probing-internal-representations"
sourceId: 6068
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-23-analyzing-model-behavior/probing-internal-representations"
title: "探查内部表示"
description: "在隐藏状态上训练简单的分类器探针，以测试编码的语言信息。"
order: 3
plots: ["plots/6068-0.json"]
sourceHash: "7e53475785a433e2036f8c35cf59c27774d839759103b54a7bc16f3ed4b271e4"
sourceCorrections: []
---

可视化注意力模式提供了一种了解模型关注点的方式，但这不能直接告诉我们每个层生成的高维隐藏状态向量 (vector)中编码了哪种语言或语义信息。这些通常具有数千维的向量，是Transformer的内部数据流通形式，将信息从一层传递到下一层。为了理解这些表示捕获了什么，我们采用一种称为*探查*的技术。

探查涉及训练简单的辅助模型，称为*探针*，以直接从LLM的内部表示中预测特定的属性。其核心思想是：如果一个简单的探针能够仅使用特定层的隐藏状态向量作为输入，准确预测某个属性（例如词性标签或依存关系），那么该信息可能被明确编码或至少在该表示中是线性可分离的。我们不太关心构建属性本身的最佳预测器；相反，我们将探针的性能用作评估LLM表示质量的诊断工具。

### 探查方法

典型的探查流程包括以下几个步骤：

1. **定义任务和获取数据：** 选择你想研究的属性（例如，识别名词的语法数，预测句法中心词）。你需要一个文本输入已标注这些属性的数据集。对于语言属性，通常使用Universal Dependencies或Penn Treebank等资源。
2. **提取表示：** 使用你想要分析的预训练 (pre-training)LLM处理已标注的数据集。重要的一点是，在此过程中LLM的权重 (weight)保持*冻结*。对于输入序列中的每个token，从你感兴趣的特定层中提取隐藏状态向量 (vector)。
3. **训练探针：** 使用提取的隐藏状态作为输入特征，已标注的属性作为目标标签，训练一个简单的分类器或回归器。探针的常见选择包括线性分类器（逻辑回归、线性支持向量机）或非常浅的多层感知机（MLP）。简单性很重要：一个复杂的探针可能会独立学习任务，而不是仅仅“读取”表示中已经存在的信息。
4. **评估探针：** 使用相关指标（例如，分类准确率，回归的均方误差）在保留的测试集上评估探针的性能。
5. **结果解读：** 高性能表明该层中的表示包含了关于所探查属性的易于访问的信息。比较不同层间的性能可以显示信息在通过模型时是如何转换或优化的。与基线探针（例如，在随机向量或静态词嵌入 (embedding)上训练的探针）进行比较有助于确定结果的重要性。

### 示例：探查词性标签

我们考虑探查预训练 (pre-training)的Transformer模型（如BERT或GPT变体）中的词性（POS）信息。

**1. 数据：** 我们需要一个带有词性标签的语料库（例如，Universal Dependencies English Web Treebank）。
**2. 表示：** 我们将语料库中的句子输入到我们冻结的LLM中，并从例如第6、12和18层收集每个token的隐藏状态向量 (vector)。
**3. 探针：** 我们选择一个简单的线性分类器。
**4. 训练与评估：** 对于每一层，我们训练一个单独的线性探针，仅根据来自该层的隐藏状态向量来预测每个token的词性标签。

这是一个PyTorch代码片段，说明了表示提取和探针定义：

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# 加载预训练模型和分词器
model_name = "bert-base-uncased" # 或任何其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval() # 将模型设置为评估模式
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False

# 示例句子和词性标签（请替换为实际数据集加载）
sentence = "Probing helps analyze model representations."
# 假设标签是 ['名词', '动词', '动词', '名词', '名词', '标点符号']
# 实际操作中，请仔细对齐分词与标签

inputs = tokenizer(sentence, return_tensors="pt")

# 从特定层（例如第8层）提取隐藏状态
target_layer = 8
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states 是一个元组：(嵌入层, 第1层, ..., 第N层)
    layer_representations = outputs.hidden_states[target_layer]
    # 形状：[批大小, 序列长度, 隐藏大小]

# 假设我们已经提取了表示和相应的
# 许多示例的词性标签ID
# representations_tensor: [示例数量, 隐藏大小]
# labels_tensor: [示例数量]

# 定义一个简单的线性探针
hidden_size = layer_representations.shape[-1]
num_pos_tags = 17 # UD EWT中唯一词性标签的示例数量
probe_classifier = nn.Linear(hidden_size, num_pos_tags)

# --- 训练循环 ---
# 标准的PyTorch训练循环如下：
# - 定义损失函数（例如 CrossEntropyLoss）
# - 定义优化器（例如 AdamW，仅优化
#   probe_classifier.parameters()）
# - 遍历批次的 (representations_tensor, labels_tensor)
# - 计算损失，反向传播，更新探针权重
# - 在验证集上评估
# --------------------

# 训练完成后，在表示的测试集上评估 probe_classifier。
```

### 探针结果解读

探查实验的结果可能非常具有启发性：

- **性能大小：** 高准确率（例如，使用线性探针达到90%以上的词性标注准确率）强烈表明该特定层以易于访问的格式编码了此信息。
- **分层比较：** 通常，词性标注或依存句法分析等句法任务的性能在Transformer的中间层达到顶点，而需要更多语义整合的任务则可能在更后面的层达到顶点。这表明信息从低级别语言特征向高级别抽象逐步演进。
- **对照任务：** 进行对照任务是好的实践。例如，训练一个探针来预测分配给token的随机标签。如果这个探针的表现明显高于随机水平，则可能表明实验设置存在偏差或问题。另一个对照是*选择性*：确保为词性标注训练的探针在例如预测依存关系时表现不佳，反之亦然。这证实了探针是专门使用目标信息。



![探针在各层上的准确率](plots/6068-0.json)



> 比较显示，词性标注准确率在模型层中比依存关系准确率更早达到峰值。

### 探查任务的范围

探查可应用于广泛的语言和语义现象：

- **形态句法：** 时态、数、格、词性标签。
- **句法：** 句法成分边界、依存关系、语法角色识别。
- **语义：** 语义角色标注（SRL）、命名实体识别（NER）、共指消解、关系抽取。
- **知识：** 尽管线性探查较难，但有些研究试图探查表示中嵌入 (embedding)的事实知识。

### 局限与考量

探查是一种有用的分析工具，但注意其局限性很重要：

- **探针的简单性与能力：** 探针必须足够简单，这样我们才能合理地将高表现归因于表示本身，而不是探针的学习能力。总会有权衡；一个稍微复杂一点的探针可能会显现出非线性可分离但仍易于提取的信息。
- **相关性，而非因果性：** 发现某一层编码了词性信息，并不一定意味着模型会以我们预期的方式将这些信息明确用于其主要目标（例如，下一个token预测）。这表明信息是存在且可访问的。
- **数据依赖性：** 用于探查的已标注数据集的质量和性质非常重要。
- **分词 (tokenization)对齐 (alignment)：** 需要仔细对齐LLM的分词和语言标注（可能基于词语）。

尽管有这些方面，探查仍为理解大型语言模型学习到的内部知识结构提供了宝贵的看法。通过系统地检视不同类型的信息在各层中是如何表示的，我们能更好地理解这些模型如何处理语言，这能为调试、模型改进以及构建更可靠、更易解释的系统提供依据。

## 参考资料

- [What Does BERT Learn about the Structure of Language?](https://arxiv.org/pdf/1904.07727.pdf) — John Hewitt and Christopher D. Manning (2019)
  Journal: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL); Pages: 872-885; DOI: [10.18653/v1/P19-1084](https://doi.org/10.18653/v1/P19-1084)
  这是一篇基础性论文，介绍了结构探针技术，用于分析BERT等上下文词表示的隐藏状态中编码的句法信息。
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/pdf/1905.05950.pdf) — Ian Tenney, Dipanjan Das, and Ellie Pavlick (2019)
  Journal: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL); Pages: 4509-4522; DOI: [10.18653/v1/P19-1443](https://doi.org/10.18653/v1/P19-1443)
  这项工作通过广泛的探针实验，系统研究了BERT表示层中出现的语言知识类型，涵盖从形态学到语义学的多个层面。
- [Linguistic Knowledge and Transferability of Contextual Representations](https://aclanthology.org/N19-1111/) — Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Volume: 1; Pages: 1104-1115; DOI: [10.18653/v1/N19-1111](https://doi.org/10.18653/v1/N19-1111)
  通过探针技术，全面分析了包括BERT在内的各种上下文嵌入中编码的语言信息，包括不同层的信息及其向下游任务的迁移能力。
- [Transformers Documentation - AutoModel](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) — Hugging Face team (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，详细说明了如何加载预训练模型和配置输出（例如隐藏状态），这对于实现探针实验是基础的。
