---
course: "introduction-to-transformer-models"
chapter: "training-implementing-transformers"
lesson: "using-pretrained-libraries-brief"
sourceId: 3035
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/using-pretrained-libraries-brief"
title: "使用预训练模型库（简述）"
description: "简要介绍Hugging Face Transformers等库如何使用预训练模型。"
order: 7
plots: []
sourceHash: "70f25cbf049ec988e2cc6125e776e500999bf7ecacc794aed4c984587dc33bbc"
sourceCorrections: []
---

尽管理解如何从头开始构建Transformer组件，例如分词 (tokenization)、损失函数 (loss function)和优化，有益，但从零开始构建和训练BERT或GPT-3这样的大规模模型是一项重大的工作。这需要庞大的数据集、大量的计算资源（通常是数百个GPU运行数周），以及相当大的工程投入。

幸运的是，机器学习 (machine learning)社区已开发出优秀的库，它们提供了对预训练 (pre-training)Transformer模型的访问，以及有效使用它们所需的工具。这些库屏蔽了大量的底层实现细节，使您能够更迅速地将强大的模型应用于您的特定任务。

在此方面，最知名的库是Hugging Face的`transformers`。它为跨越多种模态（文本、视觉、音频）和深度学习 (deep learning)框架（PyTorch、TensorFlow、JAX）的数千个预训练模型提供了一个统一的接口。

### 为何使用预训练 (pre-training)模型库？

使用Hugging Face `transformers`等库具有以下几个显著的优势：

1. **获取最先进模型：** 您可以立即获取已在大量数据集上进行预训练的模型，这些模型能捕捉丰富的语言或数据规律。自行训练这些模型通常是不可行的。
2. **标准化API：** 该库提供了加载模型、分词 (tokenization)器 (tokenizer)以及执行推理 (inference)或微调 (fine-tuning)的一致方法，无论具体的Transformer架构（例如BERT、GPT-2、T5、BART）如何。
3. **集成式分词：** 预训练模型需要它们训练时所用的特定分词方法（如WordPiece或BPE）。库会提供相应的分词器，确保输入数据得到正确处理。
4. **缩短开发时间：** 无需自己实现注意力机制 (attention mechanism)、编码器/解码器堆栈和训练循环，您只需几行代码即可加载模型和分词器。
5. **降低计算成本：** 对于许多应用而言，在较小、特定任务的数据集上微调预训练模型，比从头训练大大加快速度，且所需数据和计算资源也大大减少。
6. **社区和生态系统：** 这些库通常附带广泛的文档、教程和庞大的社区，使得查找解决方案和分享知识变得更加容易。

### 典型工作流程

从库中使用预训练 (pre-training)模型通常包含以下步骤：

1. **安装：** 安装库（例如，`pip install transformers datasets`）。
2. **加载分词 (tokenization)器 (tokenizer)：** 实例化您打算使用的特定预训练模型相关的分词器。该分词器了解如何将文本拆分成模型期望的子词 (subword)单元，并将其转换为输入ID。
3. **加载模型：** 实例化预训练模型架构，它会自动下载并缓存已学习的权重 (weight)。您可以加载一个基础模型用于特征提取，或加载一个为特定任务（如序列分类或问答）预配置的模型。
4. **准备输入：** 使用已加载的分词器将您的原始文本数据处理成模型所需的格式，包括输入ID、注意力掩码以及可能的 token 类型ID。
5. **推理 (inference)或微调 (fine-tuning)：**
   - **推理：** 将准备好的输入传递给模型，以获取预测、嵌入 (embedding)或其他输出。
   - **微调：** 添加一个特定任务的输出层（如果尚未存在），并在您的标注数据集上继续训练模型（或其部分），根据您的特定目标调整预训练权重。

以下是使用Python和Hugging Face `transformers`库的示例：

```python
# 注意：此代码用于说明工作流程。
# 实际实现可能因任务而略有不同。

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch # 如果使用TF，则为tensorflow

# 1. 选择一个预训练模型检查点
model_name = "bert-base-uncased" # 示例：BERT模型

# 2. 加载与模型关联的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 加载预训练模型（此处用于序列分类）
#    加载 'AutoModel' 将获得没有特定头的基础Transformer。
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 4. 准备输入文本
raw_text = ["This is the first sentence.", "This is another sentence."]
inputs = tokenizer(raw_text, padding=True, truncation=True, return_tensors="pt")
# 'inputs' 现在包含 input_ids、attention_mask 等，作为 PyTorch 张量（"pt"）

# 5. 执行推理（获取模型输出）
with torch.no_grad(): # 推理时禁用梯度计算
    outputs = model(**inputs)
    logits = outputs.logits # 分类头的原始分数

# （可选）进一步处理：应用softmax，映射到标签等。
probabilities = torch.softmax(logits, dim=-1)
predicted_classes = torch.argmax(probabilities, dim=-1)

print(f"输入ID形状: {inputs['input_ids'].shape}")
print(f"Logits形状: {logits.shape}")
print(f"预测类别: {predicted_classes}")
```

"这个简短的概述仅是粗略概览。像`transformers`这样的库包含针对各种任务、模型配置、训练工具（如`Trainer` API）以及与数据集集成的广泛功能。虽然构建核心组件提供了必要的认知，但在实际场景中，使用这些库通常是应用Transformer模型的最佳途径。它们代表了自然语言处理研究和发展的重要推动力。"

## 参考资料

- [Hugging Face Transformers Library Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face team (2024)
  Hugging Face `transformers`库的官方及全面文档，对实际实现和理解其全部功能至关重要。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2018)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Volume: 1; Pages: 4171-4186; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  介绍了BERT模型，这是一种基础的预训练Transformer，彻底改变了自然语言处理，并展示了预训练和微调范式的强大功能，这正是使用模型库的核心。
- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗 Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) — Lewis, Leandro, Thomas (2022)
  Publisher: O'Reilly Media
  一本实用指南，讲解如何使用Hugging Face `transformers`库应用Transformer模型，涵盖各种任务、微调以及基本概念。
