---
course: "how-to-build-a-large-language-model"
chapter: "extrinsic-evaluation-downstream-tasks"
lesson: "rationale-downstream-task-evaluation"
sourceId: 6060
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-22-extrinsic-evaluation-downstream-tasks/rationale-downstream-task-evaluation"
title: "下游任务评估的理由"
description: "解释为何内在指标无法全面衡量模型的效用。"
order: 1
plots: ["plots/6060-0.json"]
sourceHash: "f427692ae48a9a4d8e21958eb1463f6b5a639021813958887e2edbea95e1d373"
sourceCorrections: []
---

困惑度等内在评估指标能为语言模型在保留文本分布上的流畅性和预测能力提供有价值的理解，但它们通常无法充分说明模型对于特定应用有多么适用。一个模型可能获得很低的困惑度分数，表明它在根据训练数据中的模式预测序列中的下一个token方面表现出色，但当被要求执行准确总结文档或正确回答事实性问题等具体任务时，却可能表现糟糕。统计文本生成能力与实际效用之间的这种差距，使得外部评估成为必需。

外部评估衡量模型在完成特定下游任务时的表现。我们不衡量模型单独预测文本的准确度，而是衡量它在执行以下任务时的表现：

- **文本分类：** 为文本分配类别（例如，情感分析、主题标注）。
- **问答：** 根据给定背景提供问题答案。
- **摘要：** 生成较长文本 (long context)的简洁摘要。
- **机器翻译：** 将文本从一种语言翻译到另一种语言。
- **命名实体识别：** 识别文本中的实体，如名称、日期和地点。

进行下游任务评估的主要原因是为了获得一个**衡量实际关联度的指标**。困惑度是一个抽象的衡量指标；而情感分析任务上的准确率、命名实体识别上的F1分数，或摘要任务的ROUGE分数，则直接量化 (quantization)了模型达到预期结果的能力。这些指标通常更易于理解，并与部署模型的目标直接相关。

考虑两个模型，模型A和模型B，它们都经过大型文本语料库的预训练 (pre-training)。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM # 模型加载的占位符

# 假设model_A和model_B是已加载的预训练LLM
# model_A = AutoModelForCausalLM.from_pretrained("model_a_checkpoint")
# model_B = AutoModelForCausalLM.from_pretrained("model_b_checkpoint")

# --- 计算困惑度 ---
# perplexity_A = calculate_perplexity(model_A, validation_data)
# perplexity_B = calculate_perplexity(model_B, validation_data)
# print(f"Model A Perplexity: {perplexity_A:.2f}") # 输出：模型A困惑度：15.23
# print(f"Model B Perplexity: {perplexity_B:.2f}") # 输出：模型B困惑度：15.89

# --- 在下游任务上评估：情感分析 ---

# 添加分类头（简化示例）
# classification_head = nn.Linear(model_A.config.hidden_size, num_labels)

# 在情感数据上微调model_A和model_B...
# accuracy_A = evaluate_sentiment(model_A_finetuned, sentiment_test_data)
# accuracy_B = evaluate_sentiment(model_B_finetuned, sentiment_test_data)
# print(f"Model A Sentiment Accuracy: {accuracy_A:.4f}") # 输出：模型A情感准确率：0.9150
# print(f"Model B Sentiment Accuracy: {accuracy_B:.4f}") # 输出：模型B情感准确率：0.8520
```

在这个场景下，模型A和模型B的困惑度分数非常接近，这表明它们在一般意义上具有相似的语言建模能力。然而，当在特定情感分析任务上进行微调 (fine-tuning)和评估时，模型A的表现显著优于模型B。这种下游表现的差异可能源于预训练期间所获取知识的微小差异、模型在微调期间的适应能力，或其对情感表达相关语言的理解。仅仅依靠困惑度会掩盖这种实际能力上的重要区别。



![内在与外部指标比较](plots/6060-0.json)



> 比较显示两个模型困惑度相似但在下游任务准确率上有所不同。

此外，外部评估有助于**发现具体的优点和不足**。一个模型可能擅长需要广泛知识的任务（例如开放域问答），但在需要逻辑推理 (inference)或创意生成的任务上可能表现不佳。对多种下游任务进行评估，能描绘出模型能力更全面的图景，这比任何单一的内在指标所能提供的更为全面。

最后，在GLUE（通用语言理解评估）或SuperGLUE等标准化下游基准上进行评估，使得不同模型和研究成果之间能够进行**客观比较**。这些基准提供精心组织的数据集和既定指标，适用于一系列任务，它们作为衡量该方面进展的共同标准。

总之，尽管内在指标对于监测训练过程和评估一般语言建模能力很有用，但在下游任务上的外部评估对于理解和量化模型的实际效用，有意义地比较不同模型，以及指导发展以构建更具能力和更有用的LLM是必不可少的。本章的后续部分将详细说明用于此目的的具体任务、基准和方法。

## 参考资料

- [GLUE: A Multi-Task Benchmark for Natural Language Understanding](https://arxiv.org/abs/1804.07461) — Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman (2018)
  Journal: ICLR 2019; DOI: [10.48550/arXiv.1804.07461](https://doi.org/10.48550/arXiv.1804.07461)
  介绍了一种广泛采用的基准测试，用于评估跨各种下游任务的自然语言理解模型。
- [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/abs/1905.00537) — Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman (2019)
  Journal: NeurIPS 2019; DOI: [10.48550/arXiv.1905.00537](https://doi.org/10.48550/arXiv.1905.00537)
  提出了一个更具挑战性的自然语言理解模型基准，通过更难的任务扩展了GLUE的概念。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2018)
  Journal: NAACL HLT; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  介绍了BERT模型，展示了预训练后在各种下游自然语言处理任务上进行微调的有效性。
