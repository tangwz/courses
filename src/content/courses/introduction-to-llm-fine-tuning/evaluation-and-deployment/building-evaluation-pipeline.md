---
course: "introduction-to-llm-fine-tuning"
chapter: "evaluation-and-deployment"
lesson: "building-evaluation-pipeline"
sourceId: 7291
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-5-evaluation-and-deployment/building-evaluation-pipeline"
title: "构建评估流程"
description: "如何构建自动化流程，以系统地评估针对保留测试集的模型检查点。"
order: 4
plots: ["plots/7291-0.json"]
sourceHash: "780d82fd11e570e97e4a2a9132ecf62a55a514d681c162e12af0ab7e99af86c3"
sourceCorrections: []
---

虽然单独的指标计算有益，但手动逐个模型地执行它们效率低下且难以扩展。要从临时抽查转变为可重复且可靠的评估过程，您需要构建一个自动化评估流程。此流程将评估步骤规范化为脚本或工作流程，您可以在不同模型检查点、超参数 (parameter) (hyperparameter)设置或微调 (fine-tuning)方法上保持一致地运行。

精心设计的流程是模型系统化改进的根本。它确保每个模型都在相同条件下依据相同标准进行评判，使您的比较公平，结论可靠。

### 评估流程的组成

评估流程是一系列自动化步骤，接收模型和测试数据集作为输入，并输出包含性能指标的全面报告。核心组成部分处理数据加载、推理 (inference)、计算和报告。

> 自动化评估流程的标准工作流程，处理输入以生成结构化的性能报告。

我们来审视这个过程的每个阶段。

### 1. 加载测试数据集

第一步是加载您的保留测试集。此数据集在训练或验证期间模型不得接触过。使用 Hugging Face `datasets` 库，这简单直接。确保数据格式，包括任何提示模板，与模型期望的精确匹配，这一点非常重要。

```python
from datasets import load_dataset

# 从 Hugging Face Hub 或本地文件加载测试分割
test_dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="test_prefs")

# 我们选择一个较小的样本进行快速评估
test_sample = test_dataset.select(range(100))
```

### 2. 生成模型预测

接下来，流程会遍历测试集，将每个输入提示馈送给模型，并收集生成的文本。此步骤通常是流程中最耗时的部分。为了提高效率，您应该批量处理数据。

这是一个简单的推理 (inference)循环示例。它接收模型、分词 (tokenization)器 (tokenizer)和数据集，然后为每个条目生成预测。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# 假设模型和分词器已从检查点加载
# model = AutoModelForCausalLM.from_pretrained(...)
# tokenizer = AutoTokenizer.from_pretrained(...)

predictions = []
references = []

for example in tqdm(test_sample):
    # 根据您的微调任务格式化提示
    prompt = f"### User:\n{example['chosen'][0]['content']}\n\n### Assistant:\n"
    reference_output = example['chosen'][1]['content']

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成输出
    outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 清理输出，仅获取助手的回复
    # 这部分高度依赖于您的数据和提示格式
    clean_prediction = prediction_text.split("### Assistant:\n")[-1]

    predictions.append(clean_prediction)
    references.append(reference_output)
```

### 3. 计算量化 (quantization)指标

有了预测列表及其对应的真实参考，您现在可以计算我们之前讨论的自动化指标。Hugging Face 的 `evaluate` 库简化了此计算。

```python
import evaluate

# 加载所需指标
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# 计算分数
rouge_results = rouge_metric.compute(predictions=predictions, references=references)

# BLEU 要求参考资料为列表的列表
bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])

print("ROUGE Results:", rouge_results)
print("BLEU Results:", bleu_results)
```

此步骤的输出是分数字典。这些结果应以 JSON 或 CSV 等结构化格式保存，并附带有关模型版本和评估日期的元数据。这会创建性能的历史记录。

### 4. 将流程整合到训练生命周期中

评估流程直接整合到您的模型开发生命周期中时效用最大。您可以将其配置为在特定间隔自动运行，例如在每个训练周期后，而不是在训练后运行一次性脚本。

这使您能够随时间追踪模型性能，并找出模型在验证集上表现最佳的精确时间点，从而帮助您避免过拟合 (overfitting)并选择最佳检查点进行最终部署。`Weights & Biases` 或 `MLflow` 等工具非常适合记录这些指标并在仪表板中进行可视化。



![每次评估步骤的 ROUGE-L 分数](plots/7291-0.json)



> 追踪 ROUGE-L 等指标在各个评估步骤中的表现，有助于找出表现最佳的模型检查点。在此示例中，性能在步骤 3 达到峰值。

### 5. 选择样本进行定性检查

最后，自动化指标不能说明全部情况。您的流程也应辅助定性检查过程。您可以利用量化 (quantization)分数找出最值得关注的案例，而不是随意选择样本进行检查。

例如，您可以自动保存 ROUGE 分数*最低*的前五个预测。这些通常代表模型最大的不足，并提供对其弱点的有益信息。您还可以标记 (token)异常短、长或包含重复的输出。将这些困难的样本保存到单独的文件中，使人工进行有针对性且高效的检查变得更加容易。

通过构建此自动化流程，您为模型评估建立了一个严谨、可重复、可扩展的过程。这种工程规范是将随意尝试与高质量、可靠语言模型的专业开发区分开来的原因。

## 参考资料

- [Hugging Face Evaluate Library Documentation](https://huggingface.co/docs/evaluate/index) — Hugging Face (2024)
  Hugging Face `evaluate` 库的官方文档，解释如何计算模型性能的各种指标。
- [Hugging Face Transformers Library: Causal Language Modeling](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM.generate) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face `transformers` 库的官方文档，详细介绍了如何使用因果语言模型执行文本生成。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  一个评估语言模型的综合框架，解决了在不同能力上进行系统化和标准化评估的需求。
- [MLOps: A Guide to Production Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2021)
  Publisher: Google Cloud
  Google Cloud 的这份指南概述了 MLOps 原则和实践，用于构建和管理生产机器学习系统，包括持续评估。
