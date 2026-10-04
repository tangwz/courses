---
course: "introduction-to-llm-fine-tuning"
chapter: "evaluation-and-deployment"
lesson: "practice-evaluating-fine-tuned-model"
sourceId: 7294
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-5-evaluation-and-deployment/practice-evaluating-fine-tuned-model"
title: "实践：评估微调模型"
description: "一次实操练习，将定量和定性评估方法应用于您已微调的模型。"
order: 7
plots: ["plots/7294-0.json"]
sourceHash: "49fef5cff3d6f6e4891997d2545ebe4d74ab45610d47b518cbe5e377d1a9a6dc"
sourceCorrections: []
---

您已成功训练了一个模型，但在严格衡量其表现之前，此过程尚未完成。本次实践将指导您完成评估我们之前练习中微调 (fine-tuning)模型的主要步骤。我们将结合自动化定量指标与人工定性审查，以全面认识模型的长处和局限。

### 评估准备

在开始之前，请确保您的环境已备妥。我们将需要微调 (fine-tuning)模型适配器、原始基础模型，以及最重要的是一个独立测试数据集。机器学习 (machine learning)的一项基本原则是，模型必须在训练或验证期间从未见过的数据上进行评估，以获得对其泛化能力的公正评估。

模型评估的步骤包括：

1. 加载基础模型并合并已训练的LoRA适配器。
2. 为测试集中的每个条目生成输出。
3. 计算定量指标（ROUGE、BLEU、困惑度）。
4. 对部分输出进行定性审查。

> 评估过程，从数据输入到最终评估。

首先，让我们安装Hugging Face的`evaluate`库，它提供了对常用指标的便捷使用。

```bash
pip install evaluate rouge_score
```

### 加载微调 (fine-tuning)模型和数据

我们将从加载基础模型并应用我们已训练的LoRA适配器开始。在本次练习中，假设您的LoRA适配器配置和权重 (weight)保存在一个名为`./my-lora-adapter`的目录中。我们还将从训练中使用的数据集中加载测试部分。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# 定义模型和适配器路径
base_model_id = "meta-llama/Llama-2-7b-hf"
adapter_path = "./my-lora-adapter"

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 加载LoRA模型
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 加载测试数据集
test_dataset = load_dataset("samsum", split="test")
```

> **注意：** 对于推理 (inference)，通常使用`model.merge_and_unload()`将LoRA权重直接合并到基础模型中。这会创建一个标准模型产物，从而简化部署，因为您在推理时不再需要PEFT库。我们在此将它们分开，是为了说明其组成部分。

### 生成预测

模型和数据准备就绪后，下一步是为测试数据集中的每个样本生成预测。我们将遍历数据集，像训练时一样格式化输入提示，对其进行分词 (tokenization)，并将其传递给模型的`generate`方法。我们将同时存储模型的输出（预测）和真实摘要（参考）。

```python
import pandas as pd
from tqdm import tqdm

# 为了本例，我们为部分样本生成预测
test_samples = test_dataset.select(range(100))

predictions = []
references = []

for sample in tqdm(test_samples):
    prompt = f"""
    Summarize the following conversation.

    {sample["dialogue"]}

    Summary:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # 生成输出
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=50, 
            do_sample=True, 
            top_p=0.9, 
            temperature=0.1
        )

    # 解码输出和参考
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 清理输出，仅获取摘要
    summary = prediction.split("Summary:")[1].strip()

    predictions.append(summary)
    references.append(sample["summary"])
```

### 定量评估

既然我们有了`predictions`和`references`，就可以计算自动化指标了。

#### ROUGE和BLEU

ROUGE（面向召回率的摘要评估辅助工具）是一组用于评估自动摘要和机器翻译的指标。它通过将自动生成的摘要或译文与一组参考摘要（通常是人工生成的）进行比较来运作。例如，ROUGE-L衡量的是最长公共子序列。

BLEU（双语评估辅助工具）是另一个常用指标，主要用于翻译，它通过比较候选文本与参考文本的n-gram来衡量精度。

```python
import evaluate

# 加载ROUGE指标
rouge_metric = evaluate.load("rouge")

# 计算得分
rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
print(rouge_scores)
```

输出将是一个包含不同ROUGE得分的字典，例如`rouge1`、`rouge2`和`rougeL`。这些数值提供了一种标准化方法来比较模型表现。通常，更高的得分表示与参考文本的对齐 (alignment)程度更好。



![ROUGE评估得分](plots/7294-0.json)



> 某次评估运行的ROUGE得分。ROUGE-1衡量的是一元词重叠，ROUGE-2衡量的是二元词重叠，ROUGE-L衡量的是最长公共子序列。

#### 困惑度

困惑度衡量概率分布或概率模型预测样本的优劣。在语言模型中，它可以被理解为模型在遇到测试集时的“惊讶”程度。它由交叉熵损失计算得出，通常表示为 $e^{loss}$。较低的困惑度得分表示模型对其预测更有信心，这是一个理想的特性。

计算困惑度要求我们在不更新任何梯度的情况下，计算测试集上的损失。

```python
# 一个计算困惑度的简化函数
def calculate_perplexity(model, tokenizer, data, device="cuda"):
    total_loss = 0
    total_tokens = 0

    for sample in tqdm(data):
        text = sample["dialogue"] + " " + sample["summary"]
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        input_ids = inputs.input_ids.to(device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss

        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()

# 计算测试样本的困惑度
perplexity_score = calculate_perplexity(model, tokenizer, test_samples)
print(f"困惑度: {perplexity_score:.2f}")
```

较低的困惑度，例如得分`15.43`，表明模型已很好地掌握了目标领域的模式。

### 定性评估

指标提供了高层面的概括，但它们并非包罗万象。模型可能获得良好的ROUGE得分，但生成的摘要可能存在事实错误或风格生硬。此时，定性分析就显得有必要了。

通过人工检查模型的输出，您可以检查：

- **连贯性：** 输出是否合乎逻辑？
- **相关性：** 摘要是否切题并抓住了主要内容？
- **事实准确性：** 模型是否编造细节（幻觉 (hallucination)）？
- **简洁性：** 摘要是否简洁明了？

让我们并排查看几个例子。

| 输入对话 | 参考摘要 | 模型生成的摘要 | 分析 |
| --- | --- | --- | --- |
| `Amanda: Can you pick up dinner tonight? <br> Ben: Sure, what do you want? <br> Amanda: Pizza would be great. <br> Ben: Ok, I'll grab it on my way home.` | Amanda让Ben回家路上取披萨当晚餐。 | Ben下班回家路上会去取披萨当晚餐。 | **好。** 模型正确识别了主要内容，并生成了流畅、准确的摘要。 |
| `Chloe: The project deadline is Friday. <br> David: I know, I'm almost done with the report. I just need to add the final charts. <br> Chloe: Great, send it over when you're ready.` | David正在完成一份报告，其中包含图表，项目截止日期是周五。 | David需要完成他的报告。截止日期是周五。 | **一般。** 摘要是正确的，但遗漏了关于图表的具体细节。它不如参考摘要信息量大。 |
| `Eve: I'm thinking of booking a flight to Paris. <br> Frank: Nice! When are you going? <br> Eve: Maybe in May. It's lovely there in the spring.` | Eve正在计划去巴黎旅行，可能在五月。 | Eve正在预订下周去巴黎度假的机票，并将入住大酒店。 | **差（幻觉）。** 模型编造了对话中没有的细节（度假、下周、大酒店）。 |

这种人工审查是评估循环中不可或缺的一部分。它提供了定量得分无法提供的见解，帮助您找出可以通过整理更好的训练数据或调整微调 (fine-tuning)过程来解决的具体失败模式。通过结合定量和定性方法，您将获得对模型真实表现的全面理解。

## 参考资料

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out; Publisher: Association for Computational Linguistics; Pages: 74-81
  介绍了ROUGE指标，广泛用于评估自动摘要。
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311–318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  提出了BLEU评分，是评估机器翻译质量的标准指标。
- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  涵盖自然语言处理概念的教材，包括第3章中困惑度的定义和计算。
