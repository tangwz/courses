---
course: "practical-llm-quantization"
chapter: "post-training-quantization-ptq"
lesson: "practice-static-ptq"
sourceId: 4622
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq/practice-static-ptq"
title: "动手实践：应用静态PTQ"
description: "一次指引完成将静态PTQ应用到预训练LLM的实践环节。"
order: 8
plots: ["plots/4622-0.json"]
sourceHash: "97c3430d9f7b23ec215c0c340781a1a3e6617638c5bc7258f20da876c88f5f15"
sourceCorrections: []
---

将静态训练后量化 (quantization)（PTQ）应用于预训练 (pre-training)的Transformer模型，需要使用常用库并遵循特定步骤。此过程的目标是将模型的权重 (weight)和可能的激活计算转换为使用低精度整数（如INT8），通过校准数据集来确定最佳量化参数 (parameter)。请注意，此过程在模型完全训练*之后*进行。

我们将主要使用Hugging Face生态系统，具体包括 `transformers` 库来加载模型和分词 (tokenization)器 (tokenizer)，`datasets` 库来获取校准数据，以及 `optimum` 库，它提供模型优化的工具，包括量化，通常会借助ONNX Runtime等后端。

### 设置您的环境

首先，确保您已安装所需的库。您需要 `transformers`、`datasets`、`optimum`，以及 `optimum` 支持的量化 (quantization)后端，例如 `onnxruntime`。您可能还需要 `accelerate` 来更顺畅地处理模型。

```bash
pip install transformers datasets optimum[onnxruntime] accelerate torch
```

*注意：请确保也安装了PyTorch，因为 `transformers` 和 `optimum` 经常需要它。*

### 加载预训练 (pre-training)模型和分词 (tokenization)器 (tokenizer)

我们首先加载一个标准预训练模型。为了方便此示例，我们使用一个较小、广为人知的模型，如 `distilbert-base-uncased`，以保持过程易于管理。

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "distilbert-base-uncased"
# 加载适合特定任务的模型，例如序列分类
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("原始模型已加载：")
print(model)
```

这会加载模型的标准FP32（32位浮点）版本。我们的目标是将其转换为静态INT8表示。

### 准备校准数据集

静态PTQ依赖于一个小型且有代表性的数据集，即校准数据集。该数据用于在推理 (inference)期间观察模型内部激活值的分布。这些观察到的范围（最小值和最大值）对于计算将FP32值映射到INT8所需的缩放因子和零点非常重要。

让我们加载数据集的一个小部分，例如“sst2”（斯坦福情感树库）数据集，它常用于分类任务。我们只需要几百个样本进行校准。

```python
from datasets import load_dataset

# 加载适合模型任务的数据集（例如，情感分析）
calibration_dataset_name = "sst2"
num_calibration_samples = 200 # 少量样本通常足够

# 加载并选择子集
full_calibration_dataset = load_dataset(calibration_dataset_name, split="train")
calibration_indices = list(range(num_calibration_samples))
calibration_subset = full_calibration_dataset.select(calibration_indices)

# 预处理函数
def preprocess_function(examples):
    # 如果文本列名不同，请调整为实际的列名 'sentence'
    return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

# 应用预处理
processed_calibration_dataset = calibration_subset.map(preprocess_function, batched=True)

print(f"已准备好包含 {len(processed_calibration_dataset)} 个样本的校准数据集。")
# 显示一个已处理样本的键
print("已处理样本的键：", processed_calibration_dataset[0].keys())
```

关键在于，此数据集应反映模型在实际推理时将遇到的数据类型。我们使用模型的tokenizer对其进行预处理，确保输入与模型的预期相符。

### 使用Optimum配置静态量化 (quantization)

现在，我们使用 `optimum` 来定义我们的量化配置。我们指定需要静态INT8量化。`Optimum` 在底层使用外部工具包（例如本例中的ONNX Runtime的量化工具）。

```python
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig
from optimum.onnxruntime.configuration import Arm64QuantizationConfig, AutoCalibrationConfig

# 1. 定义量化策略：静态INT8
qconfig = AutoQuantizationConfig.int8_static(
    config=model.config, # 提供模型配置
    dataset=processed_calibration_dataset # 之前准备好的校准数据集
)

# 对于ONNX Runtime，我们通常先导出到ONNX
onnx_model_path = "distilbert_base_uncased_onnx"
quantized_model_path = "distilbert_base_uncased_quantized_onnx"

# 2. 使用模型ID（或路径）创建量化器
quantizer = ORTQuantizer.from_pretrained(model_id, feature=model.config.task_specific_params.get("feature", "sequence-classification"))

print("量化配置已创建：")
print(qconfig)
```

在这里，`AutoQuantizationConfig.int8_static` 方便地设置了静态INT8量化的配置。我们提供了模型的配置和我们已处理的校准数据集。`ORTQuantizer` 使用模型标识符进行初始化。

### 应用静态PTQ过程

模型加载完毕，校准数据准备就绪，配置也已定义，我们现在可以执行量化 (quantization)。`quantizer.quantize` 方法处理将模型导出为中间格式（如ONNX）、运行校准、计算量化参数 (parameter)，并生成最终的量化模型。

```python
# 3. 应用量化
quantizer.quantize(
    save_dir=quantized_model_path,
    quantization_config=qconfig,
    # 可选：指定中间ONNX模型的路径
    # onnx_model_path=onnx_model_path
)

print(f"静态PTQ完成。量化模型已保存到：{quantized_model_path}")
```

此过程可能需要几分钟，具体取决于模型大小和校准样本数量。它包括：

1. 将原始FP32模型导出为ONNX格式。
2. 使用ONNX模型在校准数据集上运行推理 (inference)，以收集激活统计信息（最小值/最大值范围）。
3. 使用这些统计信息来计算权重 (weight)和激活的INT8缩放因子和零点。
4. 将模型权重转换为INT8，并修改计算图，在必要时加入量化和反量化节点。
5. 保存最终的量化ONNX模型。

### 验证量化 (quantization)模型（大小比较）

量化的一个主要好处是模型大小的减小。让我们比较原始PyTorch模型的状态字典大小与量化后的ONNX模型文件大小。

```python
import os

# 获取目录大小的函数
def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total / (1024 * 1024) # 大小，单位MB

# 原始PyTorch模型的近似大小（可能略有不同）
original_model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

# 量化ONNX模型目录的大小
quantized_model_size_mb = get_dir_size(quantized_model_path)

print(f"原始FP32模型近似大小：{original_model_size_mb:.2f} MB")
print(f"量化INT8模型大小（ONNX）：{quantized_model_size_mb:.2f} MB")

# 计算缩减量
reduction = (1 - (quantized_model_size_mb / original_model_size_mb)) * 100
print(f"大小缩减：{reduction:.2f}%")

# 可选：可视化大小比较
```



![模型大小比较（近似值）](plots/4622-0.json)



> 原始FP32模型与静态量化INT8模型（ONNX格式）之间的近似大小比较。大小为指示性数值，可能因具体模型和保存格式而异。

您应该会观察到模型大小显著减小，从FP32到INT8通常接近4倍，因为INT8每个参数 (parameter)只使用四分之一的比特。

### 加载并运行量化 (quantization)模型（可选）

尽管详细评估将在第六章介绍，但您可以使用 `optimum` 和ONNX Runtime加载量化模型，进行快速推理 (inference)检查。

```python
# 需要安装onnxruntime
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline

# 加载量化模型
quantized_ort_model = ORTModelForSequenceClassification.from_pretrained(quantized_model_path)

# 创建一个方便推理的管道
classifier = pipeline("sentiment-analysis", model=quantized_ort_model, tokenizer=tokenizer)

# 测试推理
text = "This movie was quite good, I enjoyed it."
result = classifier(text)
print("量化模型的推理结果：", result)

text_neg = "The plot was predictable and the acting was mediocre."
result_neg = classifier(text_neg)
print("量化模型的推理结果：", result_neg)
```

这确认量化模型可以加载并生成输出。请记住，静态PTQ有时会导致精度下降，这需要针对您的具体用例进行仔细评估。

### 总结

在本实践部分中，我们使用 `optimum` 库对 `transformers` 模型应用了静态训练后量化 (quantization)。我们加载了一个预训练 (pre-training)模型，准备了校准数据集，配置了静态INT8转换的量化过程，使用ONNX Runtime作为后端执行了量化，并观察了由此产生的模型大小减小。这个实用的流程展示了PTQ如何有效应用，无需访问原始训练流程或昂贵的再训练，从而使模型更小并可能加快推理 (inference)速度。接下来的章节将介绍旨在减轻潜在精度损失的更高级PTQ技术，并介绍量化感知训练（QAT）。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.1712.05877](https://doi.org/10.48550/arXiv.1712.05877)
  一篇基础论文，介绍了静态后训练量化及其在神经网络中仅整数推理的应用。
- [Quantization in Optimum](https://huggingface.co/docs/optimum/en/concept_guides/quantization) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Optimum库量化功能的官方文档，涵盖了实践中展示的工具和工作流程。
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — Microsoft (2024)
  Publisher: Microsoft
  ONNX Runtime支持的量化技术的官方指南，它在实践示例中作为Optimum的后端。
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://ieeexplore.ieee.org/document/9404286) — Haoliang Xu, Hao Liu, Zhenyu Liu, Mengyao Li, Jing Han, Yiping Deng, Jianqiang Li, Xiangyang Xue, Jianping Fan (2021)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 3394-3407; DOI: [10.1109/TNNLS.2021.3069173](https://doi.org/10.1109/TNNLS.2021.3069173)
  一篇综合性综述，回顾了包括后训练量化在内的各种量化方法，提供了更广泛的学术背景。
