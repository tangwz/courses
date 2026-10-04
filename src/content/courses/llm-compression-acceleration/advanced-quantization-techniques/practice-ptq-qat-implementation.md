---
course: "llm-compression-acceleration"
chapter: "advanced-quantization-techniques"
lesson: "practice-ptq-qat-implementation"
sourceId: 4334
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-2-advanced-quantization-techniques/practice-ptq-qat-implementation"
title: "实践操作：PTQ 和 QAT 的实现"
description: "使用标准库将 PTQ 和 QAT 技术应用于大型 Transformer 模型。"
order: 8
plots: ["plots/4334-0.json"]
sourceHash: "efb7814828bc9747d4670bb8bdb1a1085e56b4333ff3320c6480d21dcc97aa83"
sourceCorrections: []
---

提供了实现两种主要量化 (quantization)技术（训练后量化 (PTQ) 和量化感知训练 (QAT)）针对大型语言模型的指导。这涵盖了典型工作流程、使用常用库的代码实现模式以及评估所得模型。目标是让您掌握有效运用这些强大优化方法所需的实践技能。

我们假设您熟悉 Python 和 PyTorch 等深度学习 (deep learning)框架，以及 Hugging Face Transformers 库。在此次实践练习中，我们将概述使用 PyTorch 内置量化工具的步骤，以及适用于 Hugging Face Optimum 或 Intel Neural Compressor 等库的设计思路，这些库通常提供更高级别的 API。

### 环境设置

开始前，请确保您的环境包含必要的库：

- `torch` 和 `torchvision` (PyTorch 核心与实用工具)
- `transformers` (用于加载预训练 (pre-training)大型语言模型和分词 (tokenization)器 (tokenizer))
- `datasets` (用于处理校准和评估数据)
- 可选：`optimum` 或其他专用量化 (quantization)库，以简化工作流程。

我们建议在这些示例中使用相对较小的预训练 Transformer 模型（如 `distilgpt2`、`bert-base-uncased`，或较大模型中可管理的片段），以保持合理的计算时间。

```python
# 示例环境设置
# pip install torch torchvision torchaudio
# pip install transformers datasets evaluate accelerate bitsandbytes # 根据需要添加其他库
```

### 实现训练后量化 (quantization) (PTQ)

PTQ 吸引人之处在于它不需要重新训练模型。它涉及在小型、具代表性的数据集上校准模型，以确定权重 (weight)和激活的最佳量化参数 (parameter)（缩放因子和零点）。

#### 1. 模型与分词 (tokenization)器 (tokenizer)加载

首先，加载您的目标预训练 (pre-training)模型及其对应的分词器。确保模型设置为评估模式。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "distilgpt2" # 示例模型
model_fp32 = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_fp32.eval() # 设置为评估模式
```

#### 2. 校准数据集准备

选择一个能够反映模型在推理 (inference)期间将遇到的数据类型的数据集。通常几百个样本就足够了。使用模型的分词器对这些数据进行预处理。

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# 加载小型校准数据集（例如，来自 wikitext 的 100 个示例）
calibration_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")

def preprocess_function(examples):
    # 根据您的模型/任务需要调整 max_length
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_calibration_data = calibration_data.map(preprocess_function, batched=True)
tokenized_calibration_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 使用 DataLoader 进行批处理
calibration_dataloader = DataLoader(tokenized_calibration_data, batch_size=8)
```

#### 3. PTQ 配置与校准

配置量化设置。PyTorch 的 `torch.quantization` 模块提供默认设置（如 `get_default_qconfig`）或允许精细控制（例如，权重的对称逐通道量化，激活的非对称逐张量量化）。

通过插入观察器 (`torch.quantization.prepare_ptq`) 来准备模型。这些观察器在校准期间收集有关激活范围的统计信息。

```python
# 配置 PTQ - 使用常见后端，如 'x86' 或 'qnnpack'
# 'fbgemm' 或 'qnnpack' 常用于服务器 (x86) 或移动设备 (ARM)
qconfig = torch.quantization.get_default_qconfig('fbgemm') # 适用于 x86

# 准备模型进行 PTQ：插入观察器
# 注意：`prepare_ptq` 正在演变；请查阅 PyTorch 文档以获取最新 API。
# 一些库可能使用不同的函数或上下文管理器。
model_to_quantize = copy.deepcopy(model_fp32) # 在副本上操作
model_to_quantize.qconfig = qconfig
torch.quantization.prepare(model_to_quantize, inplace=True) # 使用旧 API 模式的示例

# 通过运行数据进行模型校准
print("正在运行校准...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_to_quantize.to(device)
with torch.no_grad():
    for batch in calibration_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        try:
            model_to_quantize(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"警告：批处理在校准期间失败：{e}") # 处理潜在问题
            # 根据模型/库的不同，可能需要特定的输入格式
print("校准完成。")
```

*注意：* 校准期间正确处理模型输入和输出很重要。某些模型架构或量化后端可能具有特定要求。PyTorch 中的 `prepare_ptq` API 也在发生变化，因此请查阅您的特定版本的文档。Hugging Face Optimum 等库通常会大幅简化此过程。

#### 4. 转换为量化模型

校准后，使用 `torch.quantization.convert` 将模型转换为其量化等效物。这将用其量化对应物替换被观察的模块。

```python
model_to_quantize.cpu() # 转换通常在 CPU 上进行
model_ptq = torch.quantization.convert(model_to_quantize, inplace=False)
print("PTQ 转换完成。")
```

#### 5. 评估

评估 PTQ 模型与原始 FP32 模型。重要衡量指标包括：

- **模型大小：** 比较保存的状态字典的文件大小。
- **准确度/保真度：** 衡量相关下游任务（例如，语言模型的困惑度，分类的准确度）的性能。
- **推理速度：** 在目标硬件上基准测试延迟和吞吐量 (throughput)。

```python
import os
import time
from evaluate import load # Hugging Face Evaluate 库

# 1. 模型大小
torch.save(model_fp32.state_dict(), "distilgpt2_fp32.pth")
torch.save(model_ptq.state_dict(), "distilgpt2_ptq_int8.pth")
fp32_size = os.path.getsize("distilgpt2_fp32.pth") / 1e6
ptq_size = os.path.getsize("distilgpt2_ptq_int8.pth") / 1e6
print(f"FP32 模型大小: {fp32_size:.2f} MB")
print(f"PTQ INT8 模型大小: {ptq_size:.2f} MB")
print(f"大小缩减: {(1 - ptq_size / fp32_size) * 100:.2f}%")

# 2. 准确度 (示例：测试集上的困惑度)
perplexity = load("perplexity", module_type="metric")
test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:50]") # 小型测试集
test_encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fp32.to(device)
model_ptq.to(device) # 确保模型在正确的设备上进行评估

with torch.no_grad():
    results_fp32 = perplexity.compute(model_id=model_id, # 由 HF evaluate 内部使用
                                     model=model_fp32,
                                     tokenizer=tokenizer,
                                     predictions=test_encodings.input_ids.to(device),
                                     batch_size=4) # 调整批次大小
    results_ptq = perplexity.compute(model_id=model_id, # 提供必要的参数
                                    model=model_ptq,
                                    tokenizer=tokenizer,
                                    predictions=test_encodings.input_ids.to(device),
                                    batch_size=4)

print(f"FP32 困惑度: {results_fp32['mean_perplexity']:.4f}")
print(f"PTQ INT8 困惑度: {results_ptq['mean_perplexity']:.4f}")

# 3. 推理速度 (简单延迟示例)
dummy_input = tokenizer("This is a sample text for benchmarking.", return_tensors="pt").input_ids.to(device)
repetitions = 100

# 预热运行
for _ in range(10):
    _ = model_fp32(dummy_input)
    _ = model_ptq(dummy_input)

# 计时运行
start_time = time.time()
for _ in range(repetitions):
    _ = model_fp32(dummy_input)
fp32_latency = (time.time() - start_time) / repetitions * 1000 # 毫秒

start_time = time.time()
for _ in range(repetitions):
    _ = model_ptq(dummy_input)
ptq_latency = (time.time() - start_time) / repetitions * 1000 # 毫秒

print(f"FP32 平均延迟: {fp32_latency:.2f} 毫秒")
print(f"PTQ INT8 平均延迟: {ptq_latency:.2f} 毫秒")
print(f"加速比: {fp32_latency / ptq_latency:.2f}x")
```

PTQ 通常会带来大幅的模型大小减少和加速，尤其是在支持原生 INT8 的硬件上，但通常会伴随准确度的小幅下降。

### 实现量化 (quantization)感知训练 (QAT)

QAT 在微调 (fine-tuning)阶段模拟量化效果，使模型能够调整其权重 (weight)以适应降低的精度。这通常可以恢复 PTQ 期间损失的部分或全部准确度，代价是需要额外的训练计算资源。

#### 1. QAT 模型准备

加载预训练 (pre-training)模型。您将从训练模式开始，而不仅仅是评估模式。使用 QAT 特定的配置 (`get_default_qat_qconfig`)，并使用 `torch.quantization.prepare_qat` 准备模型。这会插入“伪量化”模块，模拟正向和反向传播 (backpropagation)期间的量化效果。

```python
# 再次加载基础 FP32 模型
model_for_qat = AutoModelForCausalLM.from_pretrained(model_id)

# 配置 QAT
# 注意：后端选择（'fbgemm', 'qnnpack'）影响支持的操作和性能。
qat_qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# 准备 QAT 模型：插入伪量化模块
model_for_qat.train() # 设置为训练模式
model_for_qat.qconfig = qat_qconfig

# 重要：层合并可以提升 QAT 的准确度和性能
# 示例：在适用情况下将 Conv-BN-ReLU、Linear-ReLU 等模块进行组合
# fuse_list = torch.quantization.fuse_modules() # 定义适用的待合并层
# model_for_qat = torch.quantization.fuse_modules(model_for_qat, fuse_list) # 应用合并

torch.quantization.prepare_qat(model_for_qat, inplace=True)
print("模型已为 QAT 准备就绪。")
```

*注意：* 对于 Transformer 模型，识别可合并的层（如线性层 -> 激活层）可能有所裨益，但这需要仔细检查模型架构。一些库可以自动化部分此过程。

#### 2. 微调阶段

在相关数据集（例如，下游任务数据集或通用文本语料库）上对准备好的模型进行少量轮次的微调。使用标准的训练循环，但请注意，与标准 FP32 微调相比，QAT 通常需要较低的学习率和仔细的超参数 (parameter) (hyperparameter)调优。梯度将流经伪量化操作，使模型能够进行调整。

```python
from torch.optim import AdamW
from transformers import get_scheduler

# 准备训练数据集（例如，针对特定任务进行微调或继续语言模型训练）
# 为求真实性，使用与校准不同的数据集或分割
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[100:1100]") # 示例：1000 个样本
tokenized_train_data = train_dataset.map(preprocess_function, batched=True)
tokenized_train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) # 如果任务需要，添加标签

train_dataloader = DataLoader(tokenized_train_data, batch_size=4, shuffle=True) # 较小的批次大小通常有帮助

# 优化器和调度器
optimizer = AdamW(model_for_qat.parameters(), lr=1e-5) # QAT 通常需要较低的学习率
num_training_steps = len(train_dataloader) # 根据需要调整 epochs（例如：1-3）
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 微调循环
print("开始 QAT 微调...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_for_qat.to(device)
model_for_qat.train()

for epoch in range(1): # 示例：1 个 epoch
    for batch in train_dataloader:
        optimizer.zero_grad()
        # 将批次数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # 假设是因果语言模型，标签通常是 input_ids 的偏移
        labels = input_ids.clone()

        outputs = model_for_qat(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward() # 梯度流经伪量化节点

        optimizer.step()
        lr_scheduler.step()

    print(f"QAT 第 {epoch+1} 轮完成。上次损失: {loss.item():.4f}")

print("QAT 微调完成。")
```

#### 3. 转换为量化模型

微调后，将 QAT 模型转换为真正的量化整数模型，类似于 PTQ 过程。模型在转换前必须处于评估模式。

```python
model_for_qat.cpu() # 转换通常在 CPU 上进行
model_for_qat.eval() # 在最终转换前设置为评估模式
model_qat = torch.quantization.convert(model_for_qat, inplace=False)
print("QAT 转换完成。")
```

#### 4. 评估

使用与 PTQ 模型相同的衡量指标（大小、准确度、速度）评估 QAT 模型。将结果与 FP32 基线模型和 PTQ 模型进行比较。

```python
# 1. 模型大小
torch.save(model_qat.state_dict(), "distilgpt2_qat_int8.pth")
qat_size = os.path.getsize("distilgpt2_qat_int8.pth") / 1e6
print(f"QAT INT8 模型大小: {qat_size:.2f} MB") # 应与 PTQ 模型大小相似

# 2. 准确度 (困惑度)
model_qat.to(device)
with torch.no_grad():
    results_qat = perplexity.compute(model_id=model_id,
                                    model=model_qat,
                                    tokenizer=tokenizer,
                                    predictions=test_encodings.input_ids.to(device),
                                    batch_size=4)

print(f"FP32 困惑度: {results_fp32['mean_perplexity']:.4f}") # 来自 PTQ 评估
print(f"PTQ INT8 困惑度: {results_ptq['mean_perplexity']:.4f}") # 来自 PTQ 评估
print(f"QAT INT8 困惑度: {results_qat['mean_perplexity']:.4f}")

# 3. 推理速度 (延迟)
# 预热
for _ in range(10):
    _ = model_qat(dummy_input)

start_time = time.time()
for _ in range(repetitions):
    _ = model_qat(dummy_input)
qat_latency = (time.time() - start_time) / repetitions * 1000

print(f"FP32 平均延迟: {fp32_latency:.2f} 毫秒") # 来自 PTQ 评估
print(f"PTQ INT8 平均延迟: {ptq_latency:.2f} 毫秒") # 来自 PTQ 评估
print(f"QAT INT8 平均延迟: {qat_latency:.2f} 毫秒")
print(f"QAT 相较 FP32 的加速比: {fp32_latency / qat_latency:.2f}x")
```

您通常会观察到 QAT 比 PTQ 获得更好的准确度，可能大幅缩小与原始 FP32 模型之间的差距，同时提供相似的模型大小减小和加速收益。

### PTQ 和 QAT 结果比较

我们根据上述结果可视化潜在的权衡。



![量化权衡：困惑度与模型大小](plots/4334-0.json)



> DistilGPT2 (FP32) 及其 PTQ 和 QAT INT8 量化 (quantization)版本在模型大小和困惑度分数上的比较。较低的困惑度表示更好的语言模型性能。PTQ 显著减小了模型大小，但略微增加了困惑度。QAT 在保持大小减小的同时，恢复了 PTQ 损失的部分性能。

### 高级考量与后续步骤

本次实践演示了使用 INT8 精度进行 PTQ 和 QAT 的核心工作流程。在实际应用中，您可以通过以下方式进行扩展：

- **尝试不同精度：** 将相同的原则应用于 INT4、NF4 或 FP4 等更低精度，可能需要使用 `bitsandbytes` 等库或专用硬件后端。
- **混合精度：** 根据敏感性分析，实施模型不同部分使用不同精度的策略。
- **硬件专用核：** 使用优化过的核（例如，通过 TensorRT、ONNX Runtime、vLLM）为特定 GPU 或加速器上的量化 (quantization)操作提供最大速度提升。此内容将在第 6 章中更详细地介绍。
- **与其他技术结合：** 将量化与剪枝或 PEFT 方法（如 QLoRA 中的 LoRA）结合，以进行进一步优化，这将在后续章节中讨论。
- **评估：** 对多个数据集和任务进行全面评估，以完全了解量化对模型能力的影响，包括公平性与检查（第 7 章）。

本次实践练习为应用高级量化打下了基础。掌握 PTQ 和 QAT 使您能够大幅降低大型语言模型的资源需求，从而在更广泛的应用和硬件平台部署时更具可行性。请记住，最佳方法（PTQ vs. QAT）和配置通常取决于具体的模型、任务、可接受的准确度权衡以及用于微调 (fine-tuning)的可用计算资源。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.1712.05877](https://doi.org/10.48550/arXiv.1712.05877)
  本文介绍了训练后量化（PTQ）和量化感知训练（QAT）的开创性工作，详细阐述了在不显著损失准确性的情况下，将浮点神经网络转换为高效纯整数网络的方法。这对于理解 PyTorch 量化实现背后的原理至关重要。
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch 原生量化工具包的官方文档。它提供了实现 PTQ 和 QAT 的详细指南和 API 参考，与本节的代码示例直接相关。
- [Hugging Face Optimum Documentation](https://huggingface.co/docs/optimum/index) — Hugging Face Team (2024)
  Publisher: Hugging Face
  Hugging Face Optimum 的官方文档，该库扩展了 Transformers 库，提供了包括量化在内的优化技术。它展示了如何使用高级 API 对大型语言模型应用 PTQ 和 QAT，为 LLM 的实际部署提供了 PyTorch 原生 API 之外的选项。
