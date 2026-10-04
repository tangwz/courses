---
course: "quantized-llm-deployment"
chapter: "addressing-advanced-challenges"
lesson: "practice-finetuning-quantization-parameters"
sourceId: 4679
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-5-addressing-advanced-challenges/practice-finetuning-quantization-parameters"
title: "实践：微调量化参数"
description: "实践环节，内容为调整量化参数（例如，校准数据集大小、量化方案），以平衡准确性和性能。"
order: 8
plots: ["plots/4679-0.json", "plots/4679-1.json"]
sourceHash: "d4b72914e2d7ef24ce5df1a371cee3c4a2f61d13344143652742f56927af9e56"
sourceCorrections: []
---

GPTQ 和 AWQ 等精密的量化 (quantization)技术，以及评估所得模型性能和准确性的方法，是优化大型语言模型 (LLM) 的重要策略。然而，应用这些技术通常不只是运行一个使用默认设置的脚本那么简单。量化库提供的默认参数 (parameter)通常是一个合理的起点，但要为您的特定应用获得模型大小、推理 (inference)速度和准确性之间的最佳平衡，往往需要仔细调整。本实践指南将引导您调整常用的量化参数，以有效应对这些权衡，尤其是在处理潜在的准确性下降问题时。

可以把量化参数看作是您可以调节的旋钮，以影响最终模型。调节一个旋钮可能会提高推理速度，但会略微降低准确性，而另一个旋钮可能会恢复准确性，但代价是模型略大或量化过程变慢。目标是为您的应用程序需求找到最佳平衡点。

我们将侧重于训练后量化 (PTQ) 方法中常见的参数，例如 GPTQ 和 AWQ，它们在 `AutoGPTQ` 或 `AutoAWQ` 等库中得到实现。

### 场景设定：一个调优情景

假设您使用默认设置的 GPTQ 将一个 70 亿参数 (parameter)的 LLM 量化 (quantization)为 INT4。初步评估（使用第 3 章的方法）显示以下结果：

- **性能：** 延迟显著降低，达到您的目标。
- **内存：** 占用空间大幅减小，符合硬件限制。
- **准确性：** 困惑度显著增加，且在特定下游任务（例如，文本摘要）上的表现下降幅度超出可接受范围。

我们现在的目标是调整量化参数以恢复部分损失的准确性，必要时，可能接受性能或模型大小的微小权衡。

### 调优参数 (parameter) 1：校准数据集大小

PTQ 方法依赖于校准数据集，通过观察激活的典型范围来确定最佳量化 (quantization)参数（如缩放因子和零点）。该数据集的大小和代表性很重要。

- **作用：** 更大的校准数据集提供了对激活分布更统计准确的视图，可能带来更好的量化参数，从而提高准确性。
- **权衡：** 使用更多的校准样本会增加量化过程本身所需的时间。非常小或不具代表性的数据集可能导致糟糕的量化选择和显著的准确性损失。
- **如何调整：** 大多数量化工具包允许您指定用于校准的样本数量。

假设我们最初的量化使用了 128 个样本。我们可以尝试增加这个数量。

```python
# 量化函数使用示例
from model_quantizer import quantize_gptq # 虚构库

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("your_base_model")
tokenizer = AutoTokenizer.from_pretrained("your_base_model")

# 加载或准备校准数据（例如，来自相关数据集的样本）
calibration_data = load_calibration_data("path/to/calibration_set.jsonl")

# 初始量化（示例）
# quantize_gptq(model, tokenizer, calibration_data, num_samples=128, output_dir="quantized_model_128")

# 实验：增加校准样本数量
print("正在使用 256 个校准样本进行量化...")
quantize_gptq(model, tokenizer, calibration_data, num_samples=256, output_dir="quantized_model_256")

print("正在使用 512 个校准样本进行量化...")
quantize_gptq(model, tokenizer, calibration_data, num_samples=512, output_dir="quantized_model_512")

# 每次运行后，评估准确性（困惑度、下游任务）和性能
```

接下来，您将评估使用 256 和 512 个样本生成的模型。准确性是否提升了？提升了多少？这种提升是否值得更长的量化时间？通常，在某个点之后回报会递减。



![校准数据集大小对困惑度的影响](plots/4679-0.json)



> 图表展示了随着校准数据集大小的增加，困惑度可能如何降低（提升），同时提升的速度通常会减缓。

### 调优参数 (parameter) 2：组大小

像 GPTQ 和 AWQ 这样的算法通常不是按张量或按通道进行量化 (quantization)，而是对层内较小的权重 (weight)组进行量化。`group_size` 参数控制有多少权重共享相同的量化参数（缩放因子和零点）。

- **作用：** 较小的组大小允许量化参数更紧密地适应权重分布中的局部变化。这对于更准确地捕捉权重范围很有帮助，可能缓解较大块中异常值引起的问题。
- **权衡：** 使用较小的组大小（例如，32 而不是 128）会增加需要存储的元数据量（缩放因子和零点），从而略微增加最终模型的大小。由于更复杂的反量化过程，它有时也可能对推理 (inference)延迟产生轻微影响，尽管这很大程度上取决于可用的特定硬件内核。组大小为 -1 通常意味着按通道量化。
- **如何调整：** 这通常是量化函数中的一个直接参数。

```python
# 调整组大小示例
from model_quantizer import quantize_gptq # 虚构库

# 假设模型、分词器、数据已按之前方式加载
num_calib_samples = 256 # 根据上一步或合理默认值选择

# 实验组大小（默认可能为 128）
print("正在使用 group_size=64 进行量化...")
quantize_gptq(model, tokenizer, calibration_data,
              num_samples=num_calib_samples,
              group_size=64,
              output_dir="quantized_model_gs64")

print("正在使用 group_size=32 进行量化...")
quantize_gptq(model, tokenizer, calibration_data,
              num_samples=num_calib_samples,
              group_size=32,
              output_dir="quantized_model_gs32")

# 评估每个组大小下的准确性和性能
```

在使用不同组大小进行量化后，评估其中的权衡。与 `group_size=64` 或 `group_size=128` 相比，`group_size=32` 是否显著提高了准确性？模型大小增加了多少？推理延迟受到怎样的影响？



![组大小与准确性及延迟的权衡](plots/4679-1.json)



> 此图显示，减小组大小（例如，从 128 减到 32）可能会提高准确性，但可能略微增加延迟。最佳选择取决于应用程序要求。

### 调优参数 (parameter) 3：算法特定的超参数 (hyperparameter)（例如，GPTQ 中的阻尼因子）

一些量化 (quantization)算法具有独特的超参数。例如，GPTQ 在计算用于权重 (weight)更新的逆 Hessian 矩阵时，会使用一个阻尼因子（通常表示为 `damp_percent`）。

- **作用：** 阻尼在 Hessian 矩阵求逆之前，向其对角线添加一个微小值。这有助于稳定计算，尤其是在 Hessian 矩阵条件不佳时。它影响了量化误差在过程中如何得到修正。
- **权衡：** 默认值（例如，0.01）通常效果不错。然而，在某些情况下，通过改变量化期间权重调整的方式，轻微增加或减少阻尼因子可能会对特定模型或数据集的准确性产生微小的提升。参数之间的改变有时可能会产生意想不到的相互作用。
- **如何调整：** 在量化函数中查找类似 `damp_percent` 或类似名称的参数。

```python
# 调整阻尼因子示例
from model_quantizer import quantize_gptq # 虚构库

# 假设模型、分词器、数据、样本数量、组大小均已设置
best_group_size = 64 # 根据上一步选择

# 实验阻尼因子（默认可能为 0.01）
print("正在使用 damp_percent=0.005 进行量化...")
quantize_gptq(model, tokenizer, calibration_data,
              num_samples=num_calib_samples,
              group_size=best_group_size,
              damp_percent=0.005,
              output_dir="quantized_model_damp005")

print("正在使用 damp_percent=0.02 进行量化...")
quantize_gptq(model, tokenizer, calibration_data,
              num_samples=num_calib_samples,
              group_size=best_group_size,
              damp_percent=0.02,
              output_dir="quantized_model_damp02")

# 评估不同阻尼因子下的准确性
```

调整算法特有的超参数（如阻尼）通常收益较小，相比校准数据集大小或组大小，且可能需要更多实验。通常在主要参数调优后，如果仍存在显著准确性问题，才会考虑尝试。

### 迭代过程

微调 (fine-tuning)量化 (quantization)参数 (parameter)很少是一次性完成的过程。它涉及一个迭代循环：

1. **选择参数：** 从默认值或一个假设开始（例如，“增加校准数据”）。
2. **量化：** 使用选定的参数运行量化过程。
3. **评估：** 使用第 3 章的技术衡量准确性（困惑度、下游任务）和性能（延迟、吞吐量 (throughput)、内存占用）。
4. **分析：** 将结果与基线和您的要求进行比较。找出主要存在的差距（例如，准确性仍然过低）。
5. **调整：** 根据您的分析修改一个参数（例如，尝试更小的组大小以提高准确性）。
6. **重复：** 返回步骤 2。

仔细记录每次实验中使用的参数以及相应的评估结果。这种系统化的方法有助于您了解模型和任务对不同量化设置的敏感程度，并收敛到最佳配置。

这种仔细的参数调整实践不可或缺，它有助于提升效率的上限，同时保持大型语言模型的能力，将量化的理论潜力转化为实际、高性能的部署。

## 参考资料

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍了AWQ算法，该算法基于激活统计数据优先保护关键权重，强调了校准数据和参数调整对准确性的重要性。
- [Quantization for LLMs](https://huggingface.co/docs/optimum/concept_guides/quantization_for_llms) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face `optimum` 库的官方指南。它提供了量化大型语言模型的实用指导，包括校准数据集大小和分组大小等关键参数的解释，与GPTQ和AWQ的实现相关。
