---
course: "practical-llm-quantization"
chapter: "evaluating-deploying-quantized-llms"
lesson: "practice-benchmarking-quantized-llm"
sourceId: 4680
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-6-evaluating-deploying-quantized-llms/practice-benchmarking-quantized-llm"
title: "实践：量化大型语言模型的基准测试"
description: "动手实践，测量量化大型语言模型的性能（速度、内存）并评估其准确性。"
order: 7
plots: ["plots/4680-0.json"]
sourceHash: "6f44b62204ee2f0cac7eeb0429574ecd4f15251226113971266afa0686d297b8"
sourceCorrections: []
---

"好的，我们将理论付诸实践。在前面的部分，我们讨论了评估量化 (quantization)模型的指标和方法。现在，你将亲自动手对一个量化大型语言模型与其原始版本进行基准测试。这项实操练习将有助于你加深对如何衡量量化对性能（速度、内存）和准确性影响的理解。"

量化本质上是关于权衡。我们的目标是减少所需的计算资源（例如内存占用 $M$ 和推理 (inference)延迟 $L$），同时尽量减小对模型预测质量的影响，预测质量通常通过准确性或困惑度来衡量。此处量化了这些确切的权衡。

### 配置你的环境

在开始之前，请确保你已安装所需的库。我们将主要使用 Hugging Face 库进行模型加载、生成和评估。你还需要 `torch` 来处理核心功能，如果加载使用 `bitsandbytes` 量化 (quantization)的模型，可能还需要 `bitsandbytes`，或者对于 GPTQ/GGUF 格式的模型则需要 `optimum`。

```bash
pip install transformers torch accelerate datasets evaluate optimum bitsandbytes sentencepiece
# 如果通过optimum中使用ctransformers后端来使用GGUF模型：
pip install optimum[exporters,ctransformers]
# 或者如果通过optimum中使用AutoGPTQ后端：
pip install optimum[exporters,auto-gptq]
```

对于本练习，你将需要一个LLM的两个版本：

1. **基础模型：** 原始的、高精度模型（例如 FP16 或 FP32）。
2. **量化模型：** 通过PTQ（例如基本的静态/动态PTQ、GPTQ，或通过 `bitsandbytes` 加载）甚至QAT获得的低精度版本（例如 INT8、INT4）。

你可以使用在先前练习中量化的模型，或者加载 Hugging Face Hub 上可用的预量化模型。为了保持一致性和加快执行速度，可以考虑使用像 `gpt2`、`distilgpt2` 这样的小型模型，或者如果原始和量化版本都可用，则使用 Mistral/Llama 的小型变体。确保你有一块支持 CUDA 的 GPU，以便进行有效的性能测量。

让我们定义模型标识符。请将它们替换为你所选模型的实际 Hugging Face Hub ID 或本地路径。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np

# --- 配置 ---
base_model_id = "gpt2"  # 示例：替换为你的基础模型（例如："meta-llama/Llama-2-7b-hf"）
quantized_model_id = "gpt2" # 示例：替换为你的量化模型ID或路径
# 如果使用bitsandbytes加载4位模型，ID可能相同，但加载方式不同
load_in_4bit = True # 如果加载时使用bitsandbytes 4位量化，则设为True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 加载分词器（通常两者相同）---
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # 如果缺失，设置填充 token

# --- 加载基础模型 ---
print(f"Loading base model: {base_model_id}")
# 如果使用GPU且模型支持，则加载半精度
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device # 简单加载到目标设备
)
base_model.eval() # 设置为评估模式
print("Base model loaded.")

# --- 加载量化模型 ---
print(f"Loading quantized model: {quantized_model_id}")
if load_in_4bit and device == "cuda":
    from transformers import BitsAndBytesConfig
    # 示例：使用bitsandbytes 4位量化加载
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 # 可选：计算数据类型
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        quantized_model_id,
        quantization_config=bnb_config,
        device_map=device # bitsandbytes 处理设备放置
    )
    print("Quantized model loaded using bitsandbytes 4-bit config.")
# 如果需要，在此处添加其他加载方法（例如，使用Optimum加载GPTQ/GGUF）
# elif quantized_model_id.endswith(".gguf"): # 示例：使用Optimum和CTransformers
#     from optimum.quanto import CtransformersModelForCausalLM
#     quantized_model = CtransformersModelForCausalLM.from_pretrained(quantized_model_id, device_map=device)
#     print("Quantized GGUF model loaded using Optimum.")
else:
    # 假设加载的是标准的Hugging Face模型（例如，如果使用了QAT或者以其他方式预量化）
    # 根据特定的量化格式，必要时调整数据类型和加载方式
    quantized_model = AutoModelForCausalLM.from_pretrained(
        quantized_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, # 可能需要调整
        device_map=device
    )
    print("Quantized model loaded (adjust loading logic if not using bitsandbytes 4-bit).")

quantized_model.eval() # 设置为评估模式
```

> 请务必根据 `quantized_model` 的量化和保存方式调整其加载逻辑。此示例展示了通过 `bitsandbytes` 加载。如果你有 GPTQ 或 GGUF 模型，通常会使用 `optimum` 等库以及相应的后端（`auto-gptq` 或 `ctransformers`）。

### 性能基准测试

现在，让我们衡量速度和内存使用情况。我们将关注延迟（每次生成所需的时间）和峰值内存消耗。

#### 测量延迟

延迟是处理单个输入所需的时间。我们可以通过对 `generate` 函数计时来测量它。为了获得准确的GPU计时，在启动和停止计时器之前进行同步很重要。

```python
# --- 延迟基准测试 ---
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 热身运行（对CUDA很重要）
_ = base_model.generate(**inputs, max_new_tokens=50)
_ = quantized_model.generate(**inputs, max_new_tokens=50)
if device == "cuda":
    torch.cuda.synchronize()

# 测量基础模型延迟
start_time = time.perf_counter()
_ = base_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
if device == "cuda":
    torch.cuda.synchronize()
end_time = time.perf_counter()
base_latency = end_time - start_time
print(f"Base Model Latency: {base_latency:.4f} seconds")

# 测量量化模型延迟
start_time = time.perf_counter()
_ = quantized_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
if device == "cuda":
    torch.cuda.synchronize()
end_time = time.perf_counter()
quantized_latency = end_time - start_time
print(f"Quantized Model Latency: {quantized_latency:.4f} seconds")

if base_latency > 0:
    speedup = base_latency / quantized_latency
    print(f"Speedup: {speedup:.2f}x")
else:
    print("Base latency was zero, cannot calculate speedup.")
```

多次运行此代码或对多个提示取平均值，以获得更稳定的结果。你应该会看到量化 (quantization)模型具有更低的延迟，尤其是在原生支持低精度操作的硬件上。

#### 测量内存使用

我们可以使用 PyTorch 的内存管理函数来估计推理 (inference)期间的 GPU 峰值内存使用量。

```python
# --- 内存基准测试 ---
def get_peak_memory_mb(model, inputs):
    """测量模型生成过程中的GPU峰值内存使用量。"""
    if device != "cuda":
        print("Memory measurement only available for CUDA devices.")
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        _ = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2) # 将字节转换为MB
    except Exception as e:
        print(f"Error during memory measurement: {e}")
        peak_memory = 0
    torch.cuda.empty_cache()
    return peak_memory

# 测量基础模型内存
base_memory_mb = get_peak_memory_mb(base_model, inputs)
print(f"Base Model Peak Memory: {base_memory_mb:.2f} MB")

# 测量量化模型内存
quantized_memory_mb = get_peak_memory_mb(quantized_model, inputs)
print(f"Quantized Model Peak Memory: {quantized_memory_mb:.2f} MB")

if base_memory_mb > 0:
    memory_reduction = base_memory_mb / quantized_memory_mb
    print(f"Memory Reduction Factor: {memory_reduction:.2f}x")
    print(f"Memory Savings: {(1 - (quantized_memory_mb / base_memory_mb)) * 100:.2f}%")
else:
    print("Base memory usage was zero or could not be measured.")
```

量化模型应显示显著更低的峰值内存使用量，这与其权重 (weight)（以及可能根据方法，激活值）的精度降低直接相关。

### 准确性基准测试

性能提升只有在模型对其预期任务保持足够准确时才有价值。我们需要使用合适的指标评估量化 (quantization)模型的质量。

#### 评估困惑度

困惑度是语言模型常用的内在指标。它衡量模型预测文本序列的能力。较低的困惑度通常表示更好的模型。我们可以使用 `evaluate` 库和像 WikiText 这样的标准数据集。

```python
import evaluate
from datasets import load_dataset
from tqdm import tqdm

# --- 困惑度评估 ---
try:
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    # 在此示例中使用一小部分数据以加快评估速度
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]")
    # 或者使用完整测试集：split="test"

    # 预处理文本数据（拼接和分块）以计算困惑度
    # 此方法将整个数据集文本作为单个序列处理
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = base_model.config.n_positions # 使用模型的最大序列长度
    stride = 512 # 块之间的重叠量

    seq_len = encodings.input_ids.size(1)
    nlls_base = []
    nlls_quantized = []
    prev_end_loc = 0

    print("Calculating perplexity...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc # 在最后一次循环中可能与步长不同
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # 忽略重叠 token 的损失计算

        with torch.no_grad():
            # 基础模型
            outputs_base = base_model(input_ids, labels=target_ids)
            neg_log_likelihood_base = outputs_base.loss * trg_len # 按目标长度缩放损失
            nlls_base.append(neg_log_likelihood_base)

            # 量化模型
            outputs_quantized = quantized_model(input_ids, labels=target_ids)
            neg_log_likelihood_quantized = outputs_quantized.loss * trg_len # 按目标长度缩放损失
            nlls_quantized.append(neg_log_likelihood_quantized)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # 计算困惑度
    ppl_base = torch.exp(torch.stack(nlls_base).sum() / end_loc)
    ppl_quantized = torch.exp(torch.stack(nlls_quantized).sum() / end_loc)

    print(f"\nBase Model Perplexity: {ppl_base.item():.4f}")
    print(f"Quantized Model Perplexity: {ppl_quantized.item():.4f}")
    print(f"Perplexity Increase: {ppl_quantized.item() - ppl_base.item():.4f}")

except ImportError:
    print("Please install 'evaluate' and 'datasets' to run perplexity benchmark.")
except Exception as e:
    print(f"Could not run perplexity evaluation: {e}")
```

> 这种困惑度计算采用了因果语言模型常用的滑动窗口方法。请注意，在大型数据集上计算困惑度可能非常耗时。使用较小的子集（`split="test[:1%]"`）可以提供快速估计。

#### 评估下游任务（可选）

另外，或者作为补充，你可以在模型可能被用于的特定任务上对其进行评估，例如问答、摘要或情感分析。这通常能更直接地衡量实际用途。像 `lm-evaluation-harness` 这样的框架就是为此目的而设计的，它提供了许多任务的标准评估设置。设置 `lm-evaluation-harness` 更复杂，超出了本次基本实操的范围，但它是严格LLM评估的标准。

对于本笔记本中更简单的基于任务的评估，你可以调整生成循环，使其在一个特定任务数据集（例如，用于是/否问题的 BoolQ）上运行并计算准确性。

### 分析结果

现在，整理你的发现。一个简单的表格通常很有效：

| 指标 | 基础模型 | 量化 (quantization)模型 | 变化 |
| --- | --- | --- | --- |
| 延迟（秒） | `base_latency` | `quant_latency` | `speedup`倍快 |
| 峰值内存（MB） | `base_memory` | `quant_memory` | `reduction`倍小 |
| 困惑度 | `ppl_base` | `ppl_quantized` | `increase`更高 |
| 任务准确率（%） | (如果已测量) | (如果已测量) | (差异) |

*（用你测量的结果替换占位符值）*

可视化权衡结果。一个简单的散点图，将速度提升/内存减少与准确性下降（例如，困惑度百分比增加）进行比较，可以提供好的参考。



![量化权衡：速度提升对比困惑度增加](plots/4680-0.json)



> 性能指标（如延迟和内存减少）对比评估指标（如困惑度增加）。示例展示了一个4位量化模型获得了1.85倍的速度提升，同时困惑度增加了5.2%。请用你自己的测量值替换这些数据。

解释结果：

- **量化是否达到了性能目标？** 速度提升或内存减少是否显著？
- **准确性影响是否可接受？** 困惑度是否显著增加？如果你测量了任务准确性，它是否降至应用程序可接受的阈值以下？
- **这种特定的量化方法合适吗？** 基于这种权衡，你会部署这个量化模型，还是尝试其他方法（例如，侵略性较低的量化级别、像 GPTQ 这样的高级 PTQ，或 QAT）？

这种亲自动手的基准测试过程提供了具体数据，以指导有关部署量化模型的决策，确保你在效率提升与所需的预测性能水平之间取得平衡。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: International Conference on Learning Representations (ICLR) 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍GPTQ，一种高效的大语言模型训练后量化方法，显著减少内存并提高推理速度，同时保持准确性。
- [Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6) — Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, Alexander Rush (2020)
  Journal: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations; Publisher: Association for Computational Linguistics; Pages: 38–45; DOI: [10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
  描述Hugging Face Transformers库，它提供数千个预训练模型和用于构建、训练和评估LLM的工具，是本实践的核心框架。
