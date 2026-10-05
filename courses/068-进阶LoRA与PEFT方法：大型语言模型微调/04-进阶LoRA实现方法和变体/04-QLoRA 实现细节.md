# QLoRA 实现细节

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/qlora-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

量化 (quantization)LoRA (QLoRA) 的实现需要具体的实践步骤。成功使用QLoRA涉及特定的模型加载流程、配置设定以及与支持库的整合。Hugging Face 生态系统，特别是 `transformers`、`peft` (参数 (parameter)高效微调 (fine-tuning))、`accelerate` 和 `bitsandbytes` 库，为此提供了工具。

### 所需库及配置

在开始QLoRA微调 (fine-tuning)之前，请确保已安装并正确配置所需库。主要组成部分包括：

1. **`transformers`**：提供对预训练 (pre-training)模型的访问以及 `Trainer` API，以简化训练循环。
2. **`peft`**：包含各种PEFT方法的实现，包括LoRA以及QLoRA所需的配置。
3. **`bitsandbytes`**：此库是QLoRA的根本，因为它负责处理低级别的量化 (quantization)操作（NF4、双重量化）以及前向和反向传播 (backpropagation)过程中的量化矩阵乘法。安装通常需要特定的CUDA版本。
4. **`accelerate`**：促进硬件管理（GPU、TPU）和分布式训练设置，简化了在多个设备上运行训练的过程。

您通常可以使用pip进行安装：

```bash
pip install transformers peft bitsandbytes accelerate datasets torch
```

*注意：`bitsandbytes`的安装可能需要根据您的CUDA环境特别留意。请查阅其文档以获取详细说明。*

### 加载量化 (quantization)后的基础模型

QLoRA实现的第一步是使用所需的量化设定加载基础大型语言模型（LLM）。这通过 `transformers` 库的 `from_pretrained` 方法实现，并辅以由 `bitsandbytes` 管理的特定参数 (parameter)。

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-hf" # 示例模型

# 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 启用4位精度加载
    bnb_4bit_quant_type="nf4", # 使用NF4量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 设置矩阵乘法的计算数据类型
    bnb_4bit_use_double_quant=True, # 启用双重量化
)

# 使用指定配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # 自动将模型层分布到可用的GPU上
)

# 可选：为训练稳定性禁用缓存（配合梯度检查点）
model.config.use_cache = False
```

我们来详细解释一下 `BitsAndBytesConfig` 中的重要参数：

- `load_in_4bit=True`：此标志通知 `transformers`（通过 `accelerate` 和 `bitsandbytes`）直接以4位精度加载模型权重 (weight)。
- `bnb_4bit_quant_type="nf4"`：指定量化方案。“nf4”（4位 NormalFloat）是QLoRA的标准，专为正态分布的权重设计。另一个选项是“fp4”（4位浮点）。
- `bnb_4bit_compute_dtype=torch.bfloat16`：虽然权重以4位存储，但计算（矩阵乘法）仍需要更高的精度。`bfloat16`通常建议用于现代GPU的性能和稳定性。`float16`是替代选项。此参数确定计算期间的临时反量化格式。
- `bnb_4bit_use_double_quant=True`：启用双重量化技术，该技术对量化常数本身进行第二次量化，进一步减少内存占用。
- `device_map="auto"`：由 `accelerate` 处理，它自动将模型的层分布到可用设备（GPU和CPU内存，如果需要）上，从而使加载无法完全放入单个GPU的模型成为可能。

### 为QLoRA训练配置LoRA

加载量化 (quantization)后的基础模型后，下一步是使用 `peft` 库的 `LoraConfig` 类定义LoRA配置。虽然许多参数 (parameter)是LoRA的标准参数，但有些参数在QLoRA设置中尤其相关或常用。

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 准备模型进行k比特训练（对于梯度检查点兼容性很重要）
model = prepare_model_for_kbit_training(model)

# 定义LoRA配置
lora_config = LoraConfig(
    r=16, # 更新矩阵的秩
    lora_alpha=32, # Alpha缩放因子
    target_modules=["q_proj", "v_proj"], # 应用LoRA的模块（通常是注意力投影）
    lora_dropout=0.05, # LoRA层的Dropout概率
    bias="none", # 将偏差设置为'none'以保持稳定性，尤其是在量化时
    task_type="CAUSAL_LM" # 指定任务类型（例如，因果语言建模）
)

# 使用LoRA配置将基础模型包装为PEFT模型
peft_model = get_peft_model(model, lora_config)

# 打印可训练参数以供验证
peft_model.print_trainable_parameters()
# 示例输出：可训练参数：4,194,304 || 所有参数：6,938,533,968 || 可训练百分比：0.0604
```

这里的重要方面包括：

- `prepare_model_for_kbit_training(model)`：这个实用函数对量化模型执行必要的预处理步骤，以确保与训练兼容，尤其是在使用梯度检查点（通常需要它来节省内存）时。
- `LoraConfig` 参数：
  - `r` 和 `lora_alpha`：标准的LoRA超参数 (hyperparameter)，控制适应的容量和缩放。
  - `target_modules`：指定基础模型中哪些线性层应使用LoRA进行适应。识别正确的模块名称（例如，Llama风格模型中的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`）很重要。您可能需要检查基础模型的架构（`print(model)`）以找到合适的名称。
  - `bias="none"`：在QLoRA中，偏置 (bias)的训练通常被禁用，因为它有时与量化结合时会引入不稳定性。LoRA的更新只专注于权重 (weight)矩阵。
  - `task_type`：告知 `peft` 模型的目标（例如，因果语言模型、序列分类），以确保适配器配置正确。
- `get_peft_model(model, lora_config)`：此函数接收（量化后的）基础模型和LoRA配置，识别 `target_modules`，并适当注入LoRA层（A和B矩阵）。它返回一个 `PeftModel` 对象。
- `print_trainable_parameters()`：一个有用的方法，用于验证总参数中只有一小部分（对应于LoRA矩阵）被标记 (token)为可训练。

以下图表展示了高级流程：

> Hugging Face库实现QLoRA微调 (fine-tuning)的高级工作流程。

### 训练QLoRA模型

创建 `PeftModel` 后，训练过程主要遵循标准的微调 (fine-tuning)流程，例如使用 `transformers.Trainer`。主要区别在于优化器（例如AdamW，或之前讨论过的分页优化器）将*只*更新注入的LoRA层内的参数 (parameter)。4位的基础模型权重 (weight)保持冻结。

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 假设已定义'train_dataset'和'tokenizer'

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./qlora-results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    max_steps=-1, # 或设置max_steps而不是训练轮数
    save_steps=100,
    fp16=True, # 使用混合精度（如果bnb_config支持并使用BF16，效果可能更好）
    # ... 其他训练参数
)

# 设置训练器
trainer = Trainer(
    model=peft_model, # 使用PEFT模型
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存训练好的LoRA适配器权重
peft_model.save_pretrained("./qlora-adapter")
```

训练过程中的重要事项：

- **优化器状态**：即使基础模型经过量化 (quantization)，优化器状态（例如Adam中的动量）仍然会消耗大量内存。这时，分页优化器（例如 `adamw_bnb_8bit`）就变得有益，它们将优化器状态卸载到CPU内存中。您可以在 `TrainingArguments` 中指定优化器。
- **梯度检查点**：通常由 `prepare_model_for_kbit_training` 自动启用，或可以在 `TrainingArguments` 中手动设置（`gradient_checkpointing=True`）。它能大幅减少激活内存，但代价是由于反向传播 (backpropagation)期间重新计算激活，计算速度会降低约20-30%。
- **精度**：确保 `bnb_4bit_compute_dtype` 与训练所用的精度（`TrainingArguments` 中的 `fp16` 或 `bf16`）保持一致。

### 保存和加载适配器

训练完成后，您只需保存LoRA适配器权重 (weight)，而无需保存整个基础模型。这通过 `PeftModel` 的 `save_pretrained` 方法完成。

```python
# 保存
peft_model.save_pretrained("my-qlora-adapter")

# 用于推理或进一步训练的加载
from peft import PeftModel, PeftConfig

# 首先加载量化后的基础模型（如前所示）
config = PeftConfig.from_pretrained("my-qlora-adapter")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config, # 使用与训练期间相同的bnb_config
    device_map="auto"
)

# 通过附加适配器加载PeftModel
loaded_model = PeftModel.from_pretrained(base_model, "my-qlora-adapter")
```

基础模型和适配器权重的这种分离是PEFT方法的一个核心优势，能够实现微调 (fine-tuning)适应的高效存储和共享。请记住，要使用已保存的适配器，您必须首先使用与训练期间使用的 *完全相同* 的 `BitsAndBytesConfig` 加载 *完全相同* 的基础模型。

通过遵循这些步骤并借助 `transformers`、`peft` 和 `bitsandbytes` 之间的集成，您可以有效地实现QLoRA，以大幅减少内存占用的方式微调大型语言模型，这与标准LoRA或完全微调相比。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA、NF4量化、双重量化和分页优化器，用于微调大型语言模型。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  提出了低秩适应（LoRA）作为一种参数高效的微调方法，是QLoRA的基础。
- [PEFT (Parameter-Efficient Fine-tuning)](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，包括LoRA配置和如`prepare_model_for_kbit_training`等实用函数。
- [bitsandbytes: 8-bit optimizers and 4-bit quantization for PyTorch](https://github.com/TimDettmers/bitsandbytes) — Tim Dettmers (2024)
  bitsandbytes库的源代码和文档，该库为QLoRA提供了核心量化（NF4、双重量化）和优化操作。

---

[上一节](03-%E9%87%8F%E5%8C%96LoRA%20%28QLoRA%29%EF%BC%9A%E5%8E%9F%E7%90%86.md) · [下一节](05-%E5%88%86%E9%A1%B5%E4%BC%98%E5%8C%96%E5%99%A8%E6%8F%90%E5%8D%87%E5%86%85%E5%AD%98%E6%95%88%E7%8E%87.md)
