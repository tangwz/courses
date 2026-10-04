# 使用 Hugging Face PEFT 库进行实现

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/implementing-peft-huggingface)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Hugging Face `peft` 库提供了一个标准化且用户友好的接口，用于实现 LoRA、QLoRA 和适配器模块等各种 PEFT 技术。它将这些方法与 `transformers` 生态系统中的模型结合，极大简化了高效适配大型模型的流程。

`peft` 的核心思想是使用 `PeftModel` 对象包装一个标准的 `transformers` 模型。这个包装器根据指定的 PEFT 配置管理基础模型的修改，确保只训练指定的适配器参数 (parameter)。

### 核心组件：`PeftConfig` 和 `get_peft_model`

库的核心是配置对象，它们是 `PeftConfig` 的子类，用于定义特定的 PEFT 方法及其超参数 (parameter) (hyperparameter)。例如，`LoraConfig` 用于 LoRA 和 QLoRA，`PromptTuningConfig` 用于 Prompt Tuning 等。

通常，您首先使用 `transformers` 加载基础的预训练 (pre-training)模型。然后，定义一个配置对象（例如 `LoraConfig`），详细说明 PEFT 策略。最后，`get_peft_model` 函数将基础模型和配置结合起来，创建启用 PEFT 的模型，可用于训练。

> 使用 Hugging Face `peft` 库创建 PEFT 模型的工作流程。

### 使用 `peft` 应用 LoRA

我们来演示如何配置模型进行 LoRA 微调 (fine-tuning)。我们将加载一个基础模型，并对其注意力层应用 LoRA 修改。

```python
# 假设已导入必要的库：
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 1. 加载基础模型（示例：使用较小的模型进行演示）
model_name = "meta-llama/Llama-2-7b-hf" # 或任何其他兼容模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    r=16, # 更新矩阵的秩。较低的秩意味着更少的参数。
    lora_alpha=32, # LoRA 缩放因子。
    target_modules=["q_proj", "v_proj"], # 将 LoRA 应用于查询和值投影
    lora_dropout=0.05, # LoRA 层的 Dropout 概率。
    bias="none", # 对于 LoRA，通常设置为 'none'。
    task_type=TaskType.CAUSAL_LM # 指定任务类型（例如，CAUSAL_LM, SEQ_2_SEQ_LM）
)

# 3. 创建 PeftModel
lora_model = get_peft_model(base_model, lora_config)

# 可选：打印可训练参数以验证效率
lora_model.print_trainable_parameters()
# 示例输出：可训练参数：4,194,304 || 所有参数：6,742,609,920 || 可训练参数百分比：0.0622
```

在此配置中：

- `r`：定义了低秩矩阵 ($W_A$ 和 $W_B$) 的秩。常见的起始值是 8、16 或 32。更高的秩会增加可训练参数 (parameter)，但在一定程度上可能提供更好的性能。
- `lora_alpha`：作为 LoRA 激活的缩放因子，通常设置为秩 `r` 的两倍。它有助于平衡原始权重 (weight)和 LoRA 更新的影响。
- `target_modules`：指定基础模型中哪些层应接收 LoRA 矩阵。识别最佳模块（通常是注意力机制 (attention mechanism)层，如查询、键、值投影）对性能很重要。您可以查看基础模型的架构（`print(base_model)`）以查找层名称。
- `task_type`：告知 `peft` 模型的目标，确保适配器正确应用（例如，用于因果语言建模）。

请注意，与基础模型的总参数数量相比，可训练参数显著减少。这突出了 LoRA 在效率方面的提升。

### 使用 `peft` 应用 QLoRA

QLoRA 在 LoRA 的基础上构建，通过将其应用于量化 (quantization)的基础模型（通常以 4 位精度加载）。这进一步减少了内存需求，使得在消费级硬件上微调 (fine-tuning)更大的模型成为可能。

`peft` 库与 `bitsandbytes` 库集成，用于量化。

```python
# 假设与之前一样，已导入必要的库

# 1. 配置量化（使用 bitsandbytes）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # 使用 NF4 (Normalized Float 4) 以获得更好的精度
    bnb_4bit_compute_dtype="bfloat16", # 或 float16，在前向传播期间计算数据类型
    bnb_4bit_use_double_quant=True, # 可选：使用双重量化
)

# 2. 加载量化后的基础模型
model_name = "meta-llama/Llama-2-7b-hf" # 示例模型
base_model_quantized = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto" # 如果需要，自动将模型层分布到设备上
)

# 准备模型进行 k 位训练（梯度检查点，输入/输出嵌入）
base_model_quantized = prepare_model_for_kbit_training(base_model_quantized)

# 3. 定义 LoRA 配置（与之前类似）
qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # 识别特定模型中的目标模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 4. 创建 PeftModel
qlora_model = get_peft_model(base_model_quantized, qlora_config)

# 可选：打印可训练参数
qlora_model.print_trainable_parameters()
# 示例输出可能与量化细节相关，因此会相似或略有不同
```

主要区别在于加载基础模型时使用了 `quantization_config` 参数 (parameter)。`prepare_model_for_kbit_training` 实用函数执行必要的设置步骤，例如启用梯度检查点以在训练期间节省内存，并确保输入/输出嵌入 (embedding)与量化模型兼容。`LoraConfig` 本身大体保持不变，但现在它应用于量化后的权重 (weight)。

### 对其他 PEFT 方法的支持

尽管 LoRA 和 QLoRA 广泛使用，`peft` 也支持其他技术：

- **适配器微调 (fine-tuning) (Adapter Tuning)：** 使用 `AdaptionPromptConfig`。需要指定适配器层和维度。
- **Prompt 微调 (Prompt Tuning)：** 使用 `PromptTuningConfig`。指定要微调的虚拟 token 数量。
- **Prefix 微调 (Prefix Tuning)：** 使用 `PrefixTuningConfig`。类似于 Prompt 微调，但向隐藏状态添加可微调的前缀。

总体工作流程保持一致：定义适当的配置对象，并将其与基础模型一同传递给 `get_peft_model`。有关每种配置类型所需的具体参数 (parameter)，请查阅 `peft` 官方文档。

### 与 `transformers.Trainer` 的集成

`peft` 的一个重要优点是它能与标准 `transformers.Trainer` API 顺畅集成。一旦您创建了 `PeftModel`（例如 `lora_model` 或 `qlora_model`），就可以像使用普通模型一样，直接将其传递给 `Trainer`。

```python
# 假设已导入：from transformers import Trainer, TrainingArguments
# 假设已准备好 'train_dataset' 和 'eval_dataset'

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    # 根据需要添加其他参数（fp16, optim 等）
)

# 使用 PeftModel 初始化 Trainer
trainer = Trainer(
    model=qlora_model, # 在此处传递 PEFT 模型
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # 如果需要，添加数据整理器、分词器
)

# 开始训练
trainer.train()
```

`Trainer` 会自动检测到它处理的是一个 `PeftModel`，并确保在反向传播 (backpropagation)期间只更新适配器参数 (parameter)（例如 LoRA 矩阵）。冻结的基础模型参数的梯度不会被计算或应用，从而带来预期的计算和内存节省。

### 保存、加载和合并适配器

训练结束后，您不需要保存整个模型（这会违背 PEFT 的目的）。相反，您只保存训练好的适配器权重 (weight)。

```python
# 训练后保存适配器
adapter_path = "./qlora_adapters"
qlora_model.save_pretrained(adapter_path)

# 可选：也保存分词器
# tokenizer.save_pretrained(adapter_path)
```

这会将适配器配置 (`adapter_config.json`) 和训练好的权重 (`adapter_model.bin` 或 `.safetensors`) 保存到指定目录。基础模型未在此处保存，因此检查点大小非常小。

要加载适配器用于推理 (inference)或进一步训练，您首先加载原始基础模型（可能使用相同的量化 (quantization)配置），然后应用已保存的适配器：

```python
# 假设已导入必要的库并已加载基础模型（量化或未量化）
# from peft import PeftModel, PeftConfig

# 首先加载基础模型（与训练期间的配置相同）
# 示例：base_model_for_inference = AutoModelForCausalLM.from_pretrained(...)

# 将适配器加载到基础模型上
inference_model = PeftModel.from_pretrained(base_model_for_inference, adapter_path)

# 合并适配器（可选，用于部署）
# merged_model = inference_model.merge_and_unload()
# 现在 'merged_model' 是一个权重已更新的标准 transformers 模型
# 注意：对于量化模型（例如 4 位），合并可能不受支持或不直接。
```

`PeftModel.from_pretrained` 函数加载适配器配置和权重，并将它们应用于提供的基础模型。

对于不需要动态适配器切换的部署场景，您可以使用 `model.merge_and_unload()` 将适配器权重直接合并到基础模型的权重中。这会创建一个具有合并权重、标准 `transformers` 模型，可能简化推理堆栈，并在移除适配器逻辑后提供轻微的速度提升。然而，这会使模型大小恢复到基础模型的大小，并且移除了轻松切换适配器的能力。合并对于非量化模型通常更简单；将适配器合并到量化模型（特别是 4 位）中可能很复杂，或者需要特定的库支持。

### 实际考量

- **选择 `target_modules`：** 选择要适配的层（例如通过 `LoraConfig` 中的 `target_modules`）会显著影响性能。Transformer 中常见的选择包括注意力投影层（`q_proj`、`k_proj`、`v_proj`、`o_proj`），有时也包括前馈网络层（`gate_proj`、`up_proj`、`down_proj`）。可能需要进行实验。对于某些模型，如果内存允许，指定 `all-linear` 可能是一个方便的起点。
- **超参数 (parameter) (hyperparameter)调优：** PEFT 方法引入了自己的超参数（例如 `r`、`lora_alpha`、`num_virtual_tokens`）。这些需要与学习率和批量大小等标准训练超参数一起进行调优。
- **库兼容性：** PEFT 涉及 `peft`、`transformers`、`accelerate` 以及可能的 `bitsandbytes` 之间的交互。请确保安装了兼容版本，以避免运行时错误。查阅每个库的文档以了解版本要求。

Hugging Face `peft` 库为实现参数高效微调 (fine-tuning)提供了一个强大且灵活的框架。通过理解其核心组件和工作流程，您可以有效地为特定任务适配大型语言模型，同时高效管理计算资源。本章后面的实操部分将提供使用该库应用 LoRA 和 QLoRA 的具体示例。

## 参考资料

- [Parameter-Efficient Fine-tuning (PEFT) Library](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Hugging Face `peft` 库的官方文档，详细介绍了其组件、API以及各种PEFT方法的使用。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  引入大语言模型低秩适应（LoRA）微调的原始研究论文，解释了其理论基础和方法。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA的研究论文，这是一种用于量化大语言模型的有效微调方法，将LoRA与4位量化相结合。
- [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face `transformers.Trainer` 类的官方文档，提供了用于训练模型的API，包括PEFT模型。

---

[上一节](07-PEFT%20%E6%96%B9%E6%B3%95%E6%AF%94%E8%BE%83.md) · [下一节](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20LoRA%20%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md)
