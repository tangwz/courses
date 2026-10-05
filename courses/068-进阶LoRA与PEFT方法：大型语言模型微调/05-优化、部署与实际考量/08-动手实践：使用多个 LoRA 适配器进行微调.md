# 动手实践：使用多个 LoRA 适配器进行微调

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/practice-multi-adapter-lora)

[返回章节目录](README.md) · [返回课程目录](../README.md)

同时使用多个 LoRA 适配器微调 (fine-tuning)单个大型语言模型，能够使基础模型适应多个不同任务或数据集，而无需复制大型基础模型权重 (weight)。这种方法在多任务训练和适配器管理场景中，可以节省大量内存和存储资源，是一种高效的策略。

我们将模拟一个场景：使预训练 (pre-training)模型适应两个任务：任务 A（例如，文本摘要）和任务 B（例如，情感分析）。虽然我们不会在这里实现特定数据集的完整数据加载和预处理，但我们将侧重于使用 Hugging Face `transformers` 和 `peft` 库来配置、添加、训练和保存多个 LoRA 适配器的主要机制。

### 前提条件

确保您已安装必要的库：

```bash
pip install transformers datasets accelerate peft bitsandbytes torch
```

我们假设您熟悉从 Hugging Face 加载模型和分词 (tokenization)器 (tokenizer)、准备数据集以及 PyTorch 训练循环或 `Trainer` API 的基本知识。

### 1. 设置和模型加载

首先，让我们导入所需的模块并加载我们的基础预训练 (pre-training)模型。为了演示，我们将使用一个较小的模型，但这些原则直接适用于更大的 LLM。我们还将以 8 位加载它，以模拟应用 PEFT 时常遇到的资源受限环境。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, set_peft_model_state_dict
from datasets import Dataset # 使用虚拟数据集进行说明

# 加载基础模型和分词器
model_name = "gpt2" # 替换为您的目标 LLM，例如 "meta-llama/Llama-2-7b-hf"
# 以 8 位加载以演示内存效率
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True, # 使用 8 位加载
    device_map="auto", # 自动分布到可用的 GPU/CPU 上
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # 如果缺失，设置填充 token
```

### 2. 定义多个 LoRA 配置

现在，我们为每个任务（适配器）定义单独的 `LoraConfig` 对象。这使我们能够根据需要为每个任务指定不同的秩、目标模块、alpha 值或其他定制的超参数 (parameter) (hyperparameter)。我们为每个适配器分配唯一的名称。

```python
# 任务 A 的配置 (例如：摘要)
lora_config_task_a = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, # 任务 A 的秩
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"], # 示例：任务 A 只针对注意力投影层
    bias="none",
    adapter_name="adapter_task_a" # 此适配器的唯一名称
)

# 任务 B 的配置 (例如：情感分析)
lora_config_task_b = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4, # 任务 B 的较低秩
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["c_proj"], # 示例：任务 B 针对不同层
    bias="none",
    adapter_name="adapter_task_b" # 此适配器的唯一名称
)
```

注意我们如何使用不同的秩（`r`）和目标模块进行演示。`adapter_name` 对于在同一个基础模型上管理多个适配器非常重要。

### 3. 将适配器添加到模型

我们使用 `peft` 库中的 `add_adapter` 方法将这些配置附加到我们的基础模型。添加的第一个适配器将使用 `get_peft_model` 隐式地封装模型。后续对 `add_adapter` 的调用会将额外的适配器附加到*相同*的基础模型结构上。

```python
# 添加第一个适配器 (任务 A)
# 如果模型还不是 PeftModel，此调用会在内部使用 get_peft_model
model.add_adapter(lora_config_task_a, adapter_name="adapter_task_a")
print(f"适配器 'adapter_task_a' 已添加。")

# 添加第二个适配器 (任务 B)
model.add_adapter(lora_config_task_b, adapter_name="adapter_task_b")
print(f"适配器 'adapter_task_b' 已添加。")

# 您可以验证已附加的适配器
print("活动适配器:", model.active_adapters)
print("PEFT 配置:", model.peft_config) # 显示所有已附加适配器的配置
```

此时，`model` 对象包含原始的预训练 (pre-training)权重 (weight)（已冻结）以及 `adapter_task_a` 和 `adapter_task_b` 的新初始化、可训练的 LoRA 矩阵。

### 4. 准备数据和训练

使用多个适配器进行训练需要仔细处理数据和训练循环。主要思想是在处理与相应任务相关的数据批次*之前*激活正确的适配器。

#### 4.1 虚拟数据准备（说明性）

让我们创建简单的虚拟数据集来代表我们的两个任务。在实际情况中，您将加载并预处理您实际的任务特定数据集。

```python
# 虚拟数据函数
def create_dummy_dataset(text_prefix, num_samples=100):
    texts = [f"{text_prefix}: Sample text number {i} for training." for i in range(num_samples)]
    # 分词 - 确保一致的处理 (填充、截断)
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    # 转换为 Dataset 对象
    return Dataset.from_dict(tokenized)

# 为每个任务创建数据集
dataset_task_a = create_dummy_dataset("Summarize", 200)
dataset_task_b = create_dummy_dataset("Analyze Sentiment", 150)

# 我们需要一种方式来识别批次属于哪个数据集。
# 一种常见方法是交错数据集或使用自定义采样器。
# 为了标准 Trainer 的简单性，我们将它们合并并添加一个任务标识符，
# 尽管自定义训练循环提供了更多控制。

# 添加适配器标识符 (演示用的简单方法)
dataset_task_a = dataset_task_a.map(lambda example: {'adapter_name': "adapter_task_a"})
dataset_task_b = dataset_task_b.map(lambda example: {'adapter_name': "adapter_task_b"})

# 合并数据集 (朴素交错) - 实际中需要仔细洗牌/采样
from datasets import concatenate_datasets
combined_dataset = concatenate_datasets([dataset_task_a, dataset_task_b]).shuffle(seed=42)

# 我们需要自定义数据收集器或进行修改来处理 'adapter_name' 列
# 或者，更常见地，使用自定义训练循环。
# 对于本例，我们仍使用 Trainer，但需要承认其在此处的局限性。
# 标准的 DataCollatorForLanguageModeling 不会使用 'adapter_name'。
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(f"合并数据集大小: {len(combined_dataset)}")
print("示例条目:", combined_dataset[0])
```

#### 4.2 自定义训练器或训练循环逻辑

标准的 Hugging Face `Trainer` 不原生支持根据批次元数据动态切换适配器。您通常需要以下两种方式之一：

1. **自定义训练循环：** 遍历批次，识别批次对应的任务/适配器，在前向传播*之前*调用 `model.set_adapter(adapter_name)`，计算损失并执行反向传播 (backpropagation)。只有活动适配器的权重 (weight)会接收梯度。
2. **修改 Trainer：** 继承 `Trainer` 类并重写 `compute_loss` 或 `training_step` 等方法，以便根据您注入批次的信息（例如我们添加的 `adapter_name` 列，尽管通过默认 collator 传递它需要注意）来包含 `model.set_adapter()` 调用。

我们来概述一下自定义训练循环中的逻辑（实际实现需要 PyTorch 样板代码）：

```python
# --- 自定义训练循环片段 ---
# 假设 'dataloader' 产生批次，每个批次包含数据和 'adapter_name'

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) # 优化器针对 PEFT 参数

# model.train()
# for batch in dataloader:
#     adapter_name_for_batch = batch.pop("adapter_name") # 提取适配器名称
#     inputs = {k: v.to(model.device) for k, v in batch.items()} # 将数据移动到设备

# *** 重要步骤：设置活动适配器 ***
#     model.set_adapter(adapter_name_for_batch)

#     # 前向传播 - 仅使用活动适配器
#     outputs = model(**inputs)
#     loss = outputs.loss

#     # 反向传播 - 梯度仅流向活动的 LoRA 权重
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     # 可选地，如果需要，在步骤后禁用适配器，
#     # 尽管在下一次迭代开始时设置就足够了。
#     # model.disable_adapter()
# --- 片段结束 ---
```

**重要考量：** 训练多个适配器时，请确保从每个任务的数据集中进行平衡采样，以防止某个适配器在训练过程中占据主导地位，或相对于其他适配器出现过拟合 (overfitting)/欠拟合 (underfitting)。这可能涉及加权采样或精心设计的训练周期。

#### 4.3 使用简化 Trainer 进行训练（仅限演示）

在本例中，我们将继续使用 `Trainer`，但为了简化起见，我们将*按顺序*训练适配器，这并非真正的同时多适配器训练，但它演示了适配器的切换和保存。这避免了修改 `Trainer` 或为此次实践编写完整自定义循环的复杂性。

```python
# 定义基础训练参数
training_args = TrainingArguments(
    output_dir="./multi_adapter_output",
    num_train_epochs=1, # 为演示目的缩短训练周期
    per_device_train_batch_size=4,
    logging_steps=50,
    save_strategy="epoch", # 在每个 epoch 结束时保存
    learning_rate=3e-4,
    weight_decay=0.01,
    report_to="none", # 为简化禁用外部日志记录
    remove_unused_columns=False # 如果之后使用修改后的 Trainer，保留 'adapter_name'
)

# --- 训练阶段 1：只训练 adapter_task_a ---
print("\n--- 训练适配器 A ---")
model.set_adapter("adapter_task_a") # 激活任务 A 适配器
# 确保只有此适配器的 LoRA 参数是可训练的
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = "adapter_task_a" in name
    else:
        param.requires_grad = False

# 为任务 A 数据集重新初始化训练器
trainer_a = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_task_a, # 使用任务 A 数据
    data_collator=data_collator,
    # 注意：为了这个简单的训练器设置，我们移除了 'adapter_name' 列
    # 因为标准 collator 会报错。
)
trainer_a.train()
print("适配器 A 训练完成。")
# 特别保存任务 A 适配器
model.save_pretrained("./multi_adapter_output/adapter_task_a", selected_adapters=["adapter_task_a"])
print("适配器 A 已保存到 ./multi_adapter_output/adapter_task_a")

# --- 训练阶段 2：只训练 adapter_task_b ---
print("\n--- 训练适配器 B ---")
model.set_adapter("adapter_task_b") # 激活任务 B 适配器
# 确保只有此适配器的 LoRA 参数是可训练的
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = "adapter_task_b" in name
    else:
        param.requires_grad = False

# 我们可能需要重置优化器状态或使用新的训练器实例
# 为了简化，我们为任务 B 创建一个新的训练器
training_args.output_dir = "./multi_adapter_output_b" # 如果需要，使用不同的输出目录
trainer_b = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_task_b, # 使用任务 B 数据
    data_collator=data_collator,
)
trainer_b.train()
print("适配器 B 训练完成。")
# 特别保存任务 B 适配器
model.save_pretrained("./multi_adapter_output/adapter_task_b", selected_adapters=["adapter_task_b"])
print("适配器 B 已保存到 ./multi_adapter_output/adapter_task_b")
```

*注意：像这样多次重新运行 `Trainer.train()` 可能不理想，特别是在优化器状态和学习率调度方面。自定义循环或修改后的 `Trainer` 为真正的交错多任务训练提供了更好的控制。*

### 5. 保存和加载多个适配器

如上所示，您可以使用 `model.save_pretrained()` 中的 `selected_adapters` 参数 (parameter)保存特定适配器。每个适配器都保存在自己的子目录中，包含 LoRA 权重 (weight)（`adapter_model.bin`）和配置（`adapter_config.json`）。

要在之后加载这些适配器进行推理 (inference)：

```python
from peft import PeftModel

# 再次加载基础模型 (如果内存中尚未存在)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

# 加载第一个适配器
model_with_adapter_a = PeftModel.from_pretrained(
    base_model,
    "./multi_adapter_output/adapter_task_a", # 保存的适配器 A 路径
    adapter_name="adapter_task_a" # 使用相同的名称
)
print("适配器 A 已加载。")

# 将第二个适配器加载到相同的基础模型实例上
# 重要：后续适配器应加载到 PeftModel 对象上
model_with_adapter_a.load_adapter(
    "./multi_adapter_output/adapter_task_b", # 保存的适配器 B 路径
    adapter_name="adapter_task_b" # 使用相同的名称
)
print("适配器 B 已加载到同一模型上。")

# 验证已加载的适配器
print("已加载适配器:", model_with_adapter_a.peft_config.keys())
```

### 6. 使用特定适配器进行推理 (inference)

在推理过程中，您可以使用 `set_adapter` 在已加载的适配器之间动态切换。

```python
# 使用适配器 A 生成文本
model_with_adapter_a.set_adapter("adapter_task_a")
print("\n--- 使用适配器 A 生成中 ---")
prompt_a = "Summarize this document: ..." # 任务 A 的示例提示
inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model_with_adapter_a.device)
# 确保模型处于 eval 模式以进行生成
model_with_adapter_a.eval()
with torch.no_grad():
    outputs_a = model_with_adapter_a.generate(**inputs_a, max_new_tokens=50)
print("适配器 A 输出:", tokenizer.decode(outputs_a[0], skip_special_tokens=True))

# 使用适配器 B 生成文本
model_with_adapter_a.set_adapter("adapter_task_b")
print("\n--- 使用适配器 B 生成中 ---")
prompt_b = "Analyze the sentiment: This movie was fantastic!" # 任务 B 的示例提示
inputs_b = tokenizer(prompt_b, return_tensors="pt").to(model_with_adapter_a.device)
model_with_adapter_a.eval()
with torch.no_grad():
    outputs_b = model_with_adapter_a.generate(**inputs_b, max_new_tokens=20)
print("适配器 B 输出:", tokenizer.decode(outputs_b[0], skip_special_tokens=True))

# 可选地禁用适配器以使用基础模型
# model_with_adapter_a.disable_adapter()
```

### 讨论和考量

- **真正的同时训练：** 本实践为了简化与标准 `Trainer` 的使用，采用了顺序训练。对于适配器权重 (weight)在同一训练运行中以交错方式更新的真正并发训练，需要实现自定义 PyTorch 循环或修改 `Trainer` 以处理每批次的适配器切换。
- **资源效率：** 这里的主要益处是内存。您在内存中只保留一份大型基础模型权重，以及每个任务的小型 LoRA 适配器权重。这与维护多个完全微调 (fine-tuning)的模型副本形成鲜明对比。
- **适配器干扰：** 在同一个基础模型上同时训练多个适配器可能会导致细微的相互影响或干扰，特别是当目标模块明显重叠或任务冲突时。可能需要对每个适配器进行仔细的监控和超参数 (parameter) (hyperparameter)调整。
- **任务调度/采样：** 在自定义循环中，如何从不同任务数据集中采样批次（例如，轮询、基于数据集大小或任务重要性的加权采样）成为影响收敛和最终性能平衡的一个重要设计选择。
- **部署：** 将多个适配器加载到单个基础模型实例上的能力对于部署而言非常高效。单个部署的模型只需在处理请求前激活相应的适配器，即可服务多个任务的请求。

“本实践练习演示了管理多个 LoRA 适配器的方法。使此框架适应数据集并实现交错训练策略是有效应用此技术的后续步骤。”

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=qV9s5B75Yq) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2022)
  Journal: International Conference on Learning Representations (ICLR); Publisher: OpenReview Foundation; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍LoRA算法的开创性研究论文，详细阐述了通过适配少量参数实现内存高效微调的原理。
- [Parameter-Efficient Fine-Tuning (PEFT) Library Documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了`LoraConfig`、`add_adapter`、`set_adapter`以及多适配器管理的API详情，与本实践练习直接相关。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，涵盖了加载预训练模型、分词器以及使用`Trainer` API进行微调等方面，这些是本实践练习的前提条件。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  这篇论文介绍了QLoRA，一种使用4位量化对大型语言模型进行内存高效微调的方法，它基于LoRA概念并为资源受限环境提供了显著的内存节省。

---

[上一节](07-%E4%BD%BF%E7%94%A8%20PEFT%20%E9%80%82%E9%85%8D%E5%99%A8%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B.md) · [下一节](../06-%E8%AF%84%E4%BC%B0%20PEFT%20%E6%80%A7%E8%83%BD%E5%92%8C%E5%B1%80%E9%99%90%E6%80%A7/01-PEFT%E8%AF%84%E4%BC%B0%E7%9A%84%E5%B8%B8%E7%94%A8%E6%8C%87%E6%A0%87.md)
