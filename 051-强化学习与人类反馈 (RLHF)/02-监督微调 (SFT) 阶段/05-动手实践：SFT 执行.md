# 动手实践：SFT 执行

来源：[原文](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf/sft-execution-practical)

[返回章节目录](README.md) · [返回课程目录](../README.md)

监督微调 (fine-tuning)（SFT）对于初始化策略模型具有重要意义。本实践指南将逐步演示在预训练 (pre-training)语言模型上执行 SFT 的方法。练习中使用了 Hugging Face 生态系统中的库，具体是 `transformers`、`datasets` 和 `trl`（Transformer 强化学习 (reinforcement learning)），极大简化了流程。

我们的目的是取一个通用大型语言模型（LLM），并在高质量的提示-响应对数据集上对其进行微调。这个调整后的模型将作为后续强化学习阶段更好的起点。

### 准备工作：导入与配置

首先，确保您已安装所需的库（`pip install transformers datasets trl torch accelerate bitsandbytes`）。我们首先导入所需组件。这些例子中将使用 PyTorch。

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig # 可选：用于量化
)
from trl import SFTTrainer
import os

# 可选：如果可用，配置 GPU 使用
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 设置为您希望的 GPU ID
```

### 步骤 1：加载基础模型和分词 (tokenization)器 (tokenizer)

"我们需要一个预训练 (pre-training)语言模型来进行微调 (fine-tuning)。为了演示，我们将使用 `gpt2` 这样的小型、公开可用的模型。在实际情况中，您可能会使用一个更大、能力更强的模型，符合您的目标应用需求。我们还加载了对应的分词器，它负责将文本转换为模型能理解的数字表示。"

为了管理内存，特别是对于较大的模型，我们可以使用 `bitsandbytes` 等量化 (quantization)技术。这是可选的，但通常很有用。

```python
# Hugging Face Hub 中的模型标识符
base_model_id = "gpt2" # 替换为您的模型，例如 "NousResearch/Llama-2-7b-hf"

# 可选：量化配置
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
# 如果未设置，则设置填充符
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    # quantization_config=bnb_config, # 如果使用量化，请取消注释
    device_map="auto", # 自动将模型加载到可用的 GPU/CPU 上
    trust_remote_code=True
)
model.config.use_cache = False # 建议用于微调
```

### 步骤 2：准备数据集

SFT 需要一个数据集，里面每个样本都代表一次高质量的交互。`trl` 库的 `SFTTrainer` 很灵活，但常见格式是带有一列（例如命名为 'text'）的数据集，其中包含完整的提示和响应，通常使用特殊标记 (token)进行结构化。

假设我们有一个数据集（也许是从 Hugging Face Hub 或本地文件加载的），其中每个条目看起来像这样：

```json
{"text": "### Prompt: Explain the concept of photosynthesis in simple terms.\n### Response: Photosynthesis is the process plants use to turn sunlight, water, and air into food (sugar) for themselves, releasing oxygen as a byproduct."}
{"text": "### Prompt: Write a short poem about a rainy day.\n### Response: Silver drops on window pane,\nWhispering secrets of the rain.\nGray clouds drift in skies above,\nA cozy day for hearth and love."}
```

您可以使用 `datasets` 库加载数据集。对于本例，让我们模拟加载一个与 `SFTTrainer` 兼容的虚构数据集。一个常用于演示的选择是 `databricks/databricks-dolly-15k` 数据集，它经过筛选，只包含遵循指令的样本。

```python
# 使用公共数据集的例子（需要 'pip install pyarrow'）
# 确保数据集有 'text' 列，或使用 formatting_func 进行调整
dataset_name = "databricks/databricks-dolly-15k"
# 为了演示，我们只使用训练集的一小部分
dataset = load_dataset(dataset_name, split="train[:500]") # 使用前 500 个样本

# --- 简单数据预处理示例（如果需要）---
# 如果您的数据集列名不同（例如 'instruction'、'response'），
# 您可能需要将其格式化为单个 'text' 字段。

# def format_dolly(example):
#    # 根据您的特定数据集结构调整格式
#    if example.get("context"):
#        return f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}\n\n### Response:\n{example['response']}"
#    else:
#        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# 如果数据集没有现成的 'text' 字段，则应用格式化
# dataset = dataset.map(lambda x: {"text": format_dolly(x)})
# --------------------------------------------------------

# 我们需要重命名 Dolly 数据集的列以符合 SFTTrainer 的预期
# 如果 'text' 列不存在。我们假设 Dolly 有 'instruction' 和 'response'。
# 如果指定，SFTTrainer 通常可以直接处理。
# 如果需要，我们将演示重命名以求清晰。
# 如果数据集已包含 'text'，则跳过重命名。

# 检查 'text' 列是否存在，否则尝试创建它
if 'text' not in dataset.column_names:
    def create_text_column(example):
        # 简单拼接，如果需要，根据实际 Dolly 格式调整
        instr = example.get('instruction', '')
        resp = example.get('response', '')
        ctx = example.get('context', '')
        if ctx:
             return {"text": f"### Instruction:\n{instr}\n\n### Context:\n{ctx}\n\n### Response:\n{resp}"}
        else:
             return {"text": f"### Instruction:\n{instr}\n\n### Response:\n{resp}"}

    dataset = dataset.map(create_text_column, remove_columns=dataset.column_names)

print("样本数据集条目：")
print(dataset[0]['text'])
```

### 步骤 3：配置训练参数 (parameter)

我们使用 `transformers.TrainingArguments` 定义训练参数。这些参数控制学习率、批次大小、训练轮数、保存频率和日志记录等方面。

```python
# 定义保存检查点和最终模型的输出目录
output_dir = "./sft_model_output"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2, # 根据 GPU 内存调整
    gradient_accumulation_steps=4, # 有效批次大小 = 批次大小 * 梯度累积
    learning_rate=2e-5,
    logging_steps=20,              # 每 20 步记录指标
    num_train_epochs=1,            # 遍历数据集的次数
    max_steps=-1,                  # 设置为 >0 以覆盖轮数
    save_strategy="epoch",         # 在每轮结束时保存检查点
    # save_steps=50,               # 或每 N 步保存
    report_to="tensorboard",       # 或 "wandb", "none"
    fp16=True,                     # 使用混合精度（需要兼容的 GPU）
    # bf16=True,                   # 使用 BF16（需要 Ampere+ GPU）- 二选一
    optim="paged_adamw_8bit",      # 内存高效优化器，尤其是在量化时
    lr_scheduler_type="cosine",    # 学习率调度器类型
    warmup_ratio=0.03,             # 调度器的预热步数
)
```

> **注意：** 超参数 (hyperparameter)，如学习率、批次大小和训练轮数，都非常重要。找到最优值通常需要根据您具体的模型、数据集和硬件情况进行实验。以上数值仅供参考。

### 步骤 4：初始化 SFT 训练器

现在我们实例化 `trl` 库中的 `SFTTrainer`。它负责协调微调 (fine-tuning)过程，处理数据整理、分词 (tokenization)填充和训练循环本身。

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text", # 包含格式化提示/响应的列名
    max_seq_length=512,        # 分词的最大序列长度
    packing=False,             # 可选：将多个短样本打包到一个序列中
                               # 对于包含许多短序列的数据集，设置 packing=True 可能会加快速度。
                               # 需要仔细准备数据集。
)
```

### 步骤 5：开始微调 (fine-tuning)

一切配置好后，开始训练就简单了。训练器将根据 `TrainingArguments` 处理优化循环、梯度更新、日志记录和保存检查点。

```python
print("开始 SFT 训练...")
trainer.train()
print("训练完成。")
```

训练期间，请监控日志（例如，在控制台或已配置的 TensorBoard/WandB 中），以观察训练损失。损失的降低通常表明模型正在从 SFT 数据集中学习。

### 6：保存最终模型

训练完成后，保存微调 (fine-tuning)后的模型权重 (weight)和分词 (tokenization)器 (tokenizer)配置。这个保存的模型是 SFT 阶段的产出。

```python
# 定义最终保存模型的路径
final_model_path = os.path.join(output_dir, "final_sft_model")

print(f"正在将最终 SFT 模型保存到 {final_model_path}")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path) # 将分词器与模型一起保存
print("模型保存成功。")
```

### 步骤 7：快速推理 (inference)检查

让我们进行一个简单的测试，定性地查看微调 (fine-tuning)模型与基础模型的响应有何不同。我们加载保存的 SFT 模型，并为示例提示生成文本。

```python
from transformers import pipeline

# 加载微调模型
print("正在加载微调模型进行推理...")
sft_pipe = pipeline("text-generation", model=final_model_path, tokenizer=final_model_path, device_map="auto")

# 定义一个符合模型预期的示例提示
# （与 SFT 数据集中使用的格式匹配）
prompt_text = "### Instruction:\nExplain the main benefit of using version control systems like Git.\n\n### Response:\n"

print(f"\n正在为提示生成响应：\n{prompt_text}")

# 生成响应
# 根据需要调整生成参数
output = sft_pipe(prompt_text, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)

print("\n生成的响应：")
print(output[0]['generated_text'])

# 如果需要，清理 GPU 内存
del sft_pipe
del model
# torch.cuda.empty_cache() # 如果稍后遇到内存不足（OOM）问题，请取消注释
```

检查生成的响应。它是否遵循指令？风格是否与 SFT 数据集中的样本一致？这个快速检查提供了 SFT 过程结果的初步评估。更严格的评估将涉及指标和可能的人工评估，正如上一节（“评估 SFT 模型性能”）和课程后面（第七章）所讨论的。

### 接下来的步骤

您现在已成功执行了监督微调 (fine-tuning)阶段。得到的模型（我们例子中的 `final_sft_model`）已根据所提供的演示进行了调整，以更好地理解期望的任务结构和输出风格。这个微调模型现在已准备好作为 RLHF 流程后续阶段的初始策略网络：训练奖励模型并进行强化学习 (reinforcement learning)优化。

## 参考资料

- [TRL library documentation - Supervised Fine-tuning](https://huggingface.co/docs/trl/main/en/sft_trainer) — Hugging Face (2024)
  提供了使用 trl 库进行监督微调（SFT）的实践指南，涵盖了实现细节。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  一篇奠基性论文，介绍了监督微调作为人类反馈强化学习（RLHF）流程的初始阶段。
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Hugging Face transformers 库的官方文档，对于加载和管理预训练模型及分词器至关重要。
- [bitsandbytes: 8-bit Optimizers and Quantization-aware Training for LLMs](https://github.com/bitsandbytes-foundation/bitsandbytes) — Tim Dettmers and Hugging Face (2024)
  详细介绍了 bitsandbytes 库，它通过量化技术实现了大型语言模型的高效内存使用和更快的计算。

---

[上一节](04-%E8%AF%84%E4%BC%B0%20SFT%20%E6%A8%A1%E5%9E%8B%E8%A1%A8%E7%8E%B0.md) · [下一节](../03-%E5%9F%BA%E4%BA%8E%E4%BA%BA%E7%B1%BB%E5%81%8F%E5%A5%BD%E6%9E%84%E5%BB%BA%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B/01-%E5%81%8F%E5%A5%BD%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%B7%AF.md)
