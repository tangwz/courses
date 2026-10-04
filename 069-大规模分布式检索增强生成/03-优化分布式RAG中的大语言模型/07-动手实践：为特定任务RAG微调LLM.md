# 动手实践：为特定任务RAG微调LLM

来源：[原文](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-3-optimizing-llms-distributed-rag/practice-llm-finetuning-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用参数 (parameter)高效微调 (fine-tuning)（PEFT）技术对大型语言模型（LLM）进行微调，是一种常见且高效的优化手段，旨在提升其在特定RAG任务中的表现。此方法的目标是改进LLM根据检索阶段提供的上下文 (context)，*严格*地综合出准确且相关答案的能力，尤其适用于专门方面或需要特定回应风格的情形。

我们将进行数据准备、模型选择、应用LoRA（低秩适应）以及评估结果的过程，这一切都将以构建生产级别RAG系统的专家视角来完成。

### 目标

完成本次实践后，您将能够：

1. 准备一个适合专门用于RAG任务的LLM微调 (fine-tuning)数据集。
2. 对基础LLM应用LoRA，以调整其上下文 (context)理解和生成能力。
3. 在类似RAG的环境中加载并使用微调后的适配器进行推断。
4. 概述评估微调LLM在忠实度和相关性方面表现的方法。

### 前提条件和设置

在开始之前，请确保您拥有一个带有较新GPU的Python环境（强烈建议，以便获得合理的训练时间）。您将需要以下核心库：

- `torch`: 用于张量运算和GPU支持。
- `transformers`: 来自Hugging Face，用于LLM模型和分词 (tokenization)器 (tokenizer)。
- `peft`: 来自Hugging Face，用于参数 (parameter)高效微调 (fine-tuning)技术，如LoRA。
- `datasets`: 来自Hugging Face，便于数据处理。
- `accelerate`: 用于简化分布式训练和混合精度（即使在单个GPU上，它也很有用）。
- `bitsandbytes`: 用于8位或4位量化 (quantization)（例如QLoRA），如果您想尝试进一步减少内存占用。

您通常可以使用pip安装这些库：

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
```

确保您的CUDA驱动和PyTorch安装与您的GPU兼容。

### 1. 为RAG准备微调 (fine-tuning)数据集

您的微调数据质量和结构对成功非常重要。对于RAG，您不仅仅是教授LLM通用知识；您是在教它*基于所提供的文本进行推理 (inference)*。一个理想的数据集由三元组组成：（查询、检索到的上下文 (context)、基于上下文的理想答案）。

微调期间LLM的输入应模仿您在RAG系统中将使用的提示结构。一种常用格式是：

```
<s>[INST] Context: {retrieved_document_chunk}
Question: {user_query} [/INST]
Answer: {ideal_answer_based_on_context}</s>
```

- `<s>` and `</s>`: 序列开始和结束标记 (token)。
- `[INST]` and `[/INST]`: 指令标签，常见于Llama和Mistral等模型。请根据您选择的基础模型的首选提示格式进行调整。
- `{retrieved_document_chunk}`: 您的检索器会提供的实际文本片段。
- `{user_query}`: 用户的提问。
- `{ideal_answer_based_on_context}`: 期望的输出。此答案*必须*仅从提供的`retrieved_document_chunk`中得出。避免需要外部知识的答案。

**示例数据点（JSONL格式）：**

```json
{
  "text": "<s>[INST] Context: Llama 2模型家族包含7B、13B和70B参数的版本。LoRA微调在使这些模型适应特定任务同时保持大多数权重冻结方面很有效。对于7B模型，LoRA秩(r)为8或16通常是一个好的起始点。 \nQuestion: Llama 2 7B建议使用哪个LoRA秩？ [/INST]\n答案：对于Llama 2 7B模型，LoRA秩为8或16通常是微调的一个良好起始点。</s>"
}
```

**制作高质量数据：**

- **来源：** 可以是人工整理的，从现有文档中合成（例如，取一个文档片段，对其提问，然后*只*根据该片段编写答案），或者从原型RAG系统的交互日志中获取（需仔细过滤）。
- **具体性：** 确保答案与上下文紧密关联。如果上下文不支持答案，微调数据应反映这一点，或许通过训练LLM说“根据提供的上下文，我无法回答...”
- **负面示例：** 尽管此处未详细说明，但对于高级情况，包含上下文不相关或误导性的示例，并训练LLM识别这些情况，会非常有益。

对于本次练习，您可以手动创建50-100个示例的小数据集，或者使用脚本从现有文档生成。将其保存为`train.jsonl`文件。

### 2. 选择基础模型和PEFT配置

基础模型的选择取决于您的性能要求、计算预算和任务的复杂程度。Mistral-7B、Llama-2-7B或Gemma-7B等模型由于其强大的基础能力以及使用LoRA进行微调 (fine-tuning)时可控的规模，是PEFT的很好起始点。

我们将使用LoRA。需要配置的LoRA参数 (parameter)有：

- `r`: 更新矩阵的秩。较小的`r`表示可训练参数更少。常用值范围为4到64。
- `lora_alpha`: 一个缩放因子，通常设为`r`或`2*r`。
- `target_modules`: 指定Transformer中要应用LoRA的线性层（例如`q_proj`、`v_proj`、`k_proj`、`o_proj`）。识别这些层通常需要检查模型架构。
- `lora_dropout`: LoRA层的Dropout概率。
- `bias`: 是否使LoRA偏置 (bias)项可训练（例如：“none”、“all”、“lora\_only”）。

我们假设使用像`mistralai/Mistral-7B-Instruct-v0.1`这样的模型。

### 3. 微调 (fine-tuning)过程

这是一个使用Hugging Face库的Python脚本概述。

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 配置
model_name = "mistralai/Mistral-7B-Instruct-v0.1" # 或您选择的模型
dataset_path = "path/to/your/train.jsonl" # 您的JSONL文件
output_dir = "./results_rag_finetune"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
# 用于QLoRA（4位量化）
use_4bit = True 
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.bfloat16 # 如果不支持bfloat16则使用torch.float16

# 2. 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 常见做法
tokenizer.padding_side = "right"

if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=True, # 可选
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} # 将模型加载到GPU 0
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0} # 将模型加载到GPU 0
    )

model.config.use_cache = False # 建议用于微调
model.config.pretraining_tp = 1 # 如果遇到此错误，请设置为1

# 3. LoRA配置
# 通过检查model.named_modules()或根据模型常识来查找目标模块
# 对于Mistral，常用目标是'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"], # 从小范围开始，如果需要再添加更多
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 检查有多少参数是可训练的

# 4. 加载数据集
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 5. 训练参数
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 根据您的GPU显存调整
    gradient_accumulation_steps=4, # 有效批量大小 = 2 * 4 = 8
    optim="paged_adamw_32bit",    # 如果不使用QLoRA，则为"adamw_torch"
    save_steps=50,                # 每50步保存检查点
    logging_steps=10,             # 记录训练进度
    learning_rate=2e-4,
    fp16=not use_4bit,            # 如果不使用4位，则使用fp16
    bf16=use_4bit and torch.cuda.is_bf16_supported(), # 如果是4位且支持，则使用bf16
    max_grad_norm=0.3,
    num_train_epochs=1,           # 对于小型数据集，从1-3个周期开始
    warmup_ratio=0.03,
    group_by_length=True,         # 通过对相似长度序列进行分组来加速训练
    lr_scheduler_type="constant", # 或 "cosine"
    report_to="tensorboard"       # 或 "wandb"
)

# 6. 初始化训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # JSONL中包含完整提示的字段
    max_seq_length=1024,        # 根据您的上下文长度和显存调整
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,              # 如果要打包多个短序列，请设置为True
)

# 7. 开始训练
print("开始训练...")
trainer.train()

# 8. 保存微调后的适配器
adapter_output_dir = f"{output_dir}/final_adapter"
trainer.model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir) # 保存分词器以保持一致性
print(f"微调后的适配器已保存到 {adapter_output_dir}")
```

**训练要点：**

- **监控损失：** 使用TensorBoard（或其他日志记录器）查看训练损失。它通常应呈下降趋势。如果在小数据集上训练过多周期，可能会发生过拟合 (overfitting)。
- **GPU显存 (VRAM)：** 调整`per_device_train_batch_size`、`gradient_accumulation_steps`、`max_seq_length`和量化 (quantization)设置（`use_4bit`），使其适应GPU的内存。
- `target_modules`: LoRA的`target_modules`选择会明显影响表现。通常需要进行实验。对于许多基于注意力的模型，定位查询、键、值和输出投影（`q_proj`、`k_proj`、`v_proj`、`o_proj`）是一个好的开始。某些架构也通过定位前馈网络层而受益。

### 4. 使用微调 (fine-tuning)后的RAG-LLM进行推断

训练后，LoRA适配器（而非完整模型）会被保存。要将其用于推断：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "mistralai/Mistral-7B-Instruct-v0.1" # 相同的基础模型
adapter_path = "./results_rag_finetune/final_adapter" # 您保存的适配器路径

# 加载基础模型（推断时也可以量化）
# 用于4位推断：
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     device_map={"": 0}
# )

# 或者不进行量化以获得全精度（显存占用更高）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16, # 或 torch.float16
    device_map={"": 0}
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path) # 从适配器目录加载分词器

# 通过将适配器合并到基础模型来加载PEFT模型
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.eval() # 设置为评估模式

# 可选：将LoRA层与基础模型合并以加快推断速度
# 这会创建一个新模型，并且最初可能需要更多显存。
# model = model.merge_and_unload() 
# print("LoRA层已合并。")

# 示例RAG风格提示
retrieved_context = "LoRA技术通过在现有层中插入可训练的低秩矩阵来调整大型预训练模型。与完全微调相比，这显著减少了可训练参数的数量，使其内存高效。"
user_query = "LoRA如何实现内存效率？"

prompt = f"<s>[INST] Context: {retrieved_context}\nQuestion: {user_query} [/INST]\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成回应
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id 
    )

response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 只提取生成的答案部分
answer_part = response_text.split("[/INST]\nAnswer:")[1].strip()
print(f"生成的答案: {answer_part}")
```

### 5. 评估对RAG表现的影响

评估非常重要。像困惑度这样的标准LLM指标对于RAG来说是不足的。您需要评估：

- **忠实度（基于上下文 (context)）：**
  - 生成的答案是否严格遵守所提供的上下文？
  - 是否存在任何编造或幻觉 (hallucination)内容，即使听起来合理？
  - **方法：** 对多样化的（查询、上下文、生成的答案）三元组进行人工审查。使用自然语言推断（NLI）模型或以LLM作为评判者的自动化方法正在出现，但需要仔细设置。
- **相关性：**
  - 答案是否*在所提供的上下文范围内*与用户查询直接相关？
  - 如果未被要求，它是否避免了上下文中的切题信息？
  - **方法：** 人工审查。如果上下文中有真实“相关片段”，则可以调整标准信息检索指标，如准确率/召回率。
- **答案质量：**
  - 清晰度、简洁性、流畅性。
  - **方法：** 人工审查，可能使用李克特量表。

**比较分析：**
The最有效方式是显示改进的，是比较微调 (fine-tuning)模型与基础模型在同一组（查询、上下文）对上的输出。

| 评估方面 | 基础模型行为（示例） | 微调模型行为（示例） |
| --- | --- | --- |
| **忠实度** | 常引入外部知识或略微误解。 | 严格遵守所提供的文本。 |
| **幻觉** | 如果上下文稀疏或模棱两可，可能会编造细节。 | 更倾向于说明“无法回答”或谨慎地保持事实性。 |
| **相关性** | 可能过度总结或错过查询中的具体细节。 | 根据上下文更精确地针对查询。 |
| **风格/方面** | 通用语言。 | 采用微调数据中的术语/风格（如果存在）。 |

> 上表说明了潜在的改进。实际结果取决于数据质量、基础模型和调优情况。

对于大规模系统中的更系统化评估，可以考虑RAGAs等框架，它们提供忠实度、答案相关性和上下文相关性等指标。构建包含有挑战性RAG查询数据集的评估套件是一种最佳实践。

### 6. 集成到分布式RAG系统中

微调 (fine-tuning)后的模型（无论是基础模型+LoRA适配器，还是合并后的模型）需要部署到您的LLM服务基础设施中（例如，vLLM、TGI、SageMaker等）。

- **基础模型 + 适配器：** 许多服务框架支持在基础模型之上加载LoRA适配器。这很灵活，因为您可以使用相同的基本模型镜像为不同任务提供多个适配器服务，从而节省显存 (VRAM)。
- **合并模型：** 如果您`merge_and_unload()`，则将所得模型作为标准LLM部署。这有时可以提供略低的推断延迟，因为没有适配器逻辑开销，但您会失去轻松切换适配器的灵活性。

考虑您的MLOps流程将如何处理再训练和部署新的适配器版本。与完全微调相比，PEFT方法明显简化了这一过程，因为适配器文件很小（兆字节而非千兆字节）。

### 结论与进一步的考虑

本次动手练习展示了PEFT，特别是LoRA，如何成为一种强大工具，用于根据RAG系统的具体要求调整LLM。通过对强调上下文 (context)关联的数据进行微调 (fine-tuning)，您可以明显改进RAG系统回应的忠实度和相关性。

**专家实践者的后续步骤：**

- **大规模数据整理：** 开发用于生成、过滤和扩充高质量RAG微调数据的流程。这可能涉及主动学习或使用LLM本身生成候选（查询、上下文、答案）对。
- **高级PEFT技术：** 考察其他PEFT方法，例如(IA)^3、AdaLoRA，或将PEFT与量化 (quantization)（例如，如前所述的QLoRA）结合，以实现更高效率。
- **课程学习：** 从简单的RAG任务开始微调，并逐步增加复杂性。
- **迭代微调：** 持续监控生产中的RAG系统，并利用观察到的失效模式或可改进之处来创建新的微调数据集。
- **成本效益分析：** 虽然微调会增加前期成本，但答案质量的潜在改进、幻觉 (hallucination)的减少以及更好的用户满意度，在生产系统中可以抵消这些成本。

通过掌握这些技术，您可以构建高性能、可靠且高效的大规模分布式RAG系统，这些系统能够真正将LLM与外部知识结合使用。

## 参考资料

- [PEFT: Parameter-Efficient Fine-tuning of Foundation Models](https://huggingface.co/docs/peft/en/index) — Hugging Face Team (2023)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了应用LoRA及其他参数高效微调方法的指导和示例。

---

[上一节](06-%E5%A4%9ALLM%20RAG%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%99%BA%E8%83%BD%E8%B7%AF%E7%94%B1.md) · [下一节](../04-%E8%A7%84%E6%A8%A1%E5%8C%96%E6%95%B0%E6%8D%AE%E6%91%84%E5%85%A5%E4%B8%8E%E5%A4%84%E7%90%86%E7%AE%A1%E9%81%93/01-%E5%88%86%E5%B8%83%E5%BC%8F%E6%95%B0%E6%8D%AE%E6%91%84%E5%8F%96%E6%A1%86%E6%9E%B6.md)
