# 合并 PEFT 适配器

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/merging-peft-adapters)

[返回章节目录](README.md) · [返回课程目录](../README.md)

参数 (parameter)高效微调 (fine-tuning)（PEFT）技术，例如 LoRA、QLoRA 和适配器，在训练阶段提供了很大的好处。与更新所有模型权重 (weight)相比，它们大幅减少了可训练参数的数量，降低了内存需求，并通常加快了微调过程。然而，这种效率在推理 (inference)时会带来轻微的代价。PEFT 方法通常需要加载原始基础模型权重*以及*独立的适配器权重。此外，前向传播涉及额外的计算，以合并基础层和适配器层的输出。

对于以推理延迟和吞吐量 (throughput)为主要考虑因素的部署情况，将学到的适应内容整合回基础模型的权重中通常是有益的。这个过程称为合并，它会创建一组单一的模型权重，这组权重表现得像一个传统微调过的模型，但包含了通过 PEFT 学到的适应内容。

### 为什么要合并 PEFT 适配器？

合并的主要原因是为了优化推理 (inference)性能和简化部署：

1. **降低计算开销：** 在使用未合并的 PEFT 模型（特别是 LoRA）进行推理时，每个适配层都需要计算原始权重 (weight)矩阵的输出和低秩适配器矩阵的输出，然后将这些结果求和。合并会预先计算这个和，将操作合并为每个层使用新的、合并后的权重进行一次矩阵乘法。这直接带来更低的延迟。
2. **简化部署文件：** 合并不再管理两组权重（基础模型 + 适配器），而是生成单个模型检查点。这简化了模型的打包、加载和提供，因为标准推理框架可以加载合并后的模型，而无需在推理时进行修改时特定 PEFT 库的集成。
3. **可能的兼容性提升：** 一些推理优化工具或服务框架可能主要为标准模型架构设计。合并后的模型表现为一个标准模型，可能实现与更广泛的部署工具和方法（例如某些类型的量化 (quantization)或编译）的兼容性，这些工具和方法可能不直接支持动态适配器加载。

### 合并的机制（以 LoRA 为例）

我们来查看低秩适应（LoRA），这是合并所应用于的最常见的 PEFT 方法之一。在 LoRA 中，预训练 (pre-training)权重 (weight)矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 保持冻结。适应是通过两个低秩矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 学习的，其中秩 $r \ll \min(d, k)$。修改后的前向传播为输入 $x$ 计算输出 $h$ 如下：


$$
h = xW_0 + x \alpha \frac{BA}{r}
$$


这里，$\alpha$ 是一个缩放因子，$r$ 是秩（尽管有时缩放 $\alpha/r$ 会被吸收到权重中）。

合并涉及计算一个新的权重矩阵 $W_{merged}$，它直接包含了适应内容：


$$
W_{merged} = W_0 + \alpha \frac{BA}{r}
$$


一旦 $W_{merged}$ 被计算出来，它就会替代 $W_0$。独立的 $B$ 和 $A$ 矩阵在推理 (inference)时不再需要。前向传播简单地变为：


$$
h = xW_{merged}
$$


对所有应用了 LoRA 适配器的层都执行此计算。结果是一个模型，它具有与基础模型相同的架构，但权重已修改。

> 图示说明了 LoRA 适配器合并前后的适配层的计算流程。合并简化了推理路径。

### 实现适配器合并

像 Hugging Face 的 `PEFT` 这样的库提供了直接的方法来执行此合并操作。通常，您会加载基础模型，然后使用 `PeftModel` 类在其之上加载适配器权重 (weight)。这个类通常包含一个类似 `merge_and_unload` 的函数。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 定义模型名称和路径
base_model_id = "meta-llama/Llama-2-7b-hf"
adapter_model_id = "path/to/your/trained-lora-adapter" # 替换为您的适配器路径
merged_model_save_path = "./merged-llama-model"

# 设置设备（如果可用，使用 CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载基础模型
print(f"正在加载基础模型: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16, # 如果适用，使用 float16 以提高内存效率
    device_map='auto' # 如有需要，自动分配模型层
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 在基础模型之上加载 PEFT 模型（适配器）
print(f"正在加载适配器: {adapter_model_id}")
model = PeftModel.from_pretrained(base_model, adapter_model_id)
print("PEFT 模型已加载。")

# 将适配器权重合并到基础模型中
print("正在合并适配器权重...")
model = model.merge_and_unload()
print("适配器已合并并卸载。")

# 'model' 变量现在包含合并后的模型。
# 这是一个标准的 Transformers 模型对象。

# 您现在可以保存合并后的模型以备后用
print(f"正在保存合并后的模型至: {merged_model_save_path}")
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)
print("合并后的模型已保存。")

# 您现在可以直接使用 'model' 进行推理或进一步处理
# 示例：生成文本
# prompt = "马来西亚的首都是什么？"
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

执行 `merge_and_unload()` 后，`model` 对象不再是 `PeftModel`，而是变回基础模型类（例如 `LlamaForCausalLM`），现在它包含更新后的权重。适配器层被移除，从架构角度看，模型表现得像一个标准的、完全微调 (fine-tuning)过的模型。

### 考量与权衡

尽管合并提供了性能优势，但理解其影响非常重要：

- **模块性丧失：** PEFT 在开发阶段的主要优势是能够轻松地在同一个基础模型上替换或堆叠不同的适配器。一旦合并，这种灵活性就会丧失。合并后的模型代表单一的适应状态。如果您以后需要尝试不同的适配器，您通常会恢复到基础模型并加载所需的适配器，而不是尝试“解除合并”。
- **与量化 (quantization)的相互影响：** 合并通常在训练后量化（PTQ）之前。您将 LoRA 权重 (weight)（通常是 $fp16$ 或 $fp32$ 等更高精度）合并到基础模型权重中，然后对得到的合并模型应用 GPTQ 或 AWQ 等量化技术，以减小其大小并可能进一步提高推理 (inference)速度。对于像 QLoRA 这样将量化集成到训练中的方法，该过程可能涉及在合并前对适配器权重进行反量化，将其合并到（可能也已反量化的）基础权重中，然后对最终模型进行重新量化。具体情况取决于具体的 QLoRA 实现和后续优化步骤。
- **模型大小：** 合并后的模型将具有与原始基础模型相同的参数 (parameter)数量，但这些参数的值已改变。合并后的模型检查点在磁盘上的大小将类似于基础模型的检查点大小，明显大于 PEFT 训练期间使用的小适配器权重文件。其好处主要是运行时性能和部署简便性，而不是与基础模型相比的磁盘空间减少。

合并 PEFT 适配器是模型训练和测试向优化部署转变中的一个实际步骤。通过将学到的修改整合到主要模型权重中，您可以精简推理过程，降低计算延迟，并简化提供您微调 (fine-tuning)过的大型语言模型服务的操作方面。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA参数高效微调的原始方法，是理解适配器技术的基石。
- [PEFT (Parameter-Efficient Fine-Tuning) Library Documentation - PeftModel](https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.merge_and_unload) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，包含`merge_and_unload`函数用于适配器合并的详细信息。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA，一种微调量化大型语言模型的方法，适用于高级PEFT和量化场景。

---

[上一节](04-%E5%BE%AE%E8%B0%83%E5%90%8E%E4%BC%98%E5%8C%96%EF%BC%9A%E5%89%AA%E6%9E%9D.md) · [下一节](06-%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E4%B8%8E%E6%89%93%E5%8C%85.md)
