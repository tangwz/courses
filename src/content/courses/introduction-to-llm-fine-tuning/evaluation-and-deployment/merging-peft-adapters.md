---
course: "introduction-to-llm-fine-tuning"
chapter: "evaluation-and-deployment"
lesson: "merging-peft-adapters"
sourceId: 7292
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-5-evaluation-and-deployment/merging-peft-adapters"
title: "将适配器与基础模型合并的方法"
description: "了解如何将训练好的 PEFT 适配器（如 LoRA）合并到基础模型权重中，以实现独立部署。"
order: 5
plots: []
sourceHash: "40c3a05ad12a21722565e93b671564f3caa82a4a6f2428961c679e6a30372d52"
sourceCorrections: []
---

使用 LoRA 等参数 (parameter)高效技术微调 (fine-tuning)模型后，您会得到两组不同的权重 (weight)：原始的、冻结的基础模型和小型、特定任务的适配器权重。尽管这种分离对于训练非常高效，并允许您为单个基础模型管理多个适配器，但它在推理 (inference)期间会引入少量开销，并可能使部署流程变得复杂。对于希望有一个单一、优化的模型的生产环境来说，将适配器权重直接合并到基础模型中是一种标准且推荐的做法。

这个过程将这两个部分组合成一个独立的模型文件。结果模型行为与完全微调的模型相同，但创建所需的计算量要少得多。合并简化了部署，因为您不再需要 PEFT 库或特殊逻辑来在运行时组合权重，并且它可以适度提升推理延迟。

### 合并的原理

合并的主要好处在于简化了前向传播计算。如果不合并，带有 LoRA 的层的输出是通过将基础模型的权重 (weight)输出添加到适配器的低秩矩阵输出中来计算的。

对于给定的输入 $x$，计算公式为：


$$
\text{输出} = W_{base}x + W_B W_A x
$$


其中，$W_{base}$ 是原始权重矩阵，而 $W_A$ 和 $W_B$ 是低秩适配器矩阵。这需要两条单独的矩阵乘法路径，然后进行求和。

通过合并，您可以预先计算一个新的统一权重矩阵 $W_{merged}$：


$$
W_{merged} = W_{base} + W_B W_A
$$


这个计算只需离线执行一次。部署模型的正向传播就变成了一次更高效的矩阵乘法：


$$
\text{输出} = W_{merged}x
$$


这带来了两个主要优势：

1. **部署简便性**：合并后的模型是标准的 `transformers` 模型。它可以使用通用推理 (inference)工具和处理程序加载和提供服务，无需将 `peft` 库作为依赖。这降低了生产环境的复杂程度。
2. **推理性能**：通过消除每个适配器层的额外矩阵乘法和求和，您可以减少计算开销。虽然对于单个推理调用，影响可能很小，但在大规模应用时会变得很明显，从而降低延迟并提高吞吐量 (throughput)。

> 合并过程将分离的基础模型和适配器权重转换为单个可部署的模型文件。

### 使用 PEFT 库合并适配器

Hugging Face PEFT 库通过 `merge_and_unload()` 方法使这个过程变得简单直接。这个函数处理权重 (weight)计算并返回一个标准的 `transformers` 模型对象。

让我们来看一下代码。首先，您加载基础模型，然后将训练好的适配器权重应用到它上面，创建一个 `PeftModel`。

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义基础模型和训练好的适配器的路径
base_model_id = "mistralai/Mistral-7B-v0.1"
adapter_path = "./outputs/mistral-lora-finetuned"

# 以4位模式加载基础模型，以在加载过程中节省内存
# 合并过程会将其反量化回原始精度
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    device_map="auto",
)

# 通过将适配器应用到基础模型来加载 PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path)
```

现在，`model` 对象是一个 `PeftModel`，它内部管理基础和适配器权重。要组合它们，您只需调用 `merge_and_unload()`。

```python
# 将适配器层合并到基础模型中
merged_model = model.merge_and_unload()
```

`merged_model` 对象现在是一个标准的 `AutoModelForCausalLM` 实例，而不是 `PeftModel`。LoRA 层已被标准的 `Linear` 层替换，其中包含新的、组合后的权重。如果您使用量化 (quantization)加载了基础模型（例如，`load_in_4bit=True`），这个方法也会对模型进行反量化，将其恢复到原始精度（如 float16 或 bfloat16），这对于权重合并是必需的。

### 保存合并模型用于推理 (inference)

模型合并后，您可以像保存任何其他 `transformers` 模型一样，使用 `save_pretrained` 方法来保存它。同样需要将对应的分词 (tokenization)器 (tokenizer)保存到同一目录，以创建一个自包含的模型文件。

```python
# 定义保存合并模型的路径
merged_model_path = "./models/mistral-7b-finetuned-merged"

# 保存合并模型
merged_model.save_pretrained(merged_model_path)

# 您也必须保存分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(merged_model_path)
```

生成的目录 `mistral-7b-finetuned-merged` 现在包含所有必要的文件（`config.json`、`model.safetensors`、`tokenizer.json` 等），用于标准的 Hugging Face 模型。您可以加载并用于推理，而无需引用 PEFT 库。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从保存的目录加载合并模型
model = AutoModelForCausalLM.from_pretrained(merged_model_path)
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# 模型现在可以用于标准推理了
# ...
```

### 合并的考量与不适用场景

合并适配器是在部署专门模型之前的最后一步，但它在实践中不可逆。以下是一些需要注意的方面：

- **丧失灵活性**：合并后，适配器被吸收到基础模型中。您不能再轻易禁用它或替换为其他适配器。如果您预计进行更多实验或需要训练其他适配器，明智的做法是保留原始的基础模型和适配器文件。
- **存储**：LoRA 在开发阶段的主要好处是存储效率。您可以有一个大型基础模型和许多小型（兆字节大小）适配器。合并后的模型与完全微调 (fine-tuning)模型具有相同的存储占用。为了部署简便性和性能，牺牲灵活性和存储空间通常在生产环境中可以接受。
- **动态适配器切换**：如果您的应用程序需要动态地在许多不同适配器之间即时切换（例如，在每个用户都有个性化模型的多租户服务中），您*不*应合并。在这种场景下，预期的架构是将基础模型保留在内存中，并按需加载合适的轻量级适配器。

总而言之，合并 PEFT 适配器是使微调模型投入运行的重要一步。它将您的工作打包成便携、高效且易于部署的格式，有效地衔接了参数 (parameter)高效训练与生产就绪推理 (inference)之间的环节。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了原始的LoRA方法，详细阐述了其数学公式和参数效率，为合并策略提供了基础。
- [Hugging Face PEFT Library Documentation](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Publisher: Hugging Face
  提供了Hugging Face PEFT库的官方文档，包括使用`merge_and_unload()`函数合并适配器权重的实用指南。
- [Hugging Face Transformers Library Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face `transformers`库的官方文档，说明了如何保存和加载标准模型，这适用于PEFT适配器合并后的模型。
- [The Hugging Face Course](https://huggingface.co/course/chapter0/1) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face提供的综合在线课程，涵盖了大型语言模型训练、微调和部署的各个方面，为生产就绪提供了背景。
