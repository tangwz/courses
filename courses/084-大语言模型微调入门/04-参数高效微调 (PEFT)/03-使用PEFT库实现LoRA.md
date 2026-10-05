# 使用PEFT库实现LoRA

来源：[原文](https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/implementing-lora-with-peft-library)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低秩适应（LoRA）可以应用于实际模型，使用Hugging Face的`PEFT`（参数 (parameter)高效微调 (fine-tuning)）库。该库提供了一个高级API，简化了将LoRA适配器注入现有Transformer模型的过程，只需几行代码。它处理修改模型架构和冻结相应权重 (weight)的复杂工作，让您能够专注于配置训练。

### `PEFT`库的工作流程

使用`PEFT`的核心思路是：取一个预训练 (pre-training)的基础模型，为您的LoRA适配器定义配置，然后将它们结合起来创建一个新的、可训练的模型。这个新模型保持原始权重 (weight)冻结，只训练注入的小型适配器层。

此过程包含三个主要步骤：

1. **加载基础模型：** 从Hugging Face Hub加载一个标准预训练模型。
2. **定义LoRA配置：** 创建一个`LoraConfig`对象，以指定适配器层的超参数 (parameter) (hyperparameter)，例如它们的秩（`r`）以及基础模型的哪些部分需要修改。
3. **创建`PeftModel`：** 使用`get_peft_model`函数，根据您的配置将LoRA适配器封装到基础模型中。

我们来逐一详细说明每个步骤。

### 使用`LoraConfig`配置LoRA适配器

`LoraConfig`类是您LoRA实现的核心控制部分。它允许您定义适配器层的所有必要参数 (parameter)。

以下是您将使用的最主要参数：

- `r`：这个整数表示低秩更新矩阵（$A$ 和 $B$）的秩。它直接控制可训练参数的数量。较小的`r`会导致更少的参数和更快的训练，但可能捕获较少的任务特定信息。较大的`r`会以更多参数为代价增加模型容量。`r`的常见取值范围是4到64。
- `lora_alpha`：这是LoRA激活的缩放因子，作用类似于适配器的学习率。LoRA更新按`lora_alpha / r`进行缩放，因此调整此值会影响适配器所做更改的幅度。常见做法是将`lora_alpha`设置为`r`值的两倍。
- `target_modules`：一个字符串列表，指定基础模型架构中哪些模块应用LoRA。对于Transformer模型，这通常是注意力机制 (attention mechanism)中的线性层，例如`query`、`key`和`value`。例如，`["q_proj", "v_proj"]`。确定正确的模块名称需要检查基础模型的架构。
- `lora_dropout`：一个浮点值，表示应用于LoRA层的Dropout概率。这作为一种正则化 (regularization)技术，以防止适配器权重 (weight)过拟合 (overfitting)。
- `task_type`：指定您正在微调 (fine-tuning)的任务类型。例如，对于文本生成模型，您可以将其设置为`TaskType.CAUSAL_LM`。这有助于`PEFT`库正确配置模型的正向传播。

```python
from peft import LoraConfig, TaskType

# 因果语言模型的LoRA配置示例
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 创建`PeftModel`

一旦您加载了基础模型并定义了`LoraConfig`，您可以使用`get_peft_model`函数将它们结合起来。该函数将基础模型和配置作为输入，并返回一个`PeftModel`对象。返回的模型已准备好进行训练，所有基础模型权重 (weight)都被冻结，只有新的LoRA适配器权重被标记 (token)为可训练。

下图说明了此工作流程。

> `get_peft_model`函数接收一个冻结的基础模型和一个`LoraConfig`来生成一个`PeftModel`，其中只有注入的小型LoRA适配器矩阵是可训练的。

### 一个实际的代码示例

我们将通过一个完整的示例来演示如何为LoRA微调 (fine-tuning)设置模型。我们将使用`meta-llama/Llama-2-7b-chat-hf`模型作为基础，但同样的原理适用于Hugging Face Hub上的任何Transformer模型。

首先，请确保您已安装必要的库：

```bash
pip install torch transformers peft accelerate bitsandbytes
```

现在，我们来编写准备模型的Python代码。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 定义模型ID并加载分词器
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 配置量化以4比特加载模型
# 这是一种节省内存的技术，我们将在QLoRA部分做进一步介绍
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 3. 加载量化的基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto" # 自动将层映射到可用硬件
)

# 4. 为k比特训练准备模型（可选但推荐）
base_model = prepare_model_for_kbit_training(base_model)

# 5. 定义LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 6. 创建PeftModel
peft_model = get_peft_model(base_model, lora_config)

# 7. 打印可训练参数的百分比
def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(peft_model)
```

运行此代码时，`print_trainable_parameters`的输出将体现PEFT的效能。对于一个70亿参数 (parameter)的模型，输出会类似这样：

```
trainable params: 8,388,608 || all params: 3,508,801,536 || trainable%: 0.24
```

这是LoRA的主要优势。我们已经成功地为微调准备了一个70亿参数的模型，而只需训练总参数的**不到0.25%**。优化器状态和梯度的内存需求大幅减少，使得在单个消费级GPU上运行训练过程成为可能。

一旦您有了`PeftModel`，您就可以像进行完整微调一样，将其与标准的Hugging Face `Trainer` API一起使用。`Trainer`会自动处理只针对可训练LoRA参数的梯度更新。这小组适配器权重 (weight)可以独立于大型基础模型进行保存和加载，从而方便管理同一基础模型的多个专门版本。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍低秩适配（LoRA）的原始论文，详细阐述其理论基础和参数高效微调机制。
- [Parameter-Efficient Fine-Tuning (PEFT) Library Documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，提供实现 LoRA 和其他 PEFT 方法的详细指南及 API 参考。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了 QLoRA，一种使用 4 比特量化的高效微调方法，有助于理解本节中讨论的内存节省技术。
- [Hugging Face Transformers Library Documentation](https://huggingface.co/docs/transformers/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers 库的官方文档，提供了加载和使用预训练模型及分词器的必要信息，这是应用 PEFT 的先决条件。

---

[上一节](02-%E4%BD%8E%E7%A7%A9%E9%80%82%E5%BA%94%EF%BC%88LoRA%EF%BC%89%EF%BC%9A%E5%8E%9F%E7%90%86%E4%B8%8E%E8%BF%90%E4%BD%9C.md) · [下一节](04-%E9%87%8F%E5%8C%96%E5%8F%8A%E5%85%B6%E5%AF%B9%E5%BE%AE%E8%B0%83%E7%9A%84%E5%BD%B1%E5%93%8D%20%28QLoRA%29.md)
