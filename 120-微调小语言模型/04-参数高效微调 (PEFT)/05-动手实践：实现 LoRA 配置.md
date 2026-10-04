# 动手实践：实现 LoRA 配置

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-4-parameter-efficient-fine-tuning-peft/hands-on-practical-implementing-lora)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将低秩自适应（Low-Rank Adaptation）的数学原理转化为可运行的 Python 代码，需要使用 Hugging Face 的 `peft` 和 `transformers` 库。这包括加载基础语言模型、应用 4 位量化 (quantization)，并将 LoRA 适配器附加到特定的神经网络 (neural network)层。这一过程为训练模型做好准备，同时将显存 (VRAM)消耗严格控制在消费级硬件的限制范围内。

### 安装所需的库

在编写实现脚本之前，请确保您的环境已安装必要的库。该实现依赖于用于加载模型的 `transformers`、用于量化 (quantization)的 `bitsandbytes` 以及用于应用适配器的 `peft`。

```python
# PEFT 配置的标准导入
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
```

### 使用 QLoRA 加载基础模型

为了最大化内存效率，您应该以 4 位精度加载基础模型。这会显著减少冻结的基础模型权重 (weight)的内存占用。您可以使用 `BitsAndBytesConfig` 类来配置此行为。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

在此配置中，`load_in_4bit` 指示库将权重压缩为 4 位整数。`bnb_4bit_quant_type` 设置为 `nf4`，代表 NormalFloat 4-bit，这是一种适用于正态分布模型权重的优化数据类型。`bnb_4bit_compute_dtype` 规定虽然权重以 4 位存储，但在前向和反向传播 (backpropagation)过程中的实际数学运算将以 `bfloat16` 进行，以保持数值稳定性。

定义好配置后，您可以将模型和分词 (tokenization)器 (tokenizer)加载到内存中。

```python
model_id = "your-chosen-slm-path"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 为参数 (parameter)高效训练准备模型

以 4 位精度加载的模型不能直接进行适配器注入。您必须对其进行处理以确保训练循环保持稳定。`peft` 库提供了一个名为 `prepare_model_for_kbit_training` 的实用函数。

该函数执行必要的结构调整。它将某些归一化 (normalization)层转换为 `float32` 以获得更好的稳定性，并使模型准备好使用梯度检查点（gradient checkpointing）。梯度检查点通过在反向传播 (backpropagation)期间重新计算某些激活值，而不是将它们全部存储在内存中，从而用计算量换取内存空间。

```python
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

### 定义 LoRA 配置

基础模型准备就绪后，下一步是使用 `LoraConfig` 对象定义低秩自适应的结构参数 (parameter)。这定义了新引入的权重 (weight)的形状以及它们的注入位置。

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

每个参数都决定了微调 (fine-tuning)数学更新的具体方面：

- `r`：低秩矩阵的秩。秩为 8 意味着更新矩阵相对较小。增加 $r$ 会为模型提供更多可训练参数和更强的学习新模式的能力，但也会增加内存使用量。
- `lora_alpha`：LoRA 权重的缩放因子。标准做法是将 $\alpha$ 设置为 $r$ 值的两倍。
- `target_modules`：指定哪些 Transformer 层接收适配器。针对自注意力 (self-attention)机制 (attention mechanism)中的查询投影（`q_proj`）和值投影（`v_proj`）是一个标准基准，通常能产生不错的效果。
- `lora_dropout`：引入 5% 的概率在训练步骤中忽略某个适配器权重。这起到正则化 (regularization)的作用，防止在小型指令数据集上产生过拟合 (overfitting)。
- `task_type`：指定任务目标。对于文本生成，这设置为因果语言建模。

> 将标准预训练 (pre-training)语言模型转换为适用于监督训练的内存高效型 PEFT 模型所需的操作序列。

### 应用适配器

最后一步是将加载的模型与定义的配置合并。`get_peft_model` 函数接收庞大的冻结模型，并将微小的、可训练的适配器矩阵拼接到指定的目标模块中。

```python
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

运行 `print_trainable_parameters()` 将直接在控制台输出摘要。对于一个拥有 70 亿个参数 (parameter)的普通小型语言模型，对注意力投影应用秩为 8 的配置通常会产生大约 400 万到 1000 万个可训练参数。这意味着更新的模型权重 (weight)不足总量的 0.2%。优化器现在只需跟踪这一小部分网络的优化状态，使整个训练流水线能够轻松装入消费级显存 (VRAM)中。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍 LoRA 方法的原始研究论文，为秩分解矩阵提供了数学基础。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems; Volume: 36; Pages: 10088-10115; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  4比特量化（NF4）的奠基性论文，介绍了代码实现中 bitsandbytes 与 PEFT 的集成方式。
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://github.com/huggingface/peft) — Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, Benjamin Bossan (2022)
  Journal: Hugging Face Documentation; Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，涵盖了 LoraConfig 和 get_peft_model 的具体实现。

---

[上一节](04-%E9%85%8D%E7%BD%AE%E7%9B%AE%E6%A0%87%E6%A8%A1%E5%9D%97%E4%B8%8E%E7%A7%A9.md) · [下一节](../05-%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B/01-%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E4%B8%8E%E8%B6%85%E5%8F%82%E6%95%B0.md)
