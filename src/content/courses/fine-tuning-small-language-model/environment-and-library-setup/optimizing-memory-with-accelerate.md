---
course: "fine-tuning-small-language-model"
chapter: "environment-and-library-setup"
lesson: "optimizing-memory-with-accelerate"
sourceId: 8510
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-3-environment-and-library-setup/optimizing-memory-with-accelerate"
title: "使用 Accelerate 优化显存"
description: "通过 Accelerate 库实现工作负载分配并高效管理显存使用。"
order: 4
plots: ["plots/8510-0.json"]
sourceHash: "504d1a6158650a66dd52ea04835324b10701fafc9c290714eda0ff89c5869be7"
sourceCorrections: []
---

训练即便是很小的语言模型也需要精细管理 GPU 显存。当你将模型、训练数据和优化器加载到显存（VRAM）中时，很快就会触及普通消费级硬件的极限。Hugging Face 的 Accelerate 库正是为了处理这一问题而设计的。它屏蔽了分布式处理和降低显存占用所需的工程细节，让你无需重写整个 PyTorch 训练循环。

在训练期间，模型权重 (weight)、优化器状态、梯度和前向激活值都会消耗内存。一个以 32 位浮点（FP32）格式存储的标准 70 亿参数 (parameter)模型，每个参数需要 4 字节。

$显存 = 7,000,000,000 \times 4 \text{ 字节} \approx 28 \text{ GB}$

这 28 GB 仅包含模型权重。像 AdamW 中使用的优化器状态以及梯度，很容易会让这一需求翻倍甚至增加两倍。为了在标准的 16 GB 或 24 GB GPU 上高效运行，必须采用 Accelerate 原生支持的显存优化技术。

### 混合精度训练

Accelerate 简化了混合精度训练的实现。混合精度不再使用 FP32 执行所有计算，而是在前向和后向传播中使用 16 位浮点数（FP16 或 BF16），同时在优化器步骤中保留一份 32 位的权重 (weight)主副本。这将激活值和梯度所需的显存 (VRAM)减半。



![70 亿参数模型的预估显存使用量](plots/8510-0.json)



> 70 亿参数 (parameter)语言模型在 32 位标准训练与 16 位混合精度训练下的显存需求对比。

虽然混合精度显著降低了显存占用，但 84 GB 对于消费级 GPU 来说仍然太大。这说明了为什么像 LoRA 这样在下一章将要配置的参数高效微调 (fine-tuning)技术是必不可少的。Accelerate 作为底层协调工具，使这些高级技术的实现成为可能。

### 将 Accelerate 集成到流水线中

在标准的 PyTorch 脚本中，你必须使用 `.to("cuda")` 手动将每个张量和模型移动到 GPU。当扩展到多个 GPU 或管理显存 (VRAM)限制时，这种手动分配设备的方式很容易出错。Accelerate 会自动处理设备分配。

你无需为设备映射编写自定义逻辑，只需将 PyTorch 组件封装在 `Accelerator` 对象中即可。

```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from torch.optim import AdamW

# 初始化 accelerator，使用 FP16 混合精度和梯度累加
accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=4)

# 加载基础模型和优化器
model = AutoModelForCausalLM.from_pretrained("your-small-model")
optimizer = AdamW(model.parameters(), lr=5e-5)
train_dataloader = get_custom_dataloader()

# 为分布式训练和显存管理准备所有对象
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
```

`accelerator.prepare()` 函数会自动检测可用的硬件环境。它会覆盖默认的 PyTorch 数据加载器，确保批次在传入模型之前被准确放置在正确的设备上。这种延迟加载可以防止因一次性向 GPU 移动过多数据批次而导致的显存溢出错误。

### 梯度累加

较小的显存 (VRAM)容量严格限制了可以处理的最大批次大小。批次大小为 1 可能刚好能装入内存，但较小的批次会导致梯度噪声过大且训练不稳定。梯度累加通过在执行单个优化步骤之前计算多个微批次的前向和后向传播，解决了这个数学难题。

当你将 `gradient_accumulation_steps` 参数 (parameter)传递给 `Accelerator` 时，它会在内部处理相关的计算。

$W_{t+1} = W_t - \eta \sum_{i=1}^{k} \nabla L_i$

在该方程中，$W_t$ 代表当前步骤的权重 (weight)，$\eta$ 是学习率，$k$ 是累加步数，而 $\nabla L_i$ 是特定微批次的损失梯度。

Accelerate 在后向传播期间累加梯度，并且只有在达到指定的微批次数量时才执行 `optimizer.step()`。你不需要手动除以损失值或编写嵌套循环来管理累加步骤。该库封装了这些操作，使你的训练循环保持简洁且高度优化。

## 参考资料

- [Accelerate: Training and Inference at Scale Made Simple, Efficient and Adaptable](https://github.com/huggingface/accelerate) — Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, Benjamin Bossan (2022)
  Publisher: Hugging Face
  Accelerate 库的官方介绍，解释了其分布式训练和内存效率的设计理念。
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍混合精度训练方法的开创性研究论文，旨在减少内存占用并提高计算速度。
- [Performance and Scalability: Training on a Single GPU](https://huggingface.co/docs/transformers/v4.40.1/en/perf_train_gpu_one) — Hugging Face (2024)
  Publisher: Hugging Face
  关于优化显存占用实用方法的文档，涵盖了梯度累积和混合精度等技术细节。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, Thomas Viehmann (2020)
  Publisher: Manning Publications; Pages: 350-385
  详细解释了 PyTorch 如何管理张量和显存，这对于理解 Accelerate 的底层运行机制非常有用。
