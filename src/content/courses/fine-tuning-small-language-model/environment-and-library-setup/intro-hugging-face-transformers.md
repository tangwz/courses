---
course: "fine-tuning-small-language-model"
chapter: "environment-and-library-setup"
lesson: "intro-hugging-face-transformers"
sourceId: 8508
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-3-environment-and-library-setup/intro-hugging-face-transformers"
title: "Hugging Face Transformers 库简介"
description: "回顾用于加载和训练模型的 Hugging Face Transformers 库的核心组件。"
order: 2
plots: []
sourceHash: "9a71f64a2c821380268b1a3f69b8ec6489dbabf0dd0743a8a76b3342d171e183"
sourceCorrections: []
---

集成 Hugging Face Transformers 库是准备本地训练环境的首要步骤。虽然 PyTorch 提供了机器学习 (machine learning)所需的张量操作和自动微分，但从零开始编写现代 Transformer 模型效率极低。手动编写多头注意力 (multi-head attention)机制 (attention mechanism)、层归一化 (normalization)和权重 (weight)初始化会带来巨大的额外工作量。Transformers 库充当了 PyTorch 之上的高级 API，它标准化了加载、交互和修改先进架构的过程，无需手动定义每个神经网络 (neural network)层。

在处理小语言模型时，你会经常使用该库提供的 AutoClasses。这些类旨在从指定的仓库名称或本地目录中自动推断正确的模型架构和分词 (tokenization)策略。你将配置的两个核心部分是分词器（tokenizer）和模型本身。

语言模型无法直接处理原始文本字符串，它们需要语言的数值表示来执行数学运算。`AutoTokenizer` 类负责将文本字符串转换为称为 token ID 的整数序列。分词器管理底层模型架构所需的特定格式规则，包括添加特殊 token 以标记 (token)序列的开始，或将用户提示与助手回答分开。它还会生成注意力掩码（attention mask），这是一个由 1 和 0 组成的辅助张量，用于告知模型哪些 token 包含实际数据，哪些是填充（padding）。

`AutoModelForCausalLM` 类将定义的神经网络权重以 PyTorch 张量的形式直接加载到机器内存中。“CausalLM”（因果语言模型）这一名称指明了模型的目标。因果语言建模涉及完全基于之前的 token 来预测序列中的下一个 token，从而防止模型提前查看后续上下文 (context)。在数学上，模型根据所有先前 token 的上下文计算下一个 token $x_t$ 的概率分布：

$P(x_t | x_1, x_2, ..., x_{t-1})$

> 使用 Transformers 库组件进行文本处理和推理 (inference)的流水线。

将模型直接加载到 RAM 或 VRAM 中时，需要密切注意数据类型，以防止内存溢出错误。默认情况下，许多预训练 (pre-training)模型以 32 位浮点精度（$FP32$）定义权重。对于一个拥有 20 亿参数 (parameter)的小语言模型，这种标准精度仅存储权重就需要大约 8 GB 的内存。这一计算尚未考虑训练激活值、梯度和优化器状态所需的额外内存开销。

你可以在 Transformers 库中通过在模型初始化阶段指定较低精度的数据类型来原生管理这一点。通过以 16 位浮点（$FP16$）或 16 位脑浮点（$BF16$）加载权重，你可以在保持同等性能的同时，立即将内存需求减半。

在 Python 脚本中集成这些组件通常如下所示：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "your-chosen-slm-path"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

在此实现中，`torch_dtype=torch.bfloat16` 强制 PyTorch 张量以内存高效的格式加载。`device_map="auto"` 参数是与 Accelerate 库的集成，它会自动评估你的硬件并优化分配模型层。如果你有独立显卡，它会将模型层放在显存（VRAM）中。如果模型超过显存容量，它会将剩余层分配给系统内存（RAM）。

虽然推理完全依赖于这种前向传播，但有监督微调 (fine-tuning)需要跟踪损失梯度并更新这些张量。由分词器和模型建立的流水线为后续的数据格式化和训练循环提供了必要的基础。

## 参考资料

- [HuggingFace's Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6) — Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, Alexander M. Rush (2020)
  Journal: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations; Pages: 38–45; DOI: [10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
  介绍该库的设计思路与技术架构的主要学术论文。
- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 5998–6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  Transformer 架构的奠基性论文，解释了注意力机制与序列建模原理。
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  解释了使用低精度数据类型以减少训练期间显存消耗的方法论。
- [Accelerate: Training and Inference at Scale Made Simple, Efficient and Adaptable](https://github.com/huggingface/accelerate) — Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, Benjamin Bossan (2022)
  Publisher: Hugging Face
  该库的文档支持了 device_map 功能，用于高效的硬件资源分配。
