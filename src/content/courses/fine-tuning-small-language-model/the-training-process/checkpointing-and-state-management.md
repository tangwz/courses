---
course: "fine-tuning-small-language-model"
chapter: "the-training-process"
lesson: "checkpointing-and-state-management"
sourceId: 8519
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-5-the-training-process/checkpointing-and-state-management"
title: "检查点与状态管理"
description: "定期保存模型检查点，以防止数据丢失并支持恢复训练。"
order: 3
plots: []
sourceHash: "55f089927354153f3d80e0e2a8238d215abb539df3e517c8e49b5b051ed90ffc"
sourceCorrections: []
---

训练语言模型是一个计算密集型的过程，可能需要运行数小时甚至数天。硬件故障、内存限制或进程中断都可能导致执行意外停止。定期保存进度可以防止数据丢失，并允许你从已知的良好状态恢复。这种机制被称为检查点（checkpointing）。

在进行标准的监督学习 (supervised learning)时，模型检查点通常包含整套模型权重 (weight)。然而，由于你使用的是 LoRA 等参数 (parameter)高效微调 (fine-tuning)（PEFT）技术，检查点的行为会有所不同。它不会复制庞大的基础模型，而只存储更新后的 LoRA 适配器权重。这显著减少了保存模型所需的时间和存储空间。

一个完整的检查点包含的内容不仅是适配器权重。它保留了特定时间点训练环境的准确状态。这包括优化器状态、学习率调度器状态、当前的训练步数以及随机数生成器（RNG）的状态。

保存这些额外元素可以确保在需要恢复训练时，数学计算能像未发生中断一样继续进行。例如，AdamW 等优化算法会维护梯度的移动平均值。如果这些动量缓存丢失，优化器必须重新开始，这可能会暂时使训练过程不稳定并改变收敛路径。

> 微调过程中存储在训练检查点内的组件。

### 配置检查点

Hugging Face 的 Trainer 类会自动处理状态管理。这种行为由你之前定义的训练参数 (parameter)中的特定项控制。

最主要的设置是保存策略，它决定了检查点的频率。你可以将策略配置为在每个 epoch 结束时保存、按特定步数保存或完全不保存。对于微调 (fine-tuning)小型语言模型，通常建议将策略设置为按步数（steps）保存，因为有时一个 epoch 需要很长时间才能完成。如果故障发生在 epoch 结束前，而你没有使用基于步数的检查点，你将损失大量的计算时间。

如果你选择基于步数的策略，则必须定义保存步数参数，以告知脚本在写入磁盘之前需要执行多少次前向和反向传播 (backpropagation)。

### 存储限制与清理

虽然 PEFT 适配器相对较小，但优化器状态却很大。适配器中的每个可训练参数 (parameter)都需要在优化器中跟踪数值。如果你的 LoRA 配置产生了数量为 $P$ 的可训练参数，AdamW 优化器需要 $2 \times P$ 个参数来存储其移动平均值。使用 32 位浮点精度，你可以使用简单的公式计算所需的存储空间（以字节为单位）：

$M_{\text{优化器}} = 2 \times P \times 4 \text{ 字节}$

如果在漫长的训练运行中每 100 步保存一个新检查点，很快就会耗尽本地磁盘空间。为了有效管理存储，你应该配置总数限制。设置限制会约束硬盘上保留的检查点最大数量。一旦达到限制，Trainer 会在写入新检查点之前自动删除最旧的检查点目录。通常的做法是仅保留最近的两到三个检查点。

### 从检查点恢复

当训练脚本因内存不足错误或远程会话断开而崩溃时，状态管理就体现了其价值。Trainer 允许立即恢复优化循环。

通过向训练方法传递布尔标志或特定的目录路径，脚本将定位最新的检查点目录。然后，它会将 LoRA 权重 (weight)加载到基础模型中，恢复优化器和调度器状态，并自动将数据加载器快速推进到正确的批次。随后，训练循环将从中断的那一步继续运行。

保存中间检查点还允许你之后评估模型的不同阶段。有时模型在训练循环早期就达到了性能峰值，并随着训练的继续开始对指令数据集产生过拟合 (overfitting)。通过保留多个检查点，你可以加载较早迭代的权重并对其进行测试，以找到对新输入泛化效果最好的版本。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  LoRA 的开创性论文，解释了在参数高效微调过程中为何仅需保存适配器权重。
- [Trainer Documentation - Checkpointing](https://huggingface.co/docs/transformers/main_classes/trainer) — Hugging Face Team (2024)
  Publisher: Hugging Face
  官方技术文档，介绍了如何使用 Transformers 库配置保存策略、管理存储限制以及恢复训练。
- [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations
  介绍了 AdamW 优化器，并解释了维持动量等优化器状态对于稳定收敛的重要性。
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://github.com/huggingface/peft) — Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul (2022)
  Journal: Hugging Face Blog
  关于 PEFT 如何管理模型状态以及相比全量微调如何降低存储需求的权威技术指南。
