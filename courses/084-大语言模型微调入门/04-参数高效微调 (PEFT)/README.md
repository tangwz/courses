# 第 4 章：参数高效微调 (PEFT)

来源：[原章节](https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft)

[返回课程目录](../README.md)

上一章讲解了全参数微调，这种办法会更新模型中的每个权重。这种办法虽然有效，但随着模型参数量达到数十亿，其计算成本变得难以承受。对GPU内存和计算能力的高要求，使全参数微调在许多实际应用中无法实现。

本章介绍参数高效微调 (PEFT)，这是一系列旨在以显著更少计算资源调整大型模型的技术。我们将重点介绍低秩适应 (LoRA)，它会冻结原始模型权重并注入更小、可训练的矩阵。LoRA并非更新庞大的原始权重矩阵 $W_0$，而是学习一个低秩更新，$$ \Delta W = BA $$，其中 $A$ 和 $B$ 中的参数数量远小于 $W_0$。你将学习如何使用Hugging Face的`PEFT`库来实现这一点。

我们还将考察量化如何进一步减少内存占用，从而引出像QLoRA这样的方法。本章最后将进行PEFT与全参数微调的比较分析，明确两者在性能和资源使用上的权衡。完成本章后，你将掌握实用技能，能够在消费级硬件上将现代、高效的微调方法应用于超大型模型。

## 小节

- 1. [参数高效微调概述](01-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A6%82%E8%BF%B0.md)
- 2. [低秩适应（LoRA）：原理与运作](02-%E4%BD%8E%E7%A7%A9%E9%80%82%E5%BA%94%EF%BC%88LoRA%EF%BC%89%EF%BC%9A%E5%8E%9F%E7%90%86%E4%B8%8E%E8%BF%90%E4%BD%9C.md)
- 3. [使用PEFT库实现LoRA](03-%E4%BD%BF%E7%94%A8PEFT%E5%BA%93%E5%AE%9E%E7%8E%B0LoRA.md)
- 4. [量化及其对微调的影响 (QLoRA)](04-%E9%87%8F%E5%8C%96%E5%8F%8A%E5%85%B6%E5%AF%B9%E5%BE%AE%E8%B0%83%E7%9A%84%E5%BD%B1%E5%93%8D%20%28QLoRA%29.md)
- 5. [其他PEFT方法：概述](05-%E5%85%B6%E4%BB%96PEFT%E6%96%B9%E6%B3%95%EF%BC%9A%E6%A6%82%E8%BF%B0.md)
- 6. [PEFT与完全微调的优劣对比](06-PEFT%E4%B8%8E%E5%AE%8C%E5%85%A8%E5%BE%AE%E8%B0%83%E7%9A%84%E4%BC%98%E5%8A%A3%E5%AF%B9%E6%AF%94.md)
- 7. [动手实践：使用 LoRA 进行微调](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20LoRA%20%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md)
