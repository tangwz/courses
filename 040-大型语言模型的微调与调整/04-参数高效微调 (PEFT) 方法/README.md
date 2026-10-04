# 第 4 章：参数高效微调 (PEFT) 方法

来源：[原章节](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning)

[返回课程目录](../README.md)

完全微调会更新大型语言模型中的每个参数，这需要大量的计算资源和内存。这通常使得在没有专用硬件的情况下调整最大型的模型变得不切实际。本章介绍参数高效微调 (PEFT) 方法，作为一种节省资源的替代方案。

PEFT 技术只修改模型参数的一小部分，大大减少了计算开销和存储需求，同时在特定任务上通常能达到与完全微调相近的性能。

你将学习几种主要 PEFT 方法的原理和实现：

*   **低秩适应 (LoRA)：** 将可训练的低秩矩阵注入到 Transformer 层中。
*   **量化 LoRA (QLoRA)：** 将 LoRA 与量化结合，以实现更大的内存节省。
*   **适配器模块 (Adapter Modules)：** 在现有架构内插入小的、可训练的瓶颈层。
*   **提示微调 (Prompt Tuning) 和前缀微调 (Prefix Tuning)：** 学习添加到输入或隐藏状态前的连续嵌入，同时保持基本模型不变。

我们将分析每种技术背后的机制，比较它们各自的优点和权衡，并通过使用常用库进行实践。在本章末尾，你将了解如何选择和运用合适的 PEFT 方法来高效调整大型语言模型。

## 小节

- 1. [参数高效性的原理](01-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E6%80%A7%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 2. [低秩适配 (LoRA)](02-%E4%BD%8E%E7%A7%A9%E9%80%82%E9%85%8D%20%28LoRA%29.md)
- 3. [量化低秩适配 (QLoRA)](03-%E9%87%8F%E5%8C%96%E4%BD%8E%E7%A7%A9%E9%80%82%E9%85%8D%20%28QLoRA%29.md)
- 4. [适配器模块](04-%E9%80%82%E9%85%8D%E5%99%A8%E6%A8%A1%E5%9D%97.md)
- 5. [提示调整](05-%E6%8F%90%E7%A4%BA%E8%B0%83%E6%95%B4.md)
- 6. [前缀微调](06-%E5%89%8D%E7%BC%80%E5%BE%AE%E8%B0%83.md)
- 7. [PEFT 方法比较](07-PEFT%20%E6%96%B9%E6%B3%95%E6%AF%94%E8%BE%83.md)
- 8. [使用 Hugging Face PEFT 库进行实现](08-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20PEFT%20%E5%BA%93%E8%BF%9B%E8%A1%8C%E5%AE%9E%E7%8E%B0.md)
- 9. [动手实践：使用 LoRA 进行微调](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20LoRA%20%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md)
- 10. [实操：使用QLoRA进行微调](10-%E5%AE%9E%E6%93%8D%EF%BC%9A%E4%BD%BF%E7%94%A8QLoRA%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md)
