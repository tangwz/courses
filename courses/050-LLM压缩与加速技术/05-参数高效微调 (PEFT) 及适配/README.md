# 第 5 章：参数高效微调 (PEFT) 及适配

来源：[原章节](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft)

[返回课程目录](../README.md)

从头训练大型语言模型需要大量资源。将这些预训练模型适配到特定的下游任务通常涉及对所有参数进行微调，这也会带来高昂的计算成本，并且如果管理多个针对特定任务的模型，还会导致存储需求巨大。

参数高效微调 (PEFT) 技术提供了一些方法，通过仅修改大型语言模型的一小部分参数，或添加少量新参数来适配它们。这种做法显著降低了微调相关的计算成本和内存占用，使得为各种应用定制大型模型成为可能，而无需完全重新训练它们或存储大量的完整副本。

本章将介绍几种突出的PEFT策略。我们将了解适配器模块的工作原理，各种基于提示的微调方法，如前缀微调 (Prefix Tuning) 和提示微调 (Prompt Tuning)，以及广泛使用的低秩适配 (LoRA)。我们还将介绍量化LoRA (QLoRA)，它通过结合量化进一步减少内存使用。您将了解这些方法的工作方式，分析它们的性能权衡，并获得实践它们的实际经验。

## 小节

- 1. [PEFT 的缘由](01-PEFT%20%E7%9A%84%E7%BC%98%E7%94%B1.md)
- 2. [适配器模块](02-%E9%80%82%E9%85%8D%E5%99%A8%E6%A8%A1%E5%9D%97.md)
- 3. [前缀微调、提示微调与P-Tuning](03-%E5%89%8D%E7%BC%80%E5%BE%AE%E8%B0%83%E3%80%81%E6%8F%90%E7%A4%BA%E5%BE%AE%E8%B0%83%E4%B8%8EP-Tuning.md)
- 4. [低秩适应（LoRA）](04-%E4%BD%8E%E7%A7%A9%E9%80%82%E5%BA%94%EF%BC%88LoRA%EF%BC%89.md)
- 5. [量化LoRA (QLoRA)](05-%E9%87%8F%E5%8C%96LoRA%20%28QLoRA%29.md)
- 6. [组合PEFT方法](06-%E7%BB%84%E5%90%88PEFT%E6%96%B9%E6%B3%95.md)
- 7. [PEFT技术性能分析](07-PEFT%E6%8A%80%E6%9C%AF%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md)
- 8. [实践：使用LoRA和QLoRA进行微调](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8LoRA%E5%92%8CQLoRA%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md)
