---
course: "llm-model-sizes-hardware"
chapter: "essential-hardware-components"
lesson: "tpu-overview"
sourceId: 4214
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-2-essential-hardware-components/tpu-overview"
title: "TPU简要介绍"
description: "简要介绍作为专用AI硬件的张量处理单元（TPU）。"
order: 5
plots: []
sourceHash: "78967eb9699030e5d92b477c606e8f8b506202107349441abb38ef927f809de1"
sourceCorrections: []
---

虽然GPU为AI中常见的并行计算提供了明显加速，但它们并非唯一可用的专用硬件。谷歌开发了自己的定制专用集成电路（ASICs），称为张量处理单元（TPU），专门用于加速基于TensorFlow构建的机器学习 (machine learning)任务，尽管它们也可用于其他框架。

可将TPU视为处理器，其架构从根本上针对大规模矩阵乘法及其他操作（统称为张量操作）进行优化，这些操作是训练和运行包括LLM在内的神经网络 (neural network)的核心。GPU是优秀的通用并行处理器，适合这些任务，而TPU则是在设计之初就将此作为主要目标。

TPU的主要思路是同时且高能效地执行大量的张量操作。这种专用性可带来明显的性能提升，特别是在大规模模型训练和推理 (inference)任务中，与仅使用CPU甚至某些GPU配置相比，尤为如此，尤其是在处理它们擅长的特定计算类型时。

然而，TPU的普及度不如GPU。它们主要通过谷歌云平台（GCP）提供，并在谷歌内部广泛用于搜索、翻译和照片等服务。对于谷歌生态系统之外的开发者和研究人员来说，GPU通常仍是加速AI任务更常见且易于获取的选择。

总而言之，TPU代表了AI硬件构成中的另一个重要部分。它们是由谷歌设计的高度专用处理器，擅长深度学习 (deep learning)基础的数学运算，为GPU提供了一种替代方案，尤其是在谷歌云环境和大规模机器学习任务中。了解它们的存在有助于完善当今AI中常用硬件（包括更普及的CPU、RAM和GPU）的概览。

## 参考资料

- [A Cloud-Scale TPU for Training Large AI Models](https://dl.acm.org/doi/10.1145/3579371.3582046) — Norman P. Jouppi, Cliff Young, Daniel S. Holt, David Agrawal, Ganesh Venkataramani, et al. (2023)
  Journal: Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA '23); Publisher: ACM; Pages: 1-14; DOI: [10.1145/3579371.3582046](https://doi.org/10.1145/3579371.3582046)
  描述谷歌第四代张量处理单元（TPU v4）的架构和性能，该单元专为云端大规模AI模型训练优化。
- [About TPUs](https://cloud.google.com/tpu/docs/tpus) — Google Cloud (2024)
  Publisher: Google Cloud
  谷歌云官方文档，概述TPU、其不同版本以及它们在谷歌云平台生态系统中的使用方式。
