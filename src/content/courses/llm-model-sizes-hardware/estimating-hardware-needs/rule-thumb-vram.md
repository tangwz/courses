---
course: "llm-model-sizes-hardware"
chapter: "estimating-hardware-needs"
lesson: "rule-thumb-vram"
sourceId: 4249
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/rule-thumb-vram"
title: "显存需求估算：参数量经验法则"
description: "一个基于参数数量和精度（如FP16）估算所需显存的基本计算方法。"
order: 1
plots: ["plots/4249-0.json"]
sourceHash: "9c5a11dcdd7560e982f9176b39e448838d07e95177b234c1558c8f6458c959fc"
sourceCorrections: []
---

估算运行大型语言模型（LLM）所需的硬件可能看起来很复杂。一个直接的估算方法可以简化这一过程。决定内存需求的最主要因素，特别是图形处理单元（GPU）的显存（VRAM），是模型本身的规模，由其参数 (parameter)数量衡量。

可以把模型中的每个参数看作一个数字，需要存储在某个可用于计算的位置。由于GPU是运行LLM的主要计算设备（如第二章所述），这些参数主要加载到GPU的专用内存——显存中。

每个参数占用多少空间？这取决于存储时使用的*精度*或*数据类型*。常见的精度类型包含：

- **FP32（单精度浮点）：** 每个参数占用32位，相当于4字节。这提供高精度，但需要更多内存。
- **FP16（半精度浮点）：** 每个参数占用16位，即2字节。这使得内存需求大约是FP32的一半，通常对推理 (inference)性能影响很小。
- **INT8（8位整数）：** 每个参数仅占用8位，即1字节。这能进一步大幅减少内存使用，但有时会导致模型准确性出现可察觉的下降。这通常通过一个称为量化 (quantization)的过程来实现（我们在第三章简要介绍过）。

最常见的估算起点假设模型将使用FP16精度运行，因为它在内存使用和性能之间提供了良好的平衡。

### 基本计算

基于此，我们可以建立一个简单的经验法则来估算仅加载模型参数 (parameter)所需的最小显存 (VRAM)：

$所需 \, 显存 \, (GB) \approx \frac{参数数量 \, (十亿) \times 每参数字节数}{1024^3 \, (字节/GB)}$

然而，一个更简单的速算方法，尤其是对于FP16，常被使用：

$所需 \, 显存 \, (GB) \approx 参数数量 \, (十亿) \times 2 \, (FP16的字节数)$

来看几个例子：

1. **一个70亿参数模型（例如，Llama 2 7B）：**

   - 使用FP16精度（2字节/参数）：
     $VRAM \approx 7 \, \text{十亿} \times 2 \, \text{字节} = 14 \, \text{十亿 字节}$
     $VRAM \approx \frac{14 \times 10^9}{1024^3} \approx 13.04 \, GB$
     因此，仅在FP16中容纳模型权重 (weight)，您大约需要**14 GB**的显存。
   - 使用FP32精度（4字节/参数）：
     $VRAM \approx 7 \, \text{十亿} \times 4 \, \text{字节} = 28 \, \text{十亿 字节}$
     $VRAM \approx \frac{28 \times 10^9}{1024^3} \approx 26.07 \, GB$
     以完整精度加载相同模型将需要约**28 GB**的显存。
   - 使用INT8精度（1字节/参数，量化 (quantization)后）：
     $VRAM \approx 7 \, \text{十亿} \times 1 \, \text{字节} = 7 \, \text{十亿 字节}$
     $VRAM \approx \frac{7 \times 10^9}{1024^3} \approx 6.52 \, GB$
     使用8位量化能大幅减少需求，大约降至**7 GB**的显存。
2. **一个700亿参数模型（例如，Llama 2 70B）：**

   - 使用FP16精度（2字节/参数）：
     $VRAM \approx 70 \, \text{十亿} \times 2 \, \text{字节} = 140 \, \text{十亿 字节}$
     $VRAM \approx \frac{140 \times 10^9}{1024^3} \approx 130.36 \, GB$
     这个更大的模型在FP16下需要约**140 GB**的显存。这通常需要多块高端GPU。



![70亿参数LLM的预估显存](plots/4249-0.json)



> 仅用于存储70亿参数模型在不同数值精度下的参数所需的预估显存。

### 重要考量

这个经验法则提供了一个存储模型参数 (parameter)的基准估算。它是确定您的硬件是否*可能*足够的一个基本起点。然而，请记住，此计算仅考虑模型权重 (weight)本身。运行模型不仅仅是存储它。正如我们接下来会讨论的，推理 (inference)时的激活、输入提示和生成输出的长度（上下文 (context)长度）以及软件开销等因素将消耗额外的显存 (VRAM)。因此，您应始终预留*更多*显存，超出这个简单计算所建议的量。

## 参考资料

- [Automatic Mixed Precision for Deep Learning](https://developer.nvidia.com/automatic-mixed-precision) — NVIDIA Developer Documentation (2023)
  Publisher: NVIDIA
  解释了在深度学习中使用混合精度（FP16和FP32）的原理和优势，这对于理解GPU上的内存和性能优化至关重要。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://ieeexplore.ieee.org/document/8578701) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Andrey Bochenin, Vitaly Tarasov, Andrew Karpov, Dianne Jouppi, Anujan Varma, Gabriel Micha (2018)
  Journal: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition; Publisher: IEEE; Pages: 683-692; DOI: [10.1109/CVPR.2018.00696](https://doi.org/10.1109/CVPR.2018.00696)
  一篇基础性学术论文，介绍了将神经网络量化为8位整数（INT8）以实现高效推理的技术，与内存减少直接相关。
- [VRAM estimation for large models](https://huggingface.co/docs/transformers/main/en/vram_estimation) — Hugging Face documentation contributors (2023)
  Publisher: Hugging Face
  一个来自领先LLM平台的实用指南，详细说明了如何估算大型语言模型的VRAM需求，包括对不同数据类型和额外内存开销的考虑。
