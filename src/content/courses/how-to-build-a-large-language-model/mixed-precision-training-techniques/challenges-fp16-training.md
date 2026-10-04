---
course: "how-to-build-a-large-language-model"
chapter: "mixed-precision-training-techniques"
lesson: "challenges-fp16-training"
sourceId: 6051
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/challenges-fp16-training"
title: "FP16训练中的挑战（范围问题）"
description: "描述激活值或梯度超出FP16可表示范围的问题。"
order: 3
plots: ["plots/6051-0.json"]
sourceHash: "4e171d40ae55dcc22e4f6ca7812e7e4bb9311b767c46b4589cc5c5854942139f"
sourceCorrections: []
---

尽管16位浮点（FP16）提供的计算速度和内存节省很有吸引力，但从标准32位精度（FP32）转换并非没有困难。主要的挑战源于FP16格式明显更窄的*动态范围*。了解这一局限性是成功进行混合精度训练的基础。

FP32数字使用1个符号位、8个指数位和23个有效数（尾数）位。这使得它们能够表示一个广泛的数值范围，大致从$1.18 \times 10^{-38}$到$3.4 \times 10^{38}$。相比之下，FP16的IEEE 754标准使用1个符号位、5个指数位和10个有效数位。这种分配方式大大减小了可表示数字的范围。最小的正规化FP16数是$2^{-14} \approx 6.1 \times 10^{-5}$，最大的是$(2 - 2^{-10}) \times 2^{15} = 65504$。小于$2^{-14}$的数字可能可以表示为*非正规数*（或非规范数），提供逐步下溢，直至大约$6.0 \times 10^{-8}$，但这些通常会带来硬件性能损失，并且与FP32相比，其表示范围仍然小得多。



![大致可表示范围（对数尺度）](plots/6051-0.json)



> FP32、FP16和BF16（BFloat16）格式的大致最大正规正值和最小正规正值在对数尺度上的比较。请注意FP16明显更小的范围。

这种有限的范围在深度学习 (deep learning)训练中带来两个主要问题：

1. **下溢为零：** 梯度，特别是在深层网络中或对于不常更新的参数 (parameter)，可能变得非常小。如果梯度的幅度低于FP16可表示的最小正值（大约$6.1 \times 10^{-5}$），它就会被向下舍入到零。当梯度变为零时，相应的权重 (weight)将不再被更新。这会有效停止模型部分区域的学习，可能阻止收敛或导致次优结果。设想一个场景，FP32中计算出的真实梯度是$5 \times 10^{-5}$。在FP16中，这个值无法被精确表示，可能变为零，从而完全丢失更新信号。
2. **溢出为无穷大：** 相反，激活值或大梯度，尤其当优化器或损失函数 (loss function)中的中间计算产生大值时，可能超出FP16可表示的最大值（65504）。当这种情况发生时，该值会溢出为无穷大（`Inf`）。涉及`Inf`的操作通常会导致非数字（`NaN`）值（例如，`Inf - Inf`，`0 * Inf`）。一旦`NaN`出现在模型的权重、激活值或损失计算中，它们往往会迅速传播，破坏整个训练过程并导致其崩溃。

在PyTorch中考虑一个简单例子：

```python
import torch

# 潜在下溢的例子
small_value_fp32 = torch.tensor(5e-5, dtype=torch.float32)
small_value_fp16 = small_value_fp32.half() # 转换为FP16
print(f"FP32 value: {small_value_fp32}")
print(f"FP16 value: {small_value_fp16}")
# 根据具体的FP16实现细节，可能变为0.0

# 潜在溢出的例子
large_value_fp32 = torch.tensor(70000.0, dtype=torch.float32)
large_value_fp16 = large_value_fp32.half() # 转换为FP16
print(f"FP32 value: {large_value_fp32}")
print(f"FP16 value: {large_value_fp16}") # 变为inf
```

这些范围问题对于LLM尤其重要，因为它们具有深度和复杂性。网络深层的激活值或通过长反向传播 (backpropagation)路径计算的梯度很容易超出狭窄的FP16范围。此外，梯度累积等技术，常用于LLM训练中，可能增加累积梯度的量级，如果处理不当，会面临溢出风险。

尽管这些挑战可能看起来令人望而生畏，但它们并未否定FP16的优点。重要的是采用稳定技术，最显著的就是损失缩放，我们将在接下来进行讨论。同样值得一提的是，`BFloat16`格式（我们将在本章后续部分介绍）与FP16相比牺牲了精度，但保留了与FP32相同的宽动态范围，这在很大程度上避免了这些特定的下溢和溢出问题，尽管精度较低可能会对收敛产生潜在影响。成功进行FP16训练需要管理好速度、内存和数值稳定性之间的这种精细平衡。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍混合精度训练，详细说明FP16的范围问题（下溢/上溢）以及提出的损失缩放解决方案。
- [A New Standard for Mixed-Precision Training: bfloat16](https://blog.google/products/ai/a-new-standard-for-mixed-precision-training-bfloat16/) — Karen Young, David Patterson, Cliff Young (2019)
  Publisher: Google AI Blog
  解释BFloat16格式，强调其比FP16更宽的动态范围，直接解决混合精度训练中的下溢和上溢问题。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture-a-quantitative-approach/hennessy/978-0-12-811905-1) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  一本基础教科书，完整解释浮点算术，包括IEEE 754标准、位分配以及FP16和FP32等各种格式的数值特性。
- [Automatic Mixed Precision for Deep Learning](https://pytorch.org/docs/stable/notes/amp_examples.html) — PyTorch Documentation (2024)
  PyTorch官方指南，解释混合精度训练，包括框架级解决方案如`torch.cuda.amp`如何应对FP16带来的数值稳定性挑战。
