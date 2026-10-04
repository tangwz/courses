# 将 LoRA 适配器与基础模型合并

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-7-model-merging-and-deployment/merging-lora-adapters-with-base-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在准备将微调 (fine-tuning)后的模型投入生产时，如果让基础架构与独立的适配器层并行运行，会产生不必要的计算开销。在训练期间，保持低秩矩阵独立是标准做法，这有助于减少内存占用并高效更新参数 (parameter)。然而，在推理 (inference)阶段，在每次前向传递时动态地将适配器层的输出添加到基础模型层会减慢文本生成速度。为了优化性能，可以将训练好的 LoRA 适配器直接合并到原始模型权重 (weight)中。

回顾低秩自适应（Low-Rank Adaptation）的数学运算。对于给定的基础权重矩阵 $W$ 和低秩适配器矩阵 $A$ 及 $B$，合并后的权重矩阵 $W'$ 计算如下：

$W' = W + \frac{\alpha}{r} (AB)$

这里，$\alpha$ 是缩放因子，$r$ 是训练前配置的秩参数。一旦完成此加法运算，就不再需要矩阵 $A$ 和 $B$ 了。新的权重矩阵 $W'$ 的运行方式与原始基础模型完全一致，但它包含了微调期间获得的特定行为。

> 将低秩适配器矩阵合并到基础模型权重中，以创建单一的可部署模型的过程。

在实际操作中，Hugging Face 的 peft 库将这一数学运算简化为了一个方法调用。你将使用 `merge_and_unload()` 函数。该函数执行矩阵乘法和加法，然后从内存中卸载独立的适配器实例。结果是一个标准的 Hugging Face Transformers 模型对象。

要进行合并，必须先加载基础模型，然后挂载适配器权重。

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_id = "your-small-base-model"
adapter_path = "./lora-adapters"

# 以 16 位精度加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载结合了基础权重和适配器的 PEFT 模型
model = PeftModel.from_pretrained(base_model, adapter_path)

# 合并权重并卸载适配器
merged_model = model.merge_and_unload()
```

合并操作要求以全精度或半精度（如 `float16` 或 `bfloat16`）加载基础模型和适配器。如果你在训练时使用了 QLoRA 且基础模型为 4 位量化 (quantization)模型，则无法直接将适配器合并回 4 位权重中。合并的数学运算需要一致的、非量化的张量类型，以防止严重的精度损失。

你必须以 `float16` 加载基础模型，应用适配器，然后执行合并。这一步对系统内存或显存 (VRAM)的需求暂时会高于训练阶段。如果你的本地机器显存不足以承载非量化模型，可以在初始加载模型时设置 `device_map="cpu"`，强制在系统 CPU 上进行合并。这样做虽然速度较慢，但可以避免在硬件受限时出现显存溢出错误。

一旦 `merge_and_unload()` 完成，生成的模型在结构上与标准的、未经微调的模型完全一致。它不再依赖 peft 库即可运行，因此与优化后的推理引擎和服务框架具有极高的兼容性。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了在不增加推理延迟的情况下将低秩矩阵添加到基础权重的数学基础的原始研究论文。
- [PEFT Documentation: Merging Adapters](https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-adapters) — Hugging Face (2024)
  PEFT 库的官方文档，详细说明了权重合并的实现以及 merge_and_unload 方法的行为。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems; Volume: 36; Pages: 10088-10115; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  提供了关于为什么与 4 位量化基础模型进行合并时需要特定的精度处理以维持模型性能的技术背景。

---

[上一节](../06-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E4%B8%8E%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95/05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%BF%90%E8%A1%8C%E8%AF%84%E4%BC%B0%E8%84%9A%E6%9C%AC.md) · [下一节](02-%E5%B0%86%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA%E4%B8%BA%20Safetensors%20%E6%A0%BC%E5%BC%8F.md)
