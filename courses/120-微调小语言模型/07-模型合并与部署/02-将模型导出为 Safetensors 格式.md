# 将模型导出为 Safetensors 格式

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-7-model-merging-and-deployment/exporting-models-to-safetensors)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将合并后的模型架构保存到磁盘需要一种可靠的序列化格式。在过去，保存 PyTorch 模型的标准方法是使用 Pickle 模块将 Python 对象序列化为 `.bin` 或 `.pt` 文件。这种方式虽然可行，但存在明显的安全风险和性能瓶颈，因此并不适合生产环境。

### Pickle 序列化的风险

Pickle 文件不仅存储数据，还可以包含任意的可执行代码。当你使用传统的 PyTorch 方法加载 Pickle 模型时，实际上是在指示 Python 解释器执行该文件中嵌入 (embedding)的所有指令。在模型可能被共享、从公共平台下载或在服务器之间传输的生产场景中，这会带来不可接受的安全隐患。恶意攻击者可能会在模型文件中注入代码，一旦模型被加载到内存中，这些代码就会立即执行。

此外，加载大型 Pickle 文件需要分配内存并逐层复制数据。操作系统必须先将整个文件读取到系统内存（RAM）中，然后再将其移动到 GPU 显存 (VRAM)空间。这种重复的复制过程会减慢模型的初始化速度，并且在系统内存有限的机器上经常导致内存溢出错误。

### 内存映射与零拷贝操作

为了解决这些安全和性能问题，Hugging Face 开发了 Safetensors 格式。Safetensors 是一种专门为存储和加载机器学习 (machine learning)张量而设计的开源序列化格式。它限制保存的文件仅包含原始张量数据和元数据，完全去除了执行任意代码的能力。

从性能角度来看，Safetensors 采用了内存映射（mmap）技术。操作系统不再将整个文件读取到 CPU 内存后再进行复制，而是将文件直接映射到虚拟内存中。只有在访问张量时，它们才会被加载到目标设备（如 GPU）中。这绕过了系统内存中的冗余复制，实现了零拷贝加载过程，从而大幅缩短了实例化语言模型所需的时间。

> 传统 Pickle 序列化与 Safetensors 内存映射的内存分配路径对比。

### 在 PyTorch 中实现 Safetensors

Hugging Face 的 Transformers 库提供了对 Safetensors 格式的原生支持。当你准备导出合并后的模型时，可以使用 `save_pretrained` 方法并明确指定序列化首选项。

```python
import os

# 为部署就绪的模型定义输出目录
output_dir = "./my-slm-production"
os.makedirs(output_dir, exist_ok=True)

# 使用 Safetensors 保存合并后的模型
merged_model.save_pretrained(
    output_dir,
    safe_serialization=True
)

# 同时保存分词器
tokenizer.save_pretrained(output_dir)
```

将 `safe_serialization=True` 设置后，可以确保模型权重 (weight)被写入 `.safetensors` 文件，而不是标准的 `.bin` 文件。检查输出目录的内容时，你会发现名为 `model.safetensors` 的文件以及必要的 JSON 配置文件。

对于较大的模型，Transformers 库会自动将权重拆分为多个文件，例如 `model-00001-of-00002.safetensors`。这种分块处理确保了单个文件的大小对于文件系统和网络传输来说是易于管理的。同时会生成一个名为 `model.safetensors.index.json` 的索引文件，用于映射各个层具体存储在哪个文件中。

### 验证导出的模型

在迁移到推理 (inference)服务设施之前，验证模型是否能从新创建的 Safetensors 文件中正确加载是一个良好的习惯。你可以直接从本地目录初始化模型。

```python
from transformers import AutoModelForCausalLM

# 直接从 Safetensors 目录加载模型
production_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    local_files_only=True,
    device_map="auto"
)
```

通过设置 `local_files_only=True`，可以阻止库尝试连接 Hugging Face Hub，从而确保本地的内存映射加载路径运行正常。`device_map="auto"` 参数 (parameter)会自动将加载的张量分配到可用的硬件上。

将微调 (fine-tuning)后的模型导出为快速、安全的格式，为部署奠定了可靠的基础。由于张量存放在磁盘上并已准备好进行快速内存映射访问，模型现在可以由高吞吐量 (throughput)的推理引擎读取使用。

## 参考资料

- [Safetensors: A Simple and Safe Way to Store and Load Tensors](https://huggingface.co/docs/safetensors/index) — Hugging Face Team (2023)
  官方文档，解释了 Safetensors 格式的设计目标、安全性优势以及实现细节。
- [Transformers Documentation: Sharing and Uploading Models](https://aclanthology.org/2020.emnlp-demos.6/) — Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, Alexander M. Rush (2020)
  Journal: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations; Publisher: Association for Computational Linguistics; Pages: 38–45; DOI: [10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)
  Transformers 库中 save_pretrained 方法及其序列化参数的详细 API 参考。
- [Memory-mapped Files](https://man7.org/tlpi/) — Michael Kerrisk (2010)
  Publisher: No Starch Press; Pages: 939-968
  关于 Safetensors 实现零拷贝加载所采用的底层操作系统机制 (mmap) 的参考资料。

---

[上一节](01-%E5%B0%86%20LoRA%20%E9%80%82%E9%85%8D%E5%99%A8%E4%B8%8E%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20vLLM%20%E9%83%A8%E7%BD%B2%20SLM.md)
