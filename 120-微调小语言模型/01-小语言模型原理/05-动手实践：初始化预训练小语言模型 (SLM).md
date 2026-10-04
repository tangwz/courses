# 动手实践：初始化预训练小语言模型 (SLM)

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-1-principles-of-small-language-models/hands-on-practical-initializing-slm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将预训练 (pre-training)的小语言模型加载到本地内存中，可以验证环境与硬件的兼容性。在编写训练循环之前，必须确认基础模型能够正常加载并生成文本。这一步在进行任何监督学习 (supervised learning)之前，确定了模型的初始性能。

在本练习中，我们将使用 Hugging Face 的 `transformers` 库与模型进行交互。我们将加载一个体积较小、采用标准架构的模型，这种模型可以轻松运行在消费级硬件上。

### 初始化分词 (tokenization)器 (Tokenizer)

语言模型无法直接处理原始文本。它们需要通过分词器将人类可读的字符串转换为数字形式的 Token ID。这些 ID 对应模型嵌入 (embedding)矩阵中的特定条目。你可以使用 `AutoTokenizer` 类来初始化分词器，并将其指向特定的模型仓库。

```python
from transformers import AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

分词器会下载与模型关联的词表文件。这确保了脚本生成的 Token ID 与模型在最初训练阶段所接触的 Token ID 完全一致。

### 加载模型权重 (weight)

接下来，将模型权重读入内存。此前，我们在梯度方程中将权重矩阵定义为 $W$。`from_pretrained` 函数会将这些权重矩阵直接加载到系统的显存 (VRAM) 中。

为了遵守上一节讨论的内存限制，我们指示 PyTorch 使用 16 位精度加载模型。

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

`device_map="auto"` 参数 (parameter)会评估可用的硬件，并自动将模型层分配到 GPU 上。如果没有可用的 GPU，它将默认为 CPU。`torch_dtype=torch.bfloat16` 参数将内存占用减少到标准 32 位浮点数的一半，这是准备参数高效微调 (fine-tuning)模型时的常规做法。

### 执行生成流水线

在模型和分词 (tokenization)器 (tokenizer)就绪后，即可执行基础的文本生成任务。这包括准备提示词 (prompt)、将其转换为张量 (Tensor)，并传递给模型。

```python
prompt = "Machine learning is a field of artificial intelligence that"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

当调用 `model.generate` 时，系统会执行一系列前向传播。在每次传播中，模型都会计算 Logits，代表下一个 Token 的概率分布。

传递给 `generate` 函数的参数 (parameter)控制模型如何从这些概率中采样。`max_new_tokens` 参数在生成 50 个 Token 后停止循环，以防止无限生成。设置 `do_sample=True` 会启用概率采样，而 `temperature` 参数则在 Softmax 操作前对 Logits 进行缩放。0.7 的温度会使分布稍微平滑，从而为生成的文本引入变化。较低的温度会使输出结果更加确定。

> 标准文本生成流水线：将人类文本转换为数字张量，然后再转回文本。

打印响应时，你会发现模型会根据其预训练 (pre-training)数据尝试补全句子。在此阶段，输出可能会偏离主题或无法遵循特定的指令。由于模型尚未针对你的特定场景进行监督微调 (fine-tuning)，它纯粹是一个统计意义上的文本续写器。这种表现说明了为什么需要格式化自定义数据集并调整内部参数，以此来构建一个可靠的、针对特定任务的应用程序。

## 参考资料

- [Generation with LLMs](https://huggingface.co/docs/transformers/main_classes/text_generation) — Hugging Face Team (2024)
  官方文档解释了 generate 函数的各项参数，包括采样和温度设置。
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) — Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Lyuwan Zhang, Xiaohuan Zhou, Junyang Zhu, Sunbin Zhu (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2309.16609](https://doi.org/10.48550/arXiv.2309.16609)
  本练习所使用的 Qwen 模型系列的官方技术报告，详细介绍了其架构和训练过程。
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one) — Hugging Face Team (2024)
  关于显存优化的综合指南，涵盖了半精度（FP16/BF16）和设备映射的相关内容。
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  描述利用降低精度来减少模型运行过程中显存占用的基础研究论文。

---

[上一节](04-%E7%A1%AC%E4%BB%B6%E8%A6%81%E6%B1%82%E4%B8%8E%E6%98%BE%E5%AD%98%E9%99%90%E5%88%B6.md) · [下一节](../02-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%E4%B8%8E%E6%A0%BC%E5%BC%8F%E5%8C%96/01-%E6%9E%84%E5%BB%BA%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E9%9B%86.md)
