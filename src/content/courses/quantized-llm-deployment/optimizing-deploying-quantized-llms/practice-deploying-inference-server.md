---
course: "quantized-llm-deployment"
chapter: "optimizing-deploying-quantized-llms"
lesson: "practice-deploying-inference-server"
sourceId: 4664
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/practice-deploying-inference-server"
title: "动手实操：通过推理服务器部署"
description: "实践环节，配置推理服务器（例如TGI或vLLM），以服务量化大语言模型。"
order: 9
plots: []
sourceHash: "204b6a13d0e4636dcf1ccbfdc19c51f8ac69ce92017f1c519556f54778f78d5a"
sourceCorrections: []
---

优化模型需要使用专用推理 (inference)服务器进行部署，这尤其适用于量化 (quantization)大语言模型 (LLM)。服务这些量化大语言模型可以高效地通过vLLM（一个高吞吐量 (throughput)服务引擎）实现。为了成功部署，需要准备好一个量化模型（例如，AWQ或GPTQ格式，已妥善保存，可能使用了第2章的方法），并且用户应熟悉Python和基本的命令行操作。

服务量化模型需要专门的推理引擎，这些引擎能理解低比特格式，并拥有用于INT4/FP4矩阵乘法等计算的优化核心。vLLM专为快速推理设计，支持流行的量化方法，使其成为本次实操的合适选择。

### 前置条件与设置

1. **Python环境：** 确保您已配置好Python环境（例如3.8+）。
2. **量化 (quantization)模型：** 您需要一个量化大语言模型 (LLM)。本示例中，我们假设您有一个AWQ量化模型，已保存到本地目录，例如 `./my-quantized-model-awq`。该模型应与Hugging Face的`AutoModelForCausalLM`加载约定兼容，并已针对量化格式进行适配。
3. **安装vLLM：** 安装支持CUDA的vLLM。确保您的NVIDIA驱动和CUDA工具包版本与您安装的vLLM版本兼容。有关具体要求，请参阅vLLM官方安装指南。典型安装如下所示：

   ```bash
   pip install vllm
   ```

   *注意：* 根据您的硬件和CUDA版本，您可能需要特定的构建或额外的依赖项。请查阅vLLM文档。

### 选项1：使用vLLM Python API

使用vLLM测试量化 (quantization)模型推理 (inference)的最简单方法是通过其Python API。这对于集成到现有Python应用程序或进行快速测试很有用。

```python
from vllm import LLM, SamplingParams

# 定义量化模型目录的路径
# 替换为您的AWQ或GPTQ模型的实际路径
model_path = "./my-quantized-model-awq"
# 指定您的模型使用的量化方法
quantization_method = "awq" # 如果您有GPTQ模型，请使用“gptq”

# 初始化大语言模型引擎
# vLLM自动检测模型类型并加载量化权重
# 'dtype="auto"' 通常效果良好。
# 'tensor_parallel_size' 可以增加以用于多GPU推理。
llm = LLM(model=model_path,
          quantization=quantization_method,
          dtype="auto",
          tensor_parallel_size=1) # 如果使用多个GPU，请进行调整

# 定义生成过程的采样参数
# 这些参数控制模型如何生成文本
sampling_params = SamplingParams(temperature=0.7,
                               top_p=0.95,
                               max_tokens=100) # 限制输出长度

# 定义您的提示
prompts = [
    "解释大型语言模型中量化的原理：",
    "写一个关于机器人学习绘画的短篇故事：",
]

# 生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}")
    print(f"生成文本: {generated_text!r}\n")
```

此脚本加载指定的量化模型，定义采样参数 (parameter)，并为提供的提示生成补全内容。vLLM在后台处理量化权重 (weight)的加载并使用其优化的计算核心。您会观察到与运行未量化模型相比，内存使用量大幅降低，并且推理速度可能更快，这取决于硬件和模型大小。

### 选项2：部署为OpenAI兼容服务器

为实现更有效的部署，vLLM提供了一个内置服务器，该服务器暴露了一个与OpenAI聊天和补全端点兼容的API。这允许您使用标准HTTP请求与您部署的量化 (quantization)模型进行交互，从而使与各种应用程序的集成变得简单。

1. **启动服务器：** 打开您的终端并运行以下命令，将占位符替换为您的具体信息：

   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model ./my-quantized-model-awq \
       --quantization awq \
       --port 8000 \
       --host 0.0.0.0
       # 如果使用N个GPU，添加 --tensor-parallel-size N
       # 添加 --dtype auto （通常为默认）
   ```

   - `--model`：您的量化模型目录路径。
   - `--quantization`：指定量化方法（'awq'、'gptq'等）。这对于vLLM正确加载模型很重要。
   - `--port`：服务器将监听的网络端口。
   - `--host`：要绑定的网络接口（`0.0.0.0`使其可在您的网络上访问）。

   服务器将加载模型，并在准备好接受请求时发出提示。

   > 使用vLLM OpenAI兼容服务器时的交互流程。客户端发送标准HTTP请求，服务器使用vLLM引擎和量化模型权重 (weight)生成响应。
2. **与服务器交互：** 您现在可以使用 `curl` 或任何HTTP客户端库等工具向服务器发送请求。以下是使用 `curl` 访问补全端点的示例：

   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "./my-quantized-model-awq",
       "prompt": "解释大型语言模型中量化的原理：",
       "max_tokens": 150,
       "temperature": 0.7
     }'
   ```

   您将收到一个包含生成文本的JSON响应，其格式类似于OpenAI API。

   您也可以使用Python的`requests`库或`openai`客户端库（通过配置`base_url`）：

   ```python
   import openai

   # 配置客户端以指向您的本地vLLM服务器
   client = openai.OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="dummy" # 必需，但vLLM不使用
   )

   response = client.completions.create(
     model="./my-quantized-model-awq", # 使用所服务模型的路径/名称
     prompt="解释大型语言模型中量化的原理：",
     max_tokens=150,
     temperature=0.7
   )

   print(response.choices[0].text)
   ```

### 性能考量

部署时，请监控服务器的性能。

- **延迟：** 单个请求需要多长时间？
- **吞吐量 (throughput)：** 服务器每秒能处理多少请求（或令牌）？
- **资源使用：** 监控GPU内存消耗和利用率。

vLLM提供通常包含性能指标的日志记录。您也可以对已部署的端点使用基准测试工具（如第3章所述）。vLLM中像`tensor_parallel_size`（用于多GPU）和批处理功能等参数 (parameter)会大幅影响这些指标。请根据您的硬件和预期负载尝试不同的设置。

### 故障排除

- **模型加载错误：** 仔细检查指定的`--model`路径和`--quantization`方法。确保模型文件完整且已为所选量化 (quantization)方法正确格式化。
- **内存不足（OOM）错误：** 量化模型减少内存使用，但大型模型或高流量仍可能超出GPU内存。尝试减少批次大小（如果可通过服务器参数 (parameter)配置，尽管vLLM在内部处理批处理）或通过张量并行（`--tensor-parallel_size`）使用更多GPU。
- **兼容性：** 确保您的vLLM版本支持您正在使用的特定量化格式和模型架构。查阅vLLM文档以了解支持的配置。

本次实践练习演示了使用vLLM部署量化大语言模型 (LLM)，包括通过其Python API和作为独立服务器。借助专门的推理 (inference)服务器，您可以有效地服务这些内存高效且可能更快的模型，使先进的大语言模型更易用于生产应用。请记住，选择合适的服务器、正确配置它并理解性能权衡是部署过程中的重要步骤。

## 参考资料

- [vLLM Documentation](https://docs.vllm.ai/en/latest/) — vLLM Authors (Ongoing)
  提供关于安装、配置和使用 vLLM 的 Python API 及 API 服务器的全面指南，实现高吞吐量 LLM 推理，包括对量化模型的支持。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: SOSP 2023; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍了 PagedAttention，这是使 vLLM 实现高吞吐量和高效内存利用率进行 LLM 推理的核心内存管理技术。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  描述了 AWQ，一种根据激活幅度保护显著权重的量化方法，可在高压缩比下为 LLM 实现良好准确性。
