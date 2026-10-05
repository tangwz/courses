# 使用文本生成推理 (TGI) 进行部署

来源：[原文](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/deploying-with-tgi)

[返回章节目录](README.md) · [返回课程目录](../README.md)

高效部署大型语言模型是一个主要目标，尤其是在对它们进行量化 (quantization)并评估其性能之后。Hugging Face 的文本生成推理 (inference) (TGI) 服务器是一个成熟的解决方案，专为高吞吐量 (throughput)文本生成设计，并且对量化模型有很好的支持，在各种部署框架中表现突出。

TGI 作为一个专门的推理服务器，简化了通过网络接口提供大型语言模型服务的过程。这在我们的场景中尤其重要，因为它直接集成了常用的量化库和优化技术，让你能以最少的障碍部署压缩模型。与简单的模型托管脚本不同，TGI 包含了旨在最大化 GPU 利用率和整体吞吐量的先进功能，这些在提供资源密集型大型语言模型服务时（即使是量化后的模型）都是重要考量。

## TGI 在量化 (quantization)模型服务方面的特点

TGI 提供了多个特点，使其非常适合部署量化大型语言模型：

1. **原生量化支持：** TGI 直接与 `bitsandbytes` 等库集成，使得使用 NF4 或 FP4 等技术量化到 8 位或 4 位精度的模型能够顺畅加载和推理 (inference)。它还支持加载使用 GPTQ 和 AWQ 等常见格式预量化的模型，通常只需在启动时设置一个简单参数 (parameter)。这种内置支持省去了为常见量化方案进行复杂手动设置的麻烦。
2. **连续批处理：** 这是一项重要的性能优化。传统的静态批处理要求批次内的序列填充到相同长度，导致计算浪费。连续批处理允许新请求动态地添加到当前运行的批次中，显著提升 GPU 利用率和整体吞吐量 (throughput)，特别是在可变负载下。量化模型对此优势感受更明显，因为每个序列的内存占用减少，使得批次可以更大、更具动态性。
3. **优化过的内核和注意力机制 (attention mechanism)：** TGI 包含了为各种操作优化过的 CUDA 内核，并且通常支持 Flash Attention（或类似的优化注意力实现）等机制。这些功能降低了内存带宽需求，并加速了 Transformer 模型中计算最密集的部分，补充了量化带来的好处。
4. **张量并行：** 对于即使量化后也过大而无法在单个 GPU 上运行的模型，TGI 支持张量并行，可以将模型的权重 (weight)分片到多个 GPU 上。
5. **与 Hugging Face Hub 方便集成：** TGI 可以直接下载和加载由其 Hugging Face Hub 标识符指定的模型（包括量化版本）。

## 使用 TGI 部署量化 (quantization)模型

使用 TGI 部署量化模型通常需要运行其 Docker 容器。如果你打算使用 GPU，则需要安装 Docker 和 NVIDIA Container Toolkit。

使用 TGI 部署的核心是 `docker run` 命令。我们来分析一个启动 GPTQ 量化模型的典型例子：

```bash
# 例子：部署一个使用 GPTQ (4 位) 量化的 Llama-2 7B 模型
MODEL_ID="TheBloke/Llama-2-7B-Chat-GPTQ"
# 为缓存模型/数据分配一个唯一的卷名
VOLUME_NAME="tgi_data_$(echo $MODEL_ID | sed 's/[^a-zA-Z0-9]/-/g')"
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $VOLUME_NAME:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id $MODEL_ID \
  --quantize gptq
```

让我们检查重要的参数 (parameter)：

- `--gpus all`：使所有可用的 GPU 都可供容器使用。如果需要，你可以指定特定的 GPU（例如，`--gpus '"device=0,1"'`）。
- `--shm-size 1g`：分配 1GB 共享内存。这对于进程间通信可能很重要，特别是对于更大的模型或张量并行。你可能需要调整此值。
- `-p 8080:80`：将你主机上的 8080 端口映射到容器内的 80 端口，这是 TGI 的默认 HTTP 端口。
- `-v $VOLUME_NAME:/data`：将一个根据模型 ID 命名的 Docker 卷挂载到容器内的 `/data` 目录。TGI 使用此目录下载和缓存模型权重 (weight)，避免在容器重启时重复下载。
- `ghcr.io/huggingface/text-generation-inference:latest`：指定 TGI Docker 镜像。在生产环境中，建议固定到特定的版本标签，而不是使用 `latest`。
- `--model-id $MODEL_ID`：Hugging Face Hub 上模型的标识符。如果模型不在 `/data` 卷中，TGI 将下载此模型。
- `--quantize gptq`：此标志明确告知 TGI 使用 GPTQ 量化方案加载模型。对于 `bitsandbytes` 量化（例如，通过 Transformers 加载的 4 位模型），你可能会使用 `--quantize bitsandbytes-nf4` 等标志，或者如果 Hugging Face Hub 上的模型保存有适当的 `quantization_config`，则可以依靠 TGI 自动检测配置。请查阅 TGI 文档，了解与不同量化方法和版本相对应的准确标志。

对于 `bitsandbytes` 集成量化（NF4, FP4），命令可能如下所示，假设 Hugging Face Hub 上的模型已为此配置：

```bash
# 例子：部署一个配置为 bitsandbytes 4 位 (NF4) 的模型
MODEL_ID="NousResearch/Llama-2-7b-chat-hf" # 假设此模型已使用 4 位配置保存
VOLUME_NAME="tgi_data_$(echo $MODEL_ID | sed 's/[^a-zA-Z0-9]/-/g')"
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $VOLUME_NAME:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id $MODEL_ID
  # TGI 通常从模型配置中自动检测 bitsandbytes 量化
  # 另外，你可能需要一个类似 --quantize bitsandbytes-nf4 的标志
```

启动容器后，TGI 将下载模型（如果需要）并启动服务器。你可以监控日志来查看进度。一旦准备就绪，它通常会记录一条消息，表明服务器正在监听 80 端口。

你可以通过查询其信息接口来验证服务器是否正在运行：

```bash
curl http://127.0.0.1:8080/info
```

这应该返回一个 JSON 对象，其中包含有关加载模型的信息，包括其类型、数据类型和量化状态。

## 配置和性能调优

TGI 提供了各种命令行参数 (parameter)和环境变量来微调 (fine-tuning)其性能，其中许多都与量化 (quantization)带来的资源节省有关联：

- `--max-concurrent-requests`：设置服务器将同时处理的最大请求数。
- `--max-input-length`：请求输入序列中允许的最大 token 数量。
- `--max-total-tokens`：输入 token 和生成 token 的最大总和。
- `--max-batch-prefill-tokens`：与连续批处理相关的限制，控制在批次初始“预填充”阶段处理的最大 token 数量。调整此参数会影响延迟和吞吐量 (throughput)。更大的值可能会提高吞吐量，但也会增加延迟和内存使用。
- `--max-batch-total-tokens`：在任何给定时间，动态批次中允许的总最大 token 数量（输入 + 生成）。这直接影响 GPU 内存使用。

量化显著减少了每个 token 和每个序列所需的内存。这让你能够潜在地增加批次大小 (`--max-batch-total-tokens`) 或处理更长的序列 (`--max-total-tokens`)，与在相同硬件上运行全精度模型相比，从而提高了吞吐量。尝试这些参数对于根据你的特定工作负载和量化模型优化 TGI 很重要。

## 与已部署模型交互

一旦 TGI 与你的量化 (quantization)模型一起运行，你可以向其生成接口发送请求（`/generate` 或用于流式传输的 `/generate_stream`）。

这是一个使用 `curl` 的简单例子：

```bash
curl http://127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is quantization in deep learning?","parameters":{"max_new_tokens":100, "temperature": 0.7, "top_p": 0.9}}' \
    -H 'Content-Type: application/json'
```

这会向模型发送一个提示，并请求最多 100 个新 token，同时使用特定的采样参数 (parameter)。响应将是一个包含生成文本的 JSON 对象。

你也可以使用 Python 的 `requests` 库：

```python
import requests
import json

tgi_endpoint = "http://127.0.0.1:8080/generate"

prompt = "解释部署量化大型语言模型的好处。"
params = {
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True
}

payload = {
    "inputs": prompt,
    "parameters": params
}

response = requests.post(tgi_endpoint, json=payload)

if response.status_code == 200:
    result = response.json()
    print("生成的文本：", result.get('generated_text'))
else:
    print(f"错误：{response.status_code}")
    print(response.text)
```

## 评估 TGI 在量化 (quantization)部署中的表现

TGI 在易用性和性能之间提供了很好的平衡，特别是对于 Hugging Face Hub 上已有的模型。

**优点：**

- **简单易用：** 使用 Docker 部署常见的量化模型（GPTQ、AWQ、`bitsandbytes`）相对简单直接。
- **性能优越：** 包含了连续批处理和 Flash Attention 等优化，以实现高吞吐量 (throughput)。
- **生态系统集成：** 兼容 Hugging Face Hub 上托管的模型。

**考虑事项：**

- **灵活性：** 尽管可配置，但与使用 vLLM 等库或直接针对 TensorRT-LLM 编码相比，它对推理 (inference)过程的控制可能不那么精细。优化与 TGI 的特定实现绑定。
- **峰值性能：** 对于特定 NVIDIA 硬件上的绝对最大性能，TensorRT-LLM 可能会提供进一步的优化机会，尽管通常会增加复杂性。
- **依赖性：** 依赖 TGI 开发团队集成新的量化技术或优化。

TGI 是一个很好的起点，通常是用于服务量化大型语言模型的足够生产方案，特别是在利用 Hugging Face 生态系统中的模型时。它有效地抽象了优化推理的许多复杂性，让你能够专注于快速部署你的量化模型。

## 参考资料

- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) — Hugging Face (2024)
  Hugging Face文本生成推理服务器的官方GitHub仓库和文档，包含其功能、部署和配置信息。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  详细介绍了一种用于LLM的训练后量化方法，特别支持4位精度，TGI可以原生加载和推理。

---

[上一节](02-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E9%83%A8%E7%BD%B2%E6%A1%86%E6%9E%B6.md) · [下一节](04-%E5%80%9F%E5%8A%A9%20vLLM%20%E5%AE%9E%E7%8E%B0%E9%AB%98%E5%90%9E%E5%90%90%E9%87%8F%E6%8E%A8%E7%90%86.md)
