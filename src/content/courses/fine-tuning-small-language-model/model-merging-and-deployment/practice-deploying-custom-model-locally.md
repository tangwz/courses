---
course: "fine-tuning-small-language-model"
chapter: "model-merging-and-deployment"
lesson: "practice-deploying-custom-model-locally"
sourceId: 8531
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-7-model-merging-and-deployment/practice-deploying-custom-model-locally"
title: "实践：在本地部署自定义模型"
description: "将最终微调后的模型部署到本地服务器，并通过 API 提交推理请求。"
order: 5
plots: []
sourceHash: "b8de44106cf13d09be5e4b08258cff8f98419d1b79e54546e8e3128aeae84721"
sourceCorrections: []
---

将微调 (fine-tuning)后的小语言模型部署到本地服务器，可以将静态的 Safetensors 文件（包含合并后的 LoRA 适配器和基础模型权重 (weight)）转换为实时服务。通过使用 vLLM，模型可以托管在本地并提供 RESTful API。随后，Python 客户端即可向该服务提交推理 (inference)请求。

vLLM 开箱即用，提供了一个兼容 OpenAI 的 API 服务器。这种兼容性意味着你可以使用现有的、为大型商业模型设计的客户端库，直接与你的本地部署进行交互。要启动服务器，请打开终端并执行以下命令，将文件路径替换为合并后的 Safetensors 模型的实际位置。

```bash
python -m vllm.entrypoints.openai.api_server --model /path/to/your/merged_model --port 8000
```

运行此命令时，vLLM 会将模型权重分配到 GPU 显存 (VRAM)中。引擎会初始化 KV 缓存，并准备好 REST API 端点以接收连接。你会看到终端输出显示服务器正在 8000 端口监听 HTTP 流量。

> 从客户端应用程序到本地推理引擎的数据流向及返回过程。

服务器运行后，你就可以与模型建立通信。由于 vLLM 模拟了 OpenAI 的 API 结构，你可以使用官方的 `openai` Python 软件包发送请求。如果你的环境中尚未安装该包，可以通过 pip 进行添加。

```python
from openai import OpenAI

# 初始化客户端，指向本地 vLLM 服务器
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-deployment" # 客户端需要 API 密钥，但本地 vLLM 会忽略它
)

system_prompt = "你是一名专门回答技术问题的得力助手。"
user_prompt = "请解释机器学习中早停法 (early stopping) 和权重衰减 (weight decay) 的区别。"

response = client.chat.completions.create(
    model="/path/to/your/merged_model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

`base_url` 参数 (parameter)将客户端导向你的本地机器地址以及服务器启动时指定的端口。`model` 参数必须与你提供给 vLLM 服务器命令的目录路径或名称完全一致。

API 调用接受控制文本生成行为的参数。`temperature` 参数在应用 softmax 函数之前对输出的 logits 进行缩放。$0.7$ 的值引入了一定的随机性，防止模型总是选择概率最高的下一个词元 (token)。`max_tokens` 参数限制了总输出长度，以防止出现死循环或过长的回答。

设 logits 为 $z_i$，温度为 $T$，则下一个词元的概率 $p_i$ 的数学计算公式为：

$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$

当 $T = 1.0$ 时，应用标准的 softmax 函数。当 $T < 1.0$ 时，概率分布变得更加集中，使得模型的输出更具确定性。

对于较长的生成内容，等待整个响应完成后再显示可能会导致明显的应用延迟。为了提高应用程序的响应速度，你可以在生成词元时进行流式传输。通过在 API 调用中设置 `stream=True` 即可实现此行为。

```python
response_stream = client.chat.completions.create(
    model="/path/to/your/merged_model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=200,
    temperature=0.7,
    stream=True
)

for chunk in response_stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

在此流模式下，服务器使用服务器发送事件 (Server-Sent Events) 将词元更新持续推送到客户端。客户端脚本遍历接收到的网络数据块，并立即将其打印到控制台。

为了验证部署能否处理多个并发请求，你可以编写一个异步测试脚本。这可以确认 vLLM 的连续批处理机制在负载下是否正常运行。

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-deployment"
)

async def fetch_response(prompt):
    response = await async_client.chat.completions.create(
        model="/path/to/your/merged_model",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content

async def main():
    prompts = [
        "什么是梯度？",
        "定义交叉熵损失。",
        "解释 ReLU 激活函数。"
    ]
    tasks = [fetch_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"响应 {i+1}: {result}\n")

asyncio.run(main())
```

执行此异步脚本会同时向服务器发送所有三个提示词 (prompt)。vLLM 引擎会动态地对传入请求进行批处理，以最大限度地利用 GPU 显存带宽并返回生成的文本。现在，你已经拥有了一个完全可运行的本地推理服务器，托管着你的微调语言模型。

## 参考资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/10.1145/3600006.3613165) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23); Publisher: ACM; Pages: 611–626; DOI: [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165)
  介绍了 vLLM 使用的 PagedAttention 算法和连续批处理技术的原始研究论文，这些技术大幅优化了推理性能。
- [vLLM Documentation](https://docs.vllm.ai/) — vLLM Team (2024)
  vLLM 推理引擎关于部署、服务器配置以及支持模型格式的官方文档。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2026)
  Publisher: Pearson
  一本涵盖语言模型数学基础的教材，包括温度缩放 Softmax 等解码策略。参考第 3 版草案。
- [OpenAI Python Library](https://github.com/openai/openai-python) — OpenAI (2020)
  用于与 OpenAI 兼容 API 交互的 Python 客户端库官方代码库，支持同步和异步请求。
