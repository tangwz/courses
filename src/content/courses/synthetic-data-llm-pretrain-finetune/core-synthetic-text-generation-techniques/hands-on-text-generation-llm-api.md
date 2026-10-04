---
course: "synthetic-data-llm-pretrain-finetune"
chapter: "core-synthetic-text-generation-techniques"
lesson: "hands-on-text-generation-llm-api"
sourceId: 6126
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-2-core-synthetic-text-generation-techniques/hands-on-text-generation-llm-api"
title: "动手实践：使用大型语言模型API生成文本"
description: "一个关于使用大型语言模型API生成合成文本样本的动手环节。"
order: 7
plots: []
sourceHash: "ec2ae19329d8601c36af5e6a6abebb8866d6df17d83ca8e2557e5f4a267d2e32"
sourceCorrections: []
---

生成合成文本有多种方法。一个灵活的办法是直接使用大型语言模型（LLMs）自身。提供了一个使用大型语言模型API生成文本样本的实践指南。侧重于如何发起API调用、构建请求以及理解响应。这构成了许多复杂合成数据生成策略的实践依据，包括那些涉及精细提示工程 (prompt engineering)的策略。

### 准备工作：你需要什么

在开始编写代码之前，请确保你已具备必要的工具：

1. **Python环境**：本次实践假设你已安装Python（推荐3.7或更高版本）。如果你在一个项目中工作，使用虚拟环境是一个好习惯。
2. **`requests` 库**：我们将使用流行的`requests`库向大型语言模型API发起HTTP调用。如果尚未安装，可以通过pip添加：

   ```bash
   pip install requests
   ```
3. **大型语言模型API访问权限**：你需要访问大型语言模型的API。许多提供商都为其模型提供API（例如OpenAI、Cohere、AI21 Labs，或者通过Hugging Face推理 (inference)端点等服务托管的开源模型，或使用Ollama等本地工具运行并提供API）。
   - **API密钥**：大多数API需要API密钥进行身份验证和使用跟踪。你通常可以在提供商网站的账户仪表板中找到此密钥。
   - **API端点**：这是你发送请求的URL。它针对不同的LLM提供商而异，通常也与你希望使用的模型有关。

> **关于API密钥的说明：**
> 你的API密钥很敏感。请像对待密码一样对待它。对于生产应用，请避免将其直接硬编码到你的脚本中。相反，应使用环境变量或安全的秘密管理服务。本次学习练习中，我们将说明如何传递它，但请记住在实际项目中要确保其安全。

### 大型语言模型API调用的结构

通过API与大型语言模型交互通常涉及向特定端点发送HTTP请求（通常是`POST`请求）。此请求包含你的指令（即提示词 (prompt)）以及控制生成过程的各种参数 (parameter)。API随后返回一个响应，通常为JSON格式，其中包含生成的文本。

让我们来分解一下常见的组成部分：

- **端点URL**：API的特定网址。例如，它可能看起来像 `https://api.examplellm.com/v1/completions`。
- **请求头**：这些为请求提供元数据。常见请求头包括：

  - `Authorization`：携带你的API密钥，通常格式为 `Bearer YOUR_API_KEY`。
  - `Content-Type`：指定你发送的数据格式，通常是 `application/json`。
- **请求体（负载）**：这是一个JSON对象，包含大型语言模型的核心信息：

  - `prompt` （或用于聊天模型的 `messages`）：引导大型语言模型生成内容的输入文本。这是你提示工程 (prompt engineering)技能发挥作用的地方。
  - `model`：（通常是端点的一部分，或在请求体中指定）你希望使用的具体大型语言模型（例如，`text-davinci-003`、`gpt-3.5-turbo`、`command-xlarge`）。
  - `max_tokens` （或 `max_new_tokens`）：大型语言模型在其响应中应生成的最大词元 (token)数（词语或词语的一部分）。适当设置此参数有助于控制输出长度和成本。
  - `temperature`：一个控制输出随机性的值（例如，0.0到2.0）。较低的值（例如，0.2）使输出更具确定性和集中性，适用于事实性任务。较高的值（例如，0.8）使其更具创造性和多样性，适用于头脑风暴或故事生成。
  - `top_p` （核采样）：一种替代temperature来控制随机性的方法。它只考虑累积概率超过`top_p`的最小词元集合。典型值为0.9。通常建议不要同时使用`temperature`和`top_p`，或者将其中一个设置为其默认值（例如，如果使用`temperature`，则将`top_p`设置为1.0）。
  - `n`：为每个提示词生成的完成数量。请求多个完成可以帮助从单个提示词获得多样化的输出。
- **响应体**：API将返回一个JSON响应。结构因提供商而异，但你通常会发现：

  - 生成的文本，通常嵌套在 `choices[0].text` 或 `choices[0].message.content` 这样的结构中。
  - 使用信息（例如，消耗的词元数）。
  - 如果出现问题，会有错误消息。

### 使用Python生成文本：一个实践范例

让我们编写一个Python脚本来与一个通用的大型语言模型API进行交互。我们的目标是生成简单的指令-响应对，这是一种常用于微调 (fine-tuning)的合成数据类型。

```python
import requests
import json
import os

# 建议从环境变量加载你的API密钥
# 例如，在你的终端中设置：export LLM_API_KEY="你的实际API密钥"
# 或者，对于本例，你可以暂时直接替换os.getenv，
# 但请谨慎将密钥提交到版本控制中。
API_KEY = os.getenv("LLM_API_KEY") 
# 替换为你选择的LLM提供商的实际API端点
API_ENDPOINT = "https://api.examplellmprovider.com/v1/completions" 

def generate_text_from_llm(prompt_text, max_tokens=150, temperature=0.7, model_name="text-davinci-003"):
    """
    使用大型语言模型API生成文本。
    根据你具体的LLM提供商调整参数和负载结构。
    """
    if not API_KEY:
        print("错误：LLM_API_KEY 环境变量未设置。")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 负载结构在不同的LLM提供商之间可能差异很大。
    # 这是完成式模型常见的结构。
    # 对于聊天模型（例如，GPT-3.5-turbo及后续版本），负载可能看起来像：
    # data = {
    #     "model": model_name,
    #     "messages": [{"role": "user", "content": prompt_text}],
    #     "max_tokens": max_tokens,
    #     "temperature": temperature,
    # }
    # 请务必查阅你LLM提供商的API文档。

    data = {
        "model": model_name, # 某些API从端点推断模型
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1, # 生成的完成数量
        # "top_p": 0.9, # 另一个参数的示例
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()  # 对于错误响应（4XX或5XX）抛出HTTPError

        response_json = response.json()

        # 生成文本的路径因API提供商而异
        # 常见路径：
        # response_json['choices'][0]['text'] (OpenAI 完成模型)
        # response_json['choices'][0]['message']['content'] (OpenAI 聊天模型)
        # response_json['generations'][0]['text'] (Cohere)
        # 查阅你的API文档！
        # 对于这个通用示例，我们假设一个常见路径：
        if 'choices' in response_json and len(response_json['choices']) > 0:
            if 'text' in response_json['choices'][0]:
                 return response_json['choices'][0]['text'].strip()
            elif 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                 return response_json['choices'][0]['message']['content'].strip()

        print("警告：未能在响应的预期位置找到生成的文本。")
        print("完整响应：", response_json)
        return None

    except requests.exceptions.RequestException as e:
        print(f"发生API请求错误：{e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容：{e.response.text}")
        return None
    except json.JSONDecodeError:
        print("解码API的JSON响应时出错。")
        print(f"响应内容：{response.text}")
        return None

if __name__ == "__main__":
    # 示例：生成简单的指令-响应对
    # 此提示词引导LLM创建一个问题及其答案
    # 基于一个角色和主题。
    instruction_prompt = """
    生成一个友好的AI助手可能提供的问题和简洁答案。
    主题应关于“在Python开发中使用虚拟环境的好处”。
    输出格式为：
    指令: [生成的问题]
    响应: [生成的答案]
    """

    print(f"发送提示词到LLM：\n{instruction_prompt}")

    generated_content = generate_text_from_llm(
        instruction_prompt,
        max_tokens=200,  # 为问题和答案留出更多词元
        temperature=0.5  # 对于此任务，使其更具确定性
    )

    if generated_content:
        print("\n--- 生成内容 ---")
        print(generated_content)
        print("-------------------------")
    else:
        print("\n未能生成内容。")

    # 示例：生成创意产品描述
    product_prompt = """
    为一款名为“Waker's Brew”的新款手工咖啡撰写两段独特且有吸引力的产品描述。
    每段描述应为2-3句话，并突出一个独特的方面（例如，产地、烘焙过程、风味）。
    以以下形式呈现：
    描述 1: ...
    描述 2: ...
    """
    print(f"\n发送提示词到LLM：\n{product_prompt}")
    creative_content = generate_text_from_llm(
        product_prompt,
        max_tokens=250,
        temperature=0.8 # 更高的temperature以获得更具创意的输出
    )

    if creative_content:
        print("\n--- 生成创意内容 ---")
        print(creative_content)
        print("----------------------------------")
    else:
        print("\n未能生成创意内容。")
```

> **运行前须知：**
>
> - 请将 `"https://api.examplellmprovider.com/v1/completions"` 替换为你所选大型语言模型提供商的实际API端点。
> - 确保你的`LLM_API_KEY`环境变量已设置，或者暂时硬编码你的API密钥（之后记得删除）。
> - 你可能需要调整 `generate_text_from_llm` 中的 `data` 负载（例如，`model`、参数 (parameter)名称、提示结构）以及从 `response_json` 中提取生成文本的方式，以匹配你特定大型语言模型提供商的API文档。代码中的注释提供了常见变体的提示。

当你运行这个脚本时，它会将定义的提示词 (prompt)发送到大型语言模型API并打印生成的文本。你将看到一个示例，说明精心构建的提示词如何引导大型语言模型生成结构化输出，如指令-响应对或多样化的产品描述。

### 试验并优化生成结果

这个动手实践提供了一个起点。真正的优势来自尝试：

- **改变提示词 (prompt)**：尝试在你的提示词中使用不同的措辞和详细程度。正如“通过有效的提示设计指导生成”一章中讨论的，提示词是你控制大型语言模型的主要工具。例如，要为特定领域生成数据，请将领域特定的术语和上下文 (context)嵌入 (embedding)到你的提示词中。
- **调整参数 (parameter)**：
  - 改变`temperature`。观察较低的`temperature`（例如0.2）如何带来更可预测、更集中的输出，而较高的`temperature`（例如0.9）则会产生更多样或出乎意料的文本。
  - 修改`max_tokens`。如果你的输出被截断，请增加`max_tokens`。如果它过于冗长，则减少它。
  - 如果你的API支持，尝试为单个提示词生成多个完成（`n > 1`），以查看一系列可能性。
- **迭代任务特定性**：如果你正在为特定任务生成数据，例如创建合成客户支持查询，请优化你的提示词，使其包含所需风格、语气和内容的示例。

### 从单个API调用进展

虽然本示例侧重于单个API调用，但在实践中，你通常会构建脚本或管道来生成大量的合成数据。这可能涉及：

- 遍历种子提示词 (prompt)或主题列表。
- 以编程方式将提示模板与可变输入结合。
- 将生成的文本保存到文件（例如JSONL、CSV）中，以便后续用于预训练 (pre-training)或微调 (fine-tuning)。

这些更高级的工作流直接建立在你在此处实践的基本API交互技术之上。

### 关于质量的简要说明

当你生成合成文本时，请记住大型语言模型有时会产生有偏见、事实不准确或重复的输出。合成数据的质量非常重要。后续章节，特别是第六章“评估合成数据和应对操作挑战”，将介绍评估和改进生成数据集质量的方法。目前，请在尝试时保持警惕。

这个动手实践使你具备了使用大型语言模型API进行文本生成的能力。这项技能对于为各种大型语言模型开发需求创建多样化的合成数据集非常必要，从扩充预训练 (pre-training)语料库到制作专门的微调 (fine-tuning)数据集。接下来的章节将在此基础上进行扩展，研究如何有效地应用这些生成的数据集。

## 参考资料

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — OpenAI (2024)
  Publisher: OpenAI
  与OpenAI大型语言模型API交互的指南，详细介绍了端点、参数和数据结构。
