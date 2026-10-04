# 实践：构建具有特定功能的LLM代理

来源：[原文](https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-2-architecting-agents-defining-roles/hands-on-llm-agent-construction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

一个实践练习将指导您构建一个具有特定角色和功能的初始LLM代理，过程中将涉及定义代理的特点、专业职能以及集成基本工具的使用。目标是构建一个能够从给定URL获取信息、进行总结并提取主要主题的“网页内容摘要器”代理。

### 定义我们的代理：网页内容摘要器

在编写任何代码之前，让我们明确定义我们的代理：

1. **特点**：“我是一名勤奋的研究助理。我的作用是处理网页文章，为您提供简洁的摘要和所讨论的主要主题列表。我追求清晰和准确。”
2. **功能**:
   - 接收URL作为输入。
   - 从该URL的网页中获取文本内容。
   - 使用LLM生成内容摘要。
   - 使用LLM从内容中识别3-5个主要主题。
   - 返回摘要和主题列表。
3. **工具**:
   - 一个网页获取工具（使用`requests`和`BeautifulSoup4`等库）。
   - 一个用于自然语言处理任务（摘要、主题提取）的LLM。

这个代理虽然简单，但它展示了核心思想，即赋予LLM驱动实体一个特定的工作描述以及完成任务的途径。

### 前提条件

请确保您已安装以下Python库：

```bash
pip install openai requests beautifulsoup4
```

您还需要一个OpenAI API密钥。将其设置为环境变量是良好的实践：

```bash
export OPENAI_API_KEY='your_api_key_here'
```

请记住，本次实践练习假设您的开发环境已按照第1章中的说明进行配置。

### 代理实现

让我们一步步构建`WebSummarizerAgent`。

#### 1. 导入库和初始化LLM客户端

首先，我们将导入必要的模块并设置我们的OpenAI客户端。

```python
import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 初始化OpenAI客户端
# 它将自动获取OPENAI_API_KEY环境变量
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    exit()

# LLM的配置
LLM_MODEL = "gpt-3.5-turbo"
```

对于这项任务，使用`gpt-3.5-turbo`能在能力和成本之间提供良好的平衡。您可以根据需要替换其他模型。

#### 2. 制作网页获取工具

我们的代理需要从URL检索内容。我们将为此创建一个函数。

```python
def fetch_web_content(url: str) -> str:
    """
    从给定URL获取并提取文本内容。
    返回文本内容或错误消息。
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 对于不良响应（4XX或5XX）引发HTTPError

        soup = BeautifulSoup(response.content, 'html.parser')

        # 从常见的文本标签中提取文本，尝试获取主要内容
        paragraphs = soup.find_all(['p', 'article', 'main'])
        if not paragraphs: # 针对简单页面的备用方案
            paragraphs = soup.find_all('body')

        text_content = ""
        for p_tag in paragraphs:
            text_content += p_tag.get_text(separator=' ', strip=True) + "\n"

        # 基本清理：减少多余的换行和空格
        text_content = '\n'.join([line.strip() for line in text_content.splitlines() if line.strip()])
        text_content = ' '.join(text_content.split())

        # 限制内容长度以避免过多的token使用
        # 这是一个简单的截断；对于非常长的文章，可能需要更复杂的块处理方法
        max_length = 15000 # 大约4k token，具体取决于文本
        return text_content[:max_length]

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"
    except Exception as e:
        return f"Error processing content from {url}: {e}"
```

这个`fetch_web_content`函数尝试获取网页的主要文本部分。网页抓取可能复杂得多，通常需要针对特定网站的解析器，但这提供了一种通用方法。内容被截断以防止LLM输入过长。

#### 3. 设计代理的核心逻辑

现在，我们将定义代理的主要处理函数。这个函数将协调内容获取，然后使用LLM进行摘要和主题提取。

```python
def process_url_with_agent(url: str) -> dict:
    """
    协调网页内容获取、摘要和主题提取。
    """
    print(f"Processing URL: {url}...")
    content = fetch_web_content(url)

    if content.startswith("Error"):
        return {"error": content}

    if not content.strip():
        return {"error": "No text content found at the URL."}

    # LLM的角色设定和指令
    system_prompt = (
        "You are a diligent Research Assistant. "
        "Your purpose is to process the provided web article text, "
        "generate a concise summary, and identify 3-5 main topics. "
        "Format your response as a JSON object with two keys: 'summary' and 'main_topics' (a list of strings)."
    )

    user_prompt = f"Please process the following web article content:\n\n{content}"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3, # 降低温度以获得更真实、更少创造性的输出
            response_format={"type": "json_object"} # 请求JSON输出
        )

        # 响应内容应为JSON字符串。
        # 我们将直接解析它。
        assistant_reply = response.choices[0].message.content
        # 如果需要在解析前进行基本验证，或者依赖OpenAI的JSON模式
        import json
        result = json.loads(assistant_reply)

        return result

    except Exception as e:
        return {"error": f"LLM processing error: {e}"}
```

在`process_url_with_agent`中，我们首先获取内容。然后，我们构建一个系统提示，明确定义代理的特点和任务，并重要地指示它返回一个JSON对象。这种结构化输出更容易在更大的系统中解析和使用。用户提示提供了实际内容。我们使用OpenAI的JSON模式以获得可靠的结构化输出。

一个简单的图表说明了我们代理的流程：

> 该代理接收一个URL，获取其内容，使用LLM处理，并生成摘要和主题列表。

#### 4. 执行代理并查看输出

让我们用一个示例URL测试我们的代理。

```python
if __name__ == "__main__":
    # 示例用法：
    # 替换为您选择的URL，最好是文章链接。
    # 请注意网站关于抓取的服务条款。
    test_url = "https://www.nature.com/articles/d41586-023-03913-3" # 示例：一篇《自然》杂志新闻文章

    print(f"Constructing agent to process: {test_url}")
    agent_output = process_url_with_agent(test_url)

    if "error" in agent_output:
        print(f"\nAgent Error: {agent_output['error']}")
    else:
        print("\nAgent Output:")
        print(f"  Summary: {agent_output.get('summary', 'Not available')}")
        topics = agent_output.get('main_topics', [])
        if topics:
            print("  Main Topics:")
            for i, topic in enumerate(topics):
                print(f"    {i+1}. {topic}")
        else:
            print("  Main Topics: Not available")
```

当您运行此脚本时，将`test_url`替换为当前的\`新闻文章或博客帖子。代理将获取内容，将其发送给LLM，并附带指定的特点和指令，然后打印生成的摘要和主题。

**示例输出（将根据URL和LLM的响应而有所不同）：**

```
正在处理URL: https://www.nature.com/articles/d41586-023-03913-3...

代理输出：
  摘要：文章讨论了探测超大质量黑洞合并产生的引力波所面临的挑战和取得的进展。尽管单独探测这些波很困难，但像NANOGrav这样的项目正在利用脉冲星计时阵列来寻找这些波的背景嗡嗡声。最近的发现表明一个与该背景一致的信号，尽管需要更多数据来证实。探测这些波将为了解星系演化和宇宙中巨大结构的形成提供见解。
  主要主题：
    1. 引力波探测
    2. 超大质量黑洞合并
    3. 脉冲星计时阵列（例如，NANOGrav）
    4. 星系演化
    5. 宇宙背景辐射
```

### 讨论与进一步完善

本次实践练习展示了如何构建一个简单的LLM代理，它具有明确的角色、功能和一个基本工具（网页获取）。我们看到系统提示如何确立代理的特点并引导其输出格式。

**与章节内容的关联：**

- **代理特点和功能专业化**：我们的代理具有明确的“研究助理”特点和专业功能（网页内容摘要）。
- **集成外部工具**：`fetch_web_content`函数是一个简单的集成工具。更复杂的代理可能会使用正式的工具注册和调用机制，如“为代理集成外部工具和函数”中所讨论的。
- **知识获取**：代理通过获取网页内容来访问外部知识。

**可能的改进：**

- **错误处理**：当前的错误处理是基础的。一个生产级代理需要更多的错误管理、网络问题的重试以及对格式错误网页内容更好的处理。
- **内容提取**：网页内容提取可以通过使用更高级的解析库或技术来改进，以更好地将主要文章文本与模板内容（广告、导航）隔离。
- **记忆**：为了后续交互或更复杂的任务，可以赋予这个代理记忆（短期或长期）来回顾之前的交互或信息，这在“LLM代理的记忆机制”中有所涉及。
- **高级工具使用**：如果代理需要执行多种类型的网页交互或使用其他API，一个更正式的工具使用框架（如LangChain或LlamaIndex中的，或自定义的函数调用）将是有益的。
- **可伸缩性和组织方式**：这个代理是独立的。在一个多代理系统中，它可以作为工作代理，或许从调度代理接收URL，并将结果返回给聚合或分析代理。这与“代理组织模型的比较分析”和“设计增加容量的代理系统”部分有关。
- **动态角色适应**：虽然此处未实现，但可以想象该代理根据输入参数 (parameter)动态调整其摘要风格或深度，这暗示了“动态角色分配策略”。

这个初步的代理可作为构建模块。随着您的进步，您将学会组合多个此类专业代理，协调它们之间的交互，并管理复杂的工作流程以解决更具挑战性的问题。本次练习应能让您具体理解单个代理架构如何提升大型多代理系统的能力。

## 参考资料

- [Generative Agents: Interactive Simulacra of Human Behavior](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7OuArdilEMCqSrhanWsObmzQL5rxBRPSLl1KczolRV7_K3WamRwVtzL0edI8t2Q-B4-3JzEjIFJsUoT0ZpKomCaZMIeMMt1HGhHhqRCE-rdXnxbQPtGy8tiE=) — Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein (2023)
  Journal: UIST; Pages: 22 pages; DOI: [10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
  介绍了一个计算框架，用于构建模拟可信人类行为的生成代理，包括代理角色和记忆的开发。
- [Structured output (JSON mode)](https://platform.openai.com/docs/guides/text-generation/json-mode) — OpenAI (2024)
  Journal: OpenAI Documentation
  解释了如何配置OpenAI模型以可靠地生成JSON对象，这用于代理的结构化输出。
- [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Journal: OpenAI Documentation
  提供了创建有效提示以指导LLM行为的指南和最佳实践，包括系统消息和输出控制。

---

[上一节](07-%E8%AE%BE%E8%AE%A1%E9%AB%98%E6%89%BF%E8%BD%BD%E5%8A%9B%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F.md) · [下一节](../03-%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E4%BD%93%E9%80%9A%E4%BF%A1%E4%B8%8E%E5%8D%8F%E8%B0%83/01-%E4%BB%A3%E7%90%86%E9%97%B4%E6%B6%88%E6%81%AF%E4%BA%A4%E6%8D%A2%E5%8D%8F%E8%AE%AE.md)
