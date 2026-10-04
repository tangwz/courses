# 实践：使用 LangSmith 评估智能体

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/practice-evaluating-agent-langsmith)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理论提供依据，但实践应用能巩固理解。学习如何使用 LangSmith 评估 LangChain 智能体。这包含配置一个简单智能体、建立评估数据集、让智能体运行于此数据集，以及部署自定义评估逻辑以程序方式评测其表现。

本练习假定您已配置好 LangSmith，并且您的 API 密钥已在环境中配置 (`LANGCHAIN_API_KEY`)。您还应基本熟悉创建 LangChain 智能体和使用工具。

### 1. 定义待测智能体

首先，我们来定义一个使用搜索工具的简单智能体。本例中我们将使用 Tavily 作为搜索工具，但您也可以替代为其他搜索工具或自定义工具。确保您已安装所需的软件包 (`langchain`, `langchain_community`, `langchain_openai`, `tavily-python`, `langsmith`)。此外，请将您的 Tavily API 密钥 (`TAVILY_API_KEY`) 和 OpenAI API 密钥 (`OPENAI_API_KEY`) 设置为环境变量。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langsmith import Client

# 确保 API 密钥已设置为环境变量
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGCHAIN_TRACING_V2"] = "true" # 确保追踪已启用
# os.environ["LANGCHAIN_PROJECT"] = "Agent Evaluation Example" # 可选：定义一个 LangSmith 项目

# 初始化 LLM 和工具
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# 获取提示模板
# 使用标准 OpenAI 工具智能体提示
prompt = hub.pull("hwchase17/openai-tools-agent")

# 创建智能体
# 此智能体设计用于支持工具调用的模型
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试调用（可选）
# print(agent_executor.invoke({"input": "What was the score of the last SF Giants game?"}))
```

现在我们有了 `agent_executor`，它代表我们要评估的系统。

### 2. 在 LangSmith 中创建评估数据集

评估需要一组输入，理想情况下还需要预期输出或评判智能体表现的标准。我们来使用客户端库直接在 LangSmith 中创建一个小型数据集。我们将包含输入（给智能体的问题）和可选的参考输出。

```python
# 初始化 LangSmith 客户端
client = Client()

dataset_name = "Simple Search Agent Questions V1"
dataset_description = "需要网页搜索的基本问题。"

# 检查数据集是否存在，如果不存在则创建
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"Dataset '{dataset_name}' already exists.")
except Exception: # 如果找不到，LangSmith 客户端会引发通用异常
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=dataset_description,
    )
    print(f"Created dataset '{dataset_name}'.")

    # 定义示例（输入问题和可选参考输出）
    examples = [
        ("What is the capital of France?", "Paris"),
        ("Who won the 2023 Formula 1 Championship?", "Max Verstappen"),
        ("What is the main component of air?", "Nitrogen"),
        ("Summarize the plot of the movie 'Inception'.", "A thief who steals information by entering people's dreams takes on the inverse task of planting an idea into a target's subconscious."), # 示例参考输出
    ]

    # 将示例添加到数据集
    for input_query, reference_output in examples:
        client.create_example(
            inputs={"input": input_query},
            outputs={"reference": reference_output}, # 使用 'reference' 键作为预期输出
            dataset_id=dataset.id,
        )
    print(f"Added {len(examples)} examples to dataset '{dataset_name}'.")
```

运行此代码后，您应该会在 LangSmith 账户中看到一个名为“Simple Search Agent Questions V1”的新数据集，其中填充了已定义的示例。`create_example` 中的 `outputs` 字典可以存储参考值、标签或任何其他对评估有帮助的信息。

### 3. 使用 LangSmith 运行评估

定义好智能体并创建数据集后，我们现在可以使用 LangSmith 的评估工具让智能体对数据集中的每个示例运行。我们将首先不使用自定义评估器，主要用于收集追踪记录和观察行为。

```python
from langsmith.evaluation import evaluate

# 定义一个封装智能体调用的函数
# 这是 evaluate 函数所需的
def agent_predictor(inputs: dict) -> dict:
    """运行智能体执行器以处理给定的输入字典。"""
    return agent_executor.invoke({"input": inputs["input"]}) # 假定数据集输入为 "input"

# 运行评估
# 这将对数据集中每个示例执行 agent_predictor
# 结果和追踪将自动记录到 LangSmith
evaluation_results = evaluate(
    agent_predictor,
    data=dataset_name, # 可以直接传递数据集名称
    description="搜索智能体的初步评估运行。",
    project_name="Agent Eval Run - Simple Search", # 可选：记录到特定项目运行
    # metadata={"agent_version": "1.0"}, # 可选：为运行添加元数据
)

print("评估运行完成。请在 LangSmith 中查看结果。")
```

导航到您的 LangSmith 项目。您应该会找到一个与数据集关联的新评估运行。点击它以查看：

- **追踪：** 数据集中每个示例的智能体执行详细日志。您可以检查 LLM 调用、工具使用、输入和输出。
- **结果表：** 显示输入、实际输出、参考输出（如果提供）、延迟和令牌计数的概览视图。

### 4. 部署自定义评估器

仅仅运行智能体和追踪记录对调试有用，但定量评估需要定义具体指标。我们来创建一个自定义评估器函数，它检查智能体的输出是否包含参考答案（不区分大小写）。

```python
from langsmith.evaluation import EvaluationResult, run_evaluator

@run_evaluator
def check_contains_reference(run, example) -> EvaluationResult:
    """
    检查智能体的输出是否包含参考答案（不区分大小写）。

    参数：
        run：智能体执行的 LangSmith 运行对象。
        example：数据集中来自 LangSmith 的示例对象。

    返回：
        一个 EvaluationResult，包含得分（包含为 1，否则为 0）
        和描述性内容。
    """
    agent_output = run.outputs.get("output") if run.outputs else None
    reference_output = example.outputs.get("reference") if example.outputs else None

    if agent_output is None or reference_output is None:
        # 处理输出或参考缺失的情况
        score = 0
        comment = "智能体输出或参考输出缺失。"
    elif str(reference_output).lower() in str(agent_output).lower():
        score = 1 # 成功：在智能体输出中找到参考
        comment = "找到参考答案。"
    else:
        score = 0 # 失败：未找到参考
        comment = f"未在输出中找到参考 '{reference_output}'。"

    return EvaluationResult(
        key="contains_reference", # 指标名称
        score=score,              # 数值分数（此处为 0 或 1）
        comment=comment           # 可选的定性反馈
    )
```

此函数使用 `@run_evaluator` 装饰器，表明它专为 LangSmith 评估设计。它访问智能体的实际输出 (`run.outputs`) 和存储在数据集中的参考输出 (`example.outputs`)。它返回一个包含指标名称和得分的 `EvaluationResult` 对象。

### 5. 使用自定义评估器运行评估

现在，我们再次运行评估，这次包含我们的自定义评估器。

```python
# 再次运行评估，这次使用自定义评估器
custom_eval_results = evaluate(
    agent_predictor,
    data=dataset_name,
    evaluators=[check_contains_reference], # 传递自定义评估器函数
    description="使用自定义 'contains_reference' 检查的评估运行。",
    project_name="Agent Eval Run - Custom Check", # 记录到不同的运行项目
    # metadata={"agent_version": "1.0", "evaluator": "contains_reference_v1"},
)

print("使用自定义评估器的评估运行完成。请检查 LangSmith。")
```

返回 LangSmith 并查看此新的评估运行。在结果表中，您现在应该会看到一个名为 `contains_reference` 的新列（与我们 `EvaluationResult` 中的引用匹配）。此列将基于我们的自定义逻辑为每个示例显示得分（0 或 1）。您可以根据此指标进行排序和筛选，以快速识别失败情况。鼠标悬停或点击反馈单元格通常会显示评估器提供的 `comment`。

如果我们要可视化此简单评估的结果（假设基于 `contains_reference` 分数），它可能看起来像这样：



[交互图表：评估结果：'contains_reference' 指标](https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/practice-evaluating-agent-langsmith#plot-n6nr2v)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "通过 (1)",
        "失败 (0)"
      ],
      "y": [
        3,
        1
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#37b24d",
          "#f03e3e"
        ]
      }
    }
  ],
  "layout": {
    "title": "评估结果：'contains_reference' 指标",
    "xaxis": {
      "title": "得分"
    },
    "yaxis": {
      "title": "示例数量"
    },
    "bargap": 0.2
  }
}
```

</details>



> 一个简单的条形图，显示通过（得分=1）和失败（得分=0）`contains_reference` 评估指标的示例数量。

### 后续步骤

本实践练习演示了使用 LangSmith 评估智能体的核心流程：定义智能体、创建数据集、运行评估和部署自定义检查。在此基础上，您可以开展更复杂的评估：

- **更精密的评估器：** 部署使用正则表达式、语义相似性检查（例如，比较实际输出和参考输出的嵌入 (embedding)），或检查输出特定结构属性的评估器。
- **LLM 作为评判者：** 使用另一个 LLM 根据诸如帮助性、正确性（与参考相比）或是否不含有害内容等标准来评估智能体的输出。LangChain 提供创建这些评估器的辅助工具。
- **多个评估器：** 将不同的评估器函数列表传递给 `evaluate`，以同时计算多个指标。
- **数据集管理：** 制定策略，随着您的应用发展，对评估数据集进行整理和版本控制。
- **与 CI/CD 集成：** 将这些评估运行自动化，作为持续集成流程的一部分，以便在部署前发现退步。

使用 LangSmith 等工具进行系统化评估对于构建可靠的 LLM 应用来说必不可少。它比零星测试更进一步，提供可量化 (quantization)指标和详细追踪，以便随着时间推移理解和提升智能体表现。

## 参考资料

- [LangSmith Documentation: Evaluation](https://docs.smith.langchain.com/evaluation) — LangChain (2024)
  LangSmith官方文档，包含评估LLM应用和智能体的核心概念与实践。
- [LangChain Documentation: Agents](https://python.langchain.com/docs/modules/agents/) — LangChain (2024)
  提供使用LangChain框架构建和配置智能体的全面信息，包括各种智能体类型和工具集成。

---

[上一节](07-%E4%BA%BA%E5%B7%A5%E5%8F%82%E4%B8%8E%E5%8F%8D%E9%A6%88%E4%B8%8E%E6%A0%87%E6%B3%A8.md) · [下一节](../06-%E4%BC%98%E5%8C%96%E5%92%8C%E6%89%A9%E5%B1%95%20LangChain%20%E5%BA%94%E7%94%A8/01-%E6%9F%A5%E6%98%8E%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md)
