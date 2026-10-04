---
course: "agentic-llm-memory-architectures"
chapter: "evaluation-optimization-agentic-systems"
lesson: "practice-evaluation-harness"
sourceId: 4552
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-6-evaluation-optimization-agentic-systems/practice-evaluation-harness"
title: "实践：构建评估框架"
description: "构建一个基础评估框架，用于测试智能体系统的各个组成部分。"
order: 9
plots: ["plots/4552-0.json"]
sourceHash: "f67bf14b11f5ccce2e6111a8498daa1720080ced7330b2676f40faddc28d3663"
sourceCorrections: []
---

在前面章节中对严格评估必要性进行阐述的基础上，我们现在转向评估框架的具体实现。理论指标和评估策略固然有价值，但将其转化为可重复、自动化的流程对于有效迭代和改进您的智能体系统而言不可或缺。本节将引导您建立一个评估基础，即一种专门用于针对预定义测试用例运行智能体并系统地衡量其表现的软件工具。

环境是指一个受控设置，可以在此观察、衡量和比较智能体行为。它使执行过程标准化，确保结果的差异归因于智能体的逻辑或配置，而非测试设置的不一致。虽然复杂的系统可能变得复杂，但我们将着重于建立一个坚实、可扩展的起点。

### 评估框架的主要构成部分

一个基础而有效的评估通常包含以下构成部分：

1. **测试用例集**：一组定义的场景，旨在考察智能体的特定能力。每个测试用例通常包含：
   - 初始输入或目标（例如，一个复杂问题、一个任务描述）。
   - 所需上下文 (context)或可用工具（例如，模拟数据库访问、特定API端点）。
   - 参考答案或成功标准（例如，预期最终答案、必须完成的特定子任务、必须遵守的限制）。
2. **智能体接口**：围绕智能体代码的标准包装器。它使得能通过一致的API与不同的智能体实现（例如，ReAct智能体与ToT智能体）进行交互，通常涉及`initialize()`和`run(input)`等方法。
3. **执行引擎**：负责加载测试用例、通过接口初始化智能体、在每个测试用例上执行智能体并收集输出的协调器。这不仅包括捕获最终结果，还可能包含中间推理 (inference)步骤、工具调用和内存交互（智能体的“轨迹”）。
4. **指标计算器**：处理原始执行输出以计算本章前面定义的性能指标的函数或类（例如，目标完成率、任务成功、规划准确性、工具调用精确率/召回率、幻觉 (hallucination)频率、以及令牌计数或延迟等资源使用情况）。
5. **结果记录器/报告器**：负责以结构化格式（例如，JSON文件、CSV、数据库）存储原始输出和计算指标，并可能生成汇总报告或可视化内容的模块。

### 构成部分的实现

实际实现细节以Python为例介绍。

#### 测试用例定义

测试用例可以用各种格式定义，例如YAML或JSON文件，或者直接作为Python字典或数据类。结构化格式更利于清晰度和加载便捷性。

```python
# 示例测试用例结构（使用Python字典）
test_case_1 = {
    "id": "TC001",
    "goal": "查找ExampleCorp现任CEO并总结他们关于AI的最新公开声明。",
    "available_tools": ["web_search", "company_database_lookup"],
    "success_criteria": {
        "ceo_identified": "Jane Doe", # 示例参考答案
        "statement_found": True,
        "summary_relevant": True, # 需要语义检查
        "constraints": ["必须至少使用一次web_search工具"]
    },
    "max_steps": 10 # 可选约束
}

test_suite = [test_case_1, ...] # 从文件加载或内联定义
```

对于专业级别的评估，测试用例集应涵盖各种情况：简单查询、多步推理 (inference)任务、需要复杂工具交互的任务、旨在触发已知故障模式（例如，歧义、冲突信息）的场景以及边缘情况。

#### 智能体接口

一个简单的基类可以定义预期接口：

```python
from abc import ABC, abstractmethod

class BaseAgentInterface(ABC):
    def __init__(self, config):
        self.config = config
        # 根据配置初始化大型语言模型、工具、内存

    @abstractmethod
    def run(self, goal: str, available_tools: list) -> dict:
        """
        执行智能体针对给定目标的逻辑。
        返回一个字典，包含最终答案、执行轨迹、
        工具调用、错误等。
        """
        pass

# 特定智能体类型的示例实现
class MyReActAgent(BaseAgentInterface):
    def run(self, goal: str, available_tools: list) -> dict:
        # 该智能体的ReAct循环实现
        trajectory = []
        final_answer = None
        tool_calls = []
        errors = []
        # ... 智能体执行逻辑 ...
        print(f"Running ReAct Agent on goal: {goal}") # 示例日志记录
        # 模拟执行
        trajectory.append("Thought: 我需要找到CEO。")
        trajectory.append("Action: company_database_lookup(company='ExampleCorp')")
        tool_calls.append({"tool": "company_database_lookup", "args": {"company": "ExampleCorp"}, "output": "CEO: Jane Doe"})
        trajectory.append("Observation: 找到CEO是Jane Doe。")
        trajectory.append("Thought: 现在搜索她最新的声明。")
        trajectory.append("Action: web_search(query='Jane Doe ExampleCorp latest AI statement')")
        tool_calls.append({"tool": "web_search", "args": {"query": "Jane Doe ExampleCorp latest AI statement"}, "output": "Snippet: ...承诺负责任的AI..."})
        trajectory.append("Observation: 找到相关声明片段。")
        final_answer = "CEO是Jane Doe。最新声明强调对负责任AI的承诺。"

        return {
            "final_answer": final_answer,
            "trajectory": trajectory,
            "tool_calls": tool_calls,
            "errors": errors,
            "steps_taken": len(trajectory) // 2 # 大约步数
        }
```

这种抽象使得能够轻松替换不同的智能体实现。

#### 指标计算

指标应基于智能体`run`方法返回的结果字典以及测试用例中的参考答案进行操作。

```python
def calculate_metrics(agent_output: dict, test_case: dict) -> dict:
    metrics = {}
    criteria = test_case["success_criteria"]

    # 示例：基本目标完成检查
    is_successful = True
    if "ceo_identified" in criteria:
         # 简单字符串检查（可更复杂）
        if criteria["ceo_identified"] not in agent_output.get("final_answer", ""):
             is_successful = False
    if criteria.get("statement_found", False):
        # 针对最终答案内容的检查占位符
        if "statement" not in agent_output.get("final_answer", "").lower():
            is_successful = False # 简化检查

    metrics["success"] = is_successful

    # 示例：工具使用检查
    required_tool_used = False
    if "constraints" in criteria:
        for constraint in criteria["constraints"]:
             if "Must use web_search" in constraint:
                 if any(call["tool"] == "web_search" for call in agent_output.get("tool_calls", [])):
                     required_tool_used = True
                 else:
                     # 约束被违反，可能标记为失败或单独记录
                     pass # 根据需要添加逻辑
    metrics["required_tool_used"] = required_tool_used

    # 示例：资源使用
    metrics["steps_taken"] = agent_output.get("steps_taken", 0)
    # 如果有跟踪，还可以添加令牌计数、延迟

    return metrics
```

对于专业用例，指标计算可能涉及复杂技术，例如使用嵌入 (embedding)进行答案相关性的语义相似性检查、解析工具参数 (parameter)以验证正确性，或分析推理轨迹以发现逻辑谬误。

#### 执行引擎和日志记录

引擎进行迭代、执行、计算和日志记录。

````python
import json
import datetime

def run_evaluation(agent_interface: BaseAgentInterface, test_suite: list):
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"evaluation_results_{timestamp}.jsonl"

    print(f"Starting evaluation run. Logging to {log_filename}")

    for i, test_case in enumerate(test_suite):
        print(f"Running Test Case {i+1}/{len(test_suite)}: {test_case['id']}")
        try:
            agent_output = agent_interface.run(
                goal=test_case["goal"],
                available_tools=test_case["available_tools"]
            )
            computed_metrics = calculate_metrics(agent_output, test_case)

            result_entry = {
                "test_case_id": test_case["id"],
                "goal": test_case["goal"],
                "agent_config": agent_interface.config, # 存储使用的智能体配置
                "raw_output": agent_output,
                "metrics": computed_metrics,
                "success": computed_metrics.get("success", False) # 提升主要指标
            }
        except Exception as e:
            print(f"Error running test case {test_case['id']}: {e}")
            result_entry = {
                "test_case_id": test_case["id"],
                "goal": test_case["goal"],
                "agent_config": agent_interface.config,
                "error": str(e),
                "success": False,
                "metrics": {"success": False} # 确保指标字典存在
            }

        results.append(result_entry)

        # 增量记录结果 (JSON Lines 格式)
        with open(log_filename, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')

    print("Evaluation run completed.")
    # 汇总并报告统计数据
    total_tests = len(results)
    successful_tests = sum(r.get('success', False) for r in results)
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSummary:")
    print(f"Total Test Cases: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success Rate: {success_rate:.2f}%")

    # 在此处添加更详细的报告或可视化生成
    generate_summary_visualization(results, timestamp) # 调用可视化函数

    return results

def generate_summary_visualization(results: list, timestamp: str):
    # 示例：按测试用例可视化成功率（简化版）
    if not results: return

    ids = [r['test_case_id'] for r in results]
    success_values = [1 if r.get('success', False) else 0 for r in results] # 成功为1，失败为0

    # 创建一个简单的柱状图，显示每个测试用例的成功/失败
    plotly_fig = {
        "data": [
            {
                "x": ids,
                "y": success_values,
                "type": "bar",
                "marker": {
                    "color": ['#37b24d' if s == 1 else '#f03e3e' for s in success_values] # 成功为绿色，失败为红色
                },
                "name": "Test Outcome"
            }
        ],
        "layout": {
            "title": f"Evaluation Results ({timestamp})",
            "xaxis": {"title": "Test Case ID", "type": "category"},
            "yaxis": {"title": "Outcome (1=Success, 0=Fail)", "tickvals": [0, 1], "ticktext": ["Fail", "Success"]},
            "template": "plotly_white" # 使用简洁模板
        }
    }

    # 保存或显示图表（实现取决于环境）
    # 对于网页输出，您可以保存此JSON或将其传递给前端组件
    viz_filename = f"evaluation_summary_{timestamp}.json"
    with open(viz_filename, 'w') as f:
        json.dump(plotly_fig, f)
    print(f"Visualization data saved to {viz_filename}")

    # 示例 Plotly JSON 结构（用于嵌入的单行）：
    # ```plotly
    # {"data": [{"x": ["TC001", "TC002"], "y": [1, 0], "type": "bar", "marker": {"color": ["#37b24d", "#f03e3e"]}, "name": "Test Outcome"}], "layout": {"title": "Evaluation Results (Example)", "xaxis": {"title": "Test Case ID", "type": "category"}, "yaxis": {"title": "Outcome (1=Success, 0=Fail)", "tickvals": [0, 1], "ticktext": ["Fail", "Success"]}, "template": "plotly_white"}}
    # ```

# 示例用法
# agent_config = {"llm": "gpt-4", "react_params": {...}}
# agent = MyReActAgent(config=agent_config)
# evaluation_results = run_evaluation(agent, test_suite)
````



![示例测试用例结果](plots/4552-0.json)



> 柱状图展示了四个不同测试用例的结果，其中绿色表示成功，红色表示失败。

### 高级考量

对于专业级别的应用，可以增强这种基础方法：

- **异步执行**：并行运行多个测试用例，特别是当智能体执行涉及大量I/O或等待时间（例如，API调用）时，以加快评估速度。使用Python的`asyncio`库。
- **依赖管理**：确保外部依赖项（数据库、API、特定库版本）在不同运行中保持一致。考虑对评估环境进行容器化（例如，Docker）。
- **故障分析**：扩展日志记录以捕获详细的错误消息和故障发生时的智能体状态。实施机制以自动分类常见故障模式。
- **比较评估**：设计系统以轻松针对同一测试用例集运行多个智能体版本或配置，并生成比较报告。
- **人机协作**：对于“答案相关性”或“推理 (inference)质量”等主观指标，加入人类评估员审查和评分智能体输出的界面。
- **成本追踪**：如果使用付费大型语言模型API，根据每次测试运行的令牌用量整合成本估算。

构建一个基础的评估流程也能带来极高价值，因为它将评估从临时性活动转变为系统化、可重复的流程。它为数据驱动的开发奠定根基，使您能够可靠地追踪进展、识别退步，并精确指出复杂系统中可优化的地方。从小处着手，并随着评估需求的复杂化逐步完善您的流程。

## 参考资料

- [AgentBench: Evaluating LLMs as Agents](https://doi.org/10.48550/arXiv.2308.03688) — Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang (2023)
  Journal: arXiv preprint arXiv:2308.03688; DOI: [10.48550/arXiv.2308.03688](https://doi.org/10.48550/arXiv.2308.03688)
  这篇论文介绍了一个评估大型语言模型作为智能体的综合基准，涵盖了各种环境和指标，与构建评估框架高度相关。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  这篇关于ReAct智能体的基础论文详细阐述了用于衡量大型语言模型智能体推理和行动能力的实验设置和评估方法。
- [OpenAI Evals](https://github.com/openai/evals) — OpenAI (2023)
  Publisher: OpenAI
  OpenAI 官方的开源框架，用于评估大型语言模型系统和智能体，提供了定义测试用例、运行评估和计算指标的实践示例。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning-systems/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书讨论了构建可靠机器学习系统的原则，包括有效的评估流程和持续改进策略，适用于生产级智能体系统。
