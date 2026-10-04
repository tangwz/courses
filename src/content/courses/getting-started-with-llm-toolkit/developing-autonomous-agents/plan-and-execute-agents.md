---
course: "getting-started-with-llm-toolkit"
chapter: "developing-autonomous-agents"
lesson: "plan-and-execute-agents"
sourceId: 7786
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents/plan-and-execute-agents"
title: "实施规划-执行型智能体"
description: "介绍一种不同的智能体架构，在这种架构中，智能体首先制定计划，然后逐步执行。"
order: 5
plots: []
sourceHash: "b689bceb0e3e62254480d9284aab783be0ead96cb1ed53450266972f8ab512eb"
sourceCorrections: []
---

对于需要动态、分步调整的任务，一种常见做法是ReAct模式。然而，有些问题通过更具结构性、更具前瞻性的方法会做得更好。另一种架构是规划-执行型智能体。与其在每一步都重复紧密的“思考-行动-观察”循环，该智能体首先制定一个完整的、多步骤的计划，然后按顺序执行。

这种规划与执行的分离对于行动序列大体上可预测，并且不依赖于每一步高度可变结果的任务很有效。

### 规划-执行型智能体的结构

该架构包含两个不同的阶段：

1. **规划阶段**：一个专门的**规划器**组件，由大型语言模型（LLM）驱动，接收高层级目标。它的唯一职责是将此目标分解成一个具体的、有序的步骤列表。此阶段的输出是一个`Plan`对象，它作为整个任务的蓝图。
2. **执行阶段**：一个**执行器**组件遍历生成计划中的步骤。对于每一步，它执行所需的操作。执行器可以是一个简单的函数调用机制，甚至是一个ReAct智能体，其任务是完成那一个步骤。

这种工作流程确保智能体在开始前有一个清晰的前进路径，降低偏离轨道的风险。

> 规划-执行工作流程将任务分为一个规划阶段，LLM在此生成一系列步骤，和一个执行阶段，智能体在此按顺序执行每个步骤。

### 创建并执行计划

`agent`模块提供规划和执行两方面的工具。`Planner`类使用LLM根据给定目标生成一个`Plan`对象。

例如，我们可以设置一个规划器，它使用一个模拟的LLM来生成步骤。在实际应用中，`llm_func`会调用像GPT-4或Claude 3这样强大的模型。

```python
from kerb.agent.planning import Planner, Plan

def planner_llm(prompt: str) -> str:
    """一个生成计划的模拟LLM。"""
    if "data analysis" in prompt.lower():
        return """1. 从源加载数据集。
2. 清理并预处理数据。
3. 执行探索性数据分析（EDA）。
4. 生成可视化和摘要报告。"""
    return "1. 步骤 A\n2. 步骤 B"

# 使用我们的LLM函数创建一个规划器
planner = Planner(llm_func=planner_llm)

# 为高层级目标生成计划
goal = "创建数据分析流程"
generated_plan = planner.create_plan(goal)

print(f"目标: {generated_plan.goal}")
print("\n生成步骤:")
for i, step in enumerate(generated_plan.steps, 1):
    print(f"  {i}. {step}")
```

`PlanAndExecuteAgent`类对整个工作流程进行了抽象。你向它提供一个用于规划的LLM函数，它统筹计划的创建和每一步的执行。

这里是一个`PlanAndExecuteAgent`的完整示例，它首先制定一个分析客户反馈的计划，然后执行该计划。

```python
"""08_agent_planning.py"""

from kerb.agent.patterns import PlanAndExecuteAgent

def planner_llm(prompt: str) -> str:
    """用于规划的LLM。"""
    if "plan" in prompt.lower() or "analyze customer feedback" in prompt.lower():
        return """1. 研究情感分析主题。
2. 分析反馈以确定主要主题。
3. 撰写调查结果摘要。
4. 审阅并敲定报告。"""
    return "正在创建计划..."

def main():
    """运行智能体规划示例。"""

    print("="*80)
    print("规划-执行型智能体示例")
    print("="*80)

    # 创建智能体
    pae_agent = PlanAndExecuteAgent(
        name="PlanExecuteAgent",
        llm_func=planner_llm,
        max_iterations=5
    )

    goal = "分析客户反馈"
    print(f"\n目标: {goal}")
    print(f"\n智能体: {pae_agent.name}")
    print("\n正在运行规划-执行型智能体...")
    print("-"*80)

    result = pae_agent.run(goal)

    print(f"\n结果:")
    print("-"*80)
    print(f"状态: {result.status.value}")
    print(f"已采取步骤: {len(result.steps)}")
    print(f"输出: {result.output[:100]}...")

    # 显示执行流程，突出规划步骤
    print("\n执行流程:")
    print("-"*80)
    for i, step in enumerate(result.steps, 1):
        print(f"\n[{i}] ", end="")
        if step.thought:
            # 第一个“思考”通常就是计划本身
            print(f"思考:\n{step.thought}")
        if step.action:
            print(f"    行动: {step.action}")

if __name__ == "__main__":
    main()
```

智能体执行的第一步通常包含一个“思考”，该“思考”会制定整个计划。随后的步骤则侧重于执行计划的每个部分。

### 失败处理与重新规划

规划-执行型智能体的一个主要难题是其刚性。如果早期步骤失败或环境意外变化，原始计划可能会失效。实现时必须包含一个**重新规划**的机制。

重新规划涉及：

1. 在步骤执行过程中发现失败。
2. 捕获来自失败步骤的观察结果或错误消息。
3. 将原始目标、失败的计划和错误反馈给规划器。
4. 生成一个新的、修正后的计划，以应对失败。

例如，如果一个“将应用程序部署到生产环境”的计划在“运行测试”步骤失败了，因为构建错误，重新规划循环将生成一个新计划，例如：

1. 调试构建问题。
2. 修正代码库中的错误。
3. 重新运行测试。
4. 构建应用程序。
5. 部署到生产环境。

尽管`PlanAndExecuteAgent`提供高级统筹，你也可以通过围绕`Planner`和执行器构建一个循环来实施自定义的重新规划逻辑。`kerb.agent.planning.replan`函数提供实用工具来辅助此过程。

### 何时选用规划-执行型智能体

这种架构并非ReAct的普适替代品。每种模式都有其优势。

**在以下情况时，选择规划-执行型智能体：**

- **任务很复杂且需要预见性。** 提前分解大问题很有好处。
- **步骤大多可预测。** 环境稳定，且一步的结果不会剧烈改变下一步。
- **可检查的计划很有价值。** 生成的计划本身可以由人审阅，或用于进度追踪。
- **你希望尽量减少迭代式LLM调用。** 单次、全面的规划调用可能比ReAct循环中的多次小型调用更高效。

**在以下情况时，坚持使用ReAct智能体：**

- **任务具有试探性。** 智能体需要在每一步对新信息作出反应。
- **环境高度动态。** 智能体必须持续调整其策略。
- **需要即时反馈。** “思考-行动-观察”循环为快速纠正提供了紧密的反馈。

例如，生成研究报告非常适合规划-执行型智能体（研究、列提纲、起草、审阅）。相反地，浏览网页以寻找复杂问题的答案更适合ReAct智能体，因为它需要在每一步对搜索结果作出反应。

## 参考资料

- [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) — Stuart Russell and Peter Norvig (2020)
  Publisher: Pearson
  这本教科书提供了智能体、经典规划和各种AI架构的基础知识，为计划-执行范式提供了背景。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  这篇论文介绍了ReAct模式，这是一种将推理和行动紧密结合在思-行-观循环中的智能体架构，常与计划-执行智能体进行对比。
