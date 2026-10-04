---
course: "multi-agent-llm-systems-design-implementation"
chapter: "advanced-orchestration-workflows"
lesson: "state-graph-orchestration"
sourceId: 6680
sourceUrl: "https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-4-advanced-orchestration-workflows/state-graph-orchestration"
title: "状态驱动与基于图的编排模型"
description: "研究用于编排代理行动序列的状态机和基于图的模型。"
order: 2
plots: []
sourceHash: "a2b54da0a32920edcfadb12ad36b9178087e623b9cbaa54b76893a977b557b71"
sourceCorrections: []
---

为了有效管理多个LLM代理的协作，特别是对于需要多个步骤或条件逻辑的任务，我们需要结构化编排模型。这些模型提供一种正式方式来规定代理如何交互、何时行动以及信息如何在它们之间流动。两种主要方法是状态驱动的编排和基于图的编排。每种都具有独特优势，适合不同种类的多代理工作流。

### 状态驱动的编排模型

状态驱动的编排通常使用有限状态机（FSM）实现，它以有限数量的状态及其之间的转换为基础来定义系统。一个代理或整个多代理系统始终处于这些预设状态之一。从一个状态到另一个状态的转换由特定事件或条件触发，通常是代理行动或外部输入的结果。

**核心组成部分：**

- **状态：** 在工作流中代表特定情形、阶段或状态。对于LLM代理，一个状态可能表示“等待输入”、“处理信息”、“生成响应”或“等待工具输出”。
- **转换：** 定义状态之间允许的移动。每次转换通常都与一个事件或条件关联，该事件或条件必须满足才能发生转换。
- **事件/条件：** 触发转换的因素。这些可以是来自其他代理的消息、任务完成、API响应或人工输入。
- **行动：** 在进入状态、退出状态或转换期间执行的操作。在LLM语境下，一个行动可能涉及使用特定提示调用LLM、调用外部工具或向另一个代理发送消息。

考虑一个由状态机管理的内容生成工作流。一个代理可能开始于`起草`状态。完成草稿（一个事件）后，它转换到`待审`状态。一个审查代理随后接手。如果需要修改（另一个事件），系统转换到`修改中`状态，将任务分配回去。如果获批，它移动到`已完成`状态。

> 一个状态机图，说明了一个简单的内容生成工作流，包含起草、待审、修改中和已完成等状态，以及基于行动的转换。

**优势：**

- **线性流程的清晰性：** 对于大部分顺序执行或具有明确、有限分支的工作流，状态机设计和理解起来都很直接。
- **可预测性：** 系统的行为通常更可预测，因为可能的状体和转换都明确规定。
- **调试：** 隔离问题可能更简单，因为问题通常可以追溯到特定的状态或转换逻辑。

**劣势：**

- **复杂性的可扩展性：** 对于具有许多可能状态、大量相互依赖或非常动态行为的高度复杂工作流，FSM可能变得难以管理。“状态爆炸”问题，即状态数量随新变量呈指数增长，是一个已知的难题。
- **并行性受限：** 与基于图的模型相比，在传统FSM结构中表示和管理任务的并行执行可能不那么直观。
- **灵活性：** 如果未明确设计到状态转换中，适应不可预见的情况或在执行中动态更改工作流可能很困难。

当操作序列得到充分理解且决策点明确时，状态驱动的模型尤其有效。例如，一个引导用户通过故障排除脚本的客户服务机器人可以通过状态机高效管理。

### 基于图的编排模型

基于图的编排将工作流表示为有向图，通常是定向无环图（DAG），其中节点表示任务、操作或代理职责，边表示这些节点之间的依赖关系、数据流或控制流。此模型具有高度灵活性，非常适合涉及多个代理和条件逻辑的复杂、非线性流程。

**核心组成部分：**

- **节点：** 代表工作单元或决策点。一个节点可以是：
  - 对特定LLM代理的调用（例如，“文本摘要代理”）。
  - 对外部工具或API的调用（例如，“获取股价工具”）。
  - 一个条件逻辑块（例如，“如果情感积极”）。
  - 一个人工审查步骤。
- **边：** 定义节点之间的关系和执行顺序。从节点A到节点B的边意味着节点A必须完成（或其输出准备就绪）后节点B才能开始。边还可以携带数据或定义条件路径。

设想一个用于研究和报告生成的多代理系统。一个图可以描绘`规划代理`定义范围，然后分派给`研究代理A`和`研究代理B`并行收集数据。它们的输出输入到`合成代理`，其结果再交给`撰写代理`起草报告。这份草稿随后交给`编辑代理`审查，可能会回溯到`撰写代理`进行修改，最终到达`最终报告`节点。

> 一个图示，说明了研究和报告生成工作流。节点代表代理或任务，边表示依赖关系，包括并行数据收集和审查循环。

**优势：**

- **处理复杂性和非线性：** 图擅长表示具有多个依赖、分支和合并的复杂工作流。
- **并行性：** 在图结构中明确定义并行任务是很自然的（例如，两个节点之间没有直接路径，但都源自同一个前置节点）。
- **可视化和模块化：** 图的视觉特性使复杂工作流更容易理解和交流。节点可以封装复杂的子工作流，促进模块化设计。
- **适应性：** 条件边和动态图修改（尽管更高级）可以允许更具适应性的工作流。

**劣势：**

- **初始设置：** 设计和实现图结构和执行引擎可能比设置简单的FSM更复杂。
- **状态管理：** 追踪复杂图执行的整体状态，包括各个节点的状态以及它们之间的数据流，需要仔细考虑。
- **调试分布式逻辑：** 虽然单个节点故障可能容易发现，但调试多个节点相互作用产生的紧急问题可能具有挑战性。

基于图的模型在数据工程中常见（例如，Apache Airflow、Prefect），并越来越多地用于LLM代理编排（例如，LangGraph）。它们为旨在构建能够规划和执行复杂系列行动的自主代理的框架提供支撑。

### 模型选择

状态驱动和基于图的编排之间的选择并非总是互斥的；存在混合方法，其中图中的一个节点本身可以作为状态机实现。然而，在选择主要模型时，考虑以下几点：

- **工作流复杂性：** 对于简单、线性的序列，状态机可能就足够了，并且实现起来更容易。对于多步骤、分支或并行流程，图通常更具优势。
- **可预测性与灵活性：** 如果流程固定，状态机提供更高的可预测性。图为动态或演变流程提供更大的灵活性。
- **并行需求：** 图模型本质上更自然地支持并行任务的表示。
- **团队熟悉度和工具：** 开发工具的可用性和团队的专业知识会影响选择。LangGraph等框架使基于图的编排在LLM应用中更易于使用。

随着多代理系统变得更复杂，基于图的方法常常因其管理复杂依赖和实现更复杂协作行为的能力而受到青睐。了解这两种模式使您能够选择最合适的结构用于编排您的代理团队，确保它们能够高效协作以实现总体目标。这为处理自适应任务规划和确保可靠性奠定基础，这些是我们将在后续讨论的话题。

## 参考资料

- [Introduction to Automata Theory, Languages, and Computation](https://www.pearson.com/us/higher-education/program/Hopcroft-Introduction-to-Automata-Theory-Languages-and-Computation-3rd-Edition/PGM9780321455369.html) — John E. Hopcroft, Jeffrey D. Ullman, Rajeev Motwani (2006)
  Publisher: Pearson
  一本基础教科书，全面介绍了计算理论，包括有限自动机和状态机。
- [Apache Airflow Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGqi73u205hpzxxkV4cqYYAIRcYsgGDYTVNY5TKdP8HhA7IiKHInd8es759jc2SfYsj8xelCMoggquarkczv0ObAC4JRf4OGOFKsnRjzPNgRB250TcmAUmGzQw1y2BkhjpSpqlrz6VPmKVYOIyzLjA=) — The Apache Software Foundation (2024)
  Publisher: The Apache Software Foundation
  Apache Airflow 的官方指南，详细介绍了使用有向无环图 (DAG) 设计和实现基于图的数据工作流。
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/index/) — LangChain (2024)
  Publisher: LangChain
  一个框架的官方文档，该框架支持使用基于图的编排，专门为LLM智能体构建有状态的多智能体应用程序。
- [An Introduction to MultiAgent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEKff29gE17hjTc-owQewP6JuxC_xUUCOamgPv10bOb_M5IyGp73rsOJJDliHNWS3-IRrYTT5Hh10AgM4AaPV0ZjzfSoyuCHqpWUyqlBg5NB_rQff1yRGPoOG4xPIXPTsCoRJf0d5yn80cU0HdReGz-PMHujIHItFS09-esWAqK-8qVOJrtPZ7MpLAxQhRktD7tmmi0RpYSsVcB) — Michael Wooldridge (2009)
  Publisher: John Wiley & Sons
  一本全面的教科书，涵盖了多智能体系统的基本概念、架构和协调机制，为智能体编排提供了背景信息。
