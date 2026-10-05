# 工具在LLM代理能力中的作用

来源：[原文](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-1-llm-agent-tooling-foundations/role-of-tools-llm-agents)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型（LLM）在理解和生成类人文本方面表现出非凡的能力。它们可以起草电子邮件、总结文章、编写代码片段，甚至进行创意叙事。然而，尽管拥有令人印象深刻的语言技能，LLM在某些固有限制下运行。它们的知识通常是静态的，基于它们最近训练时的数据，而且它们无法单独与数字或物理环境互动或采取行动。这就是工具发挥作用的地方。

工具作为LLM代理的重要扩展，显著扩大了它们的操作范围和问题解决能力。可以把LLM想象成一个高度智能的大脑。工具是让这个大脑感知当前信息并操纵其环境的感觉器官和四肢。没有工具，LLM可以思考执行一项任务，比如查询当前天气或预订航班，但它无法真正执行。

## 弥补LLM的局限性

工具直接解决了一些独立LLM的基本约束：

- **获取实时信息：** LLM有知识截止日期。如果一个LLM的训练数据截止到2023年初，它就不会知道此后发生的事情、新的发现或生成的数据。工具，比如网络搜索API包装器或新闻源连接器，可以为LLM提供最新信息，使其能够回答当前事件的问题或获取最新股价。
- **执行操作：** LLM可以生成描述操作的文本，比如“发送一封主题为‘会议更新’的电子邮件到[contact@example.com](mailto:contact@example.com)”，但它们无法自己发送电子邮件。专门的电子邮件工具，在LLM调用时，可以接收收件人、主题和正文作为输入，并执行实际的发送操作。同样，工具可以与日历、数据库、电子商务平台或任何其他可通过API访问的系统进行互动。
- **事实依据和减少幻觉 (hallucination)：** 尽管LLM通常善于从训练数据中回忆信息，但有时它们可能会“产生幻觉”或生成听起来合理但不正确的信息。工具可以从权威来源获取数据，从而帮助LLM的回复有事实依据。例如，LLM无需依赖其可能过时或不精确的关于公司现任CEO的内部知识，而是可以使用工具查询可靠的商业数据库。
- **精确计算和结构化操作：** LLM主要是语言处理器，而不是计算器或数据库查询引擎。尽管它们可能近似简单的算术，但对于复杂计算或精确逻辑操作却力不从心。计算器工具或代码解释器工具可以准确处理这些任务，为LLM提供可靠结果以纳入其回复中。设计用于查询SQL数据库的工具可以执行LLM可能只能用自然语言表达的复杂查询。
- **与专有或私有数据交互：** 许多有价值的应用要求LLM访问非公开数据，例如公司内部文件、客户关系管理（CRM）系统或个人用户数据。安全设计的工具可以充当守门员，允许LLM在受控条件下查询这些私有数据源，同时遵守隐私和安全协议。

以下图表描绘了LLM代理如何通过与外部系统交互来满足用户的请求。

> LLM代理使用工具访问外部天气API，使其能够响应用户查询提供实时天气信息。

## 扩展代理能力

通过集成工具，LLM代理从复杂的文本生成器转变为更有能力、更通用的助手。这种集成有助于：

- **与数字服务交互：** 代理可以读取网页、提取特定信息、与在线服务（如预订系统或社交媒体）交互，并管理云资源。
- **执行复杂工作流程：** 单个用户请求可能需要多个步骤或访问不同类型的信息。例如，规划一次旅行可能涉及一个工具来检查航班可用性，另一个来查找酒店住宿，第三个来查询当地景点。LLM代理可以按顺序或并行协调这些工具。
- **个性化体验：** 工具可以使代理访问用户特定的上下文 (context)，例如过往偏好、日历信息或个人文档（在获得适当权限的情况下），从而带来更相关和量身定制的交互。
- **重复任务自动化：** 配备工具的代理可以自动化常规任务，例如根据数据库查询生成报告、总结与特定主题相关的每日新闻或管理待办事项列表。

本质上，工具帮助LLM代理不再仅仅是简单的问答或文本生成。它们成为能够收集信息、执行计算并在各种数字系统中启动行动的积极参与者。随着您在本课程中的学习，您将学会如何有效地设计、构建和管理这些工具，使您的LLM代理成为强大的问题解决者。

## 参考资料

- [ReAct: Synergistic Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Shaoyang Ma, Dian Yu, Kai Sun, Yuqing Xia, Xiao Liu, Zhouxing Shi, Derek Zheng, Huadong Wang, Yie Sansan, Yongqiang Huang, Teng Xu, Hongkun Yu, Cho-Jui Hsieh, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，该框架结合了LLM代理的推理和行动能力，通过工具使用完成复杂任务。
- [Tools](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8JiXHOlLdXMNtbtQTV1zfnHTMQluvMEkKQ3Z-YfsP-2-eETXsDUkbhgbDTLfiS4YNmxiOd5swtKPsR7B3xDGmenLeAAqTaFgehWVE71e424jK5gpKn2mTkG23tbzB7mFCNg9ziBQgs915c5mbnQ==) — LangChain (2024)
  Publisher: LangChain Documentation
  提供了在LangChain框架内将工具与LLM代理集成的实践指导和示例。

---

[下一节](02-%E7%90%86%E8%A7%A3%E5%B7%A5%E5%85%B7%E8%A7%84%E8%8C%83%E4%B8%8E%E6%8F%8F%E8%BF%B0.md)
