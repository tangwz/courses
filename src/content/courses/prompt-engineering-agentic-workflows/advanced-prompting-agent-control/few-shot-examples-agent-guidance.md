---
course: "prompt-engineering-agentic-workflows"
chapter: "advanced-prompting-agent-control"
lesson: "few-shot-examples-agent-guidance"
sourceId: 6720
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-2-advanced-prompting-agent-control/few-shot-examples-agent-guidance"
title: "运用少样本示例指导智能体"
description: "在提示中应用少样本学习原则，用少量示例指引和调整智能体行为。"
order: 4
plots: []
sourceHash: "f4d5b44a16e7573c151b3480bde303ba690b5c8796e1f57b7bc949f72ec4a4ed"
sourceCorrections: []
---

尽管本章前面讨论过的通用指令和角色分配为智能体控制提供了坚实基础，但有时你需要通过更具体的实例来指引智能体的行为。这时，在提示中直接运用少样本示例就成为一种宝贵方法。你不再只是告诉智能体*做什么*，而是通过演示向它展示*如何*完成任务。

少样本学习 (few-shot learning)，在大型语言模型（LLMs）和智能体系统的背景下，是指向模型提供少量期望输入-输出行为的示例（即“样本”）。这种方法帮助智能体理解模式、格式，甚至隐含的推理 (inference)步骤，通常比单独冗长的描述性指令更有效。它是零样本提示（智能体仅依赖指令）和大规模微调 (fine-tuning)（涉及在大型数据集上重新训练模型）之间的一种中间方案。对于智能体工作流程而言，少样本示例提供了一种实际可行的方式来达成特定行为，而无需模型重新训练的额外开销。

### 少样本示例为何能提升智能体指引效果

由LLM驱动的智能体擅长模式识别。通过提供少量精心设计的示例，你可以利用这种能力，让智能体推断出期望的行动方案。这种方法对智能体控制有效，原因如下：

1. **消除歧义**：指令有时可能存在多种解读。示例提供了你期望的具体实例，从而减少模糊性。如果智能体需要调用一个带有特定参数 (parameter)格式的工具，展示该格式的示例比抽象描述清晰得多。
2. **展示复杂逻辑**：对于需要特定思维序列或特定问题分解方式的任务，示例可以隐含地说明这种逻辑。当推理 (inference)过程细致入微、难以通过明确指令完全表达时，这尤其有用。
3. **指定输出格式**：如果你的智能体需要以结构化格式（例如JSON、XML或自定义模式）生成输出，少样本示例是定义该结构的绝佳方式。智能体通过观察来学习格式。
4. **适应细节**：通过包含展示这些细节的示例，可以指引智能体采纳特定风格、语调，或处理边缘情况。例如，如果智能体应共情地回应某些用户查询，一个示例可以展示这种共情的回应风格。
5. **提高可靠性**：通过提供行为模板，少样本示例可以使智能体表现更一致、更可靠，尤其是在任务的重复性方面。

### 设计高效少样本示例

少样本提示的成功取决于所提供示例的质量和相关性。以下是设计高效智能体指导示例的一些指南：

- **重质不重量**：少量高质量、多样化、有代表性的示例（通常2到5个）通常比大量冗余或构建不佳的示例更有效。每个示例都会占用提示中的token空间，因此请精打细算。
- **相关性重要**：确保你的示例直接关联智能体需要执行的任务。不相关的示例可能会使智能体困惑或误入歧途。
- **结构一致性**：在你的示例中保持结构一致。如果一个示例展示 `输入 -> 思考 -> 行动 -> 输出`，所有示例都应遵循此模式。这有助于智能体学习期望的序列。
- **意图清晰**：示例的每个部分都应清晰展示期望的行为。如果你正在展示如何使用工具，工具调用及其参数 (parameter)应明确无误。
- **涵盖必要变体**：如果任务有常见变体或需要根据输入进行不同处理，请包含涵盖这些不同情况的示例。这有助于智能体更好地泛化。例如，展示如何处理缺失信息或不同类型的用户请求。
- **正例和反例（可选但有效）**：尽管在基础少样本设置中较少见，但对于复杂任务，你可以考虑展示一个*不*该做什么的示例，同时提供纠正方法，但这需要仔细措辞。通常情况下，你会侧重于正向范例。

### 在提示中构建少样本示例

在加入少样本示例时，你通常会将它们作为前言呈现在智能体需要处理的实际任务或查询之前。每个示例的结构应清晰，并模拟你希望智能体遵循的流程。

提示中示例的常见结构可能如下所示：

```
[可选前言: "以下是处理X的一些示例："]

示例 1:
用户查询: [用户请求或输入的示例]
思考: [可选：对智能体推理过程或计划的简要描述。如果你希望遵循特定的推理模式（如思维链），这可以指导智能体的内部“独白”。]
工具调用: [工具调用的示例，例如 search("query") 或 api_call(endpoint="...", params={...})]
工具观察: [工具返回的结果/数据的示例]
智能体响应: [对用户的最终响应或内部摘要的示例]

示例 2:
用户查询: [...]
思考: [...]
工具调用: [...]
工具观察: [...]
智能体响应: [...]

[示例结束]

当前任务:
用户查询: [智能体需处理的实际当前用户查询]
思考: [智能体填写]
工具调用: [智能体填写]
智能体响应: [智能体填写]
```

当智能体处理“当前任务”时，它预期会遵循示例确立的模式。

我们来考虑一个设计用于从文本中提取联系信息并将其格式化为JSON的智能体。

**零样本尝试（可能含糊不清）：**

```
从以下文本中提取联系信息（姓名、电子邮件、电话）并以JSON格式返回。
文本: "Contact Jane Doe at jane.doe@example.com or (555) 123-4567 for details."
```

智能体*可能*会做对，但确切的JSON结构、字段名称以及缺失信息的处理方式并未明确定义。

**少样本方法（更清晰的指引）：**

```
你是一个专业的联系信息提取器。给定一段文本，提取姓名、电子邮件和电话号码。将输出格式化为JSON对象。如果未找到某个信息，请将其值设为null。

示例 1:
文本: "Reach out to John Smith via john.smith@work.com. His number is 123-456-7890."
输出:
{
  "name": "John Smith",
  "email": "john.smith@work.com",
  "phone": "123-456-7890"
}

示例 2:
文本: "For support, email support@example.org."
输出:
{
  "name": null,
  "email": "support@example.org",
  "phone": null
}

当前任务:
文本: "Contact Jane Doe at jane.doe@example.com or (555) 123-4567 for details."
输出:
```

通过提供这些示例，智能体收到清晰的指引，明确了：

- 期望的JSON键（`name`、`email`、`phone`）。
- 如何表示缺失信息（`null`）。
- 输出的整体结构。

这显著增加了智能体持续生成正确输出的可能性。

### 何时使用少样本示例

少样本示例在以下场景中特别有益：

- **工具使用规范**：当智能体需要调用一个带有精确参数 (parameter)名称和格式的特定工具时。
- **输出格式化**：为了强制执行严格的输出结构，如JSON、XML或特定的文本模板。
- **演示推理 (inference)过程**：当你希望智能体遵循特定的推理思路或问题分解方式时（例如，展示简化版思维链）。
- **风格和语调适应**：为了指引智能体的沟通风格（例如，正式、非正式、共情）。
- **处理歧义或边缘情况**：当仅靠指令不足以涵盖所有必要变体或棘手情况时。
- **提高任务成功率**：如果零样本提示产生不一致或不正确的结果，少样本示例通常可以稳定并提升性能。

### 与其他提示技术的结合

少样本示例并非孤立存在；它们可以与本章讨论的其他高级提示策略有效结合。

- **角色与人格设定**：示例可以展示被分配的角色或人格设定在特定情况下*如何*行为。如果智能体是“乐于助人的客服助理”，示例可以展示礼貌且面向解决方案的互动。
- **思维链（CoT）/ 思维树（ToT）**：你可以在少样本示例中明确包含“思考：”或“推理 (inference)：”步骤，以展示期望的思考过程。这不仅指导智能体的最终行动或输出，还指导其中间推理步骤。

例如，将角色扮演、CoT和少样本结合应用于一个故障排除智能体：

```
你是网络医生，一名网络故障排除专家。当用户描述问题时，首先逐步思考诊断，然后提出行动建议。

示例 1:
用户: "我的互联网坏了。"
网络医生的思考:
1. 检查物理连接：调制解调器/路由器是否已通电？指示灯是否正常？
2. 检查本地网络：其他设备能否连接？
3. 检查外部服务：是否有已知的服务中断？
网络医生的行动: "首先，请检查您的调制解调器和路由器是否已通电，并且状态指示灯是否正常。您家里的其他设备也无法连接吗？"

当前问题:
用户: "我无法访问我的电子邮件。"
网络医生的思考:
网络医生的行动:
```

### 注意事项与局限

尽管有效，但少样本提示有一些注意事项：

- **Token 消耗**：示例会增加提示的长度，消耗LLM上下文 (context)窗口中的token。对于上下文限制较小的模型，或在尝试优化成本和延迟时，这是一个重要因素。
- **过拟合 (overfitting)风险**：如果你的示例过于狭窄或过于具体，智能体可能会对其“过拟合”，难以处理哪怕略有偏差的输入。目标是涵盖合理范围的期望输入的示例。
- **示例质量非常重要**：少样本提示的性能高度依赖于示例的质量。选择不当或误导性的示例会降低性能。
- **维护性**：如果任务、工具或期望的输出发生发展变化，你的少样本示例也需要更新以保持相关性。这为你的提示工程 (prompt engineering)工作增加了维护方面的工作。

通过周全地加入少样本示例，你可以对智能体的行为获得更精细的控制，使其能够更准确、更一致地执行复杂任务。这项技术是朝着构建更复杂、更可靠的智能体工作流程迈出的实用一步。在继续前进时，请记住，提示工程的技巧通常涉及对这些示例进行迭代、观察智能体行为，并完善你的演示，直到达到期望的性能。

## 参考资料

- [Language Models are Few-Shot Learners](https://doi.org/10.48550/arXiv.2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇开创性论文介绍了大型语言模型的少样本学习概念，展示了它们仅通过少量示例即可执行新任务而无需微调的能力。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  本文介绍了思维链提示技术，通过明确展示中间推理步骤来提高大型语言模型的推理能力，该技术可与少样本示例结合用于代理引导。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  本文提出了 ReAct 框架，将推理和行动结合用于大型语言模型代理。少样本示例常用于 ReAct 中，以指导复杂任务中的工具使用和决策制定。
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  Publisher: OpenAI
  来自 OpenAI 的官方指南，提供提示工程的实用技术和最佳实践，包括有效利用少样本示例来控制模型。
