---
course: "langchain-production-llm"
chapter: "evaluation-monitoring-observability"
lesson: "human-feedback-annotation"
sourceId: 3915
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/human-feedback-annotation"
title: "人工参与反馈与标注"
description: "建立使用LangSmith或自定义工具收集LLM回复的人工反馈系统。"
order: 7
plots: []
sourceHash: "9ff4ed778ce856c7c46e36a90903a629bcd7c3c43ddb3e032b8b4a4e8e27889f"
sourceCorrections: []
---

"虽然自动化指标能提供关于性能、召回率和流畅度的有用信号，但它们常常未能捕捉到生产环境中LLM应用质量的全貌。诸如实用性、特定情况下的事实准确性、安全性、语气恰当性以及与用户意图的一致性等方方面面，是众所周知难以通过算法衡量的。仅依靠自动化评估可能导致部署那些在基准测试中表现良好但在实际情况中无法满足用户需求的系统。在此，融入人类判断变得不可或缺。"

人工参与（HITL）的反馈和标注流程提供所需的定性数据，以弥补定量指标未能覆盖的不足。它们能帮助您理解应用在特定交互中成功或失败的*原因*，识别不易察觉的问题，并收集高质量数据以持续改进。整合人工参与机制是成熟、可投入生产的LLM系统的一个标志，这类系统优先考虑用户满意度和可靠性。

### 人类判断的必要性

自动化指标，例如用于摘要的ROUGE或用于翻译的BLEU，是为特定NLP任务开发的，它们与人类对现代LLM处理的生成任务质量的感知关联性通常较差。一个回复可能对参考文本获得高相似度得分，但仍然可能无用、事实不准确或不安全。

人类反馈擅长评估：

- **主观质量：** 评估语气、风格、创造力、同理心和整体用户体验。
- **事实准确性和有根据性：** 验证LLM所作的声明，特别是在涉及外部知识时（如RAG系统）。人类可以检查来源或运用其专业知识。
- **安全性和恰当性：** 识别可能绕过自动化过滤器的有害、有偏见、不道德或不恰当的内容。
- **指令遵循和意图对齐 (alignment)：** 确定LLM是否真正理解并回应了用户的潜在需求，特别是对于复杂或含糊的请求。
- **比较质量：** 对两个或更多潜在回复进行偏好判断，这对于像基于人类反馈的强化学习 (reinforcement learning)（RLHF）这样的技术来说是根本。

系统地收集这些反馈，能让您超越简单的通过/失败测试，并对应用行为有更全面的认识。

### 收集反馈的机制

反馈收集方法从被动观察到主动请求不等。

- **隐式反馈：** 这涉及分析用户行为信号，如建议操作的点击率、会话时长、任务完成成功率，或者用户在收到初始回复后是否重新措辞其查询。隐式信号虽然可扩展，但通常有噪音，只提供满意度或失败的间接证据。
- **显式反馈：** 这需要直接向用户或标注员征求意见。常见方法包括：
  - **简单评分：** 二元点赞/点踩按钮易于实施，并能提供快速的情感信号。
  - **量表评分：** 使用李克特量表（例如1-5星）衡量实用性、准确性或相关性等维度，能提供更细致的定量数据。
  - **类别标签：** 允许用户或标注员选择预定义标签（例如“事实错误”、“幻觉 (hallucination)”、“离题”、“不安全内容”），有助于确定特定的错误类型。
  - **自由形式评论：** 文本框允许用户为其评分提供详细的定性解释。
  - **修正界面：** 提供用户编辑LLM回复使其正确的选项，为监督微调 (fine-tuning)提供有用的数据。
- **专家标注：** 对于复杂领域或高风险应用，通常需要聘请领域专家或训练有素的标注员进行离线审查。他们遵循详细的指导方针（评分标准），提供高质量、一致的标签、修正或比较，通常处理精心策划的难题交互数据集。

机制的选择依据应用、用户群、可用资源和反馈过程的具体目标而定。通常，多种方法结合能产生最佳结果。

### 整合反馈收集系统

有效地收集反馈需要将这些机制整合到您的应用工作流和工具中。

#### 应用内UI元素

获取用户反馈最直接的方式是，将简单的UI元素（按钮、星级评分、评论框）直接嵌入 (embedding)应用界面中，靠近生成的回复。这能最大限度地减少用户的操作阻力。确保这些元素不显眼但可被发现。收集到的数据需要与交互的上下文 (context)信息（输入、输出、时间戳、适用的用户ID、应用状态）一同记录。

#### 使用LangSmith收集反馈

LangSmith旨在支持反馈收集和分析。它提供了一种结构化的方式，将反馈数据直接与您的LangChain应用的执行记录关联起来。

您可以使用LangSmith客户端以编程方式记录针对特定运行ID的反馈。这能将人类判断直接与生成回复的链或代理的详细运行记录关联起来，有助于调试。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import collect_runs
from langsmith import Client

# 确保LangSmith环境变量已设置
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGCHAIN_PROJECT"] = "YOUR_PROJECT_NAME"
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 初始化LangSmith客户端
client = Client()

# 定义一个简单的链
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("user", "{input}")
])
chain = prompt | llm

# 执行链并捕获运行ID
run_id = None
try:
    # 使用collect_runs来捕获运行上下文
    with collect_runs() as cb:
        response = chain.invoke({"input": "解释一下编程中的递归原理。"})
        
        # 从捕获的运行中检索运行ID
        if cb.traced_runs:
            run_id = cb.traced_runs[0].id
            
        print(f"LLM Response: {response.content[:100]}...") # 打印部分回复
        print(f"运行ID: {run_id}")

except Exception as e:
    print(f"链执行错误: {e}")

# 模拟收集用户反馈（例如，来自网页UI）
if run_id:
    user_feedback_score = 1 # 示例：1表示“好”，0表示“差”
    user_comment = "解释清晰，但可以用一个更简单的例子。"
    feedback_key = "quality_rating" # 为此类反馈定义一个一致的键

    try:
        # 将反馈记录到LangSmith，与特定运行关联
        client.create_feedback(
            run_id=run_id,
            key=feedback_key,
            score=user_feedback_score, # 可以是二进制、量表（0-1，1-5）等
            comment=user_comment,
            feedback_source_type="user", # 区分用户反馈、评审员反馈或模型反馈
            # 您还可以添加source_info，例如{"userId": "user123"}
        )
        print(f"反馈已成功记录到运行: {run_id}")

    except Exception as e:
        print(f"将反馈记录到LangSmith时出错: {e}")
else:
    print("无法获取运行ID，未记录反馈。")
```

LangSmith也提供一个网页界面，协作人员可以在其中手动审查运行记录、添加评论、分配分数和标记 (token)运行。这有助于有针对性的调试会话或手动标注工作流程。

#### 自定义和第三方标注工具

对于大规模标注工作或特殊要求，您可能需要与Label Studio、Prodigy、Scale AI等专门的数据标注平台集成，或构建自定义的内部工具。这些平台提供更精密的界面、工作流管理、质量控制功能和标注员管理能力。通过这些工具收集的数据通常可以导出并链接回LangSmith运行记录，或用于创建评估数据集。

### 标注策略与指南

收集反馈只是第一步；理解其含义需要结构化。

- **清晰的标注指南：** 为标注员（无论是提供简单评分的最终用户还是专业专家）制定详细的评分标准或一套指令。定义每个评分级别或标签的含义，提供良好和不良回复的例子，并澄清如何处理模糊性。一致性对于可靠数据不可或缺。
- **标注类型：** 选择符合您目标的标注类型：
  - 二元（正确/不正确，点赞/点踩）
  - 类别（错误类型，安全标记 (token)）
  - 序数尺度（实用性、准确性的李克特量表）
  - 自由文本（解释，修正）
  - 比较（哪个回复更好？）
- **抽样：** 标注每一次交互通常是不可行的。实施抽样策略：
  - **随机抽样：** 提供对整体性能的无偏见视图。
  - **不确定性抽样：** 侧重于模型置信度得分（如果可用）较低的交互。
  - **基于错误的抽样：** 优先标注被自动化监控器标记或收到负面隐式/显式信号的交互。
  - **边缘情况抽样：** 有意选择代表有难度输入或已知失败模式的交互。

### 使用反馈进行持续改进

收集人类反馈的最终目的是推动应用改进。

> 改进LLM应用的典型人工反馈回路。用户交互生成回复，反馈被收集并记录（通常通过LangSmith等平台），进行分析，然后用于调试问题、更新评估数据集或微调 (fine-tuning)提示词 (prompt)和模型，从而实现改进的应用部署。

反馈如何转化为行动：

1. **监控与警报：** 持续追踪汇总的反馈分数和评论。使用仪表板（在LangSmith或其他可观测性工具中）可视化用户满意度或报告问题的趋势。设置警报，以防分数突然下降或负面反馈类别激增，这可能表明生产环境出现了退步。
2. **调试：** 与特定LangSmith运行记录关联的反馈有助于诊断。当用户报告不良回复时，您可以检查导致问题的确切输入、中间步骤、工具调用和最终LLM输出，从而大幅加速根本原因分析。
   "3. **评估数据集整理：** 标注的交互，特别是通过反馈识别出的失败或边缘情况，会成为您评估数据集的高质量补充。这能确保您的自动化评估更好地反映现实中的问题。"
3. **模型改进：**
   - **提示词工程：** 一致的反馈模式常常暴露出系统提示词或任务指令中的弱点。利用这些观察结果迭代优化您的提示词。
   - **微调：** 源自人类反馈的精心策划的高质量交互集（输入+良好回复）或偏好对（输入+更好回复+更差回复），可用于监督微调（SFT）或RLHF，以提升基础LLM的功能或其与您特定任务的对齐 (alignment)度。
4. **工具/代理改进：** 反馈可能表明代理工具使用不当，或者工具本身存在缺陷或不足。这指导了工具描述、代理推理 (inference)逻辑或工具实施的改进。

### 遇到的问题与实用方法

实施成功的人工参与流程涉及处理以下问题：

- **成本与可扩展性：** 人力成本高昂。标注可能成为一项重要的运营成本，尤其是在规模化时。平衡反馈质量和数量需求与预算限制。在可能的情况下使用自动化和智能抽样。
- **主观性与偏见：** 人类判断本质上是主观的，可能受到个人偏见、文化背景或不同专业水平的影响。通过清晰的标注指南、每个任务安排多名标注员（以衡量标注员间的一致性）、招募多元化的标注员以及定期质量检查来缓解此问题。
- **延迟：** 收集反馈并将其用于部署改进之间通常存在延迟。优化反馈分析和部署流程以最大限度地减少这种滞后。
- **反馈质量：** 用户提供的反馈在质量和实用性方面可能差异很大。简单评分可能缺乏上下文 (context)，而自由形式评论可能模糊或不相关。设计反馈机制以鼓励提供具体且可操作的输入。

"尽管存在这些问题，整合人类反馈是一项重要实践，用于构建不仅功能完善，而且可靠、值得信赖、真正对用户有帮助的LLM应用在生产环境中。它将评估从纯粹的自动化检查转变为由经验驱动的持续学习过程。"

## 参考资料

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  本文介绍了InstructGPT，展示了如何通过人类反馈强化学习（RLHF）使语言模型与用户意图和偏好对齐，从而使其更具帮助性且危害性更小。
- [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.440/) — Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh (2020)
  Journal: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 4902-4912; DOI: [10.18653/v1/2020.acl-main.440](https://doi.org/10.18653/v1/2020.acl-main.440)
  本文提出了一种超越传统指标的自然语言处理模型评估方法，侧重于行为测试和人类可解释的能力，这对于理解LLM自动化评估的局限性非常相关。
- [Human-in-the-Loop Machine Learning: Active Learning and Annotation for Adaptive Algorithms](https://www.manning.com/books/human-in-the-loop-machine-learning) — Robert (Munro) Monarch (2021)
  Publisher: Manning Publications
  一本指南，涵盖了构建人机协作（Human-in-the-Loop）系统的原理、方法和策略，包括数据标注、主动学习和部署自适应机器学习解决方案。
- [How to log feedback](https://docs.smith.langchain.com/concepts/feedback/how-to-log-feedback) — LangChain (2024)
  Publisher: LangChain
  官方文档，详细介绍了如何以编程方式将人类反馈记录到LangSmith，用于跟踪和分析LangChain应用程序的运行情况。
