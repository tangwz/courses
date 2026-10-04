---
course: "prompt-engineering-agentic-workflows"
chapter: "debugging-optimizing-prompts-agentic-systems"
lesson: "organizing-versioning-prompts-agents"
sourceId: 6755
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-6-debugging-optimizing-prompts-agentic-systems/organizing-versioning-prompts-agents"
title: "智能体提示词的组织与版本管理"
description: "建立用于智能体系统中提示词的组织、记录和版本控制最佳实践。"
order: 7
plots: []
sourceHash: "5d349446f1f714ebabedd77db1c01e9b29f368e750b421f5bc1a4d3acecde80d"
sourceCorrections: []
---

随着您的智能体系统复杂性增加，以及您需要管理的提示词 (prompt)数量增多，保持秩序和追踪变化变得必不可少。就像软件代码一样，随意管理提示词会导致混乱、调试困难，并难以复现智能体的成功行为。对提示词实行系统化组织和版本管理，不仅仅是一项行政工作；它是一项基本做法，直接有助于本章所讨论的迭代改进和优化。

### 建立结构的必要性：组织您的提示词 (prompt)

想象一个智能体，它在其操作的不同阶段使用了十几种不同的提示词：规划、工具选择、信息整合和用户交互。现在，想象有多个这样的智能体，每个都有自己的一套提示词。如果没有清晰的组织策略，找到特定的提示词、了解其用途或确定当前哪个版本在活跃使用，都可能变成一个明显的阻碍。

有效的组织方式有助于您：

- **快速找到提示词：** 减少查找正确文件的时间。
- **理解提示词之间的关联：** 明确不同提示词如何影响智能体的工作流程。
- **促进协作：** 让团队成员能在共享的提示词集上无冲突地工作。
- **简化调试：** 当智能体出现异常行为时，组织良好的提示词更容易帮助您找出潜在的提示词相关问题。

以下是一些组织提示词的实用方法：

1. **逻辑目录结构：**
   创建清晰的文件夹层级结构。常见方法包括按以下方式组织：

   - **智能体：** 每个智能体都有自己的目录，其中包含所有相关提示词。

     ```
     prompts/
     ├── customer_support_agent/
     │   ├── greeting_prompt.txt
     │   ├── issue_categorization_prompt.txt
     │   └── knowledge_base_query_prompt.txt
     └── data_analysis_agent/
         ├── data_ingestion_prompt.txt
         └── report_generation_prompt_v1.txt
     ```
   - **任务或工作流程阶段：** 根据提示词在不同智能体中承担的功能进行分组（例如，将所有规划提示词放在一起）。

     ```
     prompts/
     ├── planning/
     │   ├── search_agent_plan.txt
     │   └── scheduling_agent_plan.txt
     ├── tool_use/
     │   └── common_api_interaction_format.txt
     ```
   - **提示词类型：** 按角色分组，例如人物设定、少样本示例集或特定的指令模式。
     这些方法的组合，例如先按智能体，再按任务类型，通常能提供一个好的平衡。
2. **一致的命名规范：**
   为您的提示词文件采用清晰一致的命名方案。这在一定程度上使提示词具有自我说明性。考虑包含：

   - 智能体名称或标识符
   - 提示词的任务或目的
   - 版本号或状态（例如，`_dev`，`_prod`）
   - 主要参数 (parameter)或区别特征
     示例：`search_agent_web_retrieval_main_v1.2.txt` 或 `email_generator_formal_persona_v3.json`（如果提示词以结构化格式存储）。
3. **提示词库或注册表：**
   对于大型项目，可以考虑建立一个中央提示词库。这可以是一个组织良好的共享目录，也可以是一个更复杂的内部工具。这样的库不仅应存储提示词文本，还应存储元数据：

   - 提示词目的的简要说明。
   - 使用它的智能体或工作流程。
   - 它期望的输入变量。
   - 预期的输出格式或行为。
   - 作者和最后修改日期。
   - 相关性能指标或测试结果的链接。
4. **提示词模板化：**
   许多提示词具有静态结构，其中动态部分在运行时填充（例如，用户查询、来自先前步骤的上下文 (context)）。使用模板引擎（如 Python 中的 Jinja2，或简单的字符串格式化）来管理这些。将基本模板存储在您的组织结构中。这将核心指令逻辑与可变数据分开，使提示词更清晰、更易于管理。

   ```python
   # 使用 Python f-string 作为简单模板的示例
   user_goal = "find recent AI research papers"
   planning_prompt_template = """
   目标: {goal}
   可用工具: [WebSearch, DocumentReader]
   上一步: None
   当前知识: None

   生成一个分步计划以实现目标。
   以编号列表形式输出计划。
   """
   filled_prompt = planning_prompt_template.format(goal=user_goal)
   ```

### 追踪演变：提示词 (prompt)版本管理

提示词很少能一次就完美。您会根据智能体表现进行迭代、测试和改进。版本控制对于管理这种演变来说不可或缺。它使您能够：

- **追踪变更：** 查看提示词随时间推移的演变，并了解特定修改的影响。
- **回滚到先前版本：** 如果新提示词版本导致性能下降或引入错误，您可以轻松恢复到已知良好状态。
- **安全地进行实验：** 创建分支来尝试新的提示词想法（例如，不同的人物设定、思维链结构），而不影响主要工作版本。
- **有效协作：** 多个团队成员可以处理提示词，其更改可以系统地合并和管理。

虽然您可以手动保存诸如 `prompt_v1.txt`、`prompt_v2.txt` 这样的文件，但这很快就会变得难以管理且容易出错。行业标准解决方案是**版本控制系统（VCS）**，其中 **Git** 是最常见的。

**将 Git 用于提示词：**
将您的提示词文件（无论是 `.txt`、`.md`、`.json` 还是其他任何格式）视作源代码。将其存储在 Git 仓库中。

- **频繁提交：** 对提示词进行有意义的更改后，以清晰的消息描述更改内容并提交。例如：“优化 search\_agent\_planner：添加了结果计数限制。”
- **使用分支进行实验：** 如果您正在尝试一种与众不同的提示词方法，请创建一个新分支（例如，`feature/search-agent-cot-prompt`）。这会将您的实验独立出来。如果成功，您可以将其合并回主分支。
- **用于发布的标签：** 当一组提示词稳定并随特定版本的智能体部署时，您可以在 Git 中为该提交添加标签（例如，`agent_v1.0_prompts`）。

下图展示了提示词版本在 Git 仓库中可能如何演变，包括主开发路径和一个实验分支。

> 此图显示了一个典型的版本管理流程，其中提示词沿着主分支演进，实验性更改在单独的分支中尝试，最终一个稳定版本被指定用于生产环境。

### 将组织和版本管理整合到您的工作流程中

采纳这些做法需要有意识的努力，但从长远来看会带来丰厚回报，特别是在您的智能体系统扩展时。

- **记录您的提示词 (prompt)：** 除了对提示词文本本身进行版本管理外，还要为每个重要的提示词或提示词模板维护文档。此文档应说明：

  - 其目的以及在智能体工作流程中的预期用途。
  - 它期望的任何输入变量及其格式。
  - 智能体对该提示词响应的预期结构或特性。
  - 已知限制或敏感性（例如，“如果输入上下文 (context)超过 500 字，则表现不佳”）。
    此文档可以放在提示词文件内的注释中、提示词目录中的独立 README 文件里，或在共享知识库中。
- **将提示词版本与智能体代码关联：** 智能体的行为是其代码和提示词的结合产物。当您对智能体的代码库进行版本管理时，请确保能够识别哪些版本的提示词与哪个版本的智能体一同使用。Git 子模块或仅仅是清晰的提交消息和标签策略，有助于管理这种关联。
- **定期审查和重构：** 就像代码一样，提示词也能从定期审查和重构中获益。是否存在多余的提示词？复杂的提示词能否被简化？命名规范是否依然清晰？定期整理能让您的提示词库易于管理并保持有效。
- **考虑测试驱动的提示词开发 (TDPD)：** 在您对提示词进行版本管理和改进时，请思考如何测试它们。对于某些提示词，您可能会为给定输入定义预期输出。对于其他提示词，特别是那些指导复杂智能体行为的，测试可能涉及检查智能体是否采取特定行动或避免不良行动。这直接与本章前面讨论的系统测试方法相关联。

通过建立组织和版本控制提示词的实践，您为开发、调试和优化您的 AI 智能体提供了更可靠的支撑。这些实践将提示词工程从一种临时性的艺术转变为一个更规范的工程过程，这对于构建可靠且高效的智能体工作流程来说很重要。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git是行业标准的版本控制系统，本书全面介绍了Git的使用，对跟踪提示词文件的更改和协作至关重要。
- [MLOps: From Data to Deployment](https://www.oreilly.com/library/view/mlops-from-data/9781098103857/) — Noah Gift, Alfredo Deza, and Greg Coquillo (2022)
  Publisher: O'Reilly Media
  本书提供了MLOps原则，包括工件管理和版本控制，这些原则直接适用于将提示词作为代理AI系统中重要资产进行组织和跟踪。
- [Software Engineering: A Practitioner's Approach](https://www.mheducation.com/highered/product/software-engineering-practitioner-s-approach-pressman-maxim/9781260239553.html) — Roger S. Pressman and Bruce Maxim (2020)
  Publisher: McGraw Hill
  一本广泛认可的软件工程教科书，涵盖了配置管理等软件工程原则，为将版本控制和结构化组织应用于提示词工件奠定了基础。
