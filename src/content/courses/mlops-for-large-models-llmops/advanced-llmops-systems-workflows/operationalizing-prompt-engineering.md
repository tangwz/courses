---
course: "mlops-for-large-models-llmops"
chapter: "advanced-llmops-systems-workflows"
lesson: "operationalizing-prompt-engineering"
sourceId: 3221
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/operationalizing-prompt-engineering"
title: "提示工程的运行化"
description: "整合提示的版本控制、测试和管理到MLOps流程中。"
order: 1
plots: []
sourceHash: "5c9278086bba159cefa8f36b3158be49e679b57f82f247e425b6adf3577983ff"
sourceCorrections: []
---

构建用于管理大型语言模型的完整自动化系统，需要提示处理方式的显著演进。在开发的早期阶段，提示可能被视为简单的文本字符串，并在应用程序代码中手动调整。然而，对于可扩展和可维护的LLM应用，尤其是在团队协作或复杂部署中，这种临时方法变得不足。将提示工程 (prompt engineering)运行化，意味着将系统化的MLOps原则应用于提示的生命周期，将它们视作与代码和模型同等重要的产物。

本节详细说明了在高级LLMOps工作流程中有效管理提示所需的实践和基础设施。我们将讨论提示的版本控制、测试、部署和监控，确保它们积极且稳定地提升整体系统性能。

### 系统化提示管理的需求

手动管理直接嵌入 (embedding)在应用程序代码或配置文件中的提示，会带来一些操作上的困难：

1. **缺乏版本管理：** 难以追踪改动，理解提示被修改的原因，或回滚到之前可用的版本。
2. **部署不一致：** 不同的环境（开发、测试、生产）可能会无意中使用不同的提示版本，导致不可预测的行为。
3. **协作困难：** 多个团队成员处理提示可能导致冲突和未记录的改动。
4. **性能归因困难：** 没有清晰的版本管理，难以将LLM性能的变化（延迟、质量、成本）与特定的提示修改联系起来。
5. **评估中的困难：** 系统地测试和比较不同提示变体的有效性变成一个手动且易出错的过程。
6. **治理和可审计性：** 为了合规或调试目的追踪提示改动通常是不可能的。

将提示工程 (prompt engineering)运行化，通过引入结构、自动化和可追溯性到提示的生命周期中，解决了这些问题。

### 提示工程 (prompt engineering)运行化的核心组成部分

将提示视为受管理产物，涉及一些相互关联的组成部分：

#### 1. 提示版本控制

就像应用程序代码一样，提示应存储在像Git这样的版本控制系统（VCS）中。这提供了变更历史，方便协作，并支持回滚。

- **结构化存储：** 将提示存储在专用文件中（例如，`.prompt`、`.yaml`、`.json`），而不是直接嵌入 (embedding)在代码中。这种分离使它们更容易找到、管理和更新。
- **元数据：** 考虑将提示存储在结构化格式（如YAML或JSON）中，以允许附加相关元数据，例如版本号、作者、描述、目标模型和占位符变量。

```yaml
# 示例：结构化提示文件（例如：summarize_report_v1.2.yaml）
prompt_id: summarize_report
version: 1.2
author: jane.doe@example.com
date: 2023-10-27
description: "总结技术报告，侧重于主要发现和建议。"
model_compatibility: ["gpt-4", "claude-3"]
template: |
  分析以下技术报告并提供简洁的摘要。侧重于报告中的发现和主要建议。

  报告内容：
  {report_text}

  摘要：
variables:
  - report_text
```

- **分支策略：** 使用标准的Git分支策略（例如，特性分支）来开发和测试新的提示变体，然后再将它们合并到主分支。

#### 2. 提示模板

提示通常需要动态内容（用户输入、从RAG系统获取的上下文 (context)等）。提示模板引擎将提示的静态结构与运行时插入的动态数据分离。

- **引擎：** 可以使用像Jinja2（Python）或简单的f-string这样的库。
- **优点：** 使提示更清晰、易读，并且在构建复杂输入时更不容易出错。它明确定义了给定提示所需的输入。

```python
# 示例：在Python中使用Jinja2模板
from jinja2 import Template

prompt_template_string = """
指令：分类以下客户评论的情绪。
类别：积极、消极、中立

评论：{{ customer_review }}

情绪：
"""

template = Template(prompt_template_string)
filled_prompt = template.render(customer_review="送货速度惊人地快！")

print(filled_prompt)
# 输出：
# 指令：分类以下客户评论的情绪。
# 类别：积极、消极、中立
#
# 评论：送货速度惊人地快！
#
# 情绪：
```

#### 3. 提示测试与评估

系统化测试对于确保提示有效性和防止退步是必不可少的。

- **评估数据集：** 维护用于测试提示的精选数据集：
  - *基准数据集：* 具有已知期望输出的代表性输入示例。
  - *对抗性数据集：* 设计用于测试边缘情况、鲁棒性，或发现潜在偏差/安全问题的输入。
- **测试级别：**
  - *单元测试：* 验证提示结构、模板渲染和变量注入。
  - *集成测试：* 将提示（带测试数据）发送给目标LLM，并根据预期输出或质量指标评估响应。指标可能包括准确率、F1分数、用于摘要的ROUGE分数、毒性分数或自定义业务指标。
  - *A/B测试：* 将不同提示版本部署到部分生产流量，并使用预定义指标（例如，点击率、转化率、用户满意度）比较它们的性能。
- **自动化：** 将提示测试集成到CI/CD流水线中。失败的测试应阻止有问题的提示版本被部署。

#### 4. 提示注册表

类似于模型注册表，提示注册表作为批准提示的集中式、版本化目录。

- **中心枢纽：** 为跨不同应用程序和环境使用的提示提供单一的真实来源。
- **元数据追踪：** 为每个注册的提示版本存储丰富的元数据（ID、版本号、描述、作者、创建日期、相关评估指标，以及`开发`、`测试`、`生产`等状态）。
- **生命周期管理：** 方便提示基于测试结果和审批通过不同阶段（例如，从`测试`到`生产`）。
- **发现：** 允许团队发现并重复使用现有提示。

像MLflow这样的工具可以被修改或扩展以充当提示注册表，或者可以使用专用平台。

#### 5. 提示部署策略

将提示管理集成到部署流水线中，确保应用程序持续使用预期的提示版本。

- **CI/CD集成：** 提示更新会触发运行测试的CI流水线。成功的测试允许注册新的提示版本，并可能触发CD过程。
- **运行时获取：** 应用程序应在运行时根据期望的版本或环境阶段，从提示注册表或配置服务中获取所需的提示（模板和元数据）。避免在应用程序代码中硬编码提示。
- **部署模式：** 使用标准部署模式（例如金丝雀发布或A/B测试）来安全地推出新提示版本。在发布过程中密切监控性能。

### 将提示管理融入LLMOps工作流程

提示工程 (prompt engineering)的运行化并非独立存在；它与其他LLMOps组成部分紧密结合。

> 运行化提示工程的高级工作流程。Git仓库中的更改通过CI流水线触发自动化测试。批准的提示会被版本化并存储在提示注册表中。应用程序通过配置管理获取相应的提示引用，获取模板，用动态数据填充它，并将其发送给LLM。

### 工具与平台

尽管标准工具发挥着重要作用，专业平台正在兴起：

- **VCS：** Git对于版本控制仍然是基本的。
- **CI/CD：** Jenkins、GitLab CI、GitHub Actions等工具自动化测试和部署。
- **实验追踪/注册表：** MLflow、Weights & Biases可以被修改以追踪提示实验并充当注册表。
- **提示管理平台：** 专用工具（例如PromptLayer、Helicone、Lunary、开源框架）提供针对提示的版本控制、模板、测试、日志记录和协作的集成功能。
- **配置管理：** 像HashiCorp Consul或云提供商服务（AWS Parameter Store、Azure App Configuration）这样的工具可以管理应用程序在给定环境中应使用哪个提示版本。

### 困难与高级考量

运行化提示涉及持续的困难：

- **客观评估：** 定义与业务目标保持一致的可靠、自动化提示质量指标可能很困难，通常需要初期的人工参与评估。
- **提示漂移：** 提示的有效性会随时间变化，这可能是由于LLM更新或输入数据分布的变化。持续的监控和重新评估是必需的。
- **扩展性：** 管理成百上千的用于不同任务、模型或A/B测试的提示变体，需要强大的工具和组织能力。
- **安全：** 提示可能包含敏感数据的占位符。确保模板和日志记录机制不会泄露机密信息。对提示注册表的访问控制也很重要。

通过严谨地运行提示工程 (prompt engineering)，您可以构建更可靠、适应性更强、更易于管理的LLM驱动系统。将提示视为版本化、经过测试且可部署的产物，是成熟LLMOps实践的标志，使团队能够更快地迭代，并在其AI应用中保持更高的质量标准。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Approach to Development, Deployment, and Maintenance](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供构建和维护机器学习系统的系统性视角，涵盖数据、模型和基础设施，为在MLOps工作流中将提示视为核心工件提供了实践依据。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个结构化方法，用于从多个维度评估大型语言模型，这对于在操作化工作流中系统地测试和比较提示至关重要。
- [MLflow Docs: MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) — Databricks (2024)
  Publisher: Databricks
  提供了机器学习模型生命周期管理的官方文档，提供了一个可以应用于构建提示注册表的强大概念框架，如本节所述。
