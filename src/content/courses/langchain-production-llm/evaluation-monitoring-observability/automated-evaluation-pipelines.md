---
course: "langchain-production-llm"
chapter: "evaluation-monitoring-observability"
lesson: "automated-evaluation-pipelines"
sourceId: 3907
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/automated-evaluation-pipelines"
title: "自动化评估流程"
description: "构建用于根据测试数据集自动评估应用性能的流程。"
order: 3
plots: []
sourceHash: "e57d8ef56a110930a483bb82638e63751490de9d33867b5d6661dc079e77eb81"
sourceCorrections: []
---

尽管使用LangSmith等工具进行手动测试和临时检查在开发过程中提供了有用的信息，但在生产环境中维护应用质量需要更系统化和可扩展的方法。自动化评估流程提供了这种结构，使得您的LangChain应用能够根据预设标准进行持续评估，及时发现退步，并便于长期追踪性能。构建这些流程是迈向操作成熟度的重要一步。

自动化评估流程通常协调多个组件，以根据代表性数据集运行LangChain应用，并使用特定标准衡量其性能。这使得能够进行可重复、客观的评估，并能直接整合到您的开发和部署工作流程中。

### 自动化评估流程的组成部分

构建一个高效的流程需要定义和整合以下核心要素：

1. **评估数据集：** 这是一组输入，通常还有对应的参考输出或标签，旨在测试应用的特定功能或极端情况。数据集可以从历史交互（例如，LangSmith追踪中记录的提示和响应）中整理，也可以合成生成，或由专家精心制作。LangSmith提供专用功能来创建和管理这些数据集，将输入与预期结果或标准关联起来。该数据集的质量和代表性是评估价值的基础。
2. **待测应用 (AUT)：** 这是您打算评估的LangChain链、代理或LLM配置的特定版本。为保证可复现性，对应用代码和配置进行版本控制很重要，这样流程就能持续测试预定的迭代版本。
3. **评估器：** 这些是衡量给定输入下AUT输出质量的函数或模块，通常将其与数据集中的参考输出进行比较。LangChain提供多种内置评估器，从简单的字符串比较和正确性检查，到更精密的衡量方式，如语义相似度（使用嵌入 (embedding)）或基于标准的评估（使用另一个LLM，常被称为“LLM作为判官”）。您也可以实现根据应用特定要求定制的自定义评估器，如前一节所述。
4. **执行框架：** 这是核心逻辑，用于遍历评估数据集，对每个输入运行AUT，调用已定义的评估器对生成的输出进行评估，并收集结果。此框架需要优雅地处理AUT执行或评估过程中可能出现的错误。LangSmith SDK提供`evaluate`函数，它简化了此过程，管理着应用、数据集和评估器之间的协调。
5. **结果存储与报告：** 评估结果（分数、指标、通过/失败状态、原始输出以及可能的执行追踪）需要持久存储。LangSmith自动存储与数据集和特定运行关联的评估结果。另外，结果可以记录到数据库、文件或监控平台，以便进行趋势分析、版本间比较和仪表盘展示。

### 流程实施

利用LangChain和LangSmith可以简化这些流程的创建。一个典型的工作流程包括：

1. **数据集准备：** 在LangSmith中创建或上传您的评估数据集。每个示例可能包含一个输入字典和一个可选的参考输出。

   ```python
   # LangSmith数据集条目示例结构
   example_input = {"question": "What is the capital of France?"}
   example_output = {"answer": "Paris"}
   # 或者用于基于标准的评估：
   # example_output = {"reference_answer": "Paris", "criteria": "Accuracy"}
   ```
2. **定义评估器：** 选择或定义与您的应用目标相关的评估器。您可以通过为LangSmith流程封装标准LangChain评估器来使用它们，或者编写返回分数的自定义Python函数。

   ```python
   # 使用LangSmith对LangChain评估器进行封装的示例
   from langsmith.evaluation import LangChainStringEvaluator
   from langchain_openai import ChatOpenAI

   # 用于检查与参考答案（QA）一致性的评估器
   # 这使用“cot_qa”（思维链QA）标准
   qa_evaluator = LangChainStringEvaluator(
       "cot_qa",
       config={"llm": ChatOpenAI(model="gpt-4", temperature=0)}
   )

   # 使用LLM根据特定标准进行判断的评估器
   criteria_evaluator = LangChainStringEvaluator(
       "criteria",
       config={
           "criteria": "conciseness",
           "llm": ChatOpenAI(model="gpt-4", temperature=0)
       }
   )
   ```
3. **配置运行：** 使用LangSmith SDK中的`evaluate`函数来协调运行。此函数接受您的AUT（作为函数或Runnable）、数据集名称和评估器列表。

   ```python
   # 使用LangSmith SDK运行评估的示例
   from langsmith import evaluate, Client

   # 假设 'my_chain' 是您的 LangChain Runnable/Chain/Agent
   # 定义一个目标封装器以确保正确的输入/输出格式
   def target(inputs):
       response = my_chain.invoke(inputs)
       # 确保我们返回评估器期望的特定字符串输出
       return response["output"] if isinstance(response, dict) else response

   # 假设 'my_dataset_name' 是您在 LangSmith 中数据集的名称
   evaluation_prefix = "production-test-run"

   results = evaluate(
       target, # 您的应用逻辑
       data=my_dataset_name,
       evaluators=[qa_evaluator, criteria_evaluator],
       experiment_prefix=evaluation_prefix,
       # max_concurrency=4, # 可选：并行运行
   )

   # 'results' 包含总结指标，完整的追踪记录在 LangSmith 中
   ```

### 与CI/CD的集成

当自动化评估集成到您的持续集成/持续部署（CI/CD）流程中时，它的真正威力才显现。通过在代码提交或部署前自动触发这些评估运行，您可以：

- **及早发现退步：** 自动标记 (token)与之前运行或基线相比性能指标的下降。
- **确保一致性：** 验证更改未对数据集测试的核心功能产生负面影响。
- **建立信心：** 提供客观证明，表明应用在发布前达到了质量标准。

一个典型的CI/CD集成可能涉及一个脚本，该脚本执行评估运行（类似于Python代码），然后检查结果。如果重要指标低于预设阈值，流程可能会失败，从而阻止部署可能性能下降的应用版本。

> 演示自动化评估流程如何集成到CI/CD过程中的工作流。该流程获取数据集，运行待测应用版本，应用评估器，记录结果，并根据指标阈值通知部署决策。

### 挑战与设计

自动化评估流程虽然功能强大，但需要周密的设计：

- **数据集维护：** 评估数据集需要随着应用一同发展并反映使用模式。过时或不具代表性的数据集会产生误导性结果。
- **评估器局限：** 自动化指标可能无法捕捉质量的所有方面，特别是语气或创造力等主观方面。作为判官的LLM评估器可能有效，但会增加延迟和成本，且其判断也可能不一致。
- **成本：** 运行评估，特别是涉及大量LLM调用（无论是待测应用中还是评估器中）的评估，会产生计算成本和API使用费。优化流程以提高效率。
- **阈值设置：** 定义合适的指标通过/失败阈值需要仔细考量，并根据应用目标和观察到的性能进行迭代优化。

自动化评估流程不能替代细致的监控或人工审查，但它们提供了一个必不可少的自动化质量保证层。通过系统性地使用整理好的数据集和客观指标来运行您的LangChain应用，您可以建立一个可重复的流程来验证性能、发现退步，并增强对生产部署的信心。这种做法是大规模运行可靠高效LLM应用的基础。

## 参考资料

- [LangChain Evaluation Overview](https://python.langchain.com/docs/concepts/evaluation/) — LangChain (2024)
  LangChain评估模块的官方指南，涵盖评估器、数据集以及与LangSmith集成以构建自动化评估流程。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Clemens Mewald and Evgeni Begel (2021)
  Journal: Google Cloud Architecture Center; Publisher: Google Cloud
  本资源涵盖自动化ML模型生命周期的背景，包括持续评估，这对自动化评估流程很重要。
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (2023)
  Journal: NeurIPS 2023 Datasets and Benchmarks Track; DOI: [10.48550/arXiv.2306.05685](https://doi.org/10.48550/arXiv.2306.05685)
  本文介绍使用LLM作为其他LLM评估器的方法，这是本节讨论的一种评估器类型。
- [Data Validation for Machine Learning: A Practical Guide](https://www.oreilly.com/library/view/data-validation-for/9781098101297/) — Robert Monarch (2021)
  Publisher: O'Reilly Media
  本书讨论机器学习模型数据质量和验证的重要性，影响“评估数据集”部分和“数据集维护”挑战。
