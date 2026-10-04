---
course: "mlops-for-large-models-llmops"
chapter: "advanced-llmops-systems-workflows"
lesson: "practice-prompt-management"
sourceId: 3241
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/practice-prompt-management"
title: "实践：构建提示词管理工作流程"
description: "实现一个对LLM任务的不同提示词进行版本控制和测试的简单工作流程。"
order: 8
plots: []
sourceHash: "919ad538859580e5252233267985427b4286e54407689320ebf6359309a395bb"
sourceCorrections: []
---

有效的提示词 (prompt)管理是构建可靠且适应性强的大型语言模型应用的核心。如本章前面所述，将提示词视为重要的配置资源，对其进行版本控制、测试和受控部署，对于提升运维成熟度非常重要。在生产代码中直接更改提示字符串会导致不可预测的行为，并使系统性改进变得几乎不可能。

本实践部分提供了一个动手示例，展示如何构建一个用于管理和评估提示词的基本工作流程。我们将重点构建一个可重复的流程，用于将不同提示词版本与预定义的评估数据集进行比较，这为更复杂的提示词工程操作提供支持。

### 场景：改进文本摘要提示词 (prompt)

设想我们有一个应用，它使用大型语言模型来摘要文章。我们想尝试不同的提示词，看看哪一个能生成简洁（例如，低于特定字数限制）并有效捕捉主要内容的摘要（我们在此处通过一个简单检查来近似实现）。

我们的目标是创建一个工作流程，使我们能够：

1. 存储不同版本的摘要提示词。
2. 定义一个包含文章的小型评估数据集。
3. 针对评估数据集，使用大型语言模型运行每个提示词。
4. 基于长度评估生成的摘要。
5. 比较不同提示词的性能。

假设您可以访问一个大型语言模型端点，我们将使用一个占位函数`call_llm_api`来表示这种交互。

### 设置提示词 (prompt)仓库

一种管理提示词的简单有效方法，尤其是在开发阶段和针对小型团队，是使用Git仓库。让我们来组织我们的提示词：

```text
prompt_workspace/
├── prompts/
│   └── summarization/
│       ├── v1_concise.txt
│       └── v2_detailed_points.txt
├── evaluation_data/
│   └── articles.json
└── run_evaluation.py
```

创建`prompts/summarization`目录并添加两个示例提示词文件：

**prompts/summarization/v1\_concise.txt：**

```text
用50字或更少的内容摘要以下文章：

{article_text}
```

**prompts/summarization/v2\_detailed\_points.txt：**

```text
从以下文章中提供重要内容的简要摘要。力求清晰和简洁。

文章：
{article_text}

摘要：
```

请注意使用`{article_text}`作为占位符。我们稍后会使用基本的字符串格式化或模板引擎。

### 准备评估数据

创建一个`evaluation_data`目录以及一个名为`articles.json`的文件，其中包含一些示例文章：

**evaluation\_data/articles.json：**

```json
[
  {
    "id": "article_001",
    "text": "Researchers have developed a new AI technique capable of generating realistic images from text descriptions. The model, called Imagen, shows significant improvements over previous methods, particularly in rendering complex scenes and relationships between objects. This advancement could impact fields ranging from graphic design to virtual reality."
  },
  {
    "id": "article_002",
    "text": "Global supply chains continue to face disruptions due to a combination of factors, including geopolitical tensions, lingering pandemic effects, and increased demand for certain goods. Experts predict these challenges may persist, leading to higher prices and longer wait times for consumers. Companies are exploring strategies like regionalization and diversification to mitigate risks."
  }
]
```

### 构建评估工作流程脚本

现在，我们来创建`run_evaluation.py`脚本。该脚本将协调整个过程。

```python
import os
import json
import glob
from collections import defaultdict

# --- 占位LLM交互 ---
# 在实际场景中，这将涉及对LLM端点的API调用
# （例如，使用openai, huggingface_hub, anthropic等库）
# 它需要处理认证、潜在错误和重试。

def call_llm_api(prompt):
    """
    占位函数，模拟调用LLM API。
    替换为您的实际LLM API调用。
    """
    print(f"--- 调用LLM，提示词片段：{prompt[:100]}... ---")
    # 根据提示词的提示模拟不同输出，用于演示
    if "50 words or less" in prompt:
        simulated_summary = "New AI generates images from text. Imagen model improves realism for complex scenes, impacting design and VR."
    elif "important points" in prompt:
        simulated_summary = "AI Breakthrough: Imagen generates realistic images from text descriptions, outperforming prior models.\nSupply Chain Issues: Global disruptions persist due to various factors, potentially raising prices. Companies seek mitigation strategies."
    else:
        simulated_summary = "This is a generic summary output."

    # 模拟字数
    word_count = len(simulated_summary.split())
    print(f"--- 模拟LLM响应 (字数: {word_count}): {simulated_summary} ---")
    return simulated_summary, word_count

# --- 评估逻辑 ---

def evaluate_summary(summary, word_count, max_words=50):
    """
    简单评估：检查摘要是否符合字数标准。
    更复杂的评估可能包括：
    - 检查特定关键词
    - 使用另一个LLM进行质量评估（LLM作为裁判）
    - 与原文的语义相似性（需要嵌入）
    - 人工评审打分
    """
    meets_criteria = word_count <= max_words
    return {"word_count": word_count, "meets_criteria": meets_criteria}

# --- 工作流程编排 ---

def load_prompts(prompt_dir):
    """从目录中的文本文件加载提示词。"""
    prompts = {}
    pattern = os.path.join(prompt_dir, "*.txt")
    for filepath in glob.glob(pattern):
        prompt_name = os.path.basename(filepath).replace(".txt", "")
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts[prompt_name] = f.read()
    return prompts

def load_evaluation_data(data_path):
    """从JSON文件加载评估数据。"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def run_evaluation(prompts, eval_data):
    """针对评估数据运行提示词并收集结果。"""
    results = defaultdict(list)

    for prompt_name, prompt_template in prompts.items():
        print(f"\n===== 评估提示词: {prompt_name} ====")
        prompt_results = []
        for item in eval_data:
            article_id = item['id']
            article_text = item['text']

            # 使用f-string替换进行简单模板化
            try:
                filled_prompt = prompt_template.format(article_text=article_text)
            except KeyError:
                print(f"警告：提示词 '{prompt_name}' 缺少占位符 'article_text'。跳过。")
                continue

            # 调用LLM（使用占位符）
            summary, word_count = call_llm_api(filled_prompt)

            # 评估结果（简单长度检查）
            evaluation_metrics = evaluate_summary(summary, word_count, max_words=50)

            prompt_results.append({
                "article_id": article_id,
                "summary": summary,
                "metrics": evaluation_metrics
            })

        results[prompt_name] = prompt_results
    return results

def display_results(results):
    """显示评估结果的摘要。"""
    print("\n===== 评估总结 ====")
    for prompt_name, prompt_results in results.items():
        total_items = len(prompt_results)
        if total_items == 0:
            print(f"\n--- 提示词: {prompt_name} ---")
            print("未生成结果（请检查提示词占位符？）。")
            continue

        met_criteria_count = sum(1 for r in prompt_results if r['metrics']['meets_criteria'])
        avg_word_count = sum(r['metrics']['word_count'] for r in prompt_results) / total_items

        print(f"\n--- 提示词: {prompt_name} ---")
        print(f"  评估文章数量: {total_items} 篇")
        print(f"  符合字数标准（<50字）: {met_criteria_count}/{total_items} ({met_criteria_count/total_items:.1%})")
        print(f"  平均字数: {avg_word_count:.1f}")

# --- 主程序执行 ---
if __name__ == "__main__":
    PROMPT_DIR = "prompts/summarization"
    EVAL_DATA_PATH = "evaluation_data/articles.json"

    # 1. 加载提示词
    prompts = load_prompts(PROMPT_DIR)
    if not prompts:
        print(f"错误：在 {PROMPT_DIR} 中未找到提示词")
        exit()
    print(f"已加载提示词: {list(prompts.keys())}")

    # 2. 加载评估数据
    eval_data = load_evaluation_data(EVAL_DATA_PATH)
    print(f"已加载 {len(eval_data)} 篇评估文章。")

    # 3. 运行评估
    evaluation_results = run_evaluation(prompts, eval_data)

    # 4. 显示结果
    display_results(evaluation_results)

    # 潜在的下一步：将详细结果保存到CSV或JSON
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(evaluation_results, f, indent=2)
    # print("\n详细结果已保存到evaluation_results.json")
```

### 运行评估

在您的终端中导航到`prompt_workspace`目录并运行脚本：

```bash
python run_evaluation.py
```

您应该看到类似以下的输出（具体细节取决于占位函数`call_llm_api`的逻辑）：

```text
已加载提示词: ['v1_concise', 'v2_detailed_points']
已加载 2 篇评估文章。

===== 评估提示词: v1_concise ====
--- 调用LLM，提示词片段：用50字或更少的内容摘要以下文章：

Researchers have developed a new AI technique c... ---
--- 模拟LLM响应 (字数: 19): New AI generates images from text. Imagen model improves realism for complex scenes, impacting design and VR. ---
--- 调用LLM，提示词片段：用50字或更少的内容摘要以下文章：

Global supply chains continue to face disrup... ---
--- 模拟LLM响应 (字数: 19): New AI generates images from text. Imagen model improves realism for complex scenes, impacting design and VR. ---

===== 评估提示词: v2_detailed_points ====
--- 调用LLM，提示词片段：从以下文章中提供重要内容的简要摘要。力求清晰和简洁.... ---
--- 模拟LLM响应 (字数: 43): AI Breakthrough: Imagen generates realistic images from text descriptions, outperforming prior models.
Supply Chain Issues: Global disruptions persist due to various factors, potentially raising prices. Companies seek mitigation strategies. ---
--- 调用LLM，提示词片段：从以下文章中提供重要内容的简要摘要。力求清晰和简洁.... ---
--- 模拟LLM响应 (字数: 43): AI Breakthrough: Imagen generates realistic images from text descriptions, outperforming prior models.
Supply Chain Issues: Global disruptions persist due to various factors, potentially raising prices. Companies seek mitigation strategies. ---

===== 评估总结 ====

--- 提示词: v1_concise ---
  评估文章数量: 2 篇
  符合字数标准（<50字）: 2/2 (100.0%)
  平均字数: 19.0

--- 提示词: v2_detailed_points ---
  评估文章数量: 2 篇
  符合字数标准（<50字）: 2/2 (100.0%)
  平均字数: 43.0
```

基于这些（模拟的）结果，两个提示词 (prompt)都符合我们简单的长度标准，但`v1_concise`平均生成的摘要更短，而`v2_detailed_points`更接近限制，且可能更详细（尽管我们的简单指标无法捕捉质量）。这种定量比较有助于指导哪个提示词可能更适合应用的特定需求。

### 集成到MLOps管线

该脚本代表一次单独的评估运行。要将其集成到更大的LLMOps工作流程中，可以这样做：

1. **触发：** 使用CI/CD系统（如Jenkins、GitHub Actions、GitLab CI）自动化执行。当`prompts/`目录中的提示词 (prompt)更新时（例如，推送到特定分支或通过拉取请求），自动触发运行。
2. **实验追踪：** 将结果（提示词、指标、示例输出）记录到实验追踪平台（MLflow、Weights & Biases、Comet ML）。这可以进行历史比较和更好的可视化。
3. **提示词晋升：** 根据达到特定阈值的评估结果，CI/CD管线可以自动标记 (token)相应提示词版本的Git提交为“生产就绪”，或更新部署应用使用的配置文件，使其指向新的首选提示词。
4. **可视化：** 一张图表，表示此流程的简单图表：

> 一张图表，表示与CI/CD集成的自动化提示词评估和晋升工作流程。

### 后续步骤和考量

本示例提供了一个基本架构。您可以对其进行大幅扩展：

- **更复杂的模板化：** 使用Jinja2等引擎在提示词 (prompt)中实现更复杂的逻辑（条件判断、循环）。
- **更丰富的评估：** 实施指标。可以考虑使用嵌入 (embedding)模型进行语义相似性检查、特定关键词提取，或LLM作为裁判模式进行定性评估。
- **人工干预：** 集成人工评审和输出标注步骤，特别是对于自动化指标不足的任务。
- **提示词链/智能体：** 将此思想扩展到管理用于更复杂智能体系统的提示词序列或图。
- **可扩展性：** 对于非常大的评估数据集或频繁运行，优化LLM交互（批量处理、异步调用）并可能分布式评估工作量。

通过建立这样一个基本的提示词管理工作流程，您有助于采用更系统化、数据驱动的方法来开发和运行由大型语言模型支持的应用。

## 参考资料

- [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  Publisher: OpenAI
  提供关于设计、测试和优化提示以实现最佳LLM性能的指南。
- [MLOps for LLMs: 5 best practices](https://cloud.google.com/blog/products/ai-machine-learning/mlops-for-llms-5-best-practices) — Karl Weinberger, Jeremy Hermann, Zhaojun Li, and Dale Markowitz (2023)
  Publisher: Google Cloud Blog
  提供大型语言模型操作化的策略，包括生产环境中的提示管理和评估。
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — Databricks (2024)
  MLflow官方文档，详细说明如何跟踪机器学习实验，包括提示版本和评估指标。
