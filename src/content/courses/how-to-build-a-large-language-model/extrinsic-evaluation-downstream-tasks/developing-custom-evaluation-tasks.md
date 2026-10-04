---
course: "how-to-build-a-large-language-model"
chapter: "extrinsic-evaluation-downstream-tasks"
lesson: "developing-custom-evaluation-tasks"
sourceId: 6065
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-22-extrinsic-evaluation-downstream-tasks/developing-custom-evaluation-tasks"
title: "开发定制评估任务"
description: "与特定应用程序相关的评估任务的设计指南。"
order: 6
plots: []
sourceHash: "7c7f5db3b12eebb7e1bfb789a1e991debd435734606386b8c8c699a8361d5d6f"
sourceCorrections: []
---

标准基准测试（如GLUE和SuperGLUE）提供了有益的比较依据，但它们在描述大语言模型 (LLM)在特定应用中的表现时，常常不够全面。您的具体应用场景可能涉及一个独特的专业范畴、一种新的互动模式，或者需要现有公开数据集未充分测试的能力。这些情况使得开发定制评估任务不仅有益，而且对于确认模型是否真正达到所需的性能标准，这是不可或缺的。设计、实施和分析这些定制评估的过程详细说明。

### 明确评估目标

在编写一行代码或收集任何数据之前，第一步是精确阐明您需要衡量什么。标准基准测试通常评估通用语言能力或在既定自然语言处理任务上的表现。然而，定制评估通常针对与您的应用更相关的特定行为或知识。请自问：

- **对我的应用程序来说，哪些特定能力很重要？** 是准确总结内部法律文件？根据自然语言请求生成语法正确的SQL查询？在客户服务聊天机器人中保持一致、富有同情心的人设？还是根据专有知识库回答问题？
- **这种能力下，“好的表现”是什么样的？** 这要求摆脱对质量的笼统认识。定义具体的成功标准。对于SQL生成，成功可能意味着查询可执行并返回正确数据。对于摘要，可能涉及包含特定实体或遵守长度限制。
- **这与现有基准测试有何不同？** 弄清这种差异有助于明确定制评估所提供的独特价值。

在此阶段保持清晰很重要。像“评估模型是否擅长金融”这样模糊的目标难以操作。而像“评估模型从季度财报中提取‘总收入’数据的能力，准确率超过95%”这样的具体目标，则为任务设计和指标开发提供了明确的目标。

### 设计任务格式

目标明确后，您需要设计任务格式，以引发模型所需的行为。该格式应尽可能模拟模型在生产环境中的使用方式。常见格式包括：

- **分类：** 为输入分配预定义标签（例如，将客户反馈分类为“正面”、“负面”、“中立”；识别用户查询背后的意图）。
- **提取：** 从文本中识别并提取特定信息（例如，从新闻文章中提取姓名、日期和地点；从研究论文中提取重要术语）。
- **生成：** 根据提示生成文本（例如，撰写电子邮件草稿、生成代码文档、会议纪要）。
- **问答（QA）：** 根据提供的上下文 (context)或内部知识回答问题（例如，根据公司政策回答常见问题、查询技术手册）。
- **排序：** 根据相关性或偏好对一组项目进行排序（例如，对搜索结果进行排序，对产品推荐进行排序）。
- **对话：** 模拟多轮对话以评估连贯性、有用性或任务完成能力。

考虑您的模型将接收的输入以及您期望的输出。例如，如果评估模型遵循复杂指令的能力，任务可能涉及提供一个详细的提示，阐明限制和预期输出，然后根据这些限制评估生成的文本。

### 数据收集与整理

定制评估的质量直接取决于评估数据的质量。

- **来源：** 数据可以来自各种地方。生产日志可能提供用户互动的真实示例。特定专业范畴的专家能够创建与其专业范畴高度相关的优质示例。如果真实数据稀缺，合成数据生成（可能使用另一个大语言模型 (LLM)并仔细审查）可以是一种选择，尽管它存在引入偏见或虚假信息的风险。
- **标注：** 如果您的任务需要人工判断（例如，评价回复的有用性，识别摘要是否抓住了主要观点），您将需要清晰的标注指南。这些指南应精确定义标签或分数，提供边缘案例的示例，并旨在尽量减少歧义。投入时间培训标注者，并衡量标注者间一致性（IAA），以确保一致性。科恩卡帕系数（$\kappa$）或弗莱斯卡帕系数等工具可以量化 (quantization)标注者间一致性。高IAA表明您的指南清晰且任务定义明确。
- **黄金标准：** 为您的评估集建立基本事实或“黄金标准”答案/标签。对于分类或提取任务，这通常是直截了当的。对于生成任务，则更为复杂。总结文档或回答问题可能存在多种有效方式。在这些情况下，您的黄金标准可能包含多个可接受的参考，或者您的评估指标可能需要考虑到这种可变性。
- **数据集大小：** 所需大小取决于任务和所需的统计显著性。即使是较小的、高质量的数据集（例如，100-500个精心整理的示例）对于识别特定失败模式非常有帮助，尽管更大的数据集更适合定量分析。确保您的数据集涵盖各种输入和潜在挑战。

### 制定评估指标

准确率、F1分数、BLEU或ROUGE等标准指标可以作为起点，但它们往往无法捕捉自定义任务的精细之处。您常常需要制定与您的特定评估目标相符的定制指标。

- **基于规则的指标：** 这些指标涉及基于预定义规则的程序化检查。它们对于评估格式的遵守、所需元素的包含或禁用内容的避免很有用。

  - *示例：* 检查生成的API调用是否包含正确的函数名和所需参数 (parameter)。
  - *示例：* 验证摘要是否在指定的字数范围内。

  ```python
  import re

  def check_report_format(generated_text: str) -> bool:
      """检查生成的文本是否包含“摘要：”部分和“建议：”部分"""
      has_summary = bool(re.search(r"Summary:", generated_text, re.IGNORECASE))
      has_recommendations = bool(re.search(r"Recommendations:", generated_text, re.IGNORECASE))
      return has_summary and has_recommendations

  # 示例用法：
  report = """
  Analysis Complete.
  Summary: Sales increased by 10%.
  Recommendations: Invest in marketing.
  """
  is_valid_format = check_report_format(report)
  print(f"报告格式有效: {is_valid_format}") # 输出: 报告格式有效: True
  ```
- **基于模型的指标：** 借助其他模型（可能是更小、更专业的模型）来评估输出。

  - *示例：* 使用毒性分类器来评估生成对话的安全性。
  - *示例：* 使用另一个大语言模型 (LLM)或嵌入 (embedding)模型来评估生成答案和黄金标准答案之间的语义相似性，这比单纯的词汇重叠评估（如BLEU/ROUGE）更进一步。
  - *示例：* 使用代码分析工具检查生成的代码是否存在语法错误或漏洞。
- **人工评估：** 在评估主观质量时不可或缺，例如有用性、连贯性、创造力、事实正确性（特别是模型训练数据之外的信息），或遵守特定语调/人设。设计良好的人工评估需要：

  - **清晰的评分标准：** 定义具体标准和评分量表（例如，有用性使用1-5的李克特量表，事实准确性使用二元判断）。
  - **比较评估：** 通常，要求人工比较两个输出（例如，模型A与模型B的输出）并选择更好的一个，比分配绝对分数更容易、更可靠。
  - **盲评：** 确保评估者不知道哪个模型生成了哪个输出，以避免偏见。

### 实施与执行

任务明确、数据收集完毕、指标选定后，您需要构建评估流程。

- **输入/输出处理：** 编写代码将评估数据格式化为适合模型的提示，并解析模型生成的输出。
- **模型推理 (inference)：** 与您的模型服务系统或推理库集成，以在评估数据集上运行模型。
- **指标计算：** 实现您的自定义指标（基于规则、基于模型或处理人工判断）的逻辑。
- **自动化：** 尽可能自动化流程，以便在模型更新时进行高效的重新评估。

这是使用PyTorch运行评估的简化结构：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 假设 custom_data_loader 生成 (提示, 黄金参考) 对
# 假设 custom_metric_function(生成的文本, 黄金参考) -> 分数

def run_custom_evaluation(model_name, custom_data_loader, custom_metric_function):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 将模型设置为评估模式

    results = []
    total_score = 0.0
    num_samples = 0

    with torch.no_grad(): # 禁用梯度计算以进行推理
        for prompt, gold_reference in custom_data_loader:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 生成输出（根据需要调整参数）
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 如有需要，仅提取生成部分
            generated_response = generated_text[len(prompt):]

            # 应用自定义指标
            score = custom_metric_function(generated_response, gold_reference)
            results.append({
                "prompt": prompt,
                "generated": generated_response,
                "gold": gold_reference,
                "score": score
            })
            total_score += score
            num_samples += 1

    average_score = total_score / num_samples if num_samples > 0 else 0
    print(f"平均自定义分数: {average_score:.4f}")
    return results, average_score

# --- 占位符定义 ---
# def load_my_custom_data():
#     # 在此处加载您的特定数据格式
#     # 示例： yield "为用户表生成SQL：", "SELECT * FROM users;"
#     pass
#
# def my_sql_metric(generated, gold):
#     # 示例： 检查生成的SQL是否有效并与黄金标准语义匹配
#     # 如果匹配返回 1.0，否则返回 0.0（简化版）
#     is_valid_sql = True # 占位符检查
#     matches_gold = generated.strip().lower() == gold.strip().lower() # 简化检查
#     return 1.0 if is_valid_sql and matches_gold else 0.0
#
# custom_data_loader = load_my_custom_data()
# results, avg_score = run_custom_evaluation("gpt2", custom_data_loader, my_sql_metric)
# --- 占位符定义结束 ---
```

*注意：这只是一个简化示例。实际评估通常涉及更精密的生成策略、批处理和错误处理。*

### 分析与迭代

仅仅计算一个总分数是不够的。真正的价值在于分析结果，以了解模型成功或失败的*原因*。

- **错误分析：** 人工审查评估实例的样本，特别是模型表现不佳的那些。对错误进行分类。模型是否生成虚假事实？是否未能遵循指令？是否产生了格式不正确的输出？是否表现出偏见？
- **定性分析：** 寻找成功和失败中的模式。模型是否在某些特定类型的提示或主题上存在困难？
- **迭代：** 利用分析所得的洞察来优化模型（例如，通过对与失败案例相似的数据进行进一步微调 (fine-tuning)）、评估任务本身（例如，澄清指令）、评估数据（例如，添加更具挑战性的示例），或优化指标（例如，设计一个更能捕捉特定失败模式的指标）。定制评估的开发通常是一个迭代过程。

> 定制大语言模型 (LLM)评估任务的迭代开发周期。

### 挑战与考量

开发定制评估需要仔细思考和资源投入：

- **成本与精力：** 创建高质量数据集和标注指南，特别是那些需要专业知识或大量人工标注的数据集和指南，可能耗时且昂贵。
- **指标有效性：** 确保您的自定义指标准确反映真实质量标准具有挑战性。一个指标可能很容易计算，但却无法与实际用户满意度或任务成功率良好关联。
- **偏见：** 评估数据集和指标可能无意中包含源数据或标注过程中存在的偏见。应积极寻找并减少与人口统计、观点或其他敏感属性相关的潜在偏见。
- **可扩展性：** 人工评估虽然有价值，但难以轻易扩展以进行频繁、大规模的测试。应平衡详细的人工分析与更具可扩展性的自动化指标。
- **维护：** 随着应用程序或模型的发展，定制评估可能需要更新以保持相关性。

尽管存在这些挑战，但精心设计的定制评估能够提供关于您大语言模型 (LLM)能力和不足的不可或缺的洞察，比仅仅依赖通用基准测试能更有效地指导开发工作。它们弥合了抽象的语言建模表现与实际取得的成果之间的差距。

## 参考资料

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  介绍了评估大型语言模型（LLM）的全面框架，涵盖多种场景，并说明了针对特定应用场景，多样化指标和任务设计的价值。
- [Handbook of Inter-Rater Reliability: The Definitive Guide to Measuring the Extent of Agreement Among Raters](https://books.google.com/books/about/Handbook_of_Inter_Rater_Reliability_4th.html?id=z1Y8oAEACAAJ) — Kilem Li Gwet (2014)
  Publisher: Advanced Analytics, LLC
  一本关于各种评分者间可靠性系数的全面指南，包括Cohen's Kappa和Fleiss' Kappa，对于确保标注评估数据集的质量至关重要。
