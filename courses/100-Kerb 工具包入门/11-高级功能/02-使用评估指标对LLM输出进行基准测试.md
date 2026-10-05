# 使用评估指标对LLM输出进行基准测试

来源：[原文](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-11-advanced-llm-capabilities/benchmarking-with-evaluation-metrics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

生成文本很简单，但衡量其质量却是一个较大的难题。您如何知道新的提示是否优于旧的？如何证明切换到更昂贵的模型是值得投入的？主观的人工评估既慢又昂贵。这就是自动化评估和基准测试发挥作用的地方。

通过设定一套标准化测试并使用量化 (quantization)指标，您可以系统地衡量LLM应用程序的性能。这使您能够比较不同的提示、模型或配置，记录长期改进，并在上线前发现退步。

### 使用真实值指标进行评估

任何良好基准的根本是一组可靠指标，这些指标将模型的生成输出与“真实值”或参考答案进行对比。`evaluation` 模块提供多种标准、基于参考的指标，各自适用于不同任务。

#### 词汇指标：BLEU、ROUGE和F1分数

词汇指标通过比较生成输出与参考文本之间的词语重叠来运作。

\*\*BLEU（Bilingual Evaluation Understudy）\*\*起源于机器翻译，衡量准确率。它计算生成文本中有多少个n-gram（词语序列）出现在参考文本中。它还包括一个简短惩罚，以避免生成过短的输出。分数范围从0到1，其中1表示完美匹配。

它最适用于词语选择和顺序有特定要求的任务，例如翻译或代码生成。

```python
from kerb.evaluation import calculate_bleu

reference = "The quick brown fox jumps over the lazy dog"
candidate = "The quick brown fox jumped over the lazy dog"

# 候选文本与参考文本有良好的n-gram重叠，但并非完美匹配
bleu_score = calculate_bleu(candidate, reference)
print(f"BLEU 分数: {bleu_score:.3f}")
```

\*\*ROUGE（Recall-Oriented Understudy for Gisting Evaluation）\*\*常用于评估摘要。它通过检查参考文本中有多少个n-gram出现在生成输出中来衡量召回率。最常见的变体ROUGE-L使用最长公共子序列（LCS）来评分，使其对词语顺序更具弹性。

```python
from kerb.evaluation import calculate_rouge

reference_summary = "AI and machine learning are transforming technology with deep learning."
generated_summary = "AI is revolutionizing technology through machine learning and deep learning advances."

# ROUGE-L适用于摘要，因为它能捕捉句子层面的结构
rouge_l_scores = calculate_rouge(generated_summary, reference_summary, rouge_type="rouge-l")
print(f"ROUGE-L F1分数: {rouge_l_scores['fmeasure']:.3f}")
```

**F1分数和精确匹配**常见于问答任务。精确匹配要求生成输出与参考文本完全一致。F1分数提供了一种更宽容的替代方法，通过计算词元 (token)级别准确率和召回率的调和平均值，有效地衡量词语重叠，而不敏感于它们的顺序。

```python
from kerb.evaluation import calculate_f1_score, calculate_exact_match

reference = "William Shakespeare"
candidate = "Shakespeare"

f1 = calculate_f1_score(candidate, reference)
exact_match = calculate_exact_match(candidate, reference)

print(f"F1分数: {f1:.3f}")
print(f"精确匹配: {exact_match}")
```

#### 语义相似度

词汇指标虽然快速且实用，但在生成输出语义正确但使用了不同词语时会失效。语义相似度通过将候选文本和参考文本都转换为向量 (vector)嵌入 (embedding)并衡量它们之间的余弦相似度来解决此问题。这捕捉的是意义上的相似，而非仅仅词语上的相似。

分数接近1.0表示两个文本在意义上非常相似。

```python
from kerb.evaluation import calculate_semantic_similarity

reference = "The new feature improves system performance."
candidate = "The update enhances the application's speed."

# 词语不同，但意思相同
similarity = calculate_semantic_similarity(candidate, reference)
print(f"语义相似度: {similarity:.3f}")
```

### 构建并运行基准测试

基准测试是对标准化数据集进行结构化评估运行。该过程包含创建一组测试用例，定义如何针对它们运行您的系统，以及明确如何对结果进行评分。

首先，您使用 `TestCase` 类定义您的评估数据集。每个测试用例将输入与预期输出配对，作为真实值。

```python
from kerb.evaluation import TestCase

test_cases = [
    TestCase(
        id="问答_Python创建者",
        input="Who created Python?",
        expected_output="吉多·范罗苏姆"
    ),
    TestCase(
        id="问答_机器学习定义",
        input="What is machine learning?",
        expected_output="机器学习是人工智能的一个子集，它使系统能够从数据中学习。"
    ),
]
```

接下来，您需要一个函数来接收输入并生成输出。这通常是您的LLM生成逻辑。为测试目的，我们可以模拟它。

```python
def simple_qa_generator(question: str) -> str:
    """一个简单的生成函数，用于演示。"""
    if "谁创建了python" in question.lower():
        return "Python是由吉多·范罗苏姆创建的。"
    if "机器学习" in question.lower():
        return "机器学习是人工智能的一个分支，系统在此分支中从数据中学习。"
    return "我不知道。"
```

最后，您使用 `run_benchmark` 函数将这些组件组合起来。您提供测试用例、生成函数以及一个评估函数，该函数根据预期值对每个输出进行评分。

```python
from kerb.evaluation import run_benchmark, calculate_f1_score

# 评估函数定义了如何对每个测试用例进行评分
def evaluate_answer(output: str, expected: str) -> float:
    return calculate_f1_score(output, expected)

# 运行基准测试
benchmark_result = run_benchmark(
    test_cases=test_cases,
    generator_fn=simple_qa_generator,
    evaluator_fn=evaluate_answer,
    threshold=0.5, # 分数达到0.5或更高即为“通过”
    name="问答系统基准测试"
)

# 打印结果
print(f"通过率: {benchmark_result.pass_rate:.1f}%")
print(f"平均分数: {benchmark_result.average_score:.3f}")
print(f"通过: {benchmark_result.passed_tests} / {benchmark_result.total_tests}")
```

输出为您提供了系统表现的概要。`pass_rate` 显示满足质量阈值的测试用例百分比，而 `average_score` 给出整个数据集的整体性能感知。分析单个分数可以帮助您发现系统存在问题的具体方面。

### 实际应用：比较提示模板

基准测试是在开发过程中做出数据驱动决策的很好的工具。例如，您可以使用它来确定多个提示模板中哪个表现最优。

`benchmark_prompts` 函数简化此过程。它针对一组测试输入，为每个提示模板运行基准测试并汇总结果。

让我们比较三个不同提示，用于摘要任务。

```python
from kerb.evaluation import benchmark_prompts, calculate_rouge

# 三个待测试的不同提示
prompts = [
    ("简洁", "总结: {input}"),
    ("指导", "请对以下文本生成一个简洁的单句摘要: {input}"),
    ("详细", "请分析以下文本，并生成一份涵盖主要点的详细摘要: {input}"),
]

# 测试输入及其预期输出（真实值）
test_data = [
    {
        "input": "Python is a high-level, interpreted programming language known for its clear syntax...",
        "expected": "Python是一种多功能且易读的编程语言。"
    },
    {
        "input": "Machine learning is a subset of AI that enables systems to learn from data...",
        "expected": "机器学习是人工智能的一个分支，系统在此分支中从数据中学习。"
    }
]
test_inputs = [item["input"] for item in test_data]
expected_outputs = {item["input"]: item["expected"] for item in test_data}

# 一个使用提示模板的模拟生成函数
def generator_with_template(template: str, input_text: str) -> str:
    # 在实际应用中，这将调用LLM并使用格式化的提示
    # 在此示例中，我们将根据关键词返回一个固定响应
    if "concise" in template:
        return "这是一个简洁的摘要。"
    elif "detailed" in template:
        return "这是对所提供文本的一个非常详细和全面的摘要。"
    else:
        return "这是一个摘要。"

# 一个使用ROUGE-L的评估函数
def evaluate_summary(output: str, input_text: str) -> float:
    expected = expected_outputs[input_text]
    rouge_scores = calculate_rouge(output, expected, rouge_type="rouge-l")
    return rouge_scores['fmeasure']

# 运行提示比较
results = benchmark_prompts(
    prompts,
    test_inputs,
    generator_with_template,
    evaluate_summary
)

# 分析结果
for name, result in results.items():
    print(f"\n模板 '{name}':")
    print(f"  平均分数: {result.average_score:.3f}")

best_prompt = max(results.items(), key=lambda item: item[1].average_score)
print(f"\n表现最佳的提示: '{best_prompt[0]}'")
```

通过运行此比较，您可以定量确定哪种提示结构根据您选择的指标产生最高质量的输出。这种系统方法用数据代替猜测，使您能够迭代改进和提升LLM应用程序，更加可靠。



[交互图表：提示模板性能比较](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-11-advanced-llm-capabilities/benchmarking-with-evaluation-metrics#plot-ea4j55)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "提示模板性能比较"
    },
    "xaxis": {
      "title": {
        "text": "提示模板"
      }
    },
    "yaxis": {
      "title": {
        "text": "平均分数"
      },
      "range": [
        0,
        1
      ]
    },
    "barmode": "group",
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "bar",
      "name": "分数",
      "x": [
        "简洁",
        "指导",
        "详细"
      ],
      "y": [
        0.21,
        0.85,
        0.55
      ],
      "marker": {
        "color": [
          "#4dabf7",
          "#20c997",
          "#7950f2"
        ]
      }
    }
  ]
}
```

</details>



> 指导提示在所选指标上明显优于简洁和详细版本，为改进指明了明确方向。

## 参考资料

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  这篇基础论文介绍了BLEU，一种基于n-gram重叠的机器翻译和其他文本生成任务的广泛使用的评估指标。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out; Publisher: Association for Computational Linguistics; Pages: 74-81; DOI: [10.3115/1621896.1621901](https://doi.org/10.3115/1621896.1621901)
  这篇原始论文介绍了ROUGE，这是一套常用于评估摘要质量的指标，通过n-gram和最长公共子序列重叠来衡量召回率。
- [Speech and Language Processing: An Introduction to Computational Linguistics, Speech Recognition, and Language Technology](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Stanford University Press
  这是一本自然语言处理领域的标准教科书，涵盖了文本生成评估的基本概念，包括BLEU和ROUGE等词汇重叠指标，以及F1分数和精确匹配等问答指标。

---

[上一节](01-%E4%BD%BF%E7%94%A8%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E5%A4%84%E7%90%86%E5%9B%BE%E5%83%8F%E8%BE%93%E5%85%A5.md) · [下一节](03-%E4%B8%BA%E5%BE%AE%E8%B0%83%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE%E9%9B%86.md)
