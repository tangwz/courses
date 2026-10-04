---
course: "evaluating-synthetic-data-quality"
chapter: "building-evaluation-reports"
lesson: "practice-generating-report-snippet"
sourceId: 5728
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-6-building-evaluation-reports/practice-generating-report-snippet"
title: "实践：生成质量报告片段"
description: "使用评估库（例如 SDMetrics）为样本数据集生成结构化质量报告。"
order: 6
plots: ["plots/5728-0.json"]
sourceHash: "d13de3ebd3b722a48b50b4bf1a196e42292f0d315833dffb1960d12bd7f02dca"
sourceCorrections: []
---

在对统计保真度、机器学习 (machine learning)实用性和隐私进行了各项单独评估后，下一步是将这些结果汇总成结构化且易于理解的格式。手动为每个合成数据集编译分数、图表和解释可能繁琐且容易出错。幸运的是，存在专门的库来简化此过程，让您能够以编程方式生成标准化质量报告。

`SDMetrics` 库可用于生成简洁的质量报告片段。该库提供了一种计算不同质量维度上多个指标的便捷方式，并以结构化方式呈现它们。一份完整报告通常包括根据您的具体目标定制的指标选择、详细的可视化和全面的解释。

我们假设您已将真实数据集 (`real_data`) 和合成数据集 (`synthetic_data`) 加载为 Pandas DataFrame。

```python
import pandas as pd
from sdmetrics import گزارش 품질 보고서
from sdmetrics.reports.utils import display_metadata
from sdmetrics.demos import load_single_table_demo

# 加载演示数据以确保可复现性
real_data, synthetic_data, metadata = load_single_table_demo()

# 显示元数据结构（可选，用于理解）
# display_metadata(metadata)
```

`metadata` 字典很重要。它告知 `SDMetrics` 数据类型（数值、分类、日期时间等）以及任何主键或约束。准确的元数据能带来更有意义的指标计算。对于演示数据，这已预加载。对于您自己的数据，您通常会手动定义此字典或使用检测工具。

现在，让我们创建一个质量报告对象。`SDMetrics` 提供不同的报告类。`QualityReport` 类提供涵盖多个方面的全面概览。

```python
# 初始化质量报告对象
report = QualityReport()

# 计算指标并生成报告
# 此步骤根据提供的数据和元数据执行计算
report.generate(real_data, synthetic_data, metadata)

# 报告对象现在包含计算出的分数和属性。
```

`generate` 方法运行与单表数据相关的默认指标集。这些通常包括逐列分布相似性度量、相关性相似度，以及可能的一些基本合成检测检查。

您可以检查整体质量分数，该分数汇总了各个指标分数：

```python
# 获取整体质量分数（0到1，越高越好）
overall_score = report.get_score()
print(f"整体质量分数: {overall_score:.3f}")
```

要理解该分数的构成因素，您可以查看各个属性（如列形状、列对趋势）的分数：

```python
# 获取不同属性的分数
properties = report.get_properties()
print("\n属性明细:")
print(properties)
```

这可能会产生如下输出：

```text
属性明细:
                    Property      Score
0                Column Shapes  0.851
1  Column Pair Trends  0.789
```

这些属性汇总了来自多个底层指标的结果。例如，“列形状”通常涉及使用科尔莫戈罗夫-斯米尔诺夫 (KS) 检验（针对数值数据）或卡方检验（针对分类数据）比较边际分布。“列对趋势”通常涉及比较相关矩阵或成对互信息。

要查看更多详情，您可以访问每个计算出的单独指标的结果：

```python
# 获取每个计算指标的详细结果
details = report.get_details(property_name='Column Shapes')
print("\n“列形状”的详细信息:")
print(details)
```

这提供了精细视图，显示了哪些列在分布相似性方面表现良好或不佳。

```text
“列形状”的详细信息:
        Column          Metric          Score Error
0  student_id  KSComplement  0.920   NaN
1   start_date  KSComplement  0.885   NaN
2     end_date  KSComplement  0.877   NaN
3    tuition  KSComplement  0.721   NaN
4        grade   TVComplement  0.850   NaN
...
```

这里，`KSComplement`（1 - KS 统计量）衡量数值分布相似性，而 `TVComplement`（1 - 总变异距离）衡量分类分布相似性。分数越高（越接近1）表示该特定列的分布保真度越好。

虽然 `SDMetrics` 在其报告结构内提供了一些基本绘图功能（尤其是在 Jupyter 等环境中），您也可以提取分数并使用 Plotly 等标准绘图库进行定制可视化，如“有效可视化评估结果”部分所讨论的。例如，让我们使用 Plotly 可视化属性分数。



![合成数据质量属性分数](plots/5728-0.json)



> 条形图显示汇总的“列形状”和“列对趋势”分数，提供报告主要发现的快速视觉摘要。

以编程方式生成此类报告片段对于以下方面非常有用：

1. **一致性：** 确保每次都计算相同的指标集。
2. **效率：** 自动化计算和汇总过程。
3. **基准测试：** 提供比较不同合成数据集或生成模型的标准结构，如“对不同合成数据集进行基准测试”部分所讨论的。

请记住，`SDMetrics` 等库提供默认设置，但也可定制。您可以选择特定指标、添加新指标，甚至创建完全自定义的报告结构以适应您的评估流程。本示例为将自动化报告集成到您的合成数据质量评估工作流程中提供了一个起点，将原始指标输出转化为易于理解的洞察。

## 参考资料

- [SDMetrics Documentation](https://docs.sdv.dev/sdmetrics/) — DataCebo (2024)
  Publisher: DataCebo
  SDMetrics库的官方文档，详细介绍了其用于评估合成数据质量的功能、指标和报告生成能力。
- [Benchmarking Synthetic Data Generation Models for Real-World Tabular Data](https://jmlr.org/tmlr/papers/v8/22-108.html) — Dorian Grinsztajn, Edouard Oyallon, Gaël Varoquaux (2022)
  Journal: Transactions on Machine Learning Research; Publisher: Journal of Machine Learning Research editorial board; Volume: 8; Pages: 1-33; DOI: [10.1162/tmlr_a_00049](https://doi.org/10.1162/tmlr_a_00049)
  这篇研究论文严格评估了合成数据生成模型，强调了合成数据在各种机器学习任务中的效用评估，并讨论了相关方法。
