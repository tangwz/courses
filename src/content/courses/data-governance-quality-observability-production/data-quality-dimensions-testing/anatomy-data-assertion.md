---
course: "data-governance-quality-observability-production"
chapter: "data-quality-dimensions-testing"
lesson: "anatomy-data-assertion"
sourceId: 8084
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-2-data-quality-dimensions-testing/anatomy-data-assertion"
title: "数据断言的剖析"
description: "编程数据质量断言的结构与逻辑。"
order: 2
plots: []
sourceHash: "e20b0061220c3badd303fcb3893bb4a46c259f2660590d6b630d7f2545abe442"
sourceCorrections: []
---

数据断言界定了可接受数据与损坏数据之间的界限。软件工程依赖单元测试来验证代码逻辑是否正确地将输入转换为输出。数据工程呈现出一种不同的情况：逻辑，通常是一个简单的复制或转换，通常是正确的，但输入数据本身是不稳定的。断言作为一个谓词函数，评估数据集的状态并返回一个二元结果：通过或失败。

在基本层面，一个断言 $A$ 将谓词 $P$ 应用于数据集 $D$。如果数据满足谓词，系统将继续运行。否则，将触发干预措施。


$$
A(D) = \begin{cases} 
\text{通过} & \text{若 } \forall r \in D, P(r) \text{ 为真} \\
\text{失败} & \text{否则}
\end{cases}
$$


这种二元特性是自动化管道所必需的。虽然数据分析师可能容忍“大致干净”的数据，但管道需要一个确定性信号来决定是加载表还是停止执行以防止下游污染。

### 断言的结构

每个数据断言都由三个不同的组成部分构成：选择器、谓词和失败阈值。理解这些组件使工程师能够编写模块化和可重用的测试，而不是临时的脚本。

**选择器**决定了测试的范围。它识别哪些行或列需要验证。这可能是一个特定列，例如 `user_id`，或 `status = 'active'` 的行的子集。

**谓词**是必须成立的逻辑条件。这是核心逻辑，例如确保时间戳在过去，或者字符串匹配特定的正则表达式模式。

**失败阈值**定义了断言的判定严密性。在零容忍环境中，单个失败的行会导致整个断言失败。在较宽松的环境中，您可能允许高达1%的失败率，然后才发出警报。

> 标准数据质量断言的流程是从选择到对照容忍阈值进行评估。

### 行级断言

行级断言逐点验证数据。当一个规则必须独立应用于每个单独的记录时，就会使用这些断言。常见例子包括空值检查、参照完整性检查和域约束。

考虑一个用户注册表。我们可能会断言 `email` 列绝不能为空，并且必须包含一个“@”符号。这需要遍历数据集并将谓词应用于每一行。

在 SQL 中，行级断言通常作为查询实现，该查询查找是否存在不良数据。如果查询返回零行，则断言通过。

```sql
-- 检查无效电子邮件格式的断言
-- 如果 count > 0，则断言失败
SELECT count(*) as failure_count
FROM users
WHERE email IS NULL 
   OR email NOT LIKE '%@%';
```

在像 Pandas 或 PySpark 这样的 Python 数据处理框架中，我们通过创建一个布尔掩码来实现这一点。

```python
import pandas as pd

def assert_positive_prices(df: pd.DataFrame):
    # 谓词：价格必须大于零
    mask = df['price'] <= 0

    # 评估
    failed_rows = df[mask]
    failure_count = len(failed_rows)

    if failure_count > 0:
        raise ValueError(f"断言失败：发现 {failure_count} 个非正价格的商品。")

    return True
```

这种方法允许精确识别有问题的记录。当行级断言失败时，系统可以隔离导致失败的特定 ID，这简化了调试。

### 基于集合的断言

基于集合的（或聚合）断言验证数据集的整体属性。与行级检查不同，这些无法通过查看单个记录来确定。它们依赖于组的统计属性。

这些断言对于检测技术上满足模式规则但表明数据质量问题的异常情况十分必要。例如，如果一个表通常每小时接收 10,000 行，而接收 50 行在技术上是“有效”的（模式正确），但实际上是灾难性的。

基于集合的断言依赖于计算指标 $M$ 并检查它是否落在可接受的范围 $[L, U]$ 内。


$$
L \le M(D) \le U
$$


常见的聚合维度包括：

1. **数量：** 行数或数据大小。
2. **分布：** 列的均值、中位数或标准差。
3. **唯一性：** 主键列中不同值的百分比。

以下图表显示了一个分布检查，其中断言验证每日平均交易值保持稳定。

```plotly
{
  "layout": {
    "title": "聚合断言：每日平均交易值",
    "xaxis": {"title": "日期", "showgrid": false},
    "yaxis": {"title": "平均值 ($)", "showgrid": true, "gridcolor": "#e9ecef"},
    "plot_bgcolor": "white",
    "font": {"family": "Arial"},
    "shapes": [
      {
        "type": "rect",
        "xref": "paper",
        "yref": "y",
        "x0": 0,
        "y0": 45,
        "x1": 1,
        "y1": 55,
        "fillcolor": "#b2f2bb",
        "opacity": 0.3,
        "line": {"width": 0}
      }
 राहुल  ]
  },
  "data": [
    {
      "x": ["Mon", "Tue", "Wed", "Thu", "Fri"],
      "y": [48, 52, 49, 47, 62],
      "type": "scatter",
      "mode": "lines+markers",
      "line": {"color": "#4dabf7", "width": 3},
      "marker": {"size": 8, "color": "#228be6"},
      "name": "观测值"
    },
    {
      "x": ["Mon", "Tue", "Wed", "Thu", "Fri"],
      "y": [62],
      "type": "scatter",
      "mode": "markers",
      "marker": {"size": 12, "color": "#fa5252", "symbol": "x"},
      "name": "失败"
    }
  ]
}
```

> 阴影绿色区域代表可接受的阈值 ($[45, 55]$)。周五的数据点 ($62$) 落在该范围之外，触发断言失败。

### 实现模式

在设计这些断言时，我们通常将它们分类为**强制阻断**和**软性警告**。

**强制阻断**会立即停止管道。当数据损坏到如此程度，以至于加载它将破坏下游仪表板或财务报告时，这是必要的。模式违规或主键重复通常会触发强制阻断。

**软性警告**允许数据继续处理，但记录警报。这适用于聚合断言，其中“正确”范围是主观的。如果行数下降 15%，这可能是一个假期或网络问题。我们记录异常但不停止业务流程。

有效的数据工程涉及将正确的断言类型映射到业务风险。

| 断言类型 | 应用场景 | 失败动作 |
| --- | --- | --- |
| **空值检查** | 必填字段 (ID, 时间戳) | 强制阻断 |
| **参照完整性** | 外键匹配主键 | 强制阻断 |
| **数量检查** | 行数与滚动平均值对比 | 软性警告 |
| **分布检查** | 机器学习 (machine learning)特征中的值偏移 | 软性警告 |

通过将数据断言视为严密的业务契约，我们从被动错误修复转向主动质量防御。其结构保持一致：选择数据，应用谓词，并强制执行结果。

## 参考资料

- [The DAMA Guide to the Data Management Body of Knowledge (DMBOK2)](https://technicspub.com/dmbok/) — DAMA International (2017)
  Publisher: Technics Publications
  这本综合指南提供了数据管理的标准框架，包含数据质量和治理的详细章节，对于理解数据断言的背景和目的至关重要。
- [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108269/) — Joe Reis, and Matt Housley (2022)
  Publisher: O'Reilly Media
  一本用于构建生产就绪数据系统的实用指南，涵盖数据质量、测试方法和可靠数据管道的架构模式，与断言的实现直接相关。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供了构建可靠数据系统的基础原则，为数据质量断言对于数据一致性和系统完整性为何重要提供了更广阔的背景。
- [Automatic Data Validation using Semantic Dependencies](https://dl.acm.org/doi/10.14778/3407881.3407945) — Ali Abed, Carsten Binnig, and Ahmad Khan (2020)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment Inc.; Volume: 13; Pages: 2976–2989; DOI: [10.14778/3407881.3407945](https://doi.org/10.14778/3407881.3407945)
  这篇研究论文介绍了自动化数据验证的方法，与数据管道中断言作为谓词函数的正式定义和实现相符。
