---
course: "essential-numpy-pandas"
chapter: "grouping-aggregating-data-pandas"
lesson: "split-apply-combine"
sourceId: 1818
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas/split-apply-combine"
title: "拆分-应用-组合方法"
description: "了解分组和汇总操作的原理。"
order: 1
plots: []
sourceHash: "9cbec13af6b7eb6979ccba1c29a632c3e986008c437a26134af8f6f8b617467a"
sourceCorrections: []
---

在处理数据集时，你通常需要看的不仅仅是单个行或列。一项常见而重要的工作是计算数据的*部分*的汇总统计量或对其进行操作，这些部分由一个或多个列中的值确定。例如，你可能想找出每个产品类别的平均销售额、每个网站部分的访问总数，或者每个气象站记录的最高温度。

分组分析通常遵循一种被称为“**拆分-应用-组合**”的方法。这是一种有用的思考方式，可以帮助理解Pandas等工具处理分组操作的方式。

### 1. 拆分

第一步是将原始DataFrame拆分成多个小块或组。这种划分是根据一个或多个指定列中的唯一值进行的，这些列通常被称为“分组键”。

想象你有一个包含销售数据的DataFrame，有`ProductCategory`和`SalesAmount`等列。如果你选择按`ProductCategory`分组，Pandas会划分DataFrame的行。所有`ProductCategory`为“Electronics”的行会构成一个组，所有为“Clothing”的行会构成另一个组，依此类推，适用于数据中所有独特的类别。每个部分都包含所有原始列，但只包含与特定键值对应的行。

### 2. 应用

数据被拆分成这些独立的组后，下一步就是对每个组应用一个函数。这个函数可以是：

- **聚合：** 计算每个组的单个汇总统计量（例如，`sum()`、`mean()`、`count()`、`min()`、`max()`）。这会把每个组简化为一个单一值或一组汇总值。
- **转换：** 进行一些针对组的计算，但得到一个与输入组形状相同的对象（例如，对每个组内的数据进行标准化）。
- **筛选：** 根据某些组级别的计算排除整个组（例如，只保留具有足够数据点的组）。

主要之处在于，所选函数在“拆分”阶段生成的每个组上独立运行。如果你计算的是每个类别的平均销售额，“Electronics”的平均值计算与“Clothing”的平均值计算是分开进行的。

### 3. 组合

最后，在“应用”阶段对每个组应用函数所得到的结果被收集并合并到一个新的数据形式中。通常，这种结果形式是一个新的Pandas Series或DataFrame。

这个结果对象的索引通常由“拆分”阶段确定的独特分组键构成。如果你计算了每个产品类别的平均销售额，最终结果很可能是一个Pandas Series，其索引包含独特的产产品类别（“Electronics”、“Clothing”等），而值则是每个组计算出的相应平均销售额。

> “拆分-应用-组合”过程的图示。数据首先根据键被拆分成组，然后一个函数分别应用于每个组，最后将结果组合成一个最终输出。

这种“拆分-应用-组合”方法是一种通用的模式，适用于许多数据分析问题。在Pandas中，`groupby()`方法是支持此过程的主要工具。了解这种三阶段方法提供了一种清晰的思路，来考虑如何高效地进行复杂的组级操作。本章后续部分将向你展示如何使用Pandas `groupby()`来具体操作这个方法。

## 参考资料

- [Group by: split-apply-combine](https://pandas.pydata.org/docs/user_guide/groupby.html) — The pandas development team (2024)
  官方文档，说明了 `groupby` 方法及其在 Pandas 中实现分分合合模式。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一个主要资源，详细解释了 `groupby` 操作和 Pandas 中底层分分合合的逻辑。
- [The Split-Apply-Combine Strategy for Data Analysis](https://www.jstatsoft.org/article/view/v040i01) — Wickham, H. (2011)
  Journal: Journal of Statistical Software; Volume: 40; Pages: 1-29; DOI: [10.18637/jss.v040.i01](https://doi.org/10.18637/jss.v040.i01)
  本文介绍并形式化了分分合合模式，为数据分析提供了其概念基础。
