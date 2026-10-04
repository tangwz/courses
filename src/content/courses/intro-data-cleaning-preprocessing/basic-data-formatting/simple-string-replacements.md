---
course: "intro-data-cleaning-preprocessing"
chapter: "basic-data-formatting"
lesson: "simple-string-replacements"
sourceId: 4032
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-5-basic-data-formatting/simple-string-replacements"
title: "简单字符串替换"
description: "使用基本替换功能纠正常见拼写错误或标准化术语。"
order: 4
plots: []
sourceHash: "d79bc4ed0e51a3f611cc0d3db885b50c51e431833809a121c64d8fc56564f038"
sourceCorrections: []
---

虽然对文本大小写进行标准化并移除多余空格可以解决许多格式不一致问题，但你通常还会遇到文本值本身内部需要修正的差异。例如，在单个列中，同一个事物可能使用不同的拼写、缩写或同义词。简单字符串替换是一种直接的方法，通过查找特定的文本值并用标准化的替代值替换它们来修正这些问题。

可以把它想象成文本编辑器中的“查找和替换”功能，但系统地应用于整个数据列。这对于分类列尤其有用，在这些列中，一致性对于分组和分析很重要。如果你的国家列数据中包含“USA”、“U.S.A.”和“United States”，分析工具会把它们当作三个不同的类别，除非你将它们标准化为一个值，例如“USA”。

### 简单替换如何进行

核心思想很直接：你指定要查找的精确字符串，以及你想用来替换它的精确字符串。大多数数据分析工具都提供功能，可以在整个列上高效地执行此操作。

例如，如果你有一个名为 `Status` 的列，包含“Complete”、“Completed”和“Finished”等值，你可能会决定将所有这些标准化为“Completed”。这个过程会涉及两个替换步骤：

1. 查找所有“Complete”的实例并将其替换为“Completed”。
2. 查找所有“Finished”的实例并将其替换为“Completed”。

让我们看一个小的例子。想象一个跟踪产品尺寸的列：

| Product ID | Size |
| --- | --- |
| P101 | Small |
| P102 | Med |
| P103 | Lrg |
| P104 | Small |
| P105 | Med. |
| P106 | Large |

这里，“Med”和“Med.”很可能应该标准化为“Medium”，而“Lrg”和“Large”则标准化为“Large”。应用简单替换会转换数据：

1. 将“Med”替换为“Medium”。
2. 将“Med.”替换为“Medium”。
3. 将“Lrg”替换为“Large”。

结果列会看起来整洁很多：

| Product ID | Size |
| --- | --- |
| P101 | Small |
| P102 | Medium |
| P103 | Large |
| P104 | Small |
| P105 | Medium |
| P106 | Large |

### 执行多次替换

通常，你需要在同一列中进行多次替换。你通常可以通过几种方式完成此操作：

1. **顺序地：** 一个接一个地应用替换规则，如上面的例子所示。
2. **使用映射：** 定义一组规则（通常在编程中使用字典或映射结构），指定多个“查找”和“替换”对。如果你需要进行大量替换，这通常更高效。

例如，为尺寸例子使用映射可能看起来像这样：

`replace {'Med': 'Medium', 'Med.': 'Medium', 'Lrg': 'Large'}`

将此映射应用于 `Size` 列会一次性执行所有必要的替换。

### 实际考量

- **精确匹配：** 简单字符串替换通常查找精确匹配。替换“US”不会改变“USA”，除非你明确指示它替换“USA”。请注意，替换子字符串可能会无意中改变其他词语（例如，替换“can”可能会改变“cannot”或“scan”）。使用这种基本方法时，尽可能坚持替换完整的、不同的值。
- **大小写敏感性：** 请记住，文本比较通常区分大小写（“Apple”与“apple”不同）。通常，最好在执行字符串替换*之前*标准化大小写（使用上一节的方法），以避免为“USA”、“usa”和“Usa”定义单独的规则。
- **顺序很重要（有时）：** 如果你的替换可能会重叠（例如，将“St.”替换为“Street”，同时将“Main St.”替换为“Main Street”），则应用顺序替换的顺序会影响结果。使用映射通常能更可预测地处理这种情况。
- **验证：** 应用替换后，务必检查你的数据。再次检查列中的唯一值，确保更改正确完成并且没有引入新的错误。

简单字符串替换是一种基本工具，用于清理分类数据和修正常见的文本不一致问题。通过仔细应用它们，你使你的数据集更接近变得可靠并为分析做好准备。虽然更复杂的文本操作通常需要像正则表达式这样的方法（允许模式匹配），但简单替换能有效处理各种常见的 数据标准化任务。

## 参考资料

- [Working with text data](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html) — pandas development team (2023)
  涵盖 pandas 中用于数据标准化的基本字符串操作方法，包括 `str.replace()` 和 `replace()`。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用 pandas 进行数据操作的指南，涵盖使用字符串替换进行文本数据清洗和标准化的技术。
- [Data Cleaning: A Practical Approach](https://link.springer.com/book/10.1007/978-3-030-80164-8) — Laura Monso-Velilla, Josep Maria Batalla-Busquets, and Joan R. Batalla (2021)
  Publisher: Springer; DOI: [10.1007/978-3-030-80164-8](https://doi.org/10.1007/978-3-030-80164-8)
  提供数据清洗概述，讨论字符串标准化作为确保数据质量的关键步骤。
