---
course: "advanced-python-programming-ml"
chapter: "python-performance-optimization-ml"
lesson: "efficient-pandas-usage"
sourceId: 2267
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/efficient-pandas-usage"
title: "大型数据集的Pandas高效用法"
description: "在机器学习中处理大型数据集时，优化Pandas数据操作以提高性能和内存效率。"
order: 3
plots: ["plots/2267-0.json"]
sourceHash: "6c766ff3c332f76777b2d8365f2a62801ecbcbd42cbe7551ada2c6927f4a2160"
sourceCorrections: []
---

Pandas是Python机器学习 (machine learning)中数据处理的重要组成部分，它因其富有表现力的API和便捷的DataFrame结构而备受推崇。然而，对于较小数据集运行良好的默认设置和常见使用模式，在处理千兆字节或太字节数据时，可能很快成为性能瓶颈。在样本数据上看似瞬间完成的操作，在完整数据集上可能需要数小时或耗尽系统内存。为了应对这些挑战，提供了更高效使用 Pandas 的策略，主要侧重于在数据集变大时减少内存占用和提高计算速度。

### 优化数据类型 (dtypes)

最有效但常被忽略的优化之一是为列选择合适的数据类型。Pandas通常默认为`int64`、`float64`或通用`object`类型（通常存储Python字符串）。这些默认设置会占用远超所需的内存。

- **整数:** 如果您的整数值在已知范围内，请使用最小的合适类型：`int8` (-128 到 127)、`uint8` (0 到 255)、`int16`、`uint16`、`int32`、`uint32`，而不是默认的`int64`。
- **浮点数:** 同样，如果不需要`float64`的精度，`float32`可以将内存使用量减半。
- **对象/字符串:** 包含字符串数据的列，特别是那些唯一值数量有限（低基数）的列，是优化的主要对象。

您可以使用`.info()`查看内存使用情况：

```python
import pandas as pd
import numpy as np

# 包含默认类型的示例DataFrame
data = {
    'user_id': np.random.randint(1, 10000, size=1_000_000),
    'rating': np.random.uniform(1.0, 5.0, size=1_000_000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=1_000_000)
}
df = pd.DataFrame(data)
print("内存使用情况 (默认类型):")
df.info(memory_usage='deep')

# 优化类型
df['user_id'] = df['user_id'].astype('uint16') # 最大值为9999，适合uint16
df['rating'] = df['rating'].astype('float32')
df['category'] = df['category'].astype('category')

print("\n内存使用情况 (优化后类型):")
df.info(memory_usage='deep')
```

应用适当的类型后，您通常会观察到内存占用显著减少。

### 使用分类数据类型

如上例所示，将低基数（相对于总行数而言，唯一值较少）的字符串列转换为`category` dtype是非常有效的。在内部，Pandas使用映射到唯一字符串值的整数编码来表示分类数据。这会大幅减少内存占用，相较于存储重复字符串，并且还能加快分组操作。

```python
# 假设'category'列包含许多重复的字符串，例如'Type A'、'Type B'等。
# 转换为分类类型
# df['category'] = df['category'].astype('category') # 在上一个示例中已完成

# 优点:
# 1. 减少内存使用
# 2. 加快该列上的分组和连接操作
```

### 避免按行操作和循环

使用`iterrows()`、`itertuples()`或显式`for`循环等结构迭代DataFrame行通常效率低下。这些方法通常涉及每行的类型转换和函数调用开销。

类似地，对`.apply()`使用按行应用的Python函数（`axis=1`）可能会很慢，因为它本质上在内部执行一个Python循环。虽然对于复杂逻辑很方便，但它绕过了Pandas优化的C实现。

```python
# 效率低下：对简单操作使用带有axis=1的.apply
# def custom_logic(row):
#     if row['col_a'] > 5 and row['col_b'] < 10:
#         return row['col_c'] * 2
#     else:
#         return row['col_c']
#
# df['new_col'] = df.apply(custom_logic, axis=1) # 尽可能避免此操作

# 更高效：使用布尔索引和np.where的向量化方法
import numpy as np

condition = (df['col_a'] > 5) & (df['col_b'] < 10)
df['new_col'] = np.where(condition, df['col_c'] * 2, df['col_c'])
```

总是寻找向量 (vector)化的替代方法。使用布尔索引、`.loc`/`.iloc`以及一次性对整个Series或DataFrame进行操作的内置Pandas/NumPy函数。这些操作通过优化的C或Cython代码实现，执行速度快得多。

### 分块处理数据

如果数据集太大无法完全放入内存，或者即使能放入但中间操作会生成大型临时DataFrame，那么分块处理数据是一种必要的策略。Pandas的`read_csv`（以及类似的读取函数，如`read_sql`）支持`chunksize`参数 (parameter)。这会返回一个迭代器，每次生成指定大小的DataFrame。

```python
import pandas as pd

chunk_iter = pd.read_csv('large_dataset.csv', chunksize=100_000) # 每次处理10万行

results = []
for chunk_df in chunk_iter:
    # 对数据块执行必要的处理
    # 示例：过滤行并计算新列
    processed_chunk = chunk_df[chunk_df['value'] > 0].copy() # 使用.copy()以避免SettingWithCopyWarning
    processed_chunk['log_value'] = np.log(processed_chunk['value'])

    # 汇总结果（如果需要）或存储/写入已处理的数据块
    # 示例：计算每个数据块的总和
    results.append(processed_chunk['log_value'].sum())

# 如果有必要，合并所有数据块的结果
total_sum = sum(results)
print(f"log_value的总和: {total_sum}")

# 替代方法：将处理后的数据块追加到文件或数据库
# for chunk_df in chunk_iter:
#    processed_chunk = ... # 处理数据块
#    processed_chunk.to_csv('processed_output.csv', mode='a', header=not pd.io.common.file_exists('processed_output.csv'), index=False)
```

分块处理使您能够对远超可用内存的数据集应用复杂的转换和聚合操作，这使得Pandas能够处理规模更大的问题。

### 性能比较示例

我们来图形化展示按行应用和向量 (vector)化方法之间的典型性能差异。假设我们要根据两列计算一个条件值。

```python
import pandas as pd
import numpy as np
import time

# 创建一个示例DataFrame
size = 1_000_000
df = pd.DataFrame({
    'col_a': np.random.rand(size) * 10,
    'col_b': np.random.rand(size) * 20,
    'col_c': np.random.rand(size) * 5
})

# 方法1：使用lambda的.apply（按行）
start_time = time.time()
df['apply_result'] = df.apply(lambda row: row['col_c'] * 2 if row['col_a'] > 5 and row['col_b'] < 10 else row['col_c'], axis=1)
apply_time = time.time() - start_time

# 方法2：使用np.where的向量化方法
start_time = time.time()
condition = (df['col_a'] > 5) & (df['col_b'] < 10)
df['vectorized_result'] = np.where(condition, df['col_c'] * 2, df['col_c'])
vectorized_time = time.time() - start_time

# 清理示例列
df = df.drop(columns=['apply_result', 'vectorized_result'])

print(f"使用.apply(axis=1)的时间: {apply_time:.4f} 秒")
print(f"使用向量化np.where的时间: {vectorized_time:.4f} 秒")
```



![Pandas操作时间比较](plots/2267-0.json)



> 对比在一个包含100万行的DataFrame上，按行`.apply`操作与向量化 (quantization)`np.where`操作的执行时间。注意Y轴上的对数刻度，这突出了巨大的性能差异。

结果一致表明，向量化操作的性能比按行`.apply`高出几个数量级，特别是当数据集大小增加时。

### 总结

大型数据集的Pandas高效用法在于摆脱默认行为和Python级循环。通过仔细选择数据类型、善用`category`类型、避免按行操作转而采用向量 (vector)化，并在必要时分块处理数据，您可以大幅提升机器学习 (machine learning)工作流中数据处理任务的内存效率和计算速度。这些做法在构建可扩展且高性能的机器学习应用中极为重要。

## 参考资料

- [Enhancing Performance](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancing_performance.html) — Pandas Development Team (2024)
  官方指南，涵盖Pandas操作优化，包括数据类型选择和向量化计算。
- [Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) — Pandas Development Team (2024)
  详细解释Pandas分类数据类型及其在内存和速度方面的优势。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://learning.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由Pandas创建者撰写的著作，提供了数据结构和高效数据分析操作的见解。
- [Universal functions (ufunc)](https://numpy.org/doc/stable/reference/ufuncs.html) — NumPy Developers (2024)
  NumPy通用函数参考，它们是Python中高效向量化数组计算的基础。
