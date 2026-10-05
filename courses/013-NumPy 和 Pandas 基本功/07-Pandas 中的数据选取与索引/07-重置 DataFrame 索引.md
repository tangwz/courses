# 重置 DataFrame 索引

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas/resetting-dataframe-index)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Pandas DataFrame 经常使用自定义索引，这通常是通过使用 `.set_index()` 将一个或多个列提升为索引来建立的。这种方法对于使用 `.loc` 根据有意义的标签选择数据通常很有用。然而，有时您可能需要转换自定义的 DataFrame 索引。这可能涉及将索引标签转回为普通数据列，或许是为了数据处理，或者仅仅是为了恢复默认的整数索引 (0, 1, 2, ...)。Pandas 为此提供了一个直接的方法：`.reset_index()`。

### 使用 `reset_index` 还原索引

`.reset_index()` 方法实质上是 `.set_index()` 的逆操作。它将当前索引级别移入 DataFrame 作为新列。默认情况下，它还会用一个简单的默认整数索引替换现有索引。

我们来看看实际操作。首先，我们将创建一个示例 DataFrame 并将其一列设为索引：

```python
import pandas as pd

data = {'City': ['Austin', 'Dallas', 'Houston', 'San Antonio'],
        'State': ['TX', 'TX', 'TX', 'TX'],
        'Population': [961855, 1304379, 2304580, 1434625]}
df = pd.DataFrame(data)

# 将 'City' 设置为索引
df_indexed = df.set_index('City')

print("以 'City' 为索引的 DataFrame：")
print(df_indexed)
```

这将输出：

```
以 'City' 为索引的 DataFrame：
            State  Population
City                         
Austin         TX      961855
Dallas         TX     1304379
Houston        TX     2304580
San Antonio    TX     1434625
```

请注意，'City' 不再是普通列，而是作为左侧的索引标签。

现在，我们使用 `.reset_index()` 将 'City' 索引转回为列：

```python
# 重置索引
df_reset = df_indexed.reset_index()

print("\n重置索引后的 DataFrame：")
print(df_reset)
```

输出清楚地显示了变化：

```
重置索引后的 DataFrame：
          City State  Population
0       Austin    TX      961855
1       Dallas    TX     1304379
2      Houston    TX     2304580
3  San Antonio    TX     1434625
```

如您所见，`reset_index()` 执行了两个主要操作：

1. 它将索引 ('City') 移回 DataFrame 作为普通列。
2. 它创建了一个新的默认整数索引 (0, 1, 2, 3)。

### 控制重置操作

`.reset_index()` 方法提供参数 (parameter)来控制其行为，主要是 `drop` 和 `inplace`。

#### `drop` 参数

有时，您可能希望完全丢弃旧索引，而不是将其转为列。如果索引值在重置后是多余的或不再需要，这会很有用。您可以使用 `drop=True` 参数实现这一点。

```python
# 重置索引并丢弃旧索引 ('City')
df_dropped_index = df_indexed.reset_index(drop=True)

print("\n重置并丢弃索引后的 DataFrame：")
print(df_dropped_index)
```

输出：

```
重置并丢弃索引后的 DataFrame：
  State  Population
0    TX      961855
1    TX     1304379
2    TX     2304580
3    TX     1434625
```

这里，'City' 列已消失，我们只剩下 'State' 和 'Population' 列以及新的默认整数索引。如前所述，默认行为对应于 `drop=False`。

#### `inplace` 参数

与许多 Pandas 方法一样，`.reset_index()` 也有一个 `inplace` 参数。

- `inplace=False` (默认值)：此方法返回一个索引已重置的*新* DataFrame，不改变原始 DataFrame。您通常将此新 DataFrame 赋值给一个变量（如我们对 `df_reset` 和 `df_dropped_index` 所做的那样）。
- `inplace=True`：此方法直接修改*原始* DataFrame 并返回 `None`。不会创建新的 DataFrame。

使用 `inplace=True` 可以为非常大的 DataFrame 节省内存，因为它避免了创建副本，但请谨慎使用，因为它会直接修改您的数据。

```python
# 创建一个副本以便原地修改
df_indexed_copy = df_indexed.copy()

print("\n原地重置前的原始 DataFrame：")
print(df_indexed_copy)

# 原地重置索引
result = df_indexed_copy.reset_index(inplace=True)

print("\n原地重置后的 DataFrame：")
print(df_indexed_copy)
print("\n原地重置的返回值：", result)
```

输出：

```
原地重置前的原始 DataFrame：
            State  Population
City
Austin         TX      961855
Dallas         TX     1304379
Houston        TX     2304580
San Antonio    TX     1434625

原地重置后的 DataFrame：
          City State  Population
0       Austin    TX      961855
1       Dallas    TX     1304379
2      Houston    TX     2304580
3  San Antonio    TX     1434625

原地重置的返回值： None
```

正如预期，`df_indexed_copy` 被直接修改，并且该方法返回 `None`。

### 何时重置索引

在数据清理和准备过程中，重置索引是常见的操作。一些常见情况包括：

- **过滤后：** 当您从 DataFrame 过滤行时，剩余的索引值可能会变得不连续（例如，0, 2, 5, 8）。重置索引可获得一个整齐的连续索引。
- **准备合并/连接：** 某些组合操作在默认整数索引下运行更顺畅或更符合预期，特别是当原始索引没有有意义的对齐 (alignment)时。
- **将索引值用作数据：** 您可能需要将之前存储在索引中的信息作为普通列用于计算、绘图或进一步分析。
- **简化 MultiIndex：** 如果您有一个带有分层索引 (MultiIndex) 的 DataFrame，`reset_index()` 可以将索引的一个或多个级别平展回列中，从而简化 DataFrame 结构。

了解如何设置和重置索引为在 Pandas DataFrame 中访问和组织数据提供了很大的灵活性。熟练使用 `.set_index()` 和 `.reset_index()` 可让您为特定的数据处理或分析任务选择最方便的数据表示形式。

## 参考资料

- [pandas.DataFrame.reset_index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html) — pandas development team (2024)
  官方文档提供了 `reset_index()` 方法的全面指南，详细说明了其参数、返回值和各种示例。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由 Pandas 创始人撰写的奠基性书籍，详细阐述了数据结构和操作，包括索引及其操作方法。

---

[上一节](06-%E8%AE%BE%E7%BD%AE%20DataFrame%20%E7%B4%A2%E5%BC%95.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BF%E9%97%AE%E7%89%B9%E5%AE%9A%E6%95%B0%E6%8D%AE%E5%AD%90%E9%9B%86.md)
