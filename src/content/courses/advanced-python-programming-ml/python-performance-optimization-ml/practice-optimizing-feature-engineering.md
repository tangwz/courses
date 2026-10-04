---
course: "advanced-python-programming-ml"
chapter: "python-performance-optimization-ml"
lesson: "practice-optimizing-feature-engineering"
sourceId: 2291
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/practice-optimizing-feature-engineering"
title: "动手实践：优化特征工程函数"
description: "分析并优化一个计算密集型特征工程函数，使用多种Python技术。"
order: 8
plots: ["plots/2291-0.json"]
sourceHash: "8674a1133303482e2f943d1cdd40b2d9363d0a7c3aa8239dd7e5148d10f35a93"
sourceCorrections: []
---

特征工程通常需要对数据应用自定义转换，而这些步骤可能成为重要的性能瓶颈，尤其是在处理大型数据集时。为了解决这个问题，将对一个典型的特征工程函数进行性能分析以找出效率低下的地方。随后，将应用向量 (vector)化和Numba等优化技术来加速它。

## 场景：计算成对特征交互

设想我们需要根据数据集中现有数值特征之间的成对交互来创建新特征。一种常见的交互特征是两个特征的乘积。对于具有特征 $f_1, f_2, ..., f_n$ 的数据集，我们希望计算所有 $i < j$ 对的 $f_i * f_j$。

我们先从一个使用 Pandas 和 NumPy 模拟的适中大小的数据集开始。

```python
import pandas as pd
import numpy as np
import time
import numba

# 生成样本数据
num_rows = 100000
num_features = 10
data = pd.DataFrame(np.random.rand(num_rows, num_features),
                    columns=[f'feature_{i}' for i in range(num_features)])

print("样本数据头部:")
print(data.head())
print(f"\n数据形状: {data.shape}")
```

## 基准实现：使用嵌套循环

实现成对交互计算的一种直接方式是使用嵌套循环遍历特征列。这种方法易于理解，但在 Python 中通常表现不佳，尤其是在 Pandas 等为向量 (vector)化操作优化的库中。

```python
def calculate_interactions_loops(df):
    """使用嵌套循环计算成对特征交互。"""
    feature_names = df.columns
    num_features = len(feature_names)
    interaction_data = {}

    for i in range(num_features):
        for j in range(i + 1, num_features):
            col1_name = feature_names[i]
            col2_name = feature_names[j]
            interaction_col_name = f'{col1_name}_x_{col2_name}'
            # 对该对特征进行逐元素乘法
            interaction_data[interaction_col_name] = df[col1_name] * df[col2_name]

    return pd.DataFrame(interaction_data)

# --- 循环实现的基准测试 ---
start_time = time.time()
interactions_loops = calculate_interactions_loops(data.copy()) # 使用副本以避免修改原始数据
end_time = time.time()
loop_time = end_time - start_time

print(f"\n--- 基准：嵌套循环 ---")
print(f"交互特征的形状: {interactions_loops.shape}")
print(f"执行时间: {loop_time:.4f} 秒")
print("交互特征示例:")
print(interactions_loops.head())

# 预期交互特征数量: nC2 = n * (n-1) / 2
# 对于 10 个特征: 10 * 9 / 2 = 45
assert interactions_loops.shape[1] == (num_features * (num_features - 1)) // 2
```

运行这段代码很可能会显示出明显的执行时间，即使对于 100,000 行和 10 个特征的数据也是如此。速度慢的原因是显式地迭代列并在 Python 循环中重复执行 Pandas Series 乘法。

## 基准分析

在优化之前，我们先确认时间都花在哪里。我们可以使用 IPython/Jupyter 中的 `%prun` 或 `cProfile` 模块。

```python
import cProfile
import pstats

# 对基于循环的函数进行性能分析
profiler = cProfile.Profile()
profiler.enable()
_ = calculate_interactions_loops(data.copy()) # 在性能分析器下运行函数
profiler.disable()

# 打印统计信息，按累积时间排序
stats = pstats.Stats(profiler).sort_stats('cumulative')
print("\n--- 性能分析结果（累积时间前 10 名）---")
stats.print_stats(10)
```

性能分析输出很可能会表明，大部分时间都花费在循环内部重复调用的 Pandas `__mul__`（乘法）方法中，以及 DataFrame/Series 的索引操作。这确认了 Python 循环中重复的逐元素操作是优化的主要目标。

## 优化 1：使用 NumPy/Pandas 进行向量 (vector)化

Pandas 和 NumPy 擅长向量化 (quantization)操作，它们在 C 级别一次性对整个数组执行计算，从而避免了 Python 循环的开销。我们可以调整交互计算的实现方式来发挥这一优势。一种方法是使用 `itertools.combinations` 获取列名的配对，然后执行乘法。

```python
from itertools import combinations

def calculate_interactions_vectorized(df):
    """使用向量化操作计算成对特征交互。"""
    feature_names = df.columns
    interaction_data = {}

    # 获取所有两个特征名的组合
    for col1_name, col2_name in combinations(feature_names, 2):
        interaction_col_name = f'{col1_name}_x_{col2_name}'
        # 对整列进行向量化乘法
        interaction_data[interaction_col_name] = df[col1_name] * df[col2_name]

    return pd.DataFrame(interaction_data)

# --- 向量化实现的基准测试 ---
start_time = time.time()
interactions_vectorized = calculate_interactions_vectorized(data.copy())
end_time = time.time()
vectorized_time = end_time - start_time

print(f"\n--- 优化 1：向量化 Pandas ---")
print(f"交互特征的形状: {interactions_vectorized.shape}")
print(f"执行时间: {vectorized_time:.4f} 秒")
print(f"相较循环的加速比: {loop_time / vectorized_time:.2f}倍")
# 结果校验（可选，比较一些值）
# pd.testing.assert_frame_equal(interactions_loops, interactions_vectorized) # 检查结果是否一致
```

你应该会发现速度有很大的提升。尽管我们仍然循环遍历列*对*，但耗时的逐元素乘法现在每对只对整列执行一次，并由 Pandas/NumPy 高效地完成。

## 优化 2：使用 Numba 进行 JIT 编译

有时，逻辑比较复杂，无法轻易地使用标准 NumPy/Pandas 函数进行向量 (vector)化。在这种情况下，Numba 可以通过即时 (JIT) 将 Python 代码编译为优化过的机器码来加速其运行，尤其适用于涉及循环和数值运算的代码。

为了让 Numba 有效，它最适用于主要操作 NumPy 数组并使用标准 Python 循环和数值类型的函数。让我们调整我们的交互计算，使其直接使用 DataFrame 的底层 NumPy 数组表示。

```python
@numba.njit
def calculate_interactions_numba_core(data_array):
    """Numba 优化的核心函数，用于成对交互。"""
    num_rows, num_features = data_array.shape
    num_interactions = (num_features * (num_features - 1)) // 2

    # 为提高效率预分配输出数组
    interactions_out = np.empty((num_rows, num_interactions), dtype=data_array.dtype)

    interaction_idx = 0
    # Numba 内部的嵌套循环效率很高
    for i in range(num_features):
        for j in range(i + 1, num_features):
            # 直接从 NumPy 数组访问列
            col_i = data_array[:, i]
            col_j = data_array[:, j]
            # 执行逐元素乘法（Numba 内的 NumPy 操作）
            interactions_out[:, interaction_idx] = col_i * col_j
            interaction_idx += 1

    return interactions_out

def calculate_interactions_numba(df):
    """调用 Numba 优化核心的包装函数。"""
    feature_names = df.columns
    num_features = len(feature_names)

    # 获取 NumPy 数组表示
    data_array = df.to_numpy()

    # 调用 JIT 编译的函数
    interactions_array = calculate_interactions_numba_core(data_array)

    # 为输出 DataFrame 生成列名
    interaction_col_names = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            interaction_col_names.append(f'{feature_names[i]}_x_{feature_names[j]}')

    return pd.DataFrame(interactions_array, columns=interaction_col_names, index=df.index)

# --- Numba 实现的基准测试 ---
# 首次运行可能包含编译时间
_ = calculate_interactions_numba(data.copy()) 

# 第二次运行计时（编译后）
start_time = time.time()
interactions_numba = calculate_interactions_numba(data.copy())
end_time = time.time()
numba_time = end_time - start_time

print(f"\n--- 优化 2：Numba JIT ---")
print(f"交互特征的形状: {interactions_numba.shape}")
print(f"执行时间: {numba_time:.4f} 秒")
print(f"相较循环的加速比: {loop_time / numba_time:.2f}倍")
print(f"相较向量化的加速比: {vectorized_time / numba_time:.2f}倍")
# 结果校验（可选，可能存在微小的浮点差异）
# np.allclose(interactions_vectorized.to_numpy(), interactions_numba.to_numpy())
```

Numba 通常能为难以纯粹使用 NumPy/Pandas 进行向量化 (quantization)的、包含大量循环的数值代码提供显著的加速。请注意，我们对一个直接操作 NumPy 数组的核心函数应用了 `@njit` 装饰器（这意味着 `nopython=True`，强制编译而不回退到较慢的对象模式）。包装函数处理了 Pandas DataFrames 之间的转换。与向量化 Pandas 方法相比，这里的性能提升可能因操作的复杂性和数据大小而异，但对于这种显式循环模式，它通常会超过纯 Pandas 的性能。

## 内存考量

在优化速度的同时，也要考虑内存使用。

- **基准：** 在乘法过程中创建中间 Series。如果特征很多，内存使用量可能会急剧增加。
- **向量 (vector)化：** 类似于基准，为每对特征创建中间 Series。
- **Numba：** 通过预分配 `interactions_out` NumPy 数组 (`np.empty`)，我们可以更直接地控制内存分配。与重复创建 Pandas Series 相比，使用 NumPy 数组有时可以更高效地利用内存，尤其是在仔细管理数据类型的情况下（例如，如果精度允许，使用 `np.float32`）。

`memory_profiler` 等工具可以逐行分析内存消耗，如果内存成为瓶颈：

```python
# memory_profiler 使用示例（需要安装：pip install memory_profiler）
# from memory_profiler import profile
#
# @profile # 将此装饰器添加到您想分析的函数上
# def calculate_interactions_vectorized_mem(df):
#     # ...（与 calculate_interactions_vectorized 实现相同）...
#     pass 
#
# # 运行函数以获取内存分析输出
# if __name__ == '__main__': # 在某些系统上，memory_profiler 需要此行
#     # interactions_vectorized_mem = calculate_interactions_vectorized_mem(data.copy()) 
#     pass # 实际运行的占位符
```

## 结果总结

我们来可视化您可能会观察到的典型性能差异。



![特征工程函数执行时间](plots/2291-0.json)



> 不同实现方式的执行时间比较（对数尺度）。实际时间取决于硬件和数据大小，但相对差异具有示意性。

## 讨论与要点

本次实践练习展示了优化 Python 机器学习 (machine learning)代码的常见工作流程：

1. **设定基准：** 首先编写一个清晰、正确的版本，即使它运行缓慢。
2. **分析性能：** 使用性能分析工具（`cProfile`、`line_profiler`、`memory_profiler`）找出实际的瓶颈。不要猜测。
3. **迭代优化：** 根据性能分析结果，应用有针对性的优化。
   - **向量 (vector)化：** 尽可能优先使用 NumPy/Pandas 的向量化 (quantization)操作。这通常是实现数组/DataFrame 操作良好性能的最“Pythonic”且可读性高的方式。
   - **Numba：** 对于难以直接向量化的、涉及循环的复杂数值算法，使用 Numba。它需要更接近地操作 NumPy 数组，但可以提供显著的加速。
   - **Cython（提及）：** 对于需要更多控制或与 C 库交互的情况，Cython（前面已介绍）提供静态类型和编译为 C 扩展的功能。它通常比 Numba 涉及更多的代码修改。
4. **基准测试：** 始终衡量您所做更改对性能的影响。
5. **考虑权衡：** 优化有时会降低可读性（例如，复杂的向量化）或增加依赖（Numba/Cython）。选择能够为所引入的精力投入和复杂性提供足够性能提升的技术。

优化特征工程通常很重要，因为它在开发过程中以及可能在大型生产数据集上重复应用。通过掌握这些 Python 性能技术，您可以构建更快、更具扩展性的机器学习管道。

## 参考资料

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — The NumPy community (2024)
  涵盖数组编程和广播，这是通过Python中的向量化操作实现高性能的基础。
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The pandas development team (2024)
  提供DataFrame和Series中高效数据操作的全面指导，包括向量化操作的最佳实践。
- [Numba User Manual](https://numba.readthedocs.io/en/stable/user/index.html) — The Numba developers (2024)
  Numba的官方文档，解释如何使用JIT编译来加速数值型Python函数，特别是包含循环的函数。
- [High Performance Python: Practical Performant Programming for Humans](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) — Micha Gorelick and Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  提供编写更快Python代码的实用策略，涵盖性能分析、NumPy和pandas的有效使用以及Numba优化。
- [The Python Profilers](https://docs.python.org/3/library/profile.html) — Python Software Foundation (2024)
  Python内置性能分析模块（`cProfile`和`profile`）的官方文档，它们对于识别性能瓶颈至关重要。
