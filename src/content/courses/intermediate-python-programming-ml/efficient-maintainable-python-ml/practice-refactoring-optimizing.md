---
course: "intermediate-python-programming-ml"
chapter: "efficient-maintainable-python-ml"
lesson: "practice-refactoring-optimizing"
sourceId: 2228
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/practice-refactoring-optimizing"
title: "实践：重构与优化机器学习代码片段"
description: "运用编码最佳实践，通过重构和优化用于机器学习任务的现有Python代码。"
order: 9
plots: []
sourceHash: "a199ec0ab926aae6ebf745634d60e5598eb8e35f7c37aa0e1cfdcd0c38fbe828"
sourceCorrections: []
---

理论很重要，但通过实践运用原则能巩固理解。提供实操练习，你将选取机器学习 (machine learning)工作流中常见的现有Python代码片段，并改进它们以提高可读性、效率和可维护性。

我们将注重找出可改进之处，应用重构技术，优化对性能要求高的部分，并更逻辑地组织代码。

## 练习1：提升可读性和结构

考虑以下旨在计算预测值与实际值之间均方误差（MSE）的函数。尽管功能上可行，但其清晰度和对标准做法的符合度仍可提升。

**原始代码：**

```python
# 计算MSE的函数
def calculate_error(d1, d2):
    if len(d1) != len(d2):
        print("错误：输入数组必须具有相同的长度。")
        return None
    err = 0
    for i in range(len(d1)):
        err += (d1[i] - d2[i])**2
    mse = err / len(d1)
    return mse

# 使用示例
predictions = [2.1, 3.4, 1.9, 5.0, 4.3]
actuals = [2.5, 3.0, 1.5, 5.5, 4.0]

result = calculate_error(predictions, actuals)
if result is not None:
    print(f"计算出的MSE: {result}")
```

**任务：**

1. **提升可读性：** 改进变量名（`d1`、`d2`、`err`）。添加类型提示和文档字符串，说明函数目的、参数 (parameter)和返回值。
2. **使用NumPy提升效率：** 将显式循环替换为NumPy的向量 (vector)化操作，用于计算平方差和均值。这通常对数值计算来说快得多。
3. **改进错误处理：** 对于无效输入，不要打印错误消息并返回`None`，而是抛出特定异常（如`ValueError`）。这更符合Python的惯用方式，并允许调用代码以编程方式处理错误。

**重构代码：**

```python
import numpy as np
from typing import List, Union

def mean_squared_error(y_true: Union[List[float], np.ndarray],
                       y_pred: Union[List[float], np.ndarray]) -> float:
    """
    计算真实值与预测值之间的均方误差（MSE）。

    参数：
        y_true: 真实目标值的类数组结构。
        y_pred: 预测值的类数组结构。

    返回：
        计算出的均方误差。

    抛出：
        ValueError: 如果输入数组长度不同。
    """
    # 将输入转换为NumPy数组，以实现高效计算和错误检查
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("输入数组必须具有相同的形状。")

    if y_true.ndim == 0: # 优雅地处理标量输入
         raise ValueError("输入必须是类数组，而非标量。")

    # 使用向量化操作计算MSE
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# 使用重构函数示例
predictions = np.array([2.1, 3.4, 1.9, 5.0, 4.3])
actuals = np.array([2.5, 3.0, 1.5, 5.5, 4.0])

try:
    result = mean_squared_error(actuals, predictions)
    print(f"计算出的MSE（重构后）: {result:.4f}")

    # 错误处理示例
    different_length_preds = np.array([1.0, 2.0])
    mean_squared_error(actuals, different_length_preds)

except ValueError as e:
    print(f"遇到错误: {e}")
```

**讨论：**

重构后的版本效率更高。

- **可读性：** 变量名（`y_true`、`y_pred`、`mse`）具有描述性。文档字符串清晰地说明了函数，类型提示提升了理解并支持静态分析。
- **效率：** 使用`np.array()`和`np.mean()`利用了NumPy优化的C语言实现，这在数值任务中显著优于Python循环，特别是对于大型数组。
- **可维护性：** 抛出`ValueError`提供了一种标准方式来指示无效输入，使函数更容易集成到需要捕获和处理错误的大型系统中。在开始时将输入转换为NumPy数组简化了主要逻辑。

## 练习2：使用Pandas优化数据处理

假设你有一个Pandas DataFrame，需要应用一个条件转换：根据'category'和'original\_price'计算'discounted\_price'。一个常见但通常效率低下的方法是逐行迭代。

**原始（低效）代码：**

```python
import pandas as pd
import time

# 示例DataFrame
data = {
    'product_id': range(10000),
    'category': ['Electronics', 'Clothing', 'Groceries', 'Books'] * 2500,
    'original_price': [100 * (i % 10 + 1) for i in range(10000)]
}
df = pd.DataFrame(data)

# 使用iterrows的低效方法
start_time = time.time()
discounted_prices = []
for index, row in df.iterrows():
    price = row['original_price']
    category = row['category']
    if category == 'Electronics':
        discount = 0.10 # 10% 折扣
    elif category == 'Clothing':
        discount = 0.15 # 15% 折扣
    else:
        discount = 0.05 # 5% 折扣
    discounted_prices.append(price * (1 - discount))

df['discounted_price'] = discounted_prices
end_time = time.time()

print(f"iterrows() 耗时: {end_time - start_time:.4f} 秒")
print(df.head())
# 为下一个示例清理添加的列
df = df.drop(columns=['discounted_price'])
```

**任务：**

1. **找出瓶颈：** 认识到`iterrows()`对于可以向量 (vector)化或更直接应用的操作通常较慢。
2. **应用向量化 (quantization)方法：** 使用Pandas的内置功能，如布尔索引或`apply()`方法（审慎使用），或者更好的方法是`np.select`，用于跨列的条件逻辑。

**使用`np.select`的优化代码：**

```python
import pandas as pd
import numpy as np
import time

# 示例DataFrame（同前）
data = {
    'product_id': range(10000),
    'category': ['Electronics', 'Clothing', 'Groceries', 'Books'] * 2500,
    'original_price': [100 * (i % 10 + 1) for i in range(10000)]
}
df = pd.DataFrame(data)

# 使用np.select进行条件逻辑的优化方法
start_time = time.time()

conditions = [
    df['category'] == 'Electronics',
    df['category'] == 'Clothing'
]
# 对应的折扣率（作为乘数，1 - 折扣）
choices = [
    0.90, # 1 - 0.10
    0.85  # 1 - 0.15
]
# 默认折扣率乘数
default_choice = 0.95 # 1 - 0.05

discount_multiplier = np.select(conditions, choices, default=default_choice)
df['discounted_price'] = df['original_price'] * discount_multiplier

end_time = time.time()

print(f"np.select 耗时: {end_time - start_time:.4f} 秒")
print(df.head())
```

**讨论：**

- **性能：** `np.select`方法在底层使用高度优化的NumPy操作执行条件逻辑和乘法，并一次性应用于整个Series。这避免了`iterrows()`逐行操作的开销，从而带来了显著的速度提升，特别是对于大型DataFrame。你通常会观察到向量化方法的速度快几个数量级。
- **可读性：** 尽管对于初学者来说，`np.select`方法可能比循环稍微不那么直接，但它清晰地表达了条件和对应的结果。它将逻辑（条件、选择）与应用分离。
- **Pandas最佳实践：** 避免显式逐行迭代是编写高效Pandas代码的根本原则。推荐使用`np.select`、布尔掩码和`map`（用于简单替换）等函数。虽然可以使用`apply()`，但在适用时它通常比`np.select`等完全向量化方法慢。

## 练习3：为模块化进行重构

考虑一个在主脚本体内部按顺序执行多个数据准备步骤的脚本。

**原始代码片段：**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv('some_data.csv') # 假设此文件包含数值特征和目标

# --- 步骤1：处理缺失值 ---
# 对数值列进行简单的均值填充
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'target' in numeric_cols: numeric_cols.remove('target') # 排除目标
for col in numeric_cols:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
print("缺失值已处理。")

# --- 步骤2：特征缩放 ---
scaler = StandardScaler()
# 如果目标变量存在，避免对其进行缩放
features_to_scale = [col for col in numeric_cols if col != 'target']
if features_to_scale: # 检查是否有特征需要缩放
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("特征已缩放。")

# --- 步骤3：数据划分 ---
X = df[features_to_scale]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据划分完成。")

# ... 后续代码使用X_train, X_test, y_train, y_test ...
print(f"训练集形状: {X_train.shape}")
```

**任务：**

1. **识别逻辑块：** 识别不同的数据处理步骤（加载、填充、缩放、划分）。
2. **封装为函数：** 将每个逻辑块移入各自的函数中。这能提升组织性，使代码可重用，并更容易单独测试。
3. **改进主脚本流程：** 脚本的主体现在应主要调度对这些函数的调用，使整体工作流程更清晰。

**重构代码：**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_data(filepath: str) -> pd.DataFrame:
    """从CSV文件加载数据。"""
    try:
        df = pd.read_csv(filepath)
        print(f"数据从 {filepath} 成功加载")
        return df
    except FileNotFoundError:
        print(f"错误：在 {filepath} 未找到文件")
        # 根据上下文，可能会抛出错误或返回空DataFrame
        raise

def impute_missing_numeric(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """使用均值填充数值列中的缺失值。"""
    df_copy = df.copy() # 在副本上操作，以避免意外修改原始DataFrame
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    cols_to_impute = [col for col in numeric_cols if col not in exclude_cols]

    for col in cols_to_impute:
        if df_copy[col].isnull().any():
            mean_val = df_copy[col].mean()
            df_copy[col].fillna(mean_val, inplace=True)
            print(f"已使用均值 {mean_val:.2f} 填充列 '{col}' 中的缺失值")
    return df_copy

def scale_features(df: pd.DataFrame, cols_to_scale: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """使用StandardScaler对指定的数值特征进行缩放。"""
    df_copy = df.copy()
    scaler = StandardScaler()
    if cols_to_scale: # 仅当提供了列时才进行缩放
        df_copy[cols_to_scale] = scaler.fit_transform(df_copy[cols_to_scale])
        print(f"已缩放列: {', '.join(cols_to_scale)}")
    else:
        print("未指定要缩放的列。")
    # 返回缩放器，以便稍后在测试数据上使用
    return df_copy, scaler

def split_data(df: pd.DataFrame, feature_cols: List[str], target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """将数据划分为训练集和测试集。"""
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"数据已划分为训练集（{X_train.shape[0]}个样本）和测试集（{X_test.shape[0]}个样本）")
    return X_train, X_test, y_train, y_test

# --- 主脚本工作流程 ---
if __name__ == "__main__": # 将脚本逻辑放在此处是好的实践
    try:
        FILE_PATH = 'some_data.csv' # 定义常量
        TARGET_COLUMN = 'target'

        # 如果文件不存在，则创建虚拟数据以供演示
        try:
            pd.read_csv(FILE_PATH)
        except FileNotFoundError:
             print(f"未找到“{FILE_PATH}”。正在创建虚拟数据。")
             dummy_data = pd.DataFrame({
                 'feature1': np.random.rand(100) * 10,
                 'feature2': np.random.rand(100) * 5 + np.random.choice([np.nan, 1], size=100, p=[0.1, 0.9]), # 添加一些NaN
                 'feature3': np.random.randint(0, 5, 100),
                 TARGET_COLUMN: np.random.randint(0, 2, 100)
             })
             dummy_data.to_csv(FILE_PATH, index=False)

        raw_df = load_data(FILE_PATH)

        # 动态识别特征列（排除目标）
        all_numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        feature_columns = [col for col in all_numeric_cols if col != TARGET_COLUMN]

        imputed_df = impute_missing_numeric(raw_df, exclude_cols=[TARGET_COLUMN])
        scaled_df, fitted_scaler = scale_features(imputed_df, feature_columns)

        # 注意：对于真实场景，你需要保存“fitted_scaler”，并稍后在新数据/测试数据上使用scaler.transform()。

        X_train, X_test, y_train, y_test = split_data(scaled_df, feature_columns, TARGET_COLUMN)

        print(f"\n训练特征形状: {X_train.shape}")
        print(f"测试特征形状: {X_test.shape}")
        # ... 使用这些集合进行模型训练 ...

    except FileNotFoundError:
        print("执行中止：需要输入数据文件。")
    except Exception as e:
        print(f"发生意外错误: {e}")
```

**讨论：**

- **模块化与可重用性：** 每个函数现在执行一个单一、明确的任务。`impute_missing_numeric`或`scale_features`有可能在项目的其他脚本或部分中重复使用。
- **可读性：** `if __name__ == "__main__":` 下的主脚本流程更容易理解。它读起来就像数据准备过程的高级描述。
- **可测试性：** 诸如`impute_missing_numeric`之类的单个函数可以针对特定输入进行单元测试，以验证其正确性，独立于脚本的其余部分。
- **可维护性：** 如果填充逻辑需要更改，你只需修改`impute_missing_numeric`函数。更改是局部的，不太可能破坏代码的其他部分。返回`fitted_scaler`对于后续正确处理测试数据而避免数据泄露也很重要。

这些练习表明，运用本章的原则，侧重于可读性、利用高效的库函数以及逻辑地组织代码，能为机器学习 (machine learning)项目带来明显更好的Python代码。定期练习在你的工作中找出重构和优化的机会。

## 参考资料

- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/) — Guido van Rossum, Barry Warsaw, Alyssa Coghlan (2001)
  提供编写可读、可维护 Python 代码的基本指南，包括命名约定和代码布局。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — The NumPy Community (2023)
  NumPy 官方指南，对于理解和应用矢量化操作以实现高效数值计算很有用。
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The pandas development team (2023)
  pandas 官方资源，涵盖高效数据处理技术，例如矢量化操作和条件选择。
- [Refactoring: Improving the Design of Existing Code](https://www.informit.com/store/refactoring-improving-design-of-existing-code-9780134757599) — Martin Fowler, Kent Beck, John Brant, William Opdyke, Don Roberts (2018)
  Publisher: Addison-Wesley Professional; Pages: 448
  关于重构代码以提升设计、可读性和可维护性的基本书籍。
