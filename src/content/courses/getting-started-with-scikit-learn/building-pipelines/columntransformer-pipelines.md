---
course: "getting-started-with-scikit-learn"
chapter: "building-pipelines"
lesson: "columntransformer-pipelines"
sourceId: 1902
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-6-building-pipelines/columntransformer-pipelines"
title: "使用 ColumnTransformer 构建复杂管道"
description: "使用 `ColumnTransformer` 对不同列应用不同的预处理步骤。"
order: 6
plots: []
sourceHash: "8dd16d74749fd8a74d6de61e1ce7eaf463e5efabf86a9db62f6c44db745e7502"
sourceCorrections: []
---

在机器学习 (machine learning)中，将多个预处理和建模步骤按顺序连接起来是常见的做法。数据集通常包含不同类型的特征，例如数值型、类别型和文本型，每种特征都需要独特的预处理技术。例如，将单一的缩放器应用于数值型和类别型列通常是不合适的。当转换按顺序应用时，每个步骤通常会处理前一个操作的整个输出。这种方法不适用于对原始特征的不同子集应用不同的转换。

这时，Scikit-learn 的 `ColumnTransformer` 就发挥作用了。它允许您*并行地*将不同的转换器应用于输入数据的不同列。应用每个转换器得到的结果会水平拼接，形成最终的转换后数据集，然后该数据集可以传递给更大管道中的下一步，通常是一个估计器。

### ColumnTransformer 的工作原理

主要思路是指定哪些转换器应该应用于哪些列。您需要向 `ColumnTransformer` 提供一个元组列表，每个元组包含：

1. **名称：** 一个唯一的字符串，用于标识转换步骤（例如，`'numerical_scaling'`，`'categorical_encoding'`）。
2. **转换器：** 一个已实例化的 Scikit-learn 转换器（例如，`StandardScaler()`，`OneHotEncoder()`），甚至可以是 `Pipeline` 本身。
3. **列：** 指定转换器应应用于哪些列。这可以是一个列名列表、列索引列表、一个切片、一个布尔掩码，或者通常是 `make_column_selector` 的输出。

让我们通过一个例子来说明。假设一个数据集包含数值特征（年龄、收入）和一个类别特征（城市）。我们希望对数值特征进行缩放，并对类别特征进行独热编码。

```python
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 示例数据
data = {
    'age': [25, 45, 30, 55, 40, 28],
    'income': [50000, 80000, 60000, 120000, 75000, 52000],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris'],
    'target': [0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df[['age', 'income', 'city']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 为不同列类型定义预处理步骤
# 为方便起见，使用 make_column_selector
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=object))
    ],
    remainder='passthrough' # 保留其他列（如果有）
)

# 创建包含预处理器和分类器的完整管道
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 拟合管道
model_pipeline.fit(X_train, y_train)

# 进行预测（预处理会自动应用）
predictions = model_pipeline.predict(X_test)

print("示例预测结果:", predictions)
print("管道得分:", model_pipeline.score(X_test, y_test))

# 您可以查看 ColumnTransformer 内部已拟合的转换器
# 访问预处理步骤
ct = model_pipeline.named_steps['preprocessor']
# 访问拟合在数值列上的缩放器
scaler = ct.named_transformers_['num']
print("\n拟合缩放器的均值:", scaler.mean_)
# 访问拟合在类别列上的独热编码器
encoder = ct.named_transformers_['cat']
print("拟合编码器的类别:", encoder.categories_)
```

在此示例中：

1. 我们使用 `make_column_selector(dtype_include=np.number)` 选择所有数值数据类型的列（`age`，`income`）。`StandardScaler` 应用于这些列。
2. 我们使用 `make_column_selector(dtype_include=object)` 选择对象数据类型的列（`city`），在 Pandas 中通常是字符串。`OneHotEncoder` 应用于此列。`handle_unknown='ignore'` 是一个有用的参数 (parameter)，如果编码器在测试集中遇到训练期间未曾见过的类别，它会避免错误发生；它只是将其编码为全零。
3. `ColumnTransformer` 将这些转换器应用于各自的列。
4. 输出结果（缩放后的数值列和编码后的类别列）会并排拼接。
5. 然后，这些组合的、经过预处理的数据被送入主 `Pipeline` 中的 `LogisticRegression` 分类器。

### 处理剩余列：`remainder` 参数 (parameter)

在 `transformers` 列表中，那些未被任何转换器选中的列会发生什么？这由 `ColumnTransformer` 的 `remainder` 参数控制：

- `remainder='drop'`（默认）：任何未被任何转换器明确选择的列都会从数据集中删除。使用此默认设置时请注意，因为您可能会无意中丢失特征。
- `remainder='passthrough'`：任何未被选择的列都会被保留并附加到指定转换的输出中。它们的值保持不变。如果您有不需要预处理或由后续步骤处理的列，这会很有用。
- `remainder=<估计器>`：您还可以指定一个转换器（如 `SimpleImputer` 或 `StandardScaler`）应用于所有剩余列。

选择正确的 `remainder` 设置对于确保所有必要特征都得到正确处理并最终送达估计器非常重要。

### 指定列

除了 `make_column_selector` 之外，您还可以使用以下方式为 `ColumnTransformer` 中的每个转换器指定列：

- **列名列表：** `['age', 'income']`
- **整数索引列表：** `[0, 1]`
- **切片：** `slice(0, 2)`
- **布尔掩码：** `[True, True, False]`

与使用整数索引相比，使用 `make_column_selector` 通常更受青睐，因为它对列顺序的变化或增删的适应性更强，不易出错。

### 工作流程可视化

我们可以可视化结合 `ColumnTransformer` 和 `Pipeline` 创建的结构：

> 流程图显示了输入数据如何通过 ColumnTransformer（'预处理器'）进行分割。数值列进入 StandardScaler（'数值'），类别列进入 OneHotEncoder（'类别'）。它们的输出被拼接，然后传递给 LogisticRegression 分类器（'分类器'）。

### ColumnTransformer 与网格搜索

在 `Pipeline` 中使用 `ColumnTransformer` 可以与 `GridSearchCV` 协同工作。您可以调整 `ColumnTransformer` 内转换器和最终估计器的超参数 (parameter) (hyperparameter)。参数名称通过步骤名用双下划线 (`__`) 分隔来构建。

例如，要调整 `ColumnTransformer`（名为 `'preprocessor'`）内部 `OneHotEncoder`（名为 `'cat'`）的 `handle_unknown` 参数，参数网格键将是 `'preprocessor__cat__handle_unknown'`。要调整 `LogisticRegression`（名为 `'classifier'`）的 `C` 参数，键是 `'classifier__C'`。

```python
# 上述管道的 GridSearchCV 示例参数网格
param_grid = {
    'preprocessor__num__with_mean': [True, False], # StandardScaler 的参数
    'preprocessor__cat__handle_unknown': ['ignore', 'error'], # OneHotEncoder 的参数
    'classifier__C': [0.1, 1.0, 10.0] # LogisticRegression 的参数
}

# 然后 GridSearchCV 将使用此 'param_grid' 应用于 'model_pipeline'
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(model_pipeline, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
```

`ColumnTransformer` 是 Scikit-learn 中构建实际机器学习 (machine learning)管道的基本工具，尤其是在处理常见的多样化数据集时。它允许简洁、模块化且正确地应用根据特定特征类型量身定制的预处理步骤，同时与 Scikit-learn 的模型评估和选择体系很好地结合。

## 参考资料

- [Column Transformer with Mixed Types (User Guide)](https://scikit-learn.org/stable/modules/compose.html#column-transformer) — Scikit-learn developers (2023)
  Scikit-learn官方用户指南中关于ColumnTransformer的章节，介绍了其目的、参数和用法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本关于构建机器学习管道的实践指导书，详细讨论了使用ColumnTransformer进行异构数据预处理。
- [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html) — Fabian Pedregosa, GaÃ«l Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Courn­apeau, Matthieu Brucher, Matthieu Perrot, Ã‰douard Duchesnay (2011)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 12; Pages: 2825−2830; DOI: [10.5555/1953048.2078195](https://doi.org/10.5555/1953048.2078195)
  介绍Scikit-learn库的学术论文，详细阐述了其设计原则和模块化架构，为ColumnTransformer等工具提供了基础。
