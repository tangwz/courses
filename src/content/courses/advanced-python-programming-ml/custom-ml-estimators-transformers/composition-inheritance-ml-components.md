---
course: "advanced-python-programming-ml"
chapter: "custom-ml-estimators-transformers"
lesson: "composition-inheritance-ml-components"
sourceId: 2371
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-6-custom-ml-estimators-transformers/composition-inheritance-ml-components"
title: "机器学习组件的组合与继承"
description: "构建复杂ML组件时，有效应用组合和继承等面向对象原则。"
order: 4
plots: []
sourceHash: "2decc713f82ddcdf662a94737f25176ef84a1fa3f202185bd273c268f4539d69"
sourceCorrections: []
---

在创建自定义估计器和变换器时，理解如何利用继承和组合等基本的面向对象编程（OOP）原则来组织代码，对于构建可维护、可复用和可扩展的机器学习 (machine learning)组件十分有益。Scikit-learn自身广泛应用了这些模式，使你的自定义代码与它们保持一致，能确保良好集成。

### 继承：行为特化与API遵循

继承建立“是一种”的关系。一个类继承自另一个类（父类或基类）时，会获得父类的属性和方法。在Scikit-learn中，继承主要用于两个方面：

1. **API兼容性**：为了被Scikit-learn工具（如`Pipeline`、`GridSearchCV`、`cross_val_score`）识别为有效的估计器或变换器，你的自定义类*必须*继承自特定的基类。

   - **`sklearn.base.BaseEstimator`**：这是所有估计器的基本基类。继承它会提供`get_params`和`set_params`的默认实现，这对于模型检查、克隆和超参数 (parameter) (hyperparameter)调优非常重要。你的`__init__`方法接受的任何参数都应作为具有相同名称的属性，并且不应带有前导下划线，除非它旨在只读。
   - **`sklearn.base.TransformerMixin`**：如果你正在构建一个变换器（具有`fit`和`transform`方法的组件），继承`TransformerMixin`会自动提供`fit_transform`方法，通过调用`fit`然后`transform`有效实现。
   - **`sklearn.base.RegressorMixin`**、**`sklearn.base.ClassifierMixin`**、**`sklearn.base.ClusterMixin`**：继承这些类分别提供适合回归、分类或聚类任务的默认`score`方法。
2. **行为特化**：你可能会使用继承来创建现有Scikit-learn组件的特化版本，甚至是你自己的自定义组件的特化版本。例如，你可以创建一个继承自`OutlierHandler`基类的`WinsorizingScaler`，添加特定的Winsorizing逻辑。

```python
# 示例：使用继承实现API兼容性的基本结构
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomLogTransformer(BaseEstimator, TransformerMixin):
    """一个应用log(1+x)的简单变换器。"""
    def __init__(self):
        # 这个简单变换器不需要参数
        pass

    def fit(self, X, y=None):
        # 这个变换器不需要从数据中学习任何东西，
        # 所以fit方法只返回self。
        # 输入验证可以在这里添加。
        print("Fitting CustomLogTransformer")
        return self

    def transform(self, X, y=None):
        # 应用变换
        # 输入验证可以在这里添加。
        print("Transforming with CustomLogTransformer")
        Xt = np.log1p(X)
        return Xt

    # get_params和set_params继承自BaseEstimator
    # fit_transform继承自TransformerMixin
```

虽然继承对于API兼容性是必要的，并对直接特化有用，但过度依赖深层继承体系可能导致代码紧密耦合和脆弱。基类的更改可能在派生类中产生意想不到的后果。

### 组合：用简单部分构建复杂组件

组合建立“有一个”的关系。类不继承属性，而是持有其他类的实例，并将任务委托给它们。这种方法通常更灵活，并且是像`Pipeline`和`FeatureUnion`这样复杂的Scikit-learn对象构建的核心。

当你的自定义组件需要时，可以考虑组合方式：

1. **组合多个步骤**：你的变换器可能需要在内部应用几个不同的处理步骤。与其将所有逻辑放入一个大类中，你可以创建更小、更集中的组件（这些组件本身可能是Scikit-learn兼容的变换器或更简单的Python对象），并在你的主变换器中组合它们。
2. **包装现有估计器/变换器**：创建修改或增强其他组件行为的元估计器或元变换器。例如，一个自定义特征选择变换器可以*包含*一个用于确定特征重要性的估计器。
3. **管理内部状态或资源**：一个组件可能管理内部数据结构、模型或外部连接。组合允许将这些封装在主类中。

```python
# 示例：在变换器中使用组合
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

class PreprocessingWrapper(BaseEstimator, TransformerMixin):
    """应用填充然后进行缩放。"""
    def __init__(self, strategy='mean'):
        # 存储内部组件的配置
        self.strategy = strategy
        # 内部组件在fit方法中实例化，
        # 确保如果包装器被克隆，它们会被重置。

    def fit(self, X, y=None):
        print(f"正在拟合 PreprocessingWrapper (策略={self.strategy})")
        # 根据存储的参数在此处实例化内部组件
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.scaler_ = StandardScaler()

        # 顺序拟合组件
        Xt = self.imputer_.fit_transform(X)
        self.scaler_.fit(Xt)
        return self

    def transform(self, X):
        # 使用已拟合的组件应用变换
        print("正在使用 PreprocessingWrapper 进行变换")
        Xt = self.imputer_.transform(X)
        Xt = self.scaler_.transform(Xt)
        return Xt

    # 注意：BaseEstimator 处理'strategy'的get_params/set_params。
    # 带有尾部下划线的内部组件（imputer_、scaler_）
    # 是拟合后的属性，默认不被视为超参数。
```

在这个`PreprocessingWrapper`示例中，该包装器*有一个*`SimpleImputer`和*有一个*`StandardScaler`。它协调它们的`fit`和`transform`调用。这使得包装器的逻辑更清晰，并允许复用标准、经过良好测试的Scikit-learn组件。

### 继承与组合的选择

软件设计的普遍指导原则常被表述为“优先使用组合而不是继承”。这在构建ML组件时也适用：

- **使用继承** 主要用于：

  - 满足Scikit-learn API约定（`BaseEstimator`、`TransformerMixin`等）。
  - 创建组件真正的“是一种”的特化版本，你可以在其中覆盖或微调 (fine-tuning)现有方法。
- **使用组合** 主要用于：

  - 组合多个独立组件的功能。
  - 包装现有组件以添加预处理/后处理或修改行为（元估计器/变换器）。
  - 在单个组件内部构建复杂工作流。
  - 实现更高的灵活性和更简单的各部分测试。

> 比较自定义Scikit-learn对象中继承（用于API兼容性和特化）与组合（用于组合独立组件）的用法。

通过审慎地应用继承来遵循API和进行特化，并优先使用组合来通过结合简单部分构建复杂功能，你可以创建功能强大、灵活、可维护并能良好融入更广泛生态系统的自定义Scikit-learn组件。请记住，`BaseEstimator`对`get_params`和`set_params`的处理对于使组合组件与超参数 (parameter) (hyperparameter)调优工具正确配合使用很重要，因为它允许内省和修改嵌套对象的参数。

## 参考资料

- [Rolling your own estimator](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator) — scikit-learn developers (2024)
  关于如何实现符合 scikit-learn API 的自定义估计器和转换器的官方指南，涵盖了所需的基类和方法签名。
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — scikit-learn developers (2023)
  解释了 scikit-learn 如何通过管道和特征联合使用组合模式，提供了从简单组件构建复杂机器学习工作流的具体示例。
- [Design Patterns: Elements of Reusable Object-Oriented Software](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHXMRaAXgLExiDHmCd-biTAsfofcnMIqZhSMVvMsuHTxYeAkowDWY7vRsAvDwCUYQJkKuu9_DuK02joJr27dMiU0nBFz_mm0Dha4_QV7Zqx1OdLtxY-e5Tp8oDu8XcZXRYkiJCXxv2SGmFXDDTV2YnhdamzI-cxYirXQHMgwVrAgccV9uIAH-ol) — Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides (1994)
  Publisher: Addison-Wesley
  一部开创性的著作，介绍了基本的面向对象设计模式，包括继承和组合的原则以及推荐优先使用组合。
- [Fluent Python: Clear, Concise, and Effective Programming, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056345/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  全面介绍了 Python 的数据模型和面向对象特性，为有效地构建 Python 代码提供了实用指导，这间接支持了继承和组合的原则。
