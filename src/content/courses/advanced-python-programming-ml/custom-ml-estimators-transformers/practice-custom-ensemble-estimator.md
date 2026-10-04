---
course: "advanced-python-programming-ml"
chapter: "custom-ml-estimators-transformers"
lesson: "practice-custom-ensemble-estimator"
sourceId: 2382
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-6-custom-ml-estimators-transformers/practice-custom-ensemble-estimator"
title: "实战演练：构建自定义集成估计器"
description: "使用Python开发一个自定义集成估计器，它能结合多个基础估计器并与Scikit-learn兼容。"
order: 8
plots: []
sourceHash: "b0507922620e2ad0c9e70e43ab5d04859ae1cd21a8823e00e711425a0351f508"
sourceCorrections: []
---

构建自定义堆叠集成估计器提供了一个实用而非简单的实例，以此了解如何创建符合Scikit-learn API的自定义组件。集成方法通常通过结合多个模型的输出来提升预测表现。虽然Scikit-learn提供了`StackingClassifier`和`StackingRegressor`，但亲自构建一个能帮助我们更好地理解估计器的组成方式和API要求。

我们的目标是创建一个`StackingEstimator`类，它接收一组基础估计器和一个最终的元学习器。在`fit`过程中，它会在输入数据上训练基础估计器，然后利用基础估计器生成的预测结果来训练元学习器。在`predict`过程中，它会结合来自基础估计器的预测结果，并将其输入到元学习器中以产生最终输出。

### 设计与Scikit-learn兼容性

为了与`Pipeline`和`GridSearchCV`等Scikit-learn工具顺利集成，我们的`StackingEstimator`必须遵循以下既定规范：

1. **继承：** 它应该继承自`BaseEstimator`和一个合适的混合类（例如，`ClassifierMixin`或`RegressorMixin`）。这提供了像`get_params`和`set_params`这样的必要方法。
2. **构造函数 (`__init__`)**：所有参数 (parameter)都必须是`__init__`中的显式关键字参数，并且这些参数不应在此处进行验证或修改。直接将未修改的参数作为公共属性存储（例如，`self.base_estimators = base_estimators`）。
3. **已拟合属性：** 在`fit`过程中学习到的属性（例如，已训练的基础模型和元学习器）应以一个下划线结尾（例如，`self.fitted_base_estimators_`）。
4. **`fit`方法：** 接受`X`、`y`并返回`self`。它执行主要的训练逻辑。
5. **`predict`方法：** 接受`X`并根据已拟合的模型返回预测结果。如果构建的是分类器，实现`predict_proba`通常是推荐的做法。

对于我们的堆叠估计器，主要参数将是`base_estimators`（估计器实例列表）和`meta_learner`（单个估计器实例）。

### 实现步骤

我们来开始构建`StackingEstimator`。为了说明目的，我们将重点放在分类器版本上，它继承自`ClassifierMixin`。

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# 生成元特征的辅助函数
def _generate_meta_features(estimators, X):
    """从已拟合估计器生成预测结果，作为元学习器的输入。"""
    # 检查估计器是否为非空列表
    if not isinstance(estimators, list) or len(estimators) == 0:
        raise ValueError("预期传入已拟合估计器列表。")

    # 收集预测结果。优先使用predict_proba，否则使用predict。
    predictions = []
    for name, estimator in estimators:
        try:
            # 如果可能，分类任务优先使用概率
            pred = estimator.predict_proba(X)
            # 处理predict_proba返回多列的情况（例如二分类）
            if pred.ndim > 1 and pred.shape[1] > 1:
                 # 使用正类的概率（常见约定）
                 # 或者如果元学习器可以处理，则使用所有概率。
                 # 为简单起见，这里我们取第二列，假设是二分类。
                 # 实际实现可能需要配置。
                 if pred.shape[1] == 2:
                     predictions.append(pred[:, 1].reshape(-1, 1))
                 else: # 多类别概率
                     predictions.append(pred) # 添加所有概率列
            else: # 单个概率向量或predict输出
                predictions.append(pred.reshape(-1, 1))
        except AttributeError:
            # 如果predict_proba不可用，则回退到predict
            pred = estimator.predict(X)
            predictions.append(pred.reshape(-1, 1))

    # 水平堆叠预测结果
    if not predictions:
         raise ValueError("没有从基础估计器生成预测结果。")

    return np.hstack(predictions)

class StackingEstimator(BaseEstimator, ClassifierMixin):
    """
    一个基本的堆叠集成分类器。

    训练基础估计器，并使用其预测结果
    作为最终元学习器的输入。

    参数
    ----------
    base_estimators : list of (str, estimator) tuples
        用于在数据上进行拟合的基础估计器。每个估计器
        在拟合前都会被克隆。

    meta_learner : estimator object
        用于在基础估计器预测结果上进行拟合的元学习器。
        在拟合前被克隆。

    属性
    ----------
    fitted_base_estimators_ : list of (str, estimator) tuples
        已拟合的基础估计器。

    fitted_meta_learner_ : estimator object
        已拟合的元学习器。

    classes_ : ndarray of shape (n_classes,)
        在拟合过程中观察到的类别标签。
    """
    def __init__(self, base_estimators, meta_learner):
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner

    def fit(self, X, y):
        """
        拟合堆叠估计器。

        在X、y上训练基础估计器，然后根据基础估计器的预测结果
        训练元学习器。

        参数
        ----------
        X : array-like of shape (n_samples, n_features)
            训练向量。
        y : array-like of shape (n_samples,)
            目标值。

        返回
        -------
        self : object
            返回实例本身。
        """
        # 验证输入数据
        X, y = check_X_y(X, y)

        # 存储在拟合过程中看到的类别
        self.classes_ = unique_labels(y)

        # 估计器的输入验证（基本检查）
        if not isinstance(self.base_estimators, list) or len(self.base_estimators) == 0:
            raise ValueError("`base_estimators`必须是非空的(名称, 估计器)元组列表。")
        if self.meta_learner is None:
             raise ValueError("`meta_learner`不能为None。")

        # 克隆估计器以避免修改原始对象
        self.fitted_base_estimators_ = []
        for name, estimator in self.base_estimators:
            fitted_estimator = clone(estimator).fit(X, y)
            self.fitted_base_estimators_.append((name, fitted_estimator))

        # 从基础估计器预测结果生成元特征
        X_meta = _generate_meta_features(self.fitted_base_estimators_, X)

        # 克隆并拟合元学习器
        self.fitted_meta_learner_ = clone(self.meta_learner).fit(X_meta, y)

        return self

    def predict(self, X):
        """
        预测X中样本的类别标签。

        参数
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。

        返回
        -------
        y_pred : ndarray of shape (n_samples,)
            预测的类别标签。
        """
        # 检查是否已调用fit方法
        check_is_fitted(self, ['fitted_base_estimators_', 'fitted_meta_learner_'])

        # 验证输入
        X = check_array(X)

        # 从基础估计器生成元特征
        X_meta = _generate_meta_features(self.fitted_base_estimators_, X)

        # 使用已拟合的元学习器进行预测
        return self.fitted_meta_learner_.predict(X_meta)

    def predict_proba(self, X):
        """
        预测X中样本的类别概率。

        参数
        ----------
        X : array-like of shape (n_samples, n_features)
            输入样本。

        返回
        -------
        p : ndarray of shape (n_samples, n_classes)
            输入样本的类别概率。
        """
        # 检查是否已调用fit方法
        check_is_fitted(self, ['fitted_base_estimators_', 'fitted_meta_learner_'])

        # 验证输入
        X = check_array(X)

        # 生成元特征
        X_meta = _generate_meta_features(self.fitted_base_estimators_, X)

        # 检查元学习器是否支持predict_proba
        if not hasattr(self.fitted_meta_learner_, "predict_proba"):
            raise AttributeError(
                f"元学习器 {self.fitted_meta_learner_.__class__.__name__} "
                f"不支持predict_proba。"
            )

        # 使用已拟合的元学习器预测概率
        return self.fitted_meta_learner_.predict_proba(X_meta)

    # get_params和set_params继承自BaseEstimator
    # 需要正确的__init__签名和与__init__参数匹配的公共属性。

    # 可选：如果需要特定Scikit-learn集成，实现_more_tags
    def _more_tags(self):
        # 表明如果基础估计器在预测时需要y（罕见），此估计器也需要y
        # 或如'requires_positive_X'等其他标签。
        return {'requires_y': False}
```

这个实现提供了一个基本的堆叠分类器。请注意`clone`的使用，以确保用户传入的原始估计器不被修改。辅助函数`_generate_meta_features`处理了基础模型的预测结果收集，尝试在可用时使用`predict_proba`，这通常对元学习器有益。我们加入了使用Scikit-learn验证工具（如`check_X_y`、`check_array`和`check_is_fitted`）进行的基本检查。

### 使用自定义估计器

现在，我们来看看如何使用我们的`StackingEstimator`。我们将定义一些基础模型和一个元学习器，然后将其集成到典型的Scikit-learn工作流程中。

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 生成合成分类数据
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,\n                           n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基础估计器
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', Pipeline([('scaler', StandardScaler()), # SVC对缩放敏感
                      ('svc', SVC(probability=True, random_state=42))]))
]

# 定义元学习器
meta_learner = LogisticRegression(solver='liblinear', random_state=42)

# 实例化我们的自定义StackingEstimator
stacking_clf = StackingEstimator(base_estimators=base_estimators,\n                                 meta_learner=meta_learner)

# --- 选项 1：直接拟合和预测 ---
print("正在直接拟合StackingEstimator...")
stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"StackingEstimator测试准确率: {accuracy:.4f}")

# 检查predict_proba
try:
    y_proba = stacking_clf.predict_proba(X_test)
    print(f"预测概率形状: {y_proba.shape}")
    # print("样本概率:\n", y_proba[:5]) # 取消注释以查看
except AttributeError as e:
    print(f"无法获取概率: {e}")

# --- 选项 2：使用交叉验证 ---
print("\n正在使用交叉验证评估StackingEstimator...")
# 注意：交叉验证可能较慢，因为它会多次重新拟合整个堆叠
cv_scores = cross_val_score(stacking_clf, X, y, cv=3, scoring='accuracy')
print(f"交叉验证准确率得分: {cv_scores}")
print(f"平均交叉验证准确率: {np.mean(cv_scores):.4f}")

# --- 选项 3：在管道中集成（示例） ---
# 尽管我们的基础'svc'已经包含了缩放，但这演示了其基本原理。
# 也许我们希望在任何估计器看到数据之前进行整体缩放。
print("\n正在管道中使用StackingEstimator...")
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 在输入到堆叠器之前缩放数据
    ('stacker', StackingEstimator(base_estimators=base_estimators, meta_learner=meta_learner))\n])

pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print(f"带StackingEstimator的管道测试准确率: {accuracy_pipeline:.4f}")
```

此示例显示了自定义`StackingEstimator`如何进行实例化、训练、用于预测、通过交叉验证进行评估，甚至作为一个步骤包含在更大的Scikit-learn `Pipeline`中。因为它遵循API，所以它可以与这些标准工具配合使用。

### 使用`check_estimator`进行测试

开发Scikit-learn组件的一个重要步骤是使用`check_estimator`工具。此函数运行一套全面的测试，以验证API兼容性、不变属性和预期行为。

```python
from sklearn.utils.estimator_checks import check_estimator

print("\n正在运行check_estimator（这可能需要一些时间并输出详细信息）...")
try:
    # 为了使某些检查通过，需要使用简单的基础估计器进行实例化
    simple_base = [('lr', LogisticRegression(solver='liblinear')), ('rf', RandomForestClassifier(n_estimators=5))]
    simple_meta = LogisticRegression(solver='liblinear')
    check_estimator(StackingEstimator(base_estimators=simple_base, meta_learner=simple_meta))
    print("check_estimator通过（或显示非重要警告）。")
except Exception as e:
    print(f"check_estimator失败: {e}")
```

运行`check_estimator`非常有价值，但有时难以完全通过，特别是对于集成等复杂估计器。失败通常指向细微的API违规或需要处理的边界情况。例如，我们的基本实现可能会根据所使用的基础估计器，未能通过与处理稀疏矩阵或特定元数据路由相关的检查。处理所有`check_estimator`失败通常需要与Scikit-learn内部机制进行更细致的交互。

### 潜在改进

这个动手实践的示例提供了一个基础。一些改进可以使`StackingEstimator`更有效、更灵活：

1. **交叉验证的元特征：** 避免在基础模型训练过的数据上直接使用其预测结果来训练元学习器（这有过度拟合的风险），而是在`fit`方法中使用交叉验证。在$k-1$折上训练基础模型，然后在留出折上进行预测，为整个训练集生成元特征，从而避免数据泄露。这是Scikit-learn官方`StackingClassifier`/`StackingRegressor`的默认操作方式。
2. **特征直通：** 允许将原始特征`X`与基础模型预测结果一同传递给元学习器。
3. **处理不同预测方法：** 允许配置基础模型是使用`predict`、`predict_proba`还是`decision_function`来生成元特征。
4. **并行拟合：** 使用`joblib`或`concurrent.futures`（如第5章所讨论的）并行拟合基础估计器，可能加速`fit`过程。
5. **改进参数 (parameter)验证：** 在`fit`中（或一个专门的私有验证方法中）增加更严格的检查，以确保估计器兼容。

构建像`StackingEstimator`这样的自定义估计器，能加深对Scikit-learn API的理解，并在现成方案不足时，帮助您为机器学习 (machine learning)管道构建高度专业的组件。

## 参考资料

- [Stacked Generalization](https://doi.org/10.1016/S0893-6080%2805%2980023-1) — David H. Wolpert (1992)
  Journal: Neural Networks; Publisher: Elsevier; Volume: 5; Pages: 241-259; DOI: [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080%2805%2980023-1)
  这篇开创性论文介绍了堆叠泛化（stacked generalization），这是一种结合多个模型的强大集成技术。
- [Developing custom scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html) — scikit-learn developers (2023)
  Scikit-learn官方指南，用于创建与scikit-learn API兼容的估计器，涵盖继承、方法签名和验证工具。
- [1.11. Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) — scikit-learn developers (2023)
  官方文档详细介绍了各种集成方法，包括对堆叠的解释以及StackingClassifier和StackingRegressor的使用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本全面的教材，提供了统计学习方法（包括堆叠等集成技术）的详细理论解释。
