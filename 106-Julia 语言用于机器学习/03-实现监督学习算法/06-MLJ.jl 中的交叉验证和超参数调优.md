# MLJ.jl 中的交叉验证和超参数调优

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-3-julia-supervised-learning-algorithms/julia-cross-validation-tuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练模型并在单个测试集上评估，可以得到模型性能的概览，但这种概览有时可能具有误导性。如果数据集较小，或者数据划分碰巧非常“幸运”（或“不幸运”），评估可能无法准确反映模型在新数据上的表现。为了构建更稳定的模型并获得对其泛化能力更可靠的评估，交叉验证等方法被采用。此外，大多数机器学习 (machine learning)模型都有需要在训练前设置的超参数 (parameter) (hyperparameter)；找到这些超参数的最佳组合可以明显影响模型性能。此处介绍如何使用 MLJ.jl 进行交叉验证和超参数调优。

### 通过交叉验证获得更可靠的性能评估

交叉验证是一种重采样过程，用于在有限数据样本上评估机器学习 (machine learning)模型。其主要思想是将训练数据分成多个“折”，然后多次训练和评估模型，每次使用不同的折作为测试集，其余的折作为训练集。这会为你提供多个性能评估，然后可以对这些评估取平均值，以提供对模型效果更稳定、可信的衡量。

最常见的形式是 k 折交叉验证。其工作方式如下：

1. 数据集随机打乱。
2. 数据集被分成 $k$ 个（或“折”）大小大致相等的组。
3. 对于每个独特的折：
   a. 将该折作为保留集或测试数据集。
   b. 将剩余的 $k-1$ 个折作为训练数据集。
   c. 在训练集上拟合模型，并在测试集上进行评估。
   d. 保留评估分数并丢弃模型。
4. 模型的能力通过对每次迭代的分数取平均值来概括。

> 此图展示了 K 折交叉验证过程。数据被分成 K 个折，模型被训练和评估 K 次。

在 MLJ.jl 中，交叉验证通过 `evaluate!` 函数执行，该函数接受模型、特征 `X`、目标 `y`、重采样策略以及一个或多个性能衡量指标。

让我们看看实际操作。我们将使用 `DecisionTreeClassifier` 并使用 5 折交叉验证对其进行评估。

```julia
using MLJ
using PrettyPrinting # 用于更美观地打印字典

# 加载模型和数据
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
X, y = @load_iris;

# 实例化模型
tree_model = DecisionTreeClassifier()

# 定义重采样策略：5 折交叉验证
cv_strategy = CV(nfolds=5, shuffle=true, rng=123)

# 执行评估
# 我们将使用常见的分类指标
eval_results = evaluate(tree_model, X, y,
                        resampling=cv_strategy,
                        measures=[accuracy, multiclass_f1score, multiclass_precision, multiclass_recall],
                        verbosity=0) # verbosity=0 以抑制每折输出

# 显示各折的平均性能
pprint(eval_results.measurement)
# 每折结果也可在 eval_results.per_fold 中查看
```

运行此代码时，`eval_results.measurement` 将显示一个指标列表，每个指标都在 5 个折上取了平均值。例如，你可能会看到 `[0.92, 0.918, 0.921, 0.92]`，这对应于平均准确率、F1 分数、精确率和召回率。查看 `eval_results.per_fold` 将会得到一个列表的列表，显示每个折的指标分数。相比于单次训练-测试划分，这能更好地了解模型的预期表现。MLJ.jl 支持 `CV` 之外的多种重采样策略，例如 `Holdout`（简单的训练-测试划分）和 `StratifiedCV`（确保每个折中类别比例得以保持，对于不平衡数据集很有用）。

### 精细调整模型超参数 (parameter) (hyperparameter)

大多数机器学习 (machine learning)算法都有超参数：这些设置不是从数据本身学习的，而是在训练开始前由使用者选择的。例如，决策树具有 `max_depth`（树的最大深度）或 `min_samples_split`（拆分内部节点所需的最小样本数）等超参数。超参数的选择可以明显影响模型的性能。超参数调优（或优化）是找到超参数值组合的过程，该组合能为你的具体问题带来最佳性能。

#### 调优策略

超参数调优存在多种策略。一种常见且直接的方法是**网格搜索**。在网格搜索中，你定义一个由可能超参数值组成的“网格”。算法随后会评估该网格中每个值组合下的模型，通常为每个组合使用交叉验证。选择产生最佳平均交叉验证分数的组合。

> 超参数调优的网格搜索图示。针对超参数的每个组合，都使用交叉验证进行评估，以确定最佳集合。

MLJ.jl 提供 `TunedModel` 包装器以自动化超参数调优。你需要指定基础模型、调优策略（如 `Grid`）、要查看的超参数范围、评估的重采样策略以及要优化的性能衡量指标。

让我们调整 `DecisionTreeClassifier` 的 `max_depth` 和 `min_samples_leaf` 超参数。

```julia
using MLJ

DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
X, y = @load_iris;

# 实例化基础模型
tree_model = DecisionTreeClassifier()

# 定义超参数范围
# 对于 max_depth，我们将尝试值 2, 3, 4, 5
r_max_depth = range(tree_model, :max_depth, lower=2, upper=5, scale=:linear)
# 对于 min_samples_leaf，我们将尝试值 1, 2, 5
r_min_leaf = range(tree_model, :min_samples_leaf, values=[1, 2, 5])

# 创建 TunedModel
tuned_tree = TunedModel(model=tree_model,
                        tuning=Grid(resolution=10), # Grid 的分辨率，或指定明确值
                        resampling=CV(nfolds=3, rng=456), # 用于内部调优循环的交叉验证
                        ranges=[r_max_depth, r_min_leaf],
                        measure=accuracy, # 要优化的指标
                        acceleration=CPUThreads()) # 如果可用，使用多线程

# 将 TunedModel 封装到 machine 中并进行拟合
mach_tuned_tree = machine(tuned_tree, X, y)
fit!(mach_tuned_tree, verbosity=1) # verbosity=1 显示一些调优进度

# 获取最佳模型及其超参数
best_model_params = fitted_params(mach_tuned_tree)
println("\n最佳模型超参数:")
pprint(best_model_params.best_model)

# 你也可以检查调优过程的完整报告
report_tuned = report(mach_tuned_tree)
# report_tuned.best_measurement 将给出最佳模型的准确率
# report_tuned.plotting 包含对可视化调优结果有用的数据

# 最佳模型可以直接用于预测
# 或者提取实际拟合的模型:
# fitted_model = fitted_params(mach_tuned_tree).best_fitted_model
```

在此示例中：

- 我们定义 `r_max_depth` 以将 `max_depth` 的整数值从 2 检查到 5。MLJ 中的 `range` 函数相当灵活。对于数值型超参数，你可以指定 `lower` 和 `upper` 边界以及 `scale`（例如，`:linear`、`:log`）。对于具有离散、无序值的超参数，你可以传递一个 `values` 向量 (vector)。
- `TunedModel` 配置了 `Grid` 搜索。`Grid` 的 `resolution` 参数表示如果没有明确给出 `values`，则沿每个数值范围采样多少点。
- 内部使用 3 折交叉验证（`CV(nfolds=3)`）来评估每个超参数组合。
- `measure` 设置为 `accuracy`，意味着网格搜索将尝试确定最大化准确率的超参数。
- `fit!(mach_tuned_tree)` 执行调优。
- `fitted_params(mach_tuned_tree).best_model` 为你提供一个 `DecisionTreeClassifier` 实例，其中包含已确定的最佳超参数。
- `report(mach_tuned_tree)` 提供详细报告，包含每个超参数组合的性能。

MLJ.jl 中也可用其他调优策略，如 `RandomSearch`，当超参数空间很大时，它可能比 `Grid` 搜索更高效。

### 组织调优和评估的工作流程

一个常见问题是使用整个数据集（或对整个数据集进行交叉验证）来调优超参数 (parameter) (hyperparameter)，然后报告这个已调优模型的性能。这种性能评估可能过于乐观，因为调优过程本身通过交叉验证“看到”了所有数据。

一种更直接的方法涉及对数据进行初始划分：

1. **初始数据划分**：将数据划分为较大的训练集和一个较小的、最终保留的测试集。测试集应单独保留，仅在最后使用一次。
2. **在训练集上进行超参数调优**：*仅*在训练集上进行交叉验证和超参数调优（例如，使用 `TunedModel`）。这将确定最佳超参数。
3. **最终模型训练**：在*整个*训练集上训练所选模型类型（使用步骤 2 中确定的最佳超参数）。
4. **最终评估**：在保留测试集上评估这个最终模型。这提供了对已调优模型在新数据上性能的无偏评估。

MLJ.jl 的 `machine` 抽象自然地支持这一点。你将在训练部分（`X_train`, `y_train`）上拟合 `TunedModel` machine（步骤 2）。这个 machine 的 `fitted_params` 会为你提供 `best_model`（这是一个带有最佳超参数的模型规范）。然后，你使用此 `best_model` 规范创建一个新的标准 machine，并在完整的 `X_train`, `y_train` 上拟合它（步骤 3），最后在 `X_test`, `y_test` 上进行预测和评估（步骤 4）。

通过认真应用交叉验证进行评估和超参数调优，你可以构建出不仅性能优异，而且其性能评估也值得信任的模型。这些方法是构建 Julia 或任何其他语言中有效机器学习 (machine learning)解决方案的基础。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2013)
  Publisher: Springer; Pages: 440
  一本广泛使用的教科书，涵盖了交叉验证和超参数调优策略的基础概念。
- [MLJ.jl Documentation: Model Tuning](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning/) — The MLJ.jl Contributors (2024)
  Publisher: The Alan Turing Institute
  MLJ.jl 中超参数调优工作流和可用策略的官方指南。
- [MLJ.jl Documentation: Evaluating Models](https://juliaai.github.io/MLJ.jl/dev/evaluating_models/) — The MLJ.jl Contributors (2025)
  MLJ.jl 中模型评估和重采样策略的官方参考。
- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/v13/bergstra12a.html) — James Bergstra, Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 13; Pages: 281-305
  一篇有影响力的论文，研究了随机搜索与网格搜索在超参数优化中的有效性。

---

[上一节](05-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%EF%BC%9A%E5%87%86%E7%A1%AE%E7%8E%87%E3%80%81%E7%B2%BE%E7%A1%AE%E7%8E%87%E3%80%81%E5%8F%AC%E5%9B%9E%E7%8E%87%E3%80%81F1%E5%88%86%E6%95%B0.md) · [下一节](07-%E7%9B%91%E7%9D%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9A%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5.md)
