---
course: "mastering-gradient-boosting-algorithms"
chapter: "boosting-hyperparameter-optimization"
lesson: "hyperparameter-optimization-frameworks"
sourceId: 2057
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-8-boosting-hyperparameter-optimization/hyperparameter-optimization-frameworks"
title: "超参数优化框架 (Optuna, Hyperopt)"
description: "使用 Optuna 和 Hyperopt 等库进行自动化调优。"
order: 5
plots: ["plots/2057-0.json"]
sourceHash: "2deefc28cdc39b9b8c8f8808cac3a52875f739cec20f2a90e2efdd8daf942478"
sourceCorrections: []
---

网格搜索和随机搜索提供了遍历超参数 (parameter) (hyperparameter)空间的系统方法，而贝叶斯优化则提供了一种更智能的搜索策略，但高效地实现这些方法，特别是贝叶斯方法，通常需要专门的工具。手动编写优化循环、管理试验以及可能整合诸如早期停止（剪枝）或并行化等高级功能，会非常复杂且耗时。超参数优化框架能够自动化这一过程的大部分，让你能够专注于定义搜索空间和目标函数。

为了自动化超参数优化的复杂过程，涌现 (emergence)了许多专用框架。其中两个著名的 Python 框架是 Optuna 和 Hyperopt。它们提供了各种搜索算法的实现，包括贝叶斯优化中常用的树结构 Parzen 估计器 (TPE)，以及用于管理和分析优化实验的实用工具。

### Optuna

Optuna 是一个现代且活跃开发的优化框架，正在获得广泛欢迎。它因其“即时定义 (define-by-run)”API 而备受推崇，该 API 提供了在目标函数中动态构建搜索空间的灵活性。

**主要特点：**

- **即时定义 (Define-by-Run) API：** 无需预先定义整个搜索空间，你可以直接在想要最小化的函数（目标函数）内部，使用 `trial.suggest_*` 方法（例如 `trial.suggest_float`、`trial.suggest_int`、`trial.suggest_categorical`）来指定参数 (parameter)分布。这允许存在条件超参数 (hyperparameter)，即一个参数的选择可能会影响另一个参数的范围或可用性。
- **采样器：** Optuna 支持多种采样算法。默认通常是 TPE (`TPESampler`)，它实现了贝叶斯优化。其他选项包括随机搜索 (`RandomSampler`)、网格搜索 (`GridSampler`) 和 CMA-ES (`CmaEsSampler`)，适用于不同的优化情况。
- **剪枝：** Optuna 与许多机器学习 (machine learning)库（包括 XGBoost、LightGBM 和 Scikit-learn）集成，以启用剪枝功能。剪枝是指监控试验的中间结果（例如，在一定数量的提升轮次后的验证分数），并提前停止没有前景的试验，从而节省大量的计算时间。
- **并行化：** Optuna 使得使用不同存储后端（如关系型数据库）在多个进程或机器上并行化超参数搜索变得简单。
- **可视化：** 它提供内置函数来可视化优化过程，例如优化历史、参数关系和超参数重要性。

**示例：使用 Optuna 调优 LightGBM**

这是一个演示如何使用 Optuna 调优 LightGBM 超参数以完成分类任务的示例。假设 `X_train`、`y_train`、`X_valid`、`y_valid` 已定义。

```python
import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 假设 X, y 是你的特征和目标
# 为简单起见，在此目标函数内分割训练和验证数据
# 实际应用中，请使用适当的交叉验证或固定验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
dtrain = lgb.Dataset(X_train, label=y_train)

def objective(trial):
    # 动态定义搜索空间
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
    }

    # 添加与 LightGBM 剪枝集成的示例
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss')

    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        callbacks=[pruning_callback, lgb.early_stopping(10, verbose=False)] # 使用 LightGBM 的早期停止
    )

    preds = gbm.predict(X_valid)
    pred_labels = (preds > 0.5).astype(int)
    accuracy = accuracy_score(y_valid, pred_labels)

    # Optuna 最小化目标，因此返回一个需要最小化的指标（例如，1.0 - 准确率）
    # 或者使用 study.direction='maximize' 并直接返回准确率。
    return 1.0 - accuracy # 值越低越好

# 创建一个研究对象并指定方向（最小化或最大化）
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

# 开始优化
study.optimize(objective, n_trials=100) # 运行 100 次试验

# 打印最佳试验结果
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value)) # 最佳目标值 (1.0 - 准确率)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 可视化示例（需要安装 plotly）
# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_param_importances(study).show()
```

Optuna 的可视化功能可以提供有用的见解。例如，优化历史图显示了最佳目标值如何随试验改进。



![优化历史](plots/2057-0.json)



> 该图显示了每次试验后迄今为止找到的最佳目标函数值，通常会随着优化器寻找更好的超参数配置而随时间降低。

### Hyperopt

Hyperopt 是另一个成熟的框架，尤其以其 TPE 实现而闻名。它与 Optuna 的主要区别在于搜索空间的定义方式。

**主要特点：**

- **搜索空间定义：** Hyperopt 要求你使用其特定的随机表达式函数（例如 `hp.choice`、`hp.uniform`、`hp.loguniform`、`hp.quniform`）预先将搜索空间定义为嵌套结构。
- **优化算法：** 主要支持 TPE (`tpe.suggest`) 和随机搜索 (`rand.suggest`)。
- **`fmin` 函数：** 运行优化过程的核心函数。它接收目标函数、搜索空间、优化算法、最大评估次数以及一个 `Trials` 对象（用于存储历史记录）作为输入。
- **`Trials` 对象：** 存储每个试验的详细信息，包括参数 (parameter)、状态和结果。可用于分析和恢复优化。
- **并行化：** 可以并行化，常通过 `SparkTrials` 对象使用 Apache Spark 等框架进行演示，尽管也可以进行基本的线程/多进程处理。

**示例：使用 Hyperopt 调优 XGBoost**

这是一个使用 Hyperopt 调优 XGBoost 的示例。假设 `X_train`、`y_train`、`X_valid`、`y_valid` 已定义。

```python
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# 假设 X, y 是你的特征和目标
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 定义搜索空间结构
space = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1), # 离散均匀（整数值）
    'learning_rate': hp.loguniform('learning_rate', -5, -1), # e^-5 到 e^-1 (约 0.0067 到 0.36)
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
    'lambda': hp.loguniform('lambda', -2, 2), # L2 正则，e^-2 到 e^2
    'alpha': hp.loguniform('alpha', -2, 2),    # L1 正则，e^-2 到 e^2
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 123 # 目标函数内用于重现性的固定种子
}

def objective(params):
    # Hyperopt 将整数参数作为浮点数传递，请进行转换
    params['max_depth'] = int(params['max_depth'])

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000, # 使用较大的值，依赖早期停止
        evals=watchlist,
        early_stopping_rounds=30,
        verbose_eval=False # 调优期间抑制详细输出
    )

    # 在验证集上评估
    preds = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    loss = log_loss(y_valid, preds)

    # Hyperopt 最小化返回字典中的 'loss' 值
    return {'loss': loss, 'status': STATUS_OK, 'model': model} # 可选：返回模型或其他信息

# 用于存储历史的 Trials 对象
trials = Trials()

# 运行优化
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest, # 使用树结构 Parzen 估计器
    max_evals=100,    # 试验次数
    trials=trials
)

print("Best parameters found: ", best)

# 如果需要，从 trials 对象访问详细结果
# print(trials.best_trial)
```

### 选择框架

Optuna 和 Hyperopt 都是自动化超参数 (parameter) (hyperparameter)优化的强大工具。

- **Optuna** 由于其 Python 风格的即时定义 API 以及与流行机器学习 (machine learning)库的出色集成功能（特别是剪枝），对于许多用户来说通常感觉更直观。其活跃的开发和不断壮大的社区也是优势。内置的可视化工具对于理解优化过程是一个显著的优点。
- **Hyperopt** 是一个成熟的库，其明确定义搜索空间的方式可能会吸引一些用户。它已被证明有效，尤其是在 TPE 方面。与 Spark 等并行化框架的集成可能是某些分布式计算环境中的一个决定因素。

Scikit-Optimize (`skopt`) 等其他库也存在，它们通常提供与 Scikit-learn 兼容的 API 和类似的功能。最佳选择取决于你的具体要求、编码风格偏好、对剪枝或高级可视化等特定功能的需求，以及与现有 MLOps 生态系统的集成情况。

使用这些框架大大简化了将贝叶斯优化等高级优化技术应用于梯度提升模型的过程。它们处理建议参数、管理试验以及（可选地）剪枝没有前景的运行等复杂机制，让你能够比手动调优或更简单的搜索方法更高效地找到高性能的超参数配置。

## 参考资料

- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/) — Optuna Developers (2024)
  使用 Optuna 的综合指南，详细介绍了其 API、采样器、剪枝和可视化功能。
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/) — Hyperopt Developers (2024)
  提供了关于 Hyperopt 搜索空间定义、优化算法（TPE）和 `fmin` 函数的详细信息。
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://dl.acm.org/doi/10.1145/3292500.3330705) — Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19); Publisher: Association for Computing Machinery; Pages: 2623–2631; DOI: [10.1145/3292500.3330705](https://doi.org/10.1145/3292500.3330705)
  介绍了 Optuna 的设计原则，包括其即时定义运行 API、动态搜索空间和剪枝功能。
- [Algorithms for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra, Rémi Bardenet, Yoshua Bengio, Balázs Kégl (2013)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 13; Pages: 281-308
  描述了树结构 Parzen 估计器 (TPE) 算法，该算法是 Hyperopt 中使用的贝叶斯优化的组件。
