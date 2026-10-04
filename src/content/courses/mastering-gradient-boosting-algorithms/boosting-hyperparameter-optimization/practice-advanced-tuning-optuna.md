---
course: "mastering-gradient-boosting-algorithms"
chapter: "boosting-hyperparameter-optimization"
lesson: "practice-advanced-tuning-optuna"
sourceId: 2064
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-8-boosting-hyperparameter-optimization/practice-advanced-tuning-optuna"
title: "动手实践：使用 Optuna 进行进阶调优"
description: "使用 Optuna 实现提升模型的贝叶斯超参数优化。"
order: 8
plots: ["plots/2064-0.json", "plots/2064-1.json"]
sourceHash: "a2bd9067ebbe3909a41e4479ec2d136e8229df718cc52dc09dbcbfa5427ed63b"
sourceCorrections: []
---

自动化超参数 (parameter) (hyperparameter)优化对于开发高效且高性能的机器学习 (machine learning)模型至关重要。Optuna 是一个专门为此任务设计的现代 Python 框架。它实现了复杂调优方法的实际应用，采用采样和剪枝算法，使搜索过程显著更高效。这些方法相对于网格搜索或随机搜索等简单技术具有优势，通常通过利用贝叶斯优化等方法实现。

我们将逐步指导您使用 Optuna 在标准数据集上调优 XGBoost 分类器。您将学习如何定义搜索空间，创建 Optuna 最小化或最大化的目标函数，运行优化研究，并解释结果以训练最终的优化模型。

### 环境准备

首先，请确保您已安装所需的库。您需要 `xgboost`、`optuna` 和 `scikit-learn`。如果尚未安装，可以使用 pip 进行安装：

```bash
pip install xgboost optuna scikit-learn plotly
```

现在，让我们导入所需的模块并加载数据集。在此示例中，我们将使用 scikit-learn 中熟悉的威斯康星乳腺癌数据集，并将其拆分为训练集和验证集。验证集对于在优化过程中评估每个超参数 (parameter) (hyperparameter)集的性能以及在 XGBoost 中实现早停功能非常重要。

```python
import xgboost as xgb
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import plotly # Required for Optuna visualizations

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
```

### 定义目标函数

Optuna 优化的核心组成部分是*目标函数*。此函数接收一个特殊的 `trial` 对象作为输入。在此函数内部，您可以使用 `trial.suggest_...` 方法定义要调优的超参数 (parameter) (hyperparameter)。这些方法指定参数名称、数据类型（整数、浮点数、分类）以及要尝试的范围或选项。然后，函数使用这些建议的超参数训练模型，在验证集上评估模型，并返回 Optuna 应该优化的指标分数。

在我们的例子中，我们希望最大化 XGBoost 分类器的 ROC 曲线下面积（AUC）。Optuna 默认最小化目标函数，因此我们将直接返回 AUC 分数，并在创建研究时指定 `direction='maximize'`。我们还将把早停功能加入到 XGBoost 训练过程中，以防止过拟合 (overfitting)并加快单个试验的运行速度。

```python
def objective(trial):
    """Objective function for Optuna to optimize."""

    # Define the hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',  # Use AUC for evaluation and early stopping
        'booster': 'gbtree',
        'verbosity': 0, # Suppress verbose output
        'nthread': -1, # Use all available threads
        'seed': 42,

        # Parameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Row subsampling
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Feature subsampling
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True), # L1 regularization
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True), # Min loss reduction for split
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # Min sum instance weight in child
    }

    # XGBoost DMatrix for efficiency
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Setup early stopping
    # Note: n_estimators is implicitly handled by early stopping
    early_stopping_rounds = 50
    evals = [(dtrain, 'train'), (dval, 'eval')]

    try:
        # Train the XGBoost model
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000, # Set a high value, early stopping will determine optimal rounds
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False # Suppress output for each round
        )

        # Make predictions on validation set
        preds = bst.predict(dval, iteration_range=(0, bst.best_iteration))

        # Calculate AUC
        auc = roc_auc_score(y_val, preds)

        return auc # Return the metric to maximize

    except xgb.core.XGBoostError as e:
        # Handle cases where parameters might lead to errors (e.g., empty trees)
        print(f"XGBoostError in trial {trial.number}: {e}")
        return 0.0 # Return a poor score if an error occurs

    except Exception as e:
        # Catch other potential issues
        print(f"An unexpected error occurred in trial {trial.number}: {e}")
        return 0.0 # Return a poor score
```

请注意我们如何使用 `trial.suggest_float` 和 `trial.suggest_int` 等方法。`log=True` 参数通常对 `learning_rate` 或正则化 (regularization)项等参数有益，因为它可以在不同数量级上更均匀地采样值。我们还加入了 `gamma` 和 `min_child_weight`，它们控制树的复杂度。`n_estimators` 通过基于验证 AUC 的早停功能进行有效调整。

### 创建并运行优化研究

定义目标函数后，我们创建一个 Optuna `study` 对象。我们将 `direction` 指定为“maximize”，因为我们想要尽可能高的 AUC。然后，我们调用 `study.optimize` 方法，传入我们的 `objective` 函数和所需的试验次数 (`n_trials`)。更多试验次数可以使 Optuna 更彻底地遍历搜索空间，但会增加计算时间。

```python
# Create an Optuna study
study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')

# Start the optimization
# Increase n_trials for a more thorough search (e.g., 100 or more)
n_trials = 50
study.optimize(objective, n_trials=n_trials)

# Optimization finished
print(f"\nOptimization finished after {n_trials} trials.")
```

Optuna 现在将迭代调用 `objective` 函数 `n_trials` 次。在每次试验中，它会根据先前试验的结果建议一组新的超参数 (parameter) (hyperparameter)，旨在找到产生最佳验证 AUC 的组合。

### 分析结果

优化完成后，Optuna 提供便捷的方式来访问结果。

```python
# Get the best trial
best_trial = study.best_trial

print(f"Best trial number: {best_trial.number}")
print(f"Best AUC score: {best_trial.value:.6f}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")
```

此输出显示了通过找到的最佳超参数 (parameter) (hyperparameter)组合所获得的验证 AUC，以及这些参数的具体值。

Optuna 还提供强大的可视化功能（通常需要安装 `plotly`）来帮助理解优化过程。

**优化历史：** 显示最佳分数如何在试验中提高。

```python
# Visualize optimization history
fig_history = optuna.visualization.plot_optimization_history(study)
fig_history.show()
```



![优化历史图](plots/2064-0.json)



> 优化历史图显示了每次试验的 AUC 分数（蓝色点）以及截至该试验所找到的最佳 AUC 分数（红色线）。通常，最佳分数在初期会快速提高，然后随着 Optuna 集中于有前景的区域而趋于平稳。

**参数重要性：** 帮助识别哪些超参数在搜索过程中对 AUC 分数影响最大。这使用基于平均杂质减少（MDI）的方法，该方法通过在试验结果上训练的随机森林计算。

```python
# Visualize parameter importances
fig_importance = optuna.visualization.plot_param_importances(study)
fig_importance.show()
```



![超参数重要性](plots/2064-1.json)



> 条形图说明了每个超参数在影响验证 AUC 方面的相对重要性。在此次特定优化运行中，重要性值越高的参数对于获得更好分数越关键。

其他可视化（如切片图 `plot_slice` 或等高线图 `plot_contour`）可以帮助理解特定超参数与目标值之间的关系，但参数重要性通常在初期提供最有用的信息。

### 训练最终模型

超参数 (parameter) (hyperparameter)调优过程根据验证性能识别出最佳参数集。最后一步是使用这些最佳参数训练一个新模型。通常的做法是使用最佳试验中早停功能确定的最佳提升轮数，在整个训练数据集（如果您有单独的最终测试集，甚至可以是原始训练集和验证集的组合）上训练此最终模型。

```python
# Get the best hyperparameters
best_params = study.best_params

# Add necessary fixed parameters
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['booster'] = 'gbtree'
best_params['verbosity'] = 0
best_params['nthread'] = -1
best_params['seed'] = 42

# Determine the optimal number of boosting rounds from the best trial
optimal_num_boost_round = study.best_trial.user_attrs.get('best_iteration') # Retrieve if saved
# Or re-run training briefly to get it if not saved (less ideal)
# For this example, let's use a fixed reasonable estimate or re-run quickly
# A better approach involves saving the best iteration within the objective function:
# trial.set_user_attr('best_iteration', bst.best_iteration)

# Let's assume we retrieved it or re-run the best trial training just to get best_iteration
# This part might need adjustment based on how you store the best iteration.
# For demonstration, we'll train again briefly on the train/val split
# to find the iteration count associated with best_params.

temp_dtrain = xgb.DMatrix(X_train, label=y_train)
temp_dval = xgb.DMatrix(X_val, label=y_val)
temp_evals = [(temp_dval, 'eval')]
temp_bst = xgb.train(best_params, temp_dtrain, num_boost_round=1000,
                     evals=temp_evals, early_stopping_rounds=50, verbose_eval=False)
final_num_boost_round = temp_bst.best_iteration
print(f"Optimal number of boosting rounds: {final_num_boost_round}")

# Train the final model on the full training data with best parameters and rounds
final_dtrain = xgb.DMatrix(X_train, label=y_train) # Use the original training set
final_model = xgb.train(
    best_params,
    final_dtrain,
    num_boost_round=final_num_boost_round, # Use optimal rounds
    verbose_eval=False
)

print("\nFinal model trained with optimal hyperparameters:")
print(final_model.attributes())

# Note: Evaluate this final_model on a separate, unseen test set for unbiased performance estimation.
```

*自我修正：* 原始计划没有明确地将早停中的 `best_iteration` 保存在 `objective` 函数中。正确的实现方式应该是在训练之后和返回分数之前，在 `objective` 函数内部使用 `trial.set_user_attr('best_iteration', bst.best_iteration)` 来存储此值。然后可以通过 `study.best_trial.user_attrs['best_iteration']` 检索它。上面的代码展示了一种变通方法，即用最佳参数短暂地重新训练以获取此值，但将其保存在试验期间会更简洁。

这个 `final_model` 现在可以用于部署或在保留的测试集上进行评估，以估计其泛化性能。

### 总结

使用 Optuna 提供了一种结构化且高效的方式来处理像 XGBoost 这样的梯度提升模型的复杂超参数 (parameter) (hyperparameter)空间。通过定义目标函数和搜索空间，您可以使用贝叶斯优化（或 Optuna 中的其他先进算法）来找到高性能的参数配置。这种自动化方法相比于网格搜索或随机搜索，节省了大量手动工作，并且通常会带来更好的模型性能。请记住，调优过程的质量很大程度上取决于定义合适的参数范围、选择合适的评估指标以及运行足够数量的试验。掌握 Optuna 这样的工具是构建优化梯度提升解决方案的重要一步。

## 参考资料

- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) — Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Publisher: ACM; Pages: 2623-2631; DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)
  介绍 Optuna 框架及其核心功能的原始研究论文。
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/) — Optuna Development Team (2024)
  官方文档，提供 Optuna 使用的全面指南、API 参考和示例。
- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍 XGBoost 算法的奠基性论文，本节中对此算法进行调优。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/pdf/1206.2944) — Jasper Snoek, Hugo Larochelle, Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems 25; Publisher: Neural Information Processing Systems Foundation, Inc.; Pages: 2951-2959
  关于贝叶斯优化在超参数调优中实际应用的具影响力论文。
