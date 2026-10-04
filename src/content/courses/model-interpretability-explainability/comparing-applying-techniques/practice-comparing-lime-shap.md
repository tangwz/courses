---
course: "model-interpretability-explainability"
chapter: "comparing-applying-techniques"
lesson: "practice-comparing-lime-shap"
sourceId: 4105
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-4-comparing-applying-techniques/practice-comparing-lime-shap"
title: "实践：对比 LIME 和 SHAP 输出"
description: "为相同的模型预测生成并对比 LIME 和 SHAP 解释。"
order: 8
plots: ["plots/4105-0.json"]
sourceHash: "16f2a6648aa707e47c613906acbb70f29f73b8df4afe29f5859545fbc36f7468"
sourceCorrections: []
---

LIME 和 SHAP 是可解释性技术。可以使用这两种技术为*相同的*模型预测生成解释，并对结果进行并排对比。这种对比有助于理解每种方法如何归因影响，以及它们不同方法可能产生的不同之处。

### 准备工作：模型与数据

设想我们已经在大家熟悉的鸢尾花数据集上训练了一个 `RandomForestClassifier`，用来根据萼片和花瓣的测量值预测鸢尾花的种类。我们假设此模型已经训练好，并以 `model` 的形式可用。我们的特征是“萼片长度 (cm)”、“萼片宽度 (cm)”、“花瓣长度 (cm)”和“花瓣宽度 (cm)”。

让我们从测试集中选择一朵特定的鸢尾花，模型对其预测为“virginica”种类（类别索引 2）。假设此实例具有以下特征：

- 萼片长度：6.5 厘米
- 萼片宽度：3.0 厘米
- 花瓣长度：5.8 厘米
- 花瓣宽度：2.2 厘米

我们的目标是使用 LIME 和 SHAP 来解释模型*为什么*将这朵特定的花预测为“virginica”。

### 生成 LIME 解释

首先，我们使用 LIME。正如第二章所讲，LIME 的工作原理是：对所讨论的实例进行扰动，并拟合一个更简单、可解释的模型（如线性回归），以*局部*理解黑盒 (black box)模型在该实例周围的行为。

我们将使用 `LimeTabularExplainer` 来完成此任务。

```python
import lime
import lime.lime_tabular
import numpy as np

# 假设:
# X_train: 训练数据 (numpy 数组或 pandas DataFrame)
# feature_names: 特征名称列表
# class_names: 目标类别名称列表
# model: 训练好的带 predict_proba 方法的 scikit-learn 分类器
# instance_to_explain: 单个实例的 numpy 数组 (例如, [6.5, 3.0, 5.8, 2.2])

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# 解释特定实例的预测结果 (预测类别索引为 2)
explanation_lime = explainer_lime.explain_instance(
    data_row=instance_to_explain,
    predict_fn=model.predict_proba,
    num_features=4, # 显示所有特征的重要性
    top_labels=1 # 关注预测的类别
)

# 显示解释 (表示形式)
# 实际输出可能通过可视化或 explanation_lime.as_list(label=2) 来获取
print("LIME 对 'virginica' 类别的解释:")
# 示例输出:
# [('花瓣宽度 (cm) > 1.70', 0.35),
#  ('花瓣长度 (cm) > 4.90', 0.25),
#  ('萼片长度 (cm) <= 6.80', -0.08),
#  ('萼片宽度 (cm) <= 3.10', 0.05)]
```

**LIME 的解读：** LIME 通常将其解释呈现为从其局部代理模型得出的特征贡献（权重 (weight)）列表。上面的输出表明，此实例的“花瓣宽度”大于 1.7 厘米和“花瓣长度”大于 4.9 厘米，强力地将预测推向“virginica”。根据 LIME 的局部近似，萼片测量值的影响较小且混杂。请注意，LIME 通常会将连续特征离散化以便于解释，从而形成诸如“花瓣宽度 (cm) > 1.70”之类的规则。

### 生成 SHAP 解释

接下来，我们应用 SHAP。由于我们有一个随机森林模型，我们可以使用高效的 `TreeExplainer`。如果我们不知道模型类型或它不是基于树的模型，我们可以使用 `KernelExplainer`。SHAP 计算 Shapley 值，这些值代表了每个特征在所有可能特征组合中的平均边际贡献。

```python
import shap

# 假设:
# model: 训练好的基于树的模型 (例如 RandomForestClassifier)
# X_train: 解释器所需的背景数据 (通常是训练集)
# instance_to_explain_df: 单个实例的 pandas DataFrame

explainer_shap = shap.TreeExplainer(model, data=X_train) # 对于非树模型使用 KernelExplainer

# 计算实例的 SHAP 值
# 注意: 如果是在 DataFrame 上训练的，SHAP 解释器通常期望 DataFrame 输入
shap_values = explainer_shap.shap_values(instance_to_explain_df)

# SHAP 值通常按类别返回。我们需要预测类别（'virginica'，索引 2）的值
shap_values_for_class_2 = shap_values[2][0] # [类别索引][实例索引]

# 显示 SHAP 值 (表示形式)
print("\nSHAP 对 'virginica' 类别的解释:")
# 示例输出:
# {'萼片长度 (cm)': -0.15,
#  '萼片宽度 (cm)': 0.05,
#  '花瓣长度 (cm)': 0.40,
#  '花瓣宽度 (cm)': 0.60}

# 使用力图可视化
# shap.force_plot(explainer_shap.expected_value[2], shap_values_for_class_2, instance_to_explain_df)
```

**SHAP 的解读：** SHAP 值表示每个特征值对模型输出的贡献，将其从基准值（训练集上的平均预测）推向此特定实例的最终预测。正 SHAP 值会增加对“virginica”的预测，而负值则会减少预测。此处，“花瓣宽度”（0.60）和“花瓣长度”（0.40）是推动预测偏向“virginica”的主要因素。“萼片宽度”有小的正贡献，而“萼片长度”有中等的负贡献。

### 对比 LIME 和 SHAP 输出

现在，让我们对比两种方法针对我们特定鸢尾花预测所得出的见解：

1. **最重要的特征：** LIME 和 SHAP 都将“花瓣宽度 (cm)”和“花瓣长度 (cm)”确定为推动预测偏向“virginica”的最重要特征。这种一致性增强了我们对这些特征重要性的信心。
2. **排名：** 前两个特征的排名是一致的（“花瓣宽度” > “花瓣长度”）。然而，不那么重要的特征（“萼片长度”、“萼片宽度”）的排名和感知影响可能略有不同。LIME 显示“萼片长度”略为负向，“萼片宽度”略为正向；而 SHAP 显示“萼片长度”更明显地负向，“萼片宽度”略为正向。
3. **量级/尺度：** 值本身不能直接比较。LIME 提供的是来自*局部线性模型*的权重 (weight)，而 SHAP 提供的是在模型输出尺度（例如，分类概率的对数几率）上测量的*加性贡献*（Shapley 值）。SHAP 值加起来等于模型对该实例的输出与基准期望值之间的差异。
4. **表示形式：** LIME 经常使用离散化特征或规则（例如，“花瓣宽度 (cm) > 1.70”），这可能很直观，但取决于离散化方法。SHAP 通常提供原始特征值的贡献。
5. **一致性：** SHAP 值具有局部准确性（值之和等于预测与基准值之间的差异）和一致性（当模型变得更依赖某个特征时，该特征的重要性不会反常地降低）等属性。LIME 的解释在很大程度上取决于扰动策略和局部线性近似的保真度，这有时可能导致不稳定性。

让我们使用上面示例中得出的重要性值来可视化对比。



![特征重要性对比 (单一预测)](plots/4105-0.json)



> 选定实例中对“virginica”预测有贡献的特征的 LIME 权重和 SHAP 值对比。虽然绝对值有所不同，但花瓣特征的相对重要性在两者中都清晰可见。

### 讨论要点

- **一致与分歧：** 当 LIME 和 SHAP 在最具影响力的特征上达成一致时（对于主导特征通常如此），这会增强我们对解释的信心。分歧，特别是对于影响力较小的特征，突显了所提供的不同视角：LIME 侧重于局部线性，而 SHAP 考量不同组合中的特征贡献。
- **计算成本：** 生成此单一 LIME 解释可能比使用 `TreeExplainer` 计算精确 SHAP 值要快（尽管 `TreeExplainer` 经过高度优化）。如果我们使用了 `KernelSHAP`，计算将比 LIME 密集得多，因为它也依赖于扰动，但需要更多的样本来提供理论依据。
- **用例：** LIME 对于快速局部检查可能更快。SHAP 提供了一种理论上更有依据的特征贡献衡量方法，并且除了像力图这样的局部解释外，还提供了强大的全局汇总图（如第三章所示）。

通过在同一实例上运行这两种方法，您能更全面地理解模型做出特定预测的原因，并体会这两种流行的可解释性技术之间的细微差异和潜在区别。这种对比方法有助于验证发现，并提供特征影响的更完整图像。

## 参考资料

- [Why Should I Trust You? Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 1135–1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME算法，利用局部代理模型解释黑盒模型的个体预测。
- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) — Scott Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Pages: 4765–4774; DOI: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
  提出了SHAP（SHapley Additive exPlanations），一个基于博弈论中Shapley值的统一模型可解释性框架。
- [Interpretable Machine Learning: A Guide for Making Sense of Black Box Models](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  一本全面的在线书籍，详细解释并比较了包括LIME和SHAP在内的各种可解释性方法。
