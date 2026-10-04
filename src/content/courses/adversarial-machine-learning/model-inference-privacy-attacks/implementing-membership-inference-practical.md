---
course: "adversarial-machine-learning"
chapter: "model-inference-privacy-attacks"
lesson: "implementing-membership-inference-practical"
sourceId: 4163
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-4-model-inference-privacy-attacks/implementing-membership-inference-practical"
title: "成员身份推断的实际操作"
description: "练习对机器学习模型实施基本的成员身份推断攻击。"
order: 6
plots: ["plots/4163-0.json"]
sourceHash: "1550d3c217bdfac73a9149fb59482a707dec25b04fb08c6e2c03689e2297b473"
sourceCorrections: []
---

让我们将成员身份推断攻击的理论付诸实践。正如本章前面所讨论的，目标是确定一个特定数据点是否属于目标模型的训练集，通常通过观察模型处理该数据点时的行为来判断。直观的理解是，模型对于训练中已经见过的数据，可能会与对未见过的数据有不同的响应，例如更高的置信度或特定的输出模式。

我们将关注一种常见且行之有效的方法：使用**影子模型**。这种方法涉及训练辅助模型（“影子模型”），使其模仿目标模型的行为。然后我们训练一个“攻击模型”，用来区分这些影子模型在其各自的训练（成员）数据和测试（非成员）数据上产生的输出。最后，这个攻击模型被用来对实际目标模型的输出进行分类。

### 准备工作

设想一个攻击者可以查询一个已训练的目标模型，$f_{target}$。攻击者还拥有一些数据，其分布与目标模型的私有训练数据$D_{target\_train}$相似。攻击者不知道$D_{target\_train}$，但想确定对于给定样本$x$，它是否在$D_{target\_train}$中。

基于影子模型的成员身份推断攻击的主要步骤是：

1. **训练影子模型：** 创建多个模型（$f_{shadow\_1}, f_{shadow\_2}, ..., f_{shadow\_k}$），理想情况下它们具有与目标模型$f_{target}$相同的架构并以相似方式训练。攻击者使用自己的数据$D_{attacker}$，并将其分区以训练每个影子模型。对于每个$f_{shadow\_i}$，一部分$D_{attacker}$用作其训练集（$D_{shadow\_i\_train}$），另一部分不重叠的数据则作为测试集（$D_{shadow\_i\_test}$）。
2. **生成攻击训练数据：** 使用其*自身*的训练集（$D_{shadow\_i\_train}$）和测试集（$D_{shadow\_i\_test}$）中的样本查询每个影子模型$f_{shadow\_i}$。收集输出（例如，预测概率向量 (vector)或logit值）。将来自$D_{shadow\_i\_train}$的输出标记 (token)为“成员”（标签1），将来自$D_{shadow\_i\_test}$的输出标记为“非成员”（标签0）。汇总所有影子模型的这些带标签的输出，以创建一个用于训练攻击模型的数据集。
3. **训练攻击模型：** 使用上一步生成的数据集训练一个二分类器$f_{attack}$。该分类器的特征是来自影子模型的输出向量（例如，概率），目标变量是成员/非成员标签。
4. **执行攻击：** 获取你想要测试的样本$x$在*目标模型* $f_{target}$上的输出。将此输出向量输入到已训练的攻击模型$f_{attack}$中。$f_{attack}$的预测结果（“成员”或“非成员”）就是成员身份推断攻击的结果。

### 实现概述

让我们通过伪代码片段来说明这一点，假设你拥有`define_model()`函数（返回模型架构，例如Keras/PyTorch模型）和`train_model(model, train_data, epochs)`函数（训练模型）。我们还假设数据已适当准备（例如，使用TensorFlow Datasets或PyTorch DataLoaders）。

#### 1. 训练影子模型

```python
# 假设 attacker_data 是攻击者可用的数据集
# 假设 num_shadow_models 是所需影子模型的数量
# 假设 target_model_architecture 已知或已近似

shadow_model_outputs = []
shadow_model_labels = []

for i in range(num_shadow_models):
    print(f"正在训练影子模型 {i+1}/{num_shadow_models}...")

    # 为此影子模型分割攻击者的数据
    shadow_train_data, shadow_test_data = split_data(attacker_data, ratio=0.5) # 使用不重叠的分区

    # 定义并训练影子模型
    shadow_model = define_model(architecture=target_model_architecture)
    train_model(shadow_model, shadow_train_data, epochs=50) # 示例训练轮次

    # 获取其自身训练数据（成员）的预测
    member_predictions = shadow_model.predict(shadow_train_data.x) # 获取输出向量
    member_labels = np.ones(len(member_predictions))

    # 获取其自身测试数据（非成员）的预测
    non_member_predictions = shadow_model.predict(shadow_test_data.x) # 获取输出向量
    non_member_labels = np.zeros(len(non_member_predictions))

    # 存储结果以供攻击模型训练
    shadow_model_outputs.append(np.concatenate((member_predictions, non_member_predictions)))
    shadow_model_labels.append(np.concatenate((member_labels, non_member_labels)))

# 组合所有影子模型的结果
attack_train_X = np.concatenate(shadow_model_outputs)
attack_train_y = np.concatenate(shadow_model_labels)

print(f"已生成攻击训练数据: {attack_train_X.shape}, {attack_train_y.shape}")
```

这个循环模拟了为目标模型创建代理的过程。每个影子模型在其自身数据上学习模式，其在已见过数据与未见过数据上的表现为攻击模型提供了信号。

> 使用影子模型的成员身份推断攻击流程图。攻击者在自己的数据上训练影子模型，以生成用于攻击模型的训练数据。然后，该攻击模型根据目标模型针对特定查询数据点的输出来预测成员身份。

#### 2. 训练攻击模型

我们可以使用`scikit-learn`中的简单分类器，如逻辑回归或小型多层感知机（MLP）作为攻击模型。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 可选：特征（模型输出）标准化
scaler = StandardScaler()
attack_train_X_scaled = scaler.fit_transform(attack_train_X)

# 分割攻击数据以进行评估
X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(
    attack_train_X_scaled, attack_train_y, test_size=0.3, stratify=attack_train_y, random_state=42
)

# 训练攻击模型
attack_model = LogisticRegression(solver='liblinear', random_state=42)
# 或者使用一个小型 MLP：
# attack_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)

attack_model.fit(X_train_att, y_train_att)

# 在保留的攻击数据分割上进行评估
y_pred_att = attack_model.predict(X_test_att)
print("攻击模型表现（在影子模型数据上）：")
print(f"准确率: {accuracy_score(y_test_att, y_pred_att):.4f}")
print(classification_report(y_test_att, y_pred_att))
```

这项评估告诉我们攻击模型从*影子模型*的输出中区分成员与非成员的熟练程度。

#### 3. 在目标模型上执行攻击

现在，来到重要一步：在*目标模型*的输出上使用已训练的`attack_model`。为了正确评估攻击对*目标*的成功，我们需要一些数据点相对于$D_{target\_train}$的成员身份的真实信息。这通常在我们可以控制目标模型训练的研究设置中是可行的，但在真实的黑盒 (black box)场景中则不然。

假设我们拥有`target_model`，并且可以为了评估目的而访问其原始训练集`target_train_data`和测试集`target_test_data`。

```python
# 获取已知成员（来自其训练集）的目标模型输出
target_member_outputs = target_model.predict(target_train_data.x)

# 获取已知非成员（来自其测试集）的目标模型输出
target_non_member_outputs = target_model.predict(target_test_data.x)

# 组合这些输出，形成针对目标的攻击模型的测试集
attack_test_target_X = np.concatenate((target_member_outputs, target_non_member_outputs))
attack_test_target_y = np.concatenate([np.ones(len(target_member_outputs)), np.zeros(len(target_non_member_outputs))])

# 使用在影子模型输出上拟合的相同标准化器进行标准化
attack_test_target_X_scaled = scaler.transform(attack_test_target_X) # 使用 transform，而非 fit_transform

# 使用攻击模型预测成员身份
final_predictions = attack_model.predict(attack_test_target_X_scaled)
final_probabilities = attack_model.predict_proba(attack_test_target_X_scaled)[:, 1] # 是成员的概率

# 评估攻击对目标模型的成功率
print("\n成员身份推断攻击表现（在目标模型数据上）：")
print(f"准确率: {accuracy_score(attack_test_target_y, final_predictions):.4f}")
print(classification_report(attack_test_target_y, final_predictions))

# 你也可以使用 final_probabilities 和 attack_test_target_y 计算 AUC
# from sklearn.metrics import roc_auc_score
# print(f"AUC: {roc_auc_score(attack_test_target_y, final_probabilities):.4f}")
```

### 成功评估与解读

这里的主要衡量指标是`attack_model`在目标模型数据上的最终准确率（或AUC）。

- **准确率接近0.5 (50%)：** 攻击的表现与随机猜测相似。这表明模型通过其输出泄露的成员身份信息很少，或者影子模型不是好的代理，或者攻击模型无法捕获到区分模式。
- **准确率显著高于0.5：** 攻击成功地区分了成员和非成员，且表现优于随机。这表明存在隐私泄露。准确率越高，泄露越严重。准确率为1.0将意味着完美的推断。

“成员”类别的高精度表示当攻击预测为“成员”时，其很可能是正确的。高召回率表示攻击识别出了大部分真实成员。



![MIA 表现](plots/4163-0.json)



> 示例评估结果显示攻击表现明显优于随机猜测，这表明目标模型可能存在成员身份信息泄露。

### 考量因素

- **影子模型忠实度：** 这次攻击的成功在很大程度上取决于影子模型如何良好地复刻目标模型的属性（架构、训练数据分布、超参数 (parameter) (hyperparameter)）。不匹配会降低攻击性能。
- **输出类型：** 使用完整的概率/置信度向量 (vector)通常比仅使用预测标签或预测标签的置信度效果更好。Logit值也可以是攻击模型的有效特征。
- **数据要求：** 攻击者需要足够与目标训练数据相似的数据来训练有效的影子模型。
- **防御对策：** 差分隐私（我们已提及其与推断攻击的关联）或输出扰动（向预测添加噪声）等技术可以通过模糊模型在成员和非成员之间行为的差异，使成员身份推断变得更难。目标模型训练期间的正则化 (regularization)技术也可能无意中减少信息泄露。

这种实际操作的视角表明，成员身份推断并非仅仅是理论上的担忧。在获得适当访问权限和数据的情况下，攻击者可以构建模型来探查机器学习 (machine learning)系统的训练历史，从而引发重要的隐私问题。弄清楚如何实施这些攻击是评估模型脆弱性并设计有效防御的第一步。

## 参考资料

- [Membership Inference Attacks Against Machine Learning Models](https://ieeexplore.ieee.org/document/7943037) — Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov (2017)
  Journal: 2017 IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 371-385; DOI: [10.1109/SP.2017.36](https://doi.org/10.1109/SP.2017.36)
  这篇开创性论文介绍了成员推断攻击的概念，并提出了影子模型技术，这是本节讨论的核心方法。
- [The Secret Sharer: Measuring and Improving Membership Inference Attack Ability](https://dl.acm.org/doi/10.1145/3548606.3560613) — Nicholas Carlini, Chiyuan Zhang, Joseph P. Mitchell, Andreas Terzis, Stephanie E. Zick, Thomas Steinke, Nicholas Papernot (2022)
  Journal: Proceedings of the 2022 ACM Conference on Computer and Communications Security (CCS); Publisher: Association for Computing Machinery; Pages: 1941–1955; DOI: [10.1145/3548606.3560613](https://doi.org/10.1145/3548606.3560613)
  对成员推断攻击进行了更深入和最新的分析，探讨了影响其有效性的因素以及提高攻击能力的方法。
