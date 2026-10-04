---
course: "evaluating-synthetic-data-quality"
chapter: "privacy-assessment-techniques"
lesson: "attribute-inference-attacks"
sourceId: 5687
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-4-privacy-assessment-techniques/attribute-inference-attacks"
title: "属性推断攻击"
description: "评估从合成记录中推断敏感属性的风险。"
order: 3
plots: []
sourceHash: "757cf5cf31ac03f0d0bd71fceaa4d62985d302ce07cbfe5d5d1e44b9a93d0db4"
sourceCorrections: []
---

成员推断攻击（MIAs）旨在确定*某个特定个人的记录*是否属于原始训练数据，而属性推断攻击（AIAs）则解决一个不同但同样重要的隐私问题：攻击者能否通过借助合成数据，推断出原始数据集中所代表个体的敏感属性？

即使某个个体的准确记录未被复制，且其成员身份无法通过成员推断攻击确认，合成数据仍可能泄露统计模式，使攻击者在已知一些非敏感属性（如年龄、邮政编码或职业）的情况下，推断出敏感信息（如收入、健康状况或政治立场）。风险在于，如果合成数据使得这种推断比其他方式（例如，如果攻击者可以访问，仅使用公开信息或真实数据本身）更简单或更准确。

### 属性推断威胁模型

在典型的属性推断攻击场景中，攻击者拥有：

1. **合成数据集 ($D_{syn}$):** 这是攻击的主要工具。
2. **辅助信息:** 攻击者知道目标个体（或一组个体）的*一些*非敏感属性，这些个体的记录可能在原始数据集 $D_{real}$ 中。我们将这些已知属性表示为 $X_{known}$。
3. **目标属性:** 攻击者旨在预测目标个体的一个特定敏感属性，$Y_{sensitive}$。

核心问题是：与没有 $D_{syn}$ 或仅使用 $D_{real}$（假设攻击者以某种方式获得了部分访问权限）相比，$D_{syn}$ 是否能使攻击者以明显更高的准确性从 $X_{known}$ 预测 $Y_{sensitive}$？

### 方法：训练攻击模型

属性推断攻击通常通过训练一个机器学习 (machine learning)模型（“攻击模型”）来预测敏感属性来实现。该过程通常遵循以下步骤：

1. **确定目标和特征:** 定义要预测的敏感属性 ($Y_{sensitive}$) 和假设攻击者拥有的非敏感属性 ($X_{known}$)。
2. **在合成数据上训练攻击模型:** 使用 $X_{known}$ 作为特征，$Y_{sensitive}$ 作为目标变量，仅使用合成数据集 $D_{syn}$ 来训练一个合适的分类或回归模型（例如，逻辑回归、决策树、神经网络 (neural network)）。
   
   $$
   \text{攻击模型} = \text{训练}(X_{known\_syn}, Y_{sensitive\_syn}) \quad \text{其中 } (X_{known\_syn}, Y_{sensitive\_syn}) \in D_{syn}
   $$
   
3. **在真实数据上评估攻击模型:** 在*原始*数据集 $D_{real}$ 的记录上（具体来说，是一个未用于生成 $D_{syn}$ 的保留测试集）测试训练好的`攻击模型`。其性能（例如，准确率、AUC、F1分数）表明了利用从合成数据中学到的模式，能多好地推断真实个体的属性。
   
   $$
   \text{性能}_{syn \to real} = \text{评估}(\text{攻击模型}, X_{known\_real}, Y_{sensitive\_real}) \quad \text{其中 } (X_{known\_real}, Y_{sensitive\_real}) \in D_{real\_test}
   $$
   
4. **建立基准:** 将 $\text{性能}_{syn \to real}$ 与相关基准进行比较：
   - **基准1（真实对真实）:** 仅在真实数据的分区上训练和测试一个相同的攻击模型。这代表了如果攻击者以某种方式访问了原始数据，可能达到的最大推断能力。
     
     $$
     \text{攻击模型}_{real} = \text{训练}(X_{known\_real\_train}, Y_{sensitive\_real\_train})
     $$
     
     
     $$
     \text{性能}_{real \to real} = \text{评估}(\text{攻击模型}_{real}, X_{known\_real\_test}, Y_{sensitive\_real\_test})
     $$
     
   - **基准2（先验概率/猜测）:** 仅通过猜测 $Y_{sensitive}$ 的最常见类别或使用其他先验知识，而不借助数据中存在的相关性所达到的性能。

### 量化 (quantization)属性推断风险

隐私风险并非孤立地通过 $\text{性能}_{syn \to real}$ 衡量。高性能可能仅仅意味着该属性通常很容易预测。*归因于合成数据的风险*，更好地通过攻击者使用合成数据时的性能与基准性能之间的*差异*或*比率*来理解。

- **高风险:** 如果 $\text{性能}_{syn \to real}$ 明显高于基准2并接近 $\text{性能}_{real \to real}$，则意味着合成数据有效地捕获并泄露了原始数据中存在的敏感相关性。
- **低风险:** 如果 $\text{性能}_{syn \to real}$ 接近基准2，则合成数据对攻击者没有提供太多帮助。

差值 $\Delta_{\text{AIA}} = \text{性能}_{syn \to real} - \text{性能}_{\text{baseline}}$ 量化了攻击者从合成数据中获得的优势。更大的正 $\Delta_{\text{AIA}}$ 表明更高的隐私风险。

> 该图说明了属性推断攻击的流程，即使用合成数据训练模型，然后根据真实数据对其进行评估以评估风险。

### 实现范例（伪代码）

让我们考虑一个场景，其中攻击者希望使用合成数据集中的 `age`、`zip_code` 和 `occupation` 来预测一个二元敏感属性 `has_condition`（1或0）。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 假设 D_real 和 D_syn 是 pandas DataFrame
# D_real: 原始数据
# D_syn: 合成数据

# 攻击者已知的特征和敏感目标属性
known_features = ['age', 'zip_code', 'occupation_encoded'] # 假设 occupation 是独热编码
sensitive_attribute = 'has_condition'

# --- 使用合成数据进行攻击 ---
# 1. 准备用于训练的合成数据
X_syn_train = D_syn[known_features]
y_syn_train = D_syn[sensitive_attribute]

# 2. 在合成数据上训练攻击模型
attacker_model_syn = RandomForestClassifier(n_estimators=100, random_state=42)
attacker_model_syn.fit(X_syn_train, y_syn_train)

# 3. 准备真实数据测试集（确保未用于合成）
# 假设 D_real_test 是 D_real 的一部分保留数据
X_real_test = D_real_test[known_features]
y_real_test = D_real_test[sensitive_attribute]

# 4. 评估在合成数据上训练的模型在真实测试数据上的表现
y_pred_syn_to_real = attacker_model_syn.predict(X_real_test)
accuracy_syn_to_real = accuracy_score(y_real_test, y_pred_syn_to_real)

print(f"攻击者准确率（在合成数据上训练，在真实数据上测试）：{accuracy_syn_to_real:.4f}")

# --- 基准1：使用真实数据进行攻击 ---
# （如果尚未完成，需要将 D_real 分割成训练/测试集）
# 假设 D_real_train 和 D_real_test 可用
X_real_train = D_real_train[known_features]
y_real_train = D_real_train[sensitive_attribute]

attacker_model_real = RandomForestClassifier(n_estimators=100, random_state=42)
attacker_model_real.fit(X_real_train, y_real_train)

y_pred_real_to_real = attacker_model_real.predict(X_real_test)
accuracy_real_to_real = accuracy_score(y_real_test, y_pred_real_to_real)

print(f"基准准确率（在真实数据上训练，在真实数据上测试）：{accuracy_real_to_real:.4f}")

# --- 基准2：多数类别猜测 ---
majority_class = y_real_test.mode()[0]
accuracy_majority = accuracy_score(y_real_test, [majority_class] * len(y_real_test))

print(f"基准准确率（多数类别猜测）：{accuracy_majority:.4f}")

# --- 评估风险 ---
advantage_over_majority = accuracy_syn_to_real - accuracy_majority
leakage_ratio = (accuracy_syn_to_real - accuracy_majority) / (accuracy_real_to_real - accuracy_majority) if (accuracy_real_to_real - accuracy_majority) > 0 else 0

print(f"攻击者相对于多数类别基准的优势：{advantage_over_majority:.4f}")
print(f"属性泄露比率（相对于真实数据）：{leakage_ratio:.4f}")
```

结果解读：

- 如果 `accuracy_syn_to_real` 明显高于 `accuracy_majority`，则合成数据为推断提供了有用信息。
- 如果 `leakage_ratio` 接近1.0，则合成数据泄露的关于属性相关性的信息量几乎与真实数据本身一样多。
- 如果 `leakage_ratio` 接近0.0，则与从真实数据中可学到的信息相比，合成数据提供的优势很小。

属性推断攻击提供了一种实际方法，用于评估特定类型的隐私泄露。高属性推断攻击风险表明，敏感属性和非敏感属性之间的关系在合成数据中反映得过于紧密，可能损害原始数据集中个体的隐私。此项评估是针对任何旨在发布或共享的合成数据集进行全面隐私评估的重要组成部分。

## 参考资料

- [The Algorithmic Foundations of Differential Privacy](http://dx.doi.org/10.1561/0400000042) — Cynthia Dwork, Aaron Roth (2014)
  Journal: Foundations and Trends® in Theoretical Computer Science; Publisher: Now Publishers Inc.; Volume: 9; Pages: 211-407; DOI: [10.1561/0400000042](https://doi.org/10.1561/0400000042)
  关于差分隐私的权威著作，阐释了数据发布中抵御各种推断攻击的理论原理。
