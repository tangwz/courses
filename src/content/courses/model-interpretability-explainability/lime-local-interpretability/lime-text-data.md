---
course: "model-interpretability-explainability"
chapter: "lime-local-interpretability"
lesson: "lime-text-data"
sourceId: 4063
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-2-lime-local-interpretability/lime-text-data"
title: "LIME 在文本数据上的应用"
description: "了解 LIME 在解释文本分类模型预测结果方面的应用。"
order: 4
plots: []
sourceHash: "3ce35d1a499f86bbd0b79961aed13a4862f5b313952588e74b8b2da09c5e5ca5"
sourceCorrections: []
---

与表格数据相比，解释处理文本数据的模型预测结果具有独特的挑战。我们如何确定句子或文档的哪些部分使得情感分类器、主题模型或垃圾邮件检测器得出其结论？LIME 提供了一种方法来查看单个文本预测背后的推理 (inference)过程。

由于 LIME 与模型无关，它不需要了解底层文本模型的架构，无论是带有逻辑回归的传统词袋模型、复杂的循环神经网络 (neural network) (RNN)，还是大型Transformer模型。它仅通过模型的预测函数与模型进行交互。

### 扰动文本数据

LIME 的核心思想保持不变：生成我们想要解释的实例的邻近项，获取黑盒 (black box)模型对这些邻近项的预测结果，并在这个局部数据集上训练一个简单、可解释的模型。但是，我们如何为一个文本片段创建“邻近项”呢？

LIME 中文本扰动最常见的方法是**从原始文本实例中随机移除词语**。

考虑以下句子：“This is a reasonably priced and effective product.”

为了为 LIME 生成邻近项，我们可能会创建如下变体：

1. “This is reasonably priced effective product.” （移除了 'a' 和 'and'）
2. “reasonably priced effective product.” （移除了 'This'、'is'、'a' 和 'and'）
3. “This reasonably priced product.” （移除了 'is'、'a'、'and' 和 'effective'）
4. “This is a product.” （移除了 'reasonably'、'priced'、'and' 和 'effective'）
   ……等等。

每个扰动后的句子都是原始实例在其由词语存在性定义的邻域中的一个邻近项。

### 局部模型的特征表示

我们如何表示这些扰动后的文本实例，以便简单的线性模型可以从中学习？LIME 通常为每个扰动后的实例使用一个**二进制特征向量 (vector)**。这个向量的维度与*原始*文本实例中独特词语（标记 (token)）的数量相同。向量中的每个位置都对应于原始文本中的一个特定词语。

向量中的 '1' 表示对应的词语在扰动后的句子中*存在*，而 '0' 表示它*缺失*。

对于我们的例子“This is a reasonably priced and effective product.”，独特词语是 {This, is, a, reasonably, priced, and, effective, product}。假设这是我们的词汇映射。

- 原始：“This is a reasonably priced and effective product.” -> `[1, 1, 1, 1, 1, 1, 1, 1]`
- 邻近项 1：“This is reasonably priced effective product.” -> `[1, 1, 0, 1, 1, 0, 1, 1]` （移除了词语 'a' 和 'and'）
- 邻近项 3：“This reasonably priced product.” -> `[1, 0, 0, 1, 1, 0, 0, 1]` （移除了词语 'is'、'a'、'and' 和 'effective'）

### 训练局部代理模型

LIME 获取这些二进制特征向量 (vector)以及原始黑盒 (black box)模型对每个扰动后句子的对应预测。然后，它将一个可解释模型（通常是岭回归或 Lasso 线性模型）拟合到这些局部数据上。

目标是为简单模型 $g$ 找到权重 (weight) $w$，使得对于接近原始实例 $z$ 的扰动实例 $z'$，$g(z') \approx f(z')$ 成立，其中 $f$ 是复杂的黑盒模型。$g$ 使用的特征是二进制词语存在性指示器。


$$
g(z') = \sum_{i=1}^{d'} w_i z'_i
$$


这里，$d'$ 是原始实例中独特词语的数量，$z'_i$ 在第 $i$ 个词语在扰动 $z'$ 中存在时为 1，否则为 0，$w_i$ 是第 $i$ 个词语学习到的权重。这些权重 $w_i$ 表示每个词语对被解释的特定实例预测结果的局部重要性。正权重表明该词语有助于预测的类别（或增加回归值），而负权重则表明它使预测偏离该类别（或降低值）。

### 解释 LIME 文本解释结果

LIME 针对文本的输出通常通过高亮显示原始文本中的词语来可视化。高亮显示的颜色和强度对应于局部代理模型学习到的权重 (weight) ($w_i$) 的符号和大小。

例如，在一个情感分类任务中，模型对评论“This is a reasonably priced and effective product.”预测为“Positive”时：

- 像“reasonably”、“priced”和“effective”这样的词语可能会被高亮显示为绿色（正向贡献）。
- 如果句子是“This product is not effective”，词语“not”可能会被高亮显示为红色（对“Positive”的负向贡献，或对“Negative”的正向贡献）。

这种可视化表示使得人们可以直观地看到哪些词语影响了模型对该特定输入文本的决策。

### 使用 `lime` 库实现

`lime` Python 库提供了专门为文本数据设计的工具。主要类是 `lime.lime_text.LimeTextExplainer`。

```python
# 假设 'model' 是您训练好的文本分类器
# 它应该有一个 predict_proba 方法，接受一个字符串列表
# 并返回每个类别的概率（例如，形状为 [样本数, 类别数]）

import lime
import lime.lime_text

# 要解释的文本实例
text_instance = "This is a reasonably priced and effective product."

# 1. 创建解释器对象
explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive']) # 替换为您的类别名称

# 2. 定义预测器函数
# 此函数接受一个扰动后的文本列表（字符串）
# 并返回模型对它们的概率预测
def predictor(texts):
  # 如果需要，预处理文本（例如，分词、向量化）
  # 与您的模型输入格式兼容
  # processed_texts = preprocess(texts)
  # return model.predict_proba(processed_texts)
  # 占位符：替换为您的实际模型预测逻辑
  import numpy as np
  # 模拟一个模型，如果存在 'effective'，则分配更高的“Positive”概率
  probs = []
  for t in texts:
      if 'effective' in t or 'reasonably' in t:
          probs.append([0.1, 0.9]) # “Positive”概率高
      elif 'not' in t:
           probs.append([0.8, 0.2]) # “Negative”概率高
      else:
          probs.append([0.5, 0.5]) # 中性
  return np.array(probs)


# 3. 生成解释
explanation = explainer.explain_instance(
    text_instance,
    predictor,
    num_features=6, # 解释中显示的词语数量
    num_samples=1000 # 生成的扰动数量
)

# 4. 可视化解释（通常在 Notebook 中完成）
explanation.show_in_notebook(text=True)
# 或打印到控制台
# print(explanation.as_list())
```

这段代码片段概述了典型的工作流程：创建一个解释器，定义一个接受扰动文本字符串并返回模型预测概率的函数，然后调用 `explain_instance`。`num_features` 参数 (parameter)控制解释中显示的词语数量，而 `num_samples` 控制 LIME 为构建局部模型生成的扰动数量（更多样本可以带来更稳定的解释，但耗时更长）。

### 文本处理注意事项

尽管强大，将 LIME 应用于文本有一些方面需要考虑：

- **扰动方法：** 简单地移除词语可能会创建语法不正确或无意义的句子。这带来的影响取决于底层黑盒 (black box)模型的鲁棒性。其他扰动策略也存在，但在标准的 LIME 实现中不那么常见。
- **分词 (tokenization)：** 文本如何分割成词语（标记 (token)）会影响 LIME 考虑哪些特征。使用不同的分词器 (tokenizer)可能导致不同的解释结果。
- **局部性定义：** 通过词语移除定义的“邻域”可能并非总能很好地捕捉语义相似性。
- **稳定性：** 由于扰动的随机性，解释结果在多次生成时有时会略有不同。增加 `num_samples` 通常有助于提高稳定性。

尽管存在这些方面，LIME 提供了一种有价值的模型无关方法，用于理解文本模型为何做出特定的预测，它高亮显示了输入文本中有影响力的词语。这对于调试模型行为、建立用户信任和确保公平性特别有用。

## 参考资料

- [“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) — Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin (2016)
  Journal: KDD '16: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Pages: 1135–1144; DOI: [10.48550/arXiv.1602.04938](https://doi.org/10.48550/arXiv.1602.04938)
  介绍了LIME框架，包括其在文本数据上的应用、扰动方法和局部可解释模型的概念。
- [\`lime\` documentation](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_text) — Marco Tulio Ribeiro (2021)
  提供实现LIME的实用指南和API参考，特别是使用`lime.lime_text.LimeTextExplainer`类处理文本数据。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/lime.html) — Christoph Molnar (2024)
  提供关于LIME的全面章节，解释其工作原理并用示例说明其在文本数据上的应用，为局部可解释性提供了更广阔的背景。
