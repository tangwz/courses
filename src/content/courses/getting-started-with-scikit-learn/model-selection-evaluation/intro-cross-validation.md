---
course: "getting-started-with-scikit-learn"
chapter: "model-selection-evaluation"
lesson: "intro-cross-validation"
sourceId: 1891
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-5-model-selection-evaluation/intro-cross-validation"
title: "交叉验证简介"
description: "为何交叉验证能提供更可靠的模型性能评估。"
order: 3
plots: []
sourceHash: "7225b8b777ecceb890f36775dbfbda3896de015ef9bf18e42bf2b63c189cb74a"
sourceCorrections: []
---

将数据划分为训练集和测试集，通常使用`train_test_split`等函数，有助于对过拟合 (overfitting)进行基本验证。模型在训练集上进行训练，并在未见过的测试集上进行评估。然而，单次划分得到的性能分数可能并非完全可靠。如果以不同方式划分数据，例如使用不同的`random_state`，可能会得到一个明显不同的评估分数。这种变动性是一个需要重视的问题。性能指标可能过于乐观或悲观，仅仅是因为在特定划分中，某些具体数据点恰好分到了训练集或测试集中。

这就是交叉验证的作用所在。它是一种更全面的模型评估方法，能对模型在未见过数据上的表现提供更稳定和可靠的估计。交叉验证不依赖于单次训练-测试划分，而是系统地创建数据的多次划分，并计算这些划分的平均评估分数。可以将其看作是多次重复训练-测试划分过程，每次使用不同的数据子集，然后合并结果。

交叉验证的主要好处是减少与单次训练-测试划分相关的方差。通过对数据多个不同划分的性能取平均，我们能够消除因获得特别“幸运”或“不幸”的划分所带来的影响。最终得到的平均分数能更好地反映模型的潜在泛化能力。

此外，交叉验证能更充分地使用可用数据。在简单的训练-测试划分中（例如，80%训练，20%测试），模型从未在留作测试的20%数据上进行训练。而使用交叉验证技术，通常每个数据点都有机会恰好一次出现在测试集中，同时也在其他迭代中用于训练。当处理规模不大的数据集时，这特别有益，因为它能让数据被充分用于训练（在某个阶段）和验证（在某个阶段）。

一般步骤包括将原始数据集划分为若干子集，这些子集通常称为“折”。模型随后进行迭代训练。在每次迭代中，会留出一个折作为验证集，模型则在剩余的折上进行训练。这个过程会重复进行，直到每个折都恰好一次充当过验证集。然后，将每次迭代的性能分数收集起来，通常取平均值以得到最终的交叉验证分数。

这套评估方式对于在不同模型或同一模型的不同超参数 (parameter) (hyperparameter)配置之间进行可靠比较非常重要。它有助于确保我们选择的模型或参数能够持续良好表现，而不仅仅是在某个特定的随机数据划分上。接下来各部分将详细介绍Scikit-learn中实现的具体交叉验证策略，例如K折交叉验证。

## 参考资料

- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — Scikit-learn developers (2024)
  提供Scikit-learn中实现的交叉验证策略的实际示例和详细信息。
- [An Introduction to Statistical Learning: With Applications in Python](https://link.springer.com/book/10.1007/978-3-031-38747-0) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  对交叉验证技术在模型评估中的应用提供了易于理解且全面的解释。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEiQr1LsMh1ES4z-3Sjc3ZN_xAnSHt259iAxJXZNMUiT40IPBeqxFCQBIViaT-UakCx3tMkPgPQYk0ACB2bh6lc2JhlE7SNLcvowaciYX8QK97IZBufAOeq5x7MXC6ggCJXJvImY7jN29Oxk7bZ5oQDYVWcZjYxk58e4m78JB-azMvpIWSTztCelSW9D7cDhuNqeHrTJy2JxvygppU=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.
  通过实际的Python和Scikit-learn代码示例介绍交叉验证的概念。
