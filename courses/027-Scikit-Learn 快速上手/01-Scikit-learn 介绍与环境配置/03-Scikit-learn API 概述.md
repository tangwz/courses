# Scikit-learn API 概述

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-1-intro-setup-scikit-learn/overview-scikit-learn-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Scikit-learn 的广泛应用部分归因于其高度一致且精心设计的应用程序接口（API）。一旦您掌握了其基本结构和约定，应用不同算法或预处理步骤通常感觉很直接，且所需代码改动很少。这种一致性提升了构建机器学习 (machine learning)模型的效率。

Scikit-learn API 的核心围绕几种基本对象类型及其相关方法。我们来了解主要组成部分。

### 评估器：基础对象

Scikit-learn 的中心是 `Estimator` 对象。库中几乎所有内容，无论是用于分类或回归的模型，还是用于转换数据的工具，都继承自这个基础类。评估器的决定性特点是其 `fit()` 方法。

- **`fit(X, y=None)`**: 这是最主要的方法。其目的是根据训练数据调整评估器的内部状态。
  - `X`: 表示输入数据，通常是一个二维 NumPy 数组或 Pandas DataFrame。行对应样本，列对应特征。
  - `y`: 表示目标值（用于监督学习 (supervised learning)）。通常是一个一维 NumPy 数组或 Pandas Series，包含标签（用于分类）或连续值（用于回归）。对于无监督评估器，`y` 通常被省略或忽略。
- **实例化**: 评估器像任何 Python 对象一样被实例化，通常在创建时设置配置参数 (parameter)（超参数 (hyperparameter)）。例如：`model = LinearRegression(fit_intercept=True)`。
- **学习到的参数**: 调用 `fit()` 后，评估器将拟合过程的结果存储在通常以 \_ 结尾的属性中（例如，`model.coef_`，`scaler.mean_`）。这些属性表示评估器从数据中学习到的内容。

### 预测器：进行预测

预测器是专门用于监督学习 (supervised learning)任务（分类和回归）的评估器。它们继承了 Estimator 基础类的 `fit()` 方法以从数据中学习。此外，它们提供了对新的、未见过的数据进行预测的方法。

- **`predict(X)`**: 评估器拟合完成后，此方法接受新的输入数据 `X`（与训练数据具有相同的特征结构）并根据学习到的模型返回预测的目标值。
  - 对于回归，它返回连续值。
  - 对于分类，它返回预测的类别标签。
- **`score(X, y)`**: 大多数预测器也具有 `score()` 方法，该方法评估模型在给定数据集 `X` 上，使用真实标签 `y` 的性能。它返回一个适合该任务的默认评估指标（例如，回归的 R-squared，分类的平均准确率）。

预测器的常见例子包括 `LinearRegression`、`LogisticRegression`、`KNeighborsClassifier` 和 `SVC`。

### 转换器：修改数据

转换器是用于数据预处理、特征提取或特征选择的评估器。它们也使用 `fit()` 方法从数据中学习（例如，学习用于缩放的均值和标准差）。但它们主要目标是修改或筛选输入数据。

- **`transform(X)`**: 此方法接受输入数据 `X` 并应用学习到的转换（在 `fit()` 期间确定），返回修改后的数据集。例如，缩放器可能会对特征进行中心化和缩放。
- **`fit_transform(X, y=None)`**: 为了方便和计算效率，转换器通常提供 `fit_transform()` 方法。此方法在*相同*数据上一步完成拟合和转换。这在将预处理步骤应用于训练集时尤其有用，因为它确保转换参数 (parameter)仅从训练数据中学习，然后才应用。

转换器的例子包括 `StandardScaler`（用于特征缩放）、`OneHotEncoder`（用于转换分类特征）和 `SimpleImputer`（用于处理缺失值）。

### API 一致性的实践

这种设计的优势在于其统一性。考虑以下步骤：

1. **实例化**: 创建一个评估器（无论是模型还是预处理器）的实例，并设置超参数 (parameter) (hyperparameter)。

   ```python
   # 实例化示例
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression

   scaler = StandardScaler()
   model = LogisticRegression(C=1.0)
   ```
2. **拟合**: 使用 `fit()` 方法和训练数据（`X_train`、`y_train`）来学习参数。对于转换器，通常只需要 `X_train`。

   ```python
   # 在训练特征上拟合缩放器
   scaler.fit(X_train)

   # 在训练特征和标签上拟合模型
   model.fit(X_train_scaled, y_train) # 假设 X_train_scaled 是缩放器的输出
   ```
3. **转换/预测**: 将拟合好的对象应用于新数据。对转换器使用 `transform()`，对预测器使用 `predict()`。

   ```python
   # 将拟合好的缩放器应用于训练和测试数据
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # 对缩放后的测试数据进行预测
   predictions = model.predict(X_test_scaled)
   ```

请注意核心方法（`fit`、`transform`、`predict`）是如何一致地用于不同类型的对象。这使得尝试不同的预处理技术或算法变得容易得多。您通常只需少量修改工作流代码，即可用一个评估器替换另一个兼容的评估器。这种结构也是构建 Scikit-learn 流水线（Pipelines）的基础，它将多个步骤连接在一起，我们将在后续章节中看到。

> 基本工作流程，展示了学习阶段（`fit`）创建评估器内部状态，与应用阶段（`predict` 或 `transform`）在新数据上使用该状态之间的分离。

了解这些核心 API 概念——评估器、预测器、转换器以及 `fit`、`predict`、`transform` 方法——为高效使用 Scikit-learn 提供了坚实的基础。在接下来的章节中，我们将了解这些方法需要的数据格式，并了解库中包含的数据集。

## 参考资料

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Scikit-learn developers (2024)
  一个主要资源，解释了该库的架构和API元素，包括Estimator、Predictor和Transformer。
- [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html) — Fabian Pedregosa, GaÃ«l Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Ã‰douard Duchesnay (2011)
  Journal: Journal of Machine Learning Research; Volume: 12; Pages: 2825−2830
  介绍Scikit-learn库及其设计原则的奠基性学术论文。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本被广泛认可的图书，提供了Scikit-learn API和组件的详细说明及实用示例。

---

[上一节](02-%E5%AE%89%E8%A3%85%E4%B8%8E%E7%8E%AF%E5%A2%83%E8%AE%BE%E7%BD%AE.md) · [下一节](04-Scikit-learn%20%E4%B8%AD%E7%9A%84%E6%95%B0%E6%8D%AE%E8%A1%A8%E7%A4%BA.md)
