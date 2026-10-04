---
course: "deep-learning-fundamentals-keras"
chapter: "improving-model-performance-workflow"
lesson: "introduction-tensorboard"
sourceId: 4982
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow/introduction-tensorboard"
title: "TensorBoard 简介"
description: "在 Keras 中使用 TensorBoard 可视化训练进度、模型图和其他指标。"
order: 6
plots: ["plots/4982-0.json"]
sourceHash: "99940c9bcca5a1e1c93cf80068ea850dca5ea77c0ae1a91b37d392aa97db5292"
sourceCorrections: []
---

仅凭控制台输出来监控训练过程可能存在局限，尤其是对于复杂模型或长时间训练。尽管每个时期（epoch）结束时打印的指标能提供瞬时视图，但要了解训练的*动态变化*、指标随时间如何变化、模型参数 (parameter)如何演变以及网络结构，则需要更精密的工具。TensorBoard 是为此目的专门设计的强大可视化工具包，它能以图形方式帮助您观察 TensorFlow 和 Keras 模型训练。

可以将 TensorBoard 视为机器学习 (machine learning)实验的仪表盘。它读取训练期间生成的日志文件，并在交互式网页界面中显示数据。这使您能够直观地跟踪和比较不同的训练运行，从而更有效地调试、优化和理解您的模型。

### TensorBoard 的特点

尽管 TensorBoard 提供了多种功能，但有几项对我们已经讨论过的工作流程特别有用：

1. **跟踪指标随时间的变化：** 可视化训练集和验证集在各个时期上的损失（loss）和准确率（accuracy）等标量值。这对于发现趋势、收敛速度以及过拟合 (overfitting)或欠拟合 (underfitting)的迹象必不可少。
2. **模型图的可视化：** 以图形方式查看您的 Keras 模型结构。这有助于确认层连接是否符合预期，尤其是在使用函数式 API 构建更复杂的架构时。
3. **查看权重 (weight)、偏差和激活值的直方图：** 观察参数 (parameter)值或层输出随时间的变化分布。剧烈变化或分布坍缩到零可能提示潜在问题，例如梯度消失或初始化选择不当。
4. **比较运行：** TensorBoard 允许您叠加来自不同训练实验的指标，从而轻松评估不同超参数 (hyperparameter)、优化器或模型架构的影响。

### 将 TensorBoard 与 Keras 训练结合

由于 `TensorBoard` 回调函数，在 Keras 中使用 TensorBoard 十分简单。这个回调函数属于 `keras.callbacks` 的一部分，它会在 `model.fit()` 过程中自动记录指定的信息。

要使用它，首先导入回调函数：

```python
from keras.callbacks import TensorBoard
import datetime # 可选：用于创建唯一的日志目录
```

接着，实例化回调函数，通常需要提供一个 `log_dir` 参数 (parameter)。最佳做法是为每次训练运行创建一个唯一的子目录，通常使用时间戳，以便保持实验的条理。

```python
# 定义日志路径
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建 TensorBoard 回调实例
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# histogram_freq=1 会在每个时期记录直方图（可能消耗更多资源）
```

最后，在调用 `model.fit()` 时，将此回调实例添加到 `callbacks` 列表中：

```python
# 假设模型已编译，并且 x_train、y_train、x_val、y_val 已准备好

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard_callback]) # 在此处传递回调函数
```

在训练期间，Keras 会将包含记录数据（标量、图、如果启用则包含直方图）的事件文件写入指定的 `log_dir`。

### 启动和使用 TensorBoard

训练开始（或结束）并生成日志文件后，您可以从终端启动 TensorBoard 界面。导航到*包含*日志目录的父目录（或更高级别的目录）并运行：

```bash
tensorboard --logdir logs/fit
```

将 `logs/fit` 替换为包含特定运行文件夹的目录路径（例如，如果您的运行位于 `logs/fit/RUN1`、`logs/fit/RUN2` 中，则使用 `logs`）。

TensorBoard 将启动一个本地 Web 服务器并打印 URL（通常是 `http://localhost:6006`）。在您的网页浏览器中打开此 URL。

在 TensorBoard 内部，您会看到几个选项卡：

- **标量（SCALARS）：** 显示损失和准确率等指标随时期变化的图表。您可以选择不同的运行进行比较。这通常是您查看训练进度和发现过拟合 (overfitting)等问题（即验证损失开始增加而训练损失继续减少）的首要位置。
- **图（GRAPHS）：** 显示您模型的计算图。您可以查看各层及其连接。
- **分布与直方图（DISTRIBUTIONS & HISTOGRAMS）：** 显示权重 (weight)、偏差或激活值分布随时间如何变化。对调试很有帮助。



![训练和验证损失随时期变化](plots/4982-0.json)



> TensorBoard SCALARS 面板中的典型图表，显示训练损失持续下降，而验证损失在第 25 个时期后开始上升，这表明过拟合的出现。

通过可视化训练指标和模型结构，TensorBoard 提供了远超简单打印语句的宝贵反馈。它帮助您加深对训练过程的理解，并就模型架构、超参数 (parameter) (hyperparameter)和正则化 (regularization)策略做出更明智的决策，从而直接支持提升模型性能和优化工作流程的目标。

## 参考资料

- [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/get_started) — TensorFlow Authors (2024)
  提供了TensorBoard的全面介绍，涵盖了其功能和与TensorFlow及Keras结合使用的基本方法。
- [Keras Callbacks API](https://keras.io/api/callbacks/) — Keras Team (2024)
  详细介绍了Keras中的`TensorBoard`回调函数，解释了其参数以及如何将其集成到模型训练中。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，包含关于使用Keras和TensorFlow构建和监控深度学习模型的详细章节，其中广泛涵盖了TensorBoard。
