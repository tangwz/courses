# 实践：调整RNN模型

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-10-evaluating-tuning-sequence-models/practice-tuning-rnn-model)

[返回章节目录](README.md) · [返回课程目录](../README.md)

超参数 (parameter) (hyperparameter)和正则化 (regularization)技巧对序列模型性能有显著影响。本练习将逐步演示如何调整RNN模型，并应用各种技巧和性能度量标准。

我们将假设你有一个基准序列模型，也许是我们在第七章构建的、使用LSTM或GRU的情感分析分类器。我们的目的不一定是要为一个特定数据集找出绝对*最优*的模型（因为这通常需要大量的计算），而是为了说明调整的*过程*以及不同更改如何影响结果。

**1. 建立你的基准**

首先，你需要一个起点。在你的训练数据上训练你的初始模型（例如，一个带有默认参数的单层LSTM），并在单独的验证集上进行评估。记录与你的任务相关的度量标准。对于情感分析，这可能包括验证准确率和F1分数。让我们设想我们的基准模型取得了以下成果：

- 验证准确率：78%
- 验证F1分数：0.77

这个基准为我们提供了进行调整时进行比较的参照。请记住使用验证集进行调整，以避免对测试集过拟合 (overfitting)，测试集只应用于最终评估。

**2. 确定要调整的参数**

根据我们之前的讨论，有几个值得调整的参数候选：

- **循环单元数量：** LSTM或GRU层需要多大容量？（例如：32、64、128）
- **学习率：** 模型在训练期间应该多快地适应？（例如：0.01、0.001、0.0001）
- **Dropout比率：** 需要多少正则化来防止过拟合？（例如：0.2、0.3、0.5）这包括标准dropout和循环dropout。
- **层数：** 堆叠（更深）的RNN会表现更好吗？（例如：1层对比2层）
- **批量大小：** 在更新权重 (weight)之前处理多少样本？（例如：32、64、128）
- **嵌入 (embedding)维度：** （如果文本使用嵌入）嵌入向量 (vector)应该有多大？（例如：50、100、200）

**3. 调整过程：迭代与评估**

调整是一个迭代过程。你通常一次更改一个或一小组相关的超参数，重新训练模型，并在验证集上评估其性能。

让我们使用TensorFlow/Keras语法模拟几个步骤作为示例。假设我们的基准模型是：

```python
# 基准模型（简化）
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# history = model.fit(train_data, validation_data=val_data, epochs=10, batch_size=64)
# baseline_val_accuracy = history.history['val_accuracy'][-1] # 示例：获取最终验证准确率
```

**第一次迭代：调整LSTM单元**

让我们尝试增加LSTM层的容量。

- **更改：** 将`LSTM(64)`修改为`LSTM(128)`。
- **理由：** 也许基准模型缺乏捕捉复杂模式的能力。
- **重新训练与评估：** 再次编译并拟合模型。
- **结果：** 验证准确率：79%。略有改善。

```python
# 第一次迭代：增加单元数量
model_iter1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True),
    tf.keras.layers.LSTM(128), # 更改了单元数量
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# 重新编译并重新拟合...
```

**第二次迭代：添加Dropout**

改进很小，也许随着单元数量的增加，过拟合正在成为一个问题。让我们添加dropout。

- **更改：** 向`LSTM`层添加`Dropout`和`recurrent_dropout`。
- **理由：** 对模型进行正则化以提高泛化能力。循环dropout将dropout应用于LSTM内部时间步之间的连接。
- **重新训练与评估：**
- **结果：** 验证准确率：81%。有了更明显的改善，表明正则化有帮助。

```python
# 第二次迭代：添加Dropout
model_iter2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True),
    tf.keras.layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3), # 添加了dropout
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# 重新编译并重新拟合...
```

**第三次迭代：调整学习率**

也许默认学习率对于这个修改后的架构来说不是最佳的。让我们尝试一个更小的。

- **更改：** 修改优化器的学习率，例如`Adam(learning_rate=0.0005)`。
- **理由：** 更小的学习率可能导致更精细的收敛，尤其对于更复杂的模型而言。
- **重新训练与评估：**
- **结果：** 验证准确率：81.5%。略有提升，可能表明收敛更平稳。请注意，训练可能需要稍长的时间。

```python
# 第三次迭代：调整学习率
model_iter3 = tf.keras.Sequential([
    # ... 来自第二次迭代的层 ...
])
model_iter3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # 更改了学习率
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
# 重新拟合...
```

**第四次迭代：堆叠层**

让我们看看更深的模型是否有助于捕获分层特征。

- **更改：** 添加第二个LSTM层。记住在第一个LSTM层上设置`return_sequences=True`，这样它会为下一层输出一个序列。
- **理由：** 更深的模型有时可以学习更抽象的表示。
- **重新训练与评估：**
- **结果：** 验证准确率：80.5%。性能略有下降。这可能表明增加的复杂度对该数据集没有帮助，或者它需要更多数据或进一步调整（例如，调整每层的dropout比率）。

```python
# 第四次迭代：堆叠层
model_iter4 = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True),
    tf.keras.layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True), # return_sequences=True
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2), # 第二个LSTM层（更少单元，可能更少dropout）
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# 使用先前的学习率重新编译并重新拟合...
```

**4. 跟踪进展**

记录你的实验是有益的。一个简单的表格或电子表格可以派上用场，或者你可以使用像MLflow或Weights & Biases这样的工具。可视化不同试验中的验证度量也能提供见解。



[交互图表：调整过程中的验证准确率](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-10-evaluating-tuning-sequence-models/practice-tuning-rnn-model#plot-1828jom)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "调整过程中的验证准确率",
    "xaxis": {
      "title": "调整迭代"
    },
    "yaxis": {
      "title": "验证准确率",
      "range": [
        0.75,
        0.85
      ]
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "x": [
        "基准",
        "迭代1 (单元=128)",
        "迭代2 (Dropout)",
        "迭代3 (学习率=0.0005)",
        "迭代4 (堆叠)"
      ],
      "y": [
        0.78,
        0.79,
        0.81,
        0.815,
        0.805
      ],
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 情感分析示例在不同调整迭代中的验证准确率。

**5. 系统方法**

手动调整参数有助于了解过程，但可能耗时且可能错过最佳组合。为了更严谨的调整，可以考虑：

- **网格搜索：** 为每个超参数定义一个值范围，并为每个可能的组合训练一个模型。计算成本高昂。
- **随机搜索：** 从指定分布中随机抽取超参数组合。在寻找良好组合方面通常比网格搜索更有效。
- **贝叶斯优化：** 使用先前试验的结果智能地选择下一组要尝试的超参数。通常是最有效的方法。

像Keras Tuner、Scikit-learn的`GridSearchCV`/`RandomizedSearchCV`、Optuna或Hyperopt等库可以自动化这些搜索策略。

**关于调整的最终思考**

- **使用验证集：** 始终根据在单独验证集上的性能进行调整。
- **从简单开始：** 从一个相对简单的模型开始，并根据需要逐步增加复杂度或正则化。
- **保持耐心：** 调整通常是实验性的。并非每一次更改都会带来改进。
- **考虑计算成本：** 更复杂的模型和详尽的超参数搜索需要大量时间和资源。
- **没有万能药：** 最佳超参数高度依赖于特定的数据集和任务。

这个实践练习说明了如何应用本章讨论的评估和调整技巧。通过系统地调整参数并衡量它们的影响，你可以大大改进序列模型相对于其初始基准的性能。请记住，最终的、保留的测试集*只使用一次*来报告你最佳调整模型的性能。

## 参考资料

- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 13; Pages: 281-305; DOI: [10.5555/2188385.2188395](https://doi.org/10.5555/2188385.2188395)
  介绍随机搜索作为比网格搜索更高效的超参数优化方法，与系统调优策略高度相关。
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287) — Yarin Gal and Zoubin Ghahramani (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 29; Pages: 3517-3525; DOI: [10.48550/arXiv.1512.05287](https://doi.org/10.48550/arXiv.1512.05287)
  介绍循环神经网络中用于正则化的循环 dropout 概念，该技术在调整 RNN 模型时特别提及。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本深度学习基础教材，涵盖循环神经网络、正则化技术和超参数优化策略等多个方面，与模型调优相关。
- [Keras Tuner: A Hyperparameter Tuning Library for Keras](https://keras.io/keras_tuner/) — O'Malley, Tom, Bursztein, Elie, Long, James, Chollet, François, Jin, Haifeng, Invernizzi, Luca, and others (2019)
  Publisher: GitHub
  Keras Tuner 的官方文档，该库可自动执行 Keras 模型的超参数搜索（网格搜索、随机搜索、贝叶斯优化），直接适用于讨论的系统调优方法。

---

[上一节](07-%E5%B8%B8%E8%A7%81%E8%AE%AD%E7%BB%83%E9%97%AE%E9%A2%98%E7%9A%84%E6%8E%92%E6%9F%A5.md)
