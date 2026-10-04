# 函数式API

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/functional-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`Sequential` API 提供了一种直接方式来将神经网络 (neural network)模型定义为线性层序列。然而，许多应用需要更复杂的网络结构。模型可能需要多个输入（例如图像及其元数据），产生多个输出（例如对对象进行分类*并*估计其边界框），或者包含在网络不同部分共享信息的层（例如暹罗网络或带有残差连接的模型）。对于这些情况，Keras 提供了**函数式API**。

函数式API将层视为可以在张量上调用的函数。它允许你将模型构建为层的有向无环图（DAG）。与`Sequential`模型的线性堆叠相比，这种基于图的方法提供了很大的灵活性。

### 核心思想：层即函数

可以将函数式API想象成连接乐高积木，其中每块积木是一个层，而连接是它们之间流动的张量。你从一个输入张量开始，将其通过一个层（就像调用一个函数），得到一个输出张量，然后将其传递给另一个层，依此类推，直到你定义最终的输出张量。

### 定义输入

与`Sequential`模型不同，`Sequential`模型的输入形状通常在第一个层中推断或指定，而函数式API要求你使用`keras.Input`明确定义图的起点。这会创建一个符号张量对象，其中包含模型期望的输入的形状和数据类型信息。

```python
import keras
from keras import layers

# 定义一个输入，期望28x28的灰度图像（展平后）
# 形状是一个元组，None通常表示批次大小可以变化。
# 对于像MNIST（784像素）这样的扁平向量，形状是(784,)。
input_tensor = keras.Input(shape=(784,), name='image_input')
```

在这里，`input_tensor`还不是实际数据。它是一个规范，说明模型将接收的数据类型。为输入提供有意义的名称（例如`image_input`）是一个良好的做法，特别是对于复杂模型。

### 连接层

一旦你有了输入张量，就可以通过在张量上调用层实例来连接层。该层会返回一个新的张量，你可以将其传递给下一个层。

让我们构建一个简单的多层感知器（MLP），类似于你使用`Sequential`构建的模型，但这次使用函数式API：

```python
# 从输入张量开始
inputs = keras.Input(shape=(784,), name='img_input')

# 第一个全连接层：在输入张量上调用该层
x = layers.Dense(64, activation='relu', name='dense_layer_1')(inputs)

# 第二个全连接层：在上一层（'x'）的输出上调用该层
x = layers.Dense(64, activation='relu', name='dense_layer_2')(x)

# 输出层：在第二个全连接层的输出上调用该层
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
```

注意这个模式：`output_tensor = Layer(...)(input_tensor)`。变量`x`在这里被重用，表示流经网络并被每个层转换的张量。

### 创建模型

定义了从输入到输出的层图后，你可以实例化一个`keras.Model`对象。你需要告诉`Model`构造函数哪些张量表示网络图的输入，哪些表示输出。

```python
# 通过指定输入和输出张量来创建模型
model = keras.Model(inputs=inputs, outputs=outputs, name='simple_mlp_functional')

# 现在你可以查看模型架构
model.summary()
```

运行`model.summary()`会产生类似于`Sequential`模型的输出，显示层、输出形状和参数 (parameter)数量。然而，在幕后，Keras已经构建了一个图表示。

### 函数式API的优点

当需要比简单线性序列更复杂的架构时，函数式API的真正优势便会显现出来：

1. **多输入模型：** 你可以定义多个`keras.Input`张量，并将它们送入网络的不同分支，最终将它们合并。

digraph G {
rankdir=LR;
graph [fontname="helvetica", fontsize=10];
node [shape=box, style="filled, rounded", fillcolor="#a5d8ff", fontname="helvetica", fontsize=10];
edge [color="#495057", fontname="helvetica", fontsize=10];

subgraph cluster\_input {
label = "输入";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
Input1 [label="输入A\n(形状A)", shape=cds, fillcolor="#bac8ff"];
Input2 [label="输入B\n(形状B)", shape=cds, fillcolor="#bac8ff"];
}

subgraph cluster\_processing {
label = "处理分支";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
ProcA [label="分支A层"];
ProcB [label="分支B层"];
}

subgraph cluster\_merge {
label = "合并";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
Merge [label="连接或相加等", shape=oval, fillcolor="#96f2d7"];
}

subgraph cluster\_output {
label = "输出";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
OutputDense [label="最终层"];
Output [label="预测", shape=ellipse, fillcolor="#ffc9c9"];
}

Input1 -> ProcA;
Input2 -> ProcB;
ProcA -> Merge;
ProcB -> Merge;
Merge -> OutputDense;
OutputDense -> Output;
}
\`\`\`

> 该图示意了一个模型，其具有两个不同的输入，通过独立分支处理后合并以进行最终预测。

2. **多输出模型：** 模型可以通过在创建`keras.Model`时指定输出张量的列表或字典来产生多个输出。这对于多标签分类或联合回归和分类等任务非常有用。
3. **共享层：** 单个层实例可以在不同张量上多次调用。该层实例维护一组权重 (weight)，这些权重会根据其所有使用位置进行更新。这对于暹罗网络等模型来说很基础，它们使用相同的处理分支比较两个输入。

digraph G {
rankdir=LR;
graph [fontname="helvetica", fontsize=10];
node [shape=box, style="filled, rounded", fillcolor="#a5d8ff", fontname="helvetica", fontsize=10];
edge [color="#495057", fontname="helvetica", fontsize=10];

subgraph cluster\_input {
label = "输入";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
Input1 [label="输入A", shape=cds, fillcolor="#bac8ff"];
Input2 [label="输入B", shape=cds, fillcolor="#bac8ff"];
}

subgraph cluster\_shared {
label = "共享处理";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
SharedLayer [label="共享层\n(例如，嵌入 (embedding))", fillcolor="#ffe066"];
}

subgraph cluster\_output {
label = "输出";
style="filled, rounded";
color="#e9ecef";
bgcolor="#f8f9fa";
OutputA [label="处理后的A"];
OutputB [label="处理后的B"];
FinalOutput [label="组合输出\n(可选)", shape=ellipse, fillcolor="#ffc9c9"];
}

Input1 -> SharedLayer [label="应用于A"];
Input2 -> SharedLayer [label="应用于B"];

SharedLayer -> OutputA [tailport=e, headport=w];
SharedLayer -> OutputB [tailport=e, headport=w];

# 可选的组合步骤

OutputA -> FinalOutput;
OutputB -> FinalOutput;
}
\`\`\`

> 该图示意了如何将单个层实例（共享层）应用于多个输入，产生独立的输出，这些输出可以独立使用或稍后组合。

4. **非线性结构：** 你可以创建复杂的图，例如带有残差连接（层的输出被加回到其输入）或带有并行卷积分支的Inception风格模块的网络。

虽然`Sequential` API 便于构建简单的线性模型，但函数式API是定义当今大多数复杂深度学习 (deep learning)架构的首选工具。掌握它可以实现更广的网络设计范围。在接下来的小节和章节中，你将看到函数式API被广泛使用，特别是在构建具有特定结构要求的卷积神经网络 (neural network)（CNN）和循环神经网络（RNN）等更复杂模型时。

## 参考资料

- [Guide to the Keras Functional API](https://keras.io/guides/functional_api_for_beyond_sequential_models/) — Keras Team (2024)
  Publisher: Keras Documentation
  Keras 函数式 API 的官方文档，涵盖高级用法模式和最佳实践。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras 创建者撰写的综合指南，提供了函数式 API 的详细解释和实用示例。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习原理的奠基性教科书，为函数式 API 支持的各种网络架构提供了理论背景。

---

[上一节](04-%E6%8C%87%E5%AE%9A%E8%BE%93%E5%85%A5%E5%BD%A2%E7%8A%B6.md) · [下一节](06-%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%A7%88%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96.md)
