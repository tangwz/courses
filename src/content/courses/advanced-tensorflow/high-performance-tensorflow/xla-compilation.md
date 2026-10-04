---
course: "advanced-tensorflow"
chapter: "high-performance-tensorflow"
lesson: "xla-compilation"
sourceId: 2831
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-2-high-performance-tensorflow/xla-compilation"
title: "XLA（加速线性代数）编译"
description: "了解 XLA 如何通过合并操作并为特定硬件编译来优化 TensorFlow 图。"
order: 5
plots: []
sourceHash: "1e7f40734f979cab717399f76fa99276149ba50be42a50c5f04e5d31e6fff799"
sourceCorrections: []
---

TensorFlow 在图级别提供了一个有效的优化层：XLA（加速线性代数）。与混合精度训练等直接修改计算中数值表示的方法不同，XLA 专注于优化计算图本身。XLA 是一种特定领域编译器，旨在通过将计算图转换为高效的机器码来优化 TensorFlow 计算，这些机器码针对特定硬件（如 CPU、GPU，尤其是 TPU）进行了优化。

XLA 不会像图定义的那样逐个执行 TensorFlow 操作（这会因启动单独的计算核而产生大量开销），而是会分析图，进行多种优化，并将图的片段（或整个图）编译成数量较少、经过合并和优化的计算核。

### XLA 如何优化性能

XLA 采用多种策略来加快您的 TensorFlow 代码的运行：

1. **操作合并：** 这可以说是 XLA 最重要的优化。它将多个独立的 TensorFlow 操作（如矩阵乘法、偏置 (bias)相加和激活函数 (activation function)）合并为一个更大、单一的计算核。

   - **优点：** 合并操作大幅减少了为每个操作启动单独计算核的开销。它还通过将中间结果保存在处理器的寄存器或缓存中，最大限度地减少对主内存的缓慢读写，从而提高内存使用效率。设想将 `tf.matmul`、`tf.nn.bias_add` 和 `tf.nn.relu` 结合成一个合并操作，而不是三个独立步骤。

digraph G {
rankdir=LR;
node [shape=box, style=rounded, fontname="Arial", fontsize=10, margin="0.1,0.05"];
edge [fontname="Arial", fontsize=10];

subgraph cluster\_0 {
label = "标准执行";
bgcolor="#e9ecef";
a [label="输入"];
b [label="矩阵乘法"];
c [label="偏置相加";
d [label="ReLU激活";
e [label="输出"];
a -> b;
b -> c;
c -> d;
d -> e;
}

subgraph cluster\_1 {
label = "XLA 合并执行";
bgcolor="#d0bfff";
x [label="输入"];
y [label="合并操作 (矩阵乘法 + 偏置相加 + ReLU激活)"];
z [label="输出"];
x -> y;
y -> z;
}
}
\`\`\`

> 此图对比了标准逐操作执行与 XLA 合并操作方法的视图。合并操作减少了计算核启动开销，并提高了数据局部性。

2. **常量折叠：** XLA 会分析图，识别只依赖于常量输入的部分，并在编译时计算它们的结果，将结果直接嵌入 (embedding)到编译后的代码中。
3. **缓冲区分析：** XLA 进行精细分析以优化内存缓冲区的分配和使用，旨在最大程度地减少内存占用，并在可能的情况下复用缓冲区。
4. **硬件特定代码生成：** XLA 生成针对目标硬件特定架构和指令集优化的机器码（例如，特定的 GPU 指令，TPU 矩阵单元操作）。

### 启用 XLA 编译

为 TensorFlow 代码的特定部分启用 XLA 编译最直接和推荐的方法是，通过在 `tf.function` 中使用 `jit_compile` 参数 (parameter)。

```python
import tensorflow as tf
import timeit

# 定义一个简单计算
def complex_computation(a, b):
  x = tf.matmul(a, b)
  y = tf.nn.relu(x)
  z = tf.reduce_sum(y)
  return z

# 创建一些输入张量
input_a = tf.random.normal((1000, 1000), dtype=tf.float32)
input_b = tf.random.normal((1000, 1000), dtype=tf.float32)

# 未使用 XLA 的版本（标准 tf.function）
@tf.function
def standard_func(a, b):
  return complex_computation(a, b)

# 启用 XLA JIT 编译的版本
@tf.function(jit_compile=True)
def xla_compiled_func(a, b):
  return complex_computation(a, b)

# 预热运行（重要！）
_ = standard_func(input_a, input_b)
_ = xla_compiled_func(input_a, input_b)

# 计时执行
n_runs = 10
standard_time = timeit.timeit(lambda: standard_func(input_a, input_b), number=n_runs)
xla_time = timeit.timeit(lambda: xla_compiled_func(input_a, input_b), number=n_runs)

print(f"标准 tf.function 时间: {standard_time / n_runs:.6f} 秒/次运行")
# 注意：XLA 编译在第一次调用时发生，后续调用会更快。
# timeit 在其测量中包含了第一次迭代的编译时间。
# 为了更公平地比较*持续*性能，请在预热后测量。
xla_time_post_compile = timeit.timeit(lambda: xla_compiled_func(input_a, input_b), number=n_runs)
print(f"XLA (jit_compile=True) 时间（含首次编译）: {xla_time / n_runs:.6f} 秒/次运行")
print(f"XLA (jit_compile=True) 时间（编译后）: {xla_time_post_compile / n_runs:.6f} 秒/次运行")

# 示例输出（实际时间会因硬件而异）：
# Standard tf.function time: 0.008512 seconds per run
# XLA (jit_compile=True) time (incl. 1st compile): 0.152345 seconds per run (包含编译时间！)
# XLA (jit_compile=True) time (post-compile): 0.001876 seconds per run (编译后执行更快)
```

设置 `jit_compile=True` 指示 TensorFlow 尝试使用 XLA 编译整个函数，在首次执行时（或使用新的输入签名首次执行时）。首次调用会产生编译开销，但后续使用兼容输入形状和类型的调用将执行高度优化的编译核，通常会带来显著的速度提升。

尽管 TensorFlow 也有“自动聚类”机制，它可以尝试在没有显式注解的情况下自动寻找适合 XLA 的子图，但使用 `tf.function(jit_compile=True)` 提供了更可预测的行为和明确的控制，以确定计算图的哪些部分会被编译。

### 考量与权衡

XLA 是一种有效的工具，但并非适用于所有情况的万能解决方案。请考量以下几点：

1. **编译开销：** XLA 编译需要时间。这种开销在首次调用具有给定输入签名的 `jit_compile=True` 函数时产生。如果一个函数只被调用几次，或者其输入形状经常变化（从而触发重新编译），编译成本可能会抵消执行速度的提升。XLA 最适合那些以一致输入形状重复调用的函数，例如模型在训练或推理 (inference)期间的前向传播。

- **操作支持：** 尽管 XLA 支持广泛的 TensorFlow 操作，但对每个操作的支持可能不是普遍的，也不是同样优化的，特别是对于较新或更不常用的操作。严重依赖不支持操作的计算可能无法获得显著益处，甚至可能阻碍编译。动态控制流（例如函数内部依赖于数据的 `if` 条件）有时会给 XLA 编译带来挑战，尽管支持已显著改善。
- **数值精度：** 由于合并等优化以及潜在的不同操作顺序，XLA 编译函数的数值结果可能与标准 TensorFlow 执行的结果略有不同（通常在可接受的浮点容差范围内）。这类似于在不同硬件（CPU 与 GPU）上运行时观察到的差异。在启用 XLA 后，仔细验证模型的输出和指标。
- **调试：** 调试在 XLA 编译核内部运行的代码本身比调试标准 TensorFlow 逐操作执行更困难。如果您遇到问题，暂时禁用 `jit_compile=True` 通常会有帮助，以确认问题是出在 XLA 编译过程还是原始 Python 逻辑中。
- **内存使用：** 尽管合并可以降低内存*带宽*需求，但与逐操作执行相比，大型合并计算核执行期间的峰值内存使用量有时可能会增加。然而，由于更好的缓冲区管理，总内存使用量通常会减少。

### 何时使用 XLA

XLA 最有可能提供显著益处的情况是：

- 您的模型或计算是**计算密集型**而非输入密集型（即 GPU/TPU 是瓶颈，而不是数据加载管道）。
- 您针对的是 **GPU 或 TPU**，XLA 的硬件特定优化在这些设备上最有效。尤其是 TPU，它们高度依赖 XLA。
- 您的计算涉及一系列**可合并操作**（例如，逐元素操作、卷积、矩阵乘法）。
- 您以兼容的输入形状**重复**执行相同函数（如模型的 `call` 方法或 `train_step`）。

### 将 XLA 应用于 Keras 模型

您可以将 XLA 编译直接应用于自定义 Keras 层或模型的 `call` 方法，或者在使用自定义训练循环时应用于整个 `train_step` 或 `test_step` 函数。

```python
import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )

    # 将 XLA 编译应用于前向传播
    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = tf.matmul(inputs, self.w)
        x = tf.nn.bias_add(x, self.b)
        x = tf.nn.relu(x)
        return x

# --- 或者应用于 train_step 函数 ---

class MyCustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = MyDenseLayer(128) # 假设 MyDenseLayer 如上所述定义
        self.dense2 = tf.keras.layers.Dense(10) # 标准层

    # 这里不使用 JIT，JIT 应用于 train_step
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyCustomModel()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 将 XLA 编译应用于整个训练步骤
@tf.function(jit_compile=True)
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# In your training loop:
# for x_batch, y_batch in dataset:
#     loss_value = train_step(x_batch, y_batch) # 此步骤受益于 XLA
```

通过策略性地应用 `@tf.function(jit_compile=True)`，您指示 TensorFlow 借助 XLA 以获得潜在的显著性能提升。与任何优化一样，在启用 XLA 前后（使用之前讨论过的 TensorBoard Profiler 等工具）配置文件您的应用程序很重要，以量化 (quantization)其对特定工作负载和硬件的影响。彻底测试以确保数值稳定性和正确性保持在可接受的范围内。

## 参考资料

- [Optimize TensorFlow performance with XLA](https://www.tensorflow.org/xla) — TensorFlow authors (2024)
  Publisher: TensorFlow (Google)
  官方指南，解释了XLA在TensorFlow中的作用、如何启用它、其优势以及性能方面的实际考量。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Publisher: USENIX Association; Pages: 578-594
  介绍了另一个重要的深度学习领域专用编译器，展示了XLA也采用的类似图优化技术（如融合）。
