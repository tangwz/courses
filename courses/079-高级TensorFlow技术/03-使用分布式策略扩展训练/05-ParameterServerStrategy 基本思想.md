# ParameterServerStrategy 基本思想

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies/parameter-server-strategy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管像 `MirroredStrategy` 和 `MultiWorkerMirroredStrategy` 这样的同步策略会复制模型并在所有设备间同步梯度更新，但参数 (parameter)服务器（PS）架构提供了一种本质不同的方式，通常与异步训练相关联。它将梯度计算任务与参数存储和更新分离。

### 参数 (parameter)服务器架构

在典型的参数服务器配置中，集群包含两种类型不同的任务：

1. **工作节点**：这些节点执行核心训练逻辑。每个工作节点通常会：

   - 读取一部分（分片）训练数据。
   - 从参数服务器拉取当前版本的模型参数。
   - 对其数据批次执行前向和后向传播，以计算梯度。
   - 将这些梯度推回参数服务器。
2. **参数服务器（PS）**：这些节点保存模型的参数（变量）。它们的主要职责包括：

   - 存储模型的参数。
   - 接收来自多个工作节点的梯度更新。
   - 应用这些梯度来更新参数（通常使用运行在PS上的优化器实例）。
   - 应请求将更新后的参数提供给工作节点。

> 典型的参数服务器配置中，工作节点负责获取参数并发送梯度，而参数服务器节点则负责存储参数和应用更新。

### 异步训练的运行特点

参数 (parameter)服务器架构最常见的运行模式是异步的。在这种模式下：

- 工作节点独立运行，互不等待。
- 工作节点从PS获取最新参数，基于当前数据批次计算梯度，并将这些梯度发送回PS。
- PS异步接收来自工作节点的梯度，并在它们到达时应用。

这可能导致一个被称为**陈旧梯度**的问题。工作节点可能基于参数版本 $W_t$计算梯度，但当这些梯度到达PS并被应用时，参数可能已被其他工作节点多次更新，达到了版本 $W_{t+k}$。使用 $W_t$ 计算的梯度随后被应用到 $W_{t+k}$ 上。

从数学角度看，PS应用来自工作节点 $i$ 的梯度 $\nabla L(w_{\text{local}}, x_i)$（工作节点 $i$ 使用的是之前获取的局部参数 $w_{\text{local}}$）的异步更新的简化形式可以是：


$$
w_{\text{global}} \leftarrow w_{\text{global}} - \eta \cdot \nabla L(w_{\text{local}}, x_i)
$$


其中 $w_{\text{global}}$ 是PS上的当前状态，而 $w_{\text{local}}$ 表示工作节点使用的可能是较旧的参数。

### 优点与缺点

**优点：**

- **可能实现更高的数据处理能力：** 工作节点之间不会相互等待，这可能在单位时间内带来更多的梯度更新，特别是当工作节点处理速度不同或网络延迟较高时。
- **提升的容错能力：** 如果一个工作节点出现故障，训练过程通常可以由其余工作节点继续进行，这与同步方法不同，同步方法中一个工作节点失败通常会停止整个过程。
- **可扩展性（工作节点）：** 可以扩展到大量工作节点。

**缺点：**

- **陈旧梯度：** 与同步训练相比，可能对收敛速度产生负面影响，并可能导致次优解。其有效性可能对学习率和更新规则很敏感。
- **参数 (parameter)服务器瓶颈：** 如果工作节点数量非常大，或者模型大小需要大量的参数和梯度通信带宽，PS节点可能会成为瓶颈。
- **可重现性与调试：** 异步特性使得训练过程的确定性降低，并且可能比同步方法更难调试。
- **优化方面的复杂性：** 需要仔细调整学习率，并且PS上可能需要更精细的优化方案来减轻陈旧梯度的影响。

### TensorFlow 实现：`tf.distribute.ParameterServerStrategy`

TensorFlow 提供了 `tf.distribute.ParameterServerStrategy` 来实现这种架构。与 `MirroredStrategy` 不同，配置 `ParameterServerStrategy` 需要显式定义集群配置，并为每个任务指定角色（`'chief'`、`'worker'`、`'ps'`）。

- `'chief'` 工作节点除了履行常规工作职责外，通常还负责处理保存检查点和写入摘要等协调任务。
- 变量会自动放置在指定的PS任务上（如果存在多个PS任务，则在它们之间进行分片）。
- 操作通常在工作节点上执行。

使用 `ParameterServerStrategy` 通常需要编写自定义训练循环，因为工作节点之间获取参数 (parameter)和推送梯度的协调，需要比简单策略中 `model.fit` 提供的抽象更明确的管理。

### 何时采用参数 (parameter)服务器训练

在以下情况下，通常会采用参数服务器策略：

- 训练极大型模型，其参数无法全部放入单个工作节点的内存中（参数在PS节点间分片是必要的）。
- 网络延迟可能较高或连接不可靠的环境中，同步等待会降低效率。
- 涉及大量工作节点的情形，其中同步协调的开销变得过高。
- 研究异步优化算法。

尽管像 `MultiWorkerMirroredStrategy` 这样的同步策略因其更简单的收敛特性以及NVLink和InfiniBand等现代互连的有效性而变得越来越流行，但了解参数服务器架构对于特定的大规模训练情景以及理解分布式机器学习 (machine learning)中的不同权衡仍然很有价值。

## 参考资料

- [Scaling Distributed Machine Learning with the Parameter Server](https://www.usenix.org/conference/osdi14/technical-tracks/presentation/li) — Mu Li, David G. Andersen, Jun Woo Park, Alexander J. Smola, Amr Ahmed, Vanja Josifovski, James Long, Eugene J. Shekita, Bor-Yiing Su (2014)
  Journal: OSDI '14: Proceedings of the 11th USENIX Symposium on Operating Systems Design and Implementation; Publisher: USENIX Association; Pages: 583-598; DOI: [10.5555/2717390.2717435](https://doi.org/10.5555/2717390.2717435)
  介绍Parameter Server架构及其核心概念的创始论文。
- [tf.distribute.ParameterServerStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/ParameterServerStrategy) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow Parameter Server策略的官方API文档。
- [Distributed Deep Learning: A Comprehensive Review](https://dl.acm.org/doi/10.1145/3397914) — Kuan-Ching Li, Zizhong Chen, Albert Y. Zomaya, Yang Zhang (2020)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 53; Pages: 1-36; DOI: [10.1145/3397914](https://doi.org/10.1145/3397914)
  一篇综述文章，讨论了包括Parameter Server在内的各种分布式深度学习技术。
- [Distributed training in TensorFlow](https://www.tensorflow.org/guide/distributed_training) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow分布式训练的官方指南，包括设置和不同策略。

---

[上一节](04-%E7%94%A8%E4%BA%8E%E5%A4%9A%E8%8A%82%E7%82%B9%E8%AE%AD%E7%BB%83%E7%9A%84%20MultiWorkerMirroredStrategy.md) · [下一节](06-%E7%94%A8%E4%BA%8E%E5%9C%A8%20TPU%20%E4%B8%8A%E8%AE%AD%E7%BB%83%E7%9A%84%20TPUStrategy.md)
