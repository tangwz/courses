---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-neural-networks-architectures-training"
lesson: "quantum-convolutional-neural-networks"
sourceId: 1065
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-5-quantum-neural-networks-architectures-training/quantum-convolutional-neural-networks"
title: "量子卷积神经网络 (QCNNs)"
description: "审视受经典CNN启发的QCNN架构，其中运用量子电路进行卷积和池化操作。"
order: 3
plots: []
sourceHash: "a9e5eb31aab4bc255db709c548d04ff1415e649387977297ec63852ae31d3f67"
sourceCorrections: []
---

经典卷积神经网络 (neural network)（CNN）在处理图像识别等网格状数据结构的任务中表现出色。它们通过应用卷积滤波器捕获局部模式的层以及在保留重要特征的同时降低维度的池化层来实现这一目的，从而形成输入的层次化表示。量子卷积神经网络（QCNN）的目标是将这些成功方法应用于量子环境，运用量子电路进行类似卷积和池化的操作。

主要原因在于处理量子态或编码为量子态的经典数据，这可能发现经典CNN无法获取的特征或关联。QCNN尤其适合处理具有固有空间结构的问题，例如分析多体系统量子模拟的数据或某些经典图像处理任务。

### QCNN的核心组成部分

QCNN通常包含与经典对应物相似的量子层：

1. **量子卷积层：** 这一层在输入量子比特上局部施加参数 (parameter)化量子电路（通常是少量子比特酉算符）。可以设想一个小型量子电路在输入寄存器上“滑动”，如同经典滤波核。这些电路，即“量子滤波器”，通常是参数化的，并在不同空间位置共享相同参数，如同经典CNN中的权重 (weight)共享。其作用是获取局部特征，并可能在相邻量子比特间产生纠缠。
2. **量子池化层：** 卷积之后，池化层减少表示数据的量子比特数量，从而降低空间维度。一种常见方法是测量特定量子比特（通常是卷积步骤中纠缠的辅助量子比特或用于池化的量子比特），并根据测量结果应用后续的受控酉操作。这个过程能有效选择并保留特征，同时减小系统规模。与经典池化（例如最大池化）不同，量子池化由于测量而可能具有固有的概率性，并且可以根据结果进一步改变状态。

这些层通常呈分层排列，一个池化层的输出量子比特作为下一个卷积层的输入。

### 构建量子卷积和池化层

#### 量子卷积

设输入状态由 $N$ 个量子比特表示，它们可能线性排列或在2D网格上。量子卷积层对 $k$ 个相邻量子比特的一小部分施加参数 (parameter)化酉算符 $U(\theta)$（例如 $k=2$ 或 $k=3$）。这个酉算符在输入寄存器上重复应用，可能带步长，类似于经典卷积。

For instance, acting on qubits $i$ and $i+1$:
$|\psi_{out}\rangle = \dots U_{i+2, i+3}(\theta) \dots U_{i, i+1}(\theta) \dots |\psi_{in}\rangle$

参数 $\theta$ 在 $U$ 的所有应用中共享，这大大减少了参数数量，与作用于所有量子比特的通用PQC相比。$U(\theta)$ 的选择决定了滤波器；它可能是一种硬件高效的拟设或旨在产生特定关联的电路。

#### 量子池化

池化操作会减少量子比特的数量。一个简单的方案可以是：根据相邻量子比特的测量结果应用受控酉操作。例如，对于量子比特 $i$ 和 $i+1$，我们可以施加一个CNOT门使它们纠缠，测量量子比特 $i+1$，然后将其舍弃。量子比特 $i$ 的状态现在可能根据 $i$ 和 $i+1$ 之间共享的属性而被改变，且量子比特总数也减少了。

更复杂的池化操作也存在，有时会涉及多个量子比特、测量以及受经典控制的量子门。核心在于在减少量子比特数量的同时，保留剩余量子态中编码的相关特征。

> 一个QCNN层作用于4个输入量子比特的简化示意。局部参数化酉算符 $U(\theta)$ 作为卷积滤波器。池化操作（包含测量和受控门）减少量子比特数量，舍弃量子比特q2和q4，最终得到2个输出量子比特。

### 架构与训练

一个完整的QCNN架构通常是堆叠多个卷积层和池化层。最终层的输出可以直接测量以获得用于分类的经典结果，或者传递给另一个量子电路（如全连接层PQC），甚至是一个经典神经网络 (neural network)进行最终处理。

QCNN的训练沿用前面讨论的变分方法。

1. 根据最终量子比特的测量结果设定一个成本函数（例如，用于分类的交叉熵）。
2. 计算成本函数关于卷积层中共享参数 (parameter) $\theta$ 的梯度（例如，通过参数移位规则）。
3. 运用经典优化器（例如Adam, SGD）更新参数 $\theta$。

分层结构和参数共享使得QCNN与通用PQC相比，可能更能扩展到更多输入量子比特，尽管在深层架构中仍可能出现贫瘠高原等难题。卷积酉算符 $U(\theta)$ 的选择和池化策略是影响网络表达能力和可训练性的重要设计环节。

### 潜在应用与展望

QCNNs正在被积极研究，以应用于以下方面：

- **量子相识别：** 识别量子系统模拟的物质相。
- **量子纠错：** 可能学习解码策略。
- **处理传感器数据：** 对量子传感器阵列的数据进行分析。
- **经典数据分析：** 将QCNNs用于编码到量子态的经典数据集（例如图像），尽管在此处体现优势存在困难。

具有吸引力，但构建高效且具有优势的QCNN面临挑战。设计最佳的卷积和池化操作，减轻近期硬件上的噪声，以及避免贫瘠高原等可训练性问题，这些都是重要的研究方向。QCNN的潜力在于它能够在受经典CNN成功启发的结构化、分层处理框架内，利用纠缠等量子现象。

## 参考资料

- [Quantum convolutional neural networks](https://www.nature.com/articles/s41567-019-0648-8) — Iris Cong, Soonwon Choi, and Mikhail D. Lukin (2019)
  Journal: Nature Physics; Volume: 15; Pages: 97-102; DOI: [10.1038/s41567-019-0648-8](https://doi.org/10.1038/s41567-019-0648-8)
  这篇开创性论文介绍了量子卷积神经网络，阐述了其架构和核心组成部分。
- [Quantum Machine Learning](https://link.springer.com/book/10.1007/978-3-319-96424-9) — Maria Schuld, Francesco Petruccione (2018)
  Publisher: Springer; DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9)
  这本书为量子机器学习提供了基础介绍，涵盖了变分量子算法和量子神经网络。
- [Variational quantum algorithms](https://doi.org/10.1038/s42254-021-00348-9) — Marco Cerezo, Andrew Thomas Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, Patrick J. Coles (2021)
  Journal: Nature Reviews Physics; Publisher: Springer Nature; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  这篇综述涵盖了变分量子算法，包括其训练、优化和相关挑战（如贫瘠高原），与QCNNs密切相关。
