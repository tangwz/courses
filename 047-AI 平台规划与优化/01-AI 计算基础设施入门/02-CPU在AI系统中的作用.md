# CPU在AI系统中的作用

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/role-of-cpus-in-ai)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管图形处理器（GPU）常是人工智能硬件讨论的焦点，中央处理器（CPU）仍是任何机器学习 (machine learning)系统的基础组成部分。它是一种通用处理器，管理着从数据加载到代码最终执行的整个工作流程。与高度专用的GPU不同，CPU的设计注重灵活性和各类任务的低延迟执行，使其成为人工智能基础设施堆栈中不可或缺的一部分。

### 专业化时代中的通用者

现代CPU包含少量高度复杂的处理核心，通常为4到64个。每个核心都为出色的单线程性能而设计，具备大容量缓存、复杂的指令集和高级分支预测能力。这种架构使其擅长处理顺序任务和复杂条件逻辑，这些在机器学习 (machine learning)生命周期中普遍存在。

GPU擅长在数千个数据点上同时执行相同的简单计算，而CPU则擅长快速执行一系列独特且相互依赖的指令。这一区别确定了它在人工智能中的作用。

### AI流程中的主要职责

CPU的多功能性意味着它处理许多不适合GPU大规模并行架构的重要工作。

#### 数据预处理与增强

在模型训练之前，数据必须从存储中获取、解码并转换为合适的格式。这个预处理流程几乎总是在CPU上执行。这些任务包括：

- **数据摄取：** 从磁盘读取文件，无论是图像、CSV文件还是大型文本语料库。这些I/O操作由CPU上的操作系统管理。
- **解析与转换：** 解码JPEG图像、解析JSON对象或将句子分词 (tokenization)成单词，都涉及一系列顺序步骤和条件逻辑，这些都非常适合CPU。
- **数据增强：** 创建训练数据的变体以提高模型泛化能力，例如随机旋转、裁剪或翻转图像，这涉及由CPU高效处理的逻辑和随机数生成。

如果CPU不能足够快地向GPU提供数据，昂贵的加速器将会闲置，造成数据瓶颈并浪费宝贵的计算时间。一台多核的强大CPU可以并行运行这些预处理任务，确保训练流程始终获得即时处理的数据。

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 使用pandas和scikit-learn的典型CPU密集型预处理工作流程
def prepare_tabular_data(csv_path):
    # 1. I/O和解析：CPU读取CSV文件并将其解析为DataFrame
    df = pd.read_csv(csv_path)

    # 2. 特征工程：CPU应用条件逻辑创建新特征
    df['feature_c'] = df['feature_a'] / (df['feature_b'] + 1e-6)

    # 3. 缩放：CPU对数据执行scikit-learn的缩放器
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    return scaled_features
```

#### 系统编排与控制

CPU是整个AI系统的主导者。它运行操作系统、Python解释器以及你的机器学习 (machine learning)脚本（使用PyTorch或TensorFlow等框架编写）的高级逻辑。CPU负责：

- 初始化GPU并分配内存。
- 执行定义模型架构和训练循环的Python代码。
- 向GPU发送预处理数据批次和计算核心（例如，“执行矩阵乘法”）。
- 从GPU收集结果，例如损失值，用于日志记录或进一步处理。
- 将模型检查点保存到磁盘。

下图显示了CPU和GPU在典型训练循环中如何协作。CPU准备数据并管理整体流程，而GPU专注于繁重的数值计算。

> 一个AI训练循环，突出显示了CPU和GPU的各自作用。CPU管理工作流程和数据准备，而GPU加速密集数学运算。

#### 低延迟推理 (inference)与传统机器学习

对于许多推理场景，特别是那些每次处理单个请求（批次大小为1）的场景，CPU可能比GPU更有效。将少量数据传输到GPU再传回的开销，引入的延迟可能超过通过更快计算节省的延迟。因此，提供实时预测的Web后端常常依赖CPU。

此外，机器学习的很大一部分不涉及深度学习 (deep learning)。像逻辑回归、梯度提升树（XGBoost）和支持向量 (vector)机（SVM）等算法，在多核CPU上运行速度相同，甚至更快。这些算法通常是Scikit-learn等库的一部分，这些库都针对CPU执行进行了优化。

总而言之，在加速计算时代，CPU不仅仅是一个遗留组件。它是管理、准备并指导整个过程的核心且灵活的组件，使GPU等专用硬件能够发挥最佳性能。

## 参考资料

- [Computer Organization and Design RISC-V Edition: The Hardware/Software Interface](https://www.elsevier.com/books-and-journals/book-companion/9780128203316) — David A. Patterson and John L. Hennessy (2020)
  Publisher: Morgan Kaufmann; Pages: 736
  涵盖CPU架构、指令集、内存层次结构和I/O，为理解AI系统中CPU操作提供了基础。
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — The scikit-learn developers (2024)
  广泛使用的机器学习库的官方指南，展示了缩放等CPU密集型数据预处理任务。
- [Loading Data in PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — Sasank Chilamkurthy (2024)
  Publisher: PyTorch Foundation
  官方教程，解释了数据如何由CPU加载和预处理，然后发送到GPU进行深度学习训练。
- [Benchmarking the inference performance of deep learning models on CPUs and GPUs](https://www.ijser.org/researchpaper/Benchmarking-the-inference-performance-of-deep-learning-models-on-CPUs-and-GPUs.pdf) — Shishir Shivam (2020)
  Journal: International Journal of Scientific & Engineering Research; Publisher: International Journal of Scientific & Engineering Research; Volume: 11; Pages: 1178-1182
  比较了CPU和GPU在深度学习推理方面的性能，突出了CPU在低延迟任务中有效的情景。

---

[上一节](01-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E6%A6%82%E8%A7%88.md) · [下一节](03-GPU%20%E5%9C%A8%E5%8A%A0%E9%80%9F%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md)
