---
course: "getting-started-with-tensorflow"
chapter: "tensorflow-fundamentals-setup"
lesson: "verifying-installation"
sourceId: 1681
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-1-tensorflow-fundamentals-setup/verifying-installation"
title: "验证您的安装"
description: "确认您的TensorFlow安装正常工作的步骤，包括GPU检测。"
order: 4
plots: []
sourceHash: "2a0d24082a75465a5d92de86743ae43925a9f15a07c8a8442c43aff35c5c3edf"
sourceCorrections: []
---

确认TensorFlow已正确安装并在您的Python环境中可用非常重要。此验证步骤有助于确保代码示例按预期运行，并能让您检查特定的硬件加速功能，例如GPU支持。

### TensorFlow基本导入与版本检查

验证您安装的最简单方法是尝试在Python解释器或脚本中导入TensorFlow库并打印其版本。

打开您的Python解释器（或创建一个新的Python脚本文件，例如 `verify_tf.py`）：

```python
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
```

执行此代码。如果TensorFlow已正确安装，您应该会看到类似以下内容的输出（具体版本号将取决于您的安装情况）：

```text
TensorFlow Version: 2.1x.x
```

如果您遇到 `ImportError` 或 `ModuleNotFoundError`，这表明TensorFlow要么未在当前Python环境中安装，要么环境未正确激活。请仔细检查您的安装步骤，并确保您从安装TensorFlow的同一环境中（例如conda环境、virtualenv）运行Python。

### 执行一个简单操作

我们来执行一个基本的TensorFlow操作，以进一步确认功能是否正常。将以下行添加到您的脚本中，或输入到您的解释器中：

```python
# 创建一个常量张量
hello = tf.constant("Hello, TensorFlow!")
print(hello.numpy())

# 执行一个简单的数学运算
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a + b
print(f"Result of addition: {c.numpy()}")
```

运行此代码应生成以下输出：

```text
b'Hello, TensorFlow!'
Result of addition: 5.0
```

`tf.constant()` 函数创建一个张量，它是TensorFlow中主要的数据结构。我们在这里使用 `.numpy()` 将张量的值转换为NumPy兼容的格式，以便于打印。看到正确的字符串和加法结果（5.0），这证实了TensorFlow的核心操作正在正常运行。

### 检查GPU可用性

如果您特意安装了支持GPU的TensorFlow版本，并且拥有兼容的NVIDIA GPU以及所需的驱动程序和CUDA工具包，您应该验证TensorFlow是否能检测和使用GPU。TensorFlow在合适的GPU上执行计算密集型操作时，速度明显快于CPU。

使用以下代码列出TensorFlow可以检测到的物理设备，特别是筛选出GPU：

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    # 当前，内存增长需要在所有GPU上保持一致
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Detected {len(gpus)} Physical GPUs, Configured {len(logical_gpus)} Logical GPUs")
    print("GPU详细信息：")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: Name={gpu.name}, Type={gpu.device_type}")
  except RuntimeError as e:
    # 内存增长必须在GPU初始化之前设置
    print(f"Error during GPU configuration: {e}")
else:
  print("TensorFlow未检测到GPU。正在CPU上运行。")

# 验证操作的设备放置（可选）
with tf.device('/CPU:0'):
    cpu_tensor = tf.random.normal([10, 10])
    print("在CPU上创建了张量")

if gpus:
    try:
        with tf.device('/GPU:0'):
            gpu_tensor = tf.random.normal([10, 10])
            print("在GPU:0上创建了张量")
    except RuntimeError as e:
        print(f"Could not create tensor on GPU: {e}")
```

**预期输出（有GPU时）：**

如果找到兼容的GPU并正确配置，输出将类似这样：

```text
检测到 1 个物理GPU，配置了 1 个逻辑GPU
GPU详细信息：
  GPU 0: Name=/physical_device:GPU:0, Type=GPU
在CPU上创建了张量
在GPU:0上创建了张量
```

GPU的数量及其名称可能因您的系统配置而异。`tf.config.experimental.set_memory_growth(gpu, True)` 这一行通常很重要，可以防止TensorFlow一次性分配所有GPU内存，从而允许其他进程（甚至多个TensorFlow进程）使用GPU。看到“在GPU上创建了张量”的消息，证实了GPU操作放置成功。

**预期输出（无GPU或配置问题时）：**

如果您安装了仅限CPU的版本、没有兼容的NVIDIA GPU，或者驱动程序/CUDA配置错误，您可能会看到：

```text
TensorFlow未检测到GPU。正在CPU上运行。
在CPU上创建了张量
```

如果您期望有GPU支持但看到此消息，请重新查看“CPU与GPU考量”部分，并确保您的NVIDIA驱动程序、CUDA工具包和cuDNN库版本与您安装的TensorFlow版本兼容。强烈建议查阅TensorFlow官方安装指南，以获取具体的版本兼容性表格。

成功运行这些验证步骤表明您的TensorFlow环境已准备就绪。您现在可以继续查看TensorFlow的核心思想，例如张量和自动微分，在下一章中。

## 参考资料

- [Install TensorFlow](https://www.tensorflow.org/install) — TensorFlow Authors (2025)
  Publisher: Google
  TensorFlow的官方安装指南，包含CPU/GPU设置细节、环境配置和常见问题排查。
- [Get started with TensorFlow](https://www.tensorflow.org/guide/basics) — TensorFlow Authors (2024)
  介绍TensorFlow的核心数据结构（张量）和基础操作，对验证基本功能至关重要。
- [TensorFlow: A System for Large-Scale Machine Learning](https://arxiv.org/abs/1603.04467) — Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, Xiaoqiang Zheng (2016)
  Journal: OSDI '16: 12th USENIX Symposium on Operating Systems Design and Implementation; Pages: 265-283; DOI: [10.48550/arXiv.1603.04467](https://doi.org/10.48550/arXiv.1603.04467)
  描述TensorFlow系统架构及其设计原则的基础性学术论文。
