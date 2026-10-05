# 动手实践：CPU 与 GPU 性能对比

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/practical-benchmarking-cpu-gpu)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本次动手实验着重于直接的性能测量。一个简单的 Python 脚本将执行常见的机器学习 (machine learning)操作——矩阵乘法，并对比其在 CPU 和 GPU 上的运行时间。这项练习将提供对 GPU 为并行任务带来的加速的直观感受。

目的不是进行严谨的科学研究，而是观察导致人工智能硬件选择的量级差异。

### 对比任务：矩阵乘法

大多数神经网络 (neural network)的核心是一个基本操作：矩阵乘法。神经网络的一层通常可以表示为：


$$
\text{输出} = \text{激活}(\text{权重} \cdot \text{输入} + \text{偏置})
$$


操作 $\text{权重} \cdot \text{输入}$ 是一个大型矩阵乘法。计算结果矩阵中的每个元素可以独立于其他元素进行，这使其成为一个“高度并行”的问题。这正是 GPU 架构凭借其数千个核心，预计能胜过 CPU 少数但更强大核心的工作负载类型。

### 前提条件

要进行此操作，您需要一个安装了 PyTorch 的 Python 环境。PyTorch 是一个流行的深度学习 (deep learning)框架，它为在不同硬件设备上运行计算提供了简单的接口。

如果您有 NVIDIA GPU，请确保安装支持 CUDA 的 PyTorch 版本。您可以在 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)上找到正确的命令。它通常看起来像这样：

```bash
# 示例命令，请查看官网获取最新版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果您没有本地 GPU，可以在 Google Colab 上免费运行整个练习，它提供了支持 GPU 的环境。只需打开一个新的 Colab 笔记本，在菜单中选择 `Runtime > Change runtime type` 并选择 `GPU` 作为硬件加速器。

### 对比代码

我们将编写一个脚本，执行相同的任务——一系列大型矩阵乘法，首先在 CPU 上，然后在 GPU 上。我们两者都使用 PyTorch 以确保公平对比，因为操作的底层实现是相同的；只有执行它的硬件发生了变化。

创建一个名为 `benchmark.py` 的文件，并添加以下代码：

```python
import torch
import time

def benchmark(device_name, matrix_size=4096, iterations=10):
    """
    在指定设备上执行矩阵乘法对比测试。

    参数:
        device_name (str): 要运行的设备，'cpu' 或 'cuda'。
        matrix_size (int): 方形矩阵的维度。
        iterations (int): 重复乘法的次数。
    """
    print(f"--- 在 {device_name.upper()} 上进行对比测试 ---")

    # 设置张量分配的设备
    device = torch.device(device_name)

    # 在指定设备上创建两个大型随机矩阵
    # 使用 .to(device) 将张量移动到目标硬件
    try:
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
    except torch.cuda.OutOfMemoryError:
        print(f"GPU 内存不足。请尝试更小的 matrix_size。")
        return
    except Exception as e:
        print(f"发生错误: {e}")
        return

    # 预热运行，以处理任何初始设置成本
    _ = torch.matmul(a, b)

    # 对于 GPU，在开始计时前同步，以确保
    # 所有先前的操作都已完成。
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(iterations):
        c = torch.matmul(a, b)

    # 对于 GPU，再次同步以确保乘法
    # 在我们停止计时器前完成。
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_iter = (total_time / iterations) * 1000  # 单位：毫秒

    print(f"矩阵大小: {matrix_size}x{matrix_size}")
    print(f"迭代次数: {iterations}")
    print(f"总时间: {total_time:.4f} 秒")
    print(f"每次乘法的平均时间: {avg_time_per_iter:.4f} 毫秒\n")

if __name__ == "__main__":
    # 在 CPU 上进行对比测试
    benchmark('cpu')

    # 检查 CUDA-启用 的 GPU 是否可用
    if torch.cuda.is_available():
        # 在 GPU 上进行对比测试
        benchmark('cuda')
    else:
        print("--- 此系统上没有可用的 CUDA (GPU)。---")
        print("跳过 GPU 对比测试。")
```

#### 理解代码

- **设备处理:** `torch.device(device_name)` 调用会创建一个设备对象。当我们创建张量 `a` 和 `b` 时，我们传递 `device=device` 以确保它们直接在目标硬件（CPU 内存或 GPU 显存 (VRAM)）上创建。
- **`torch.cuda.synchronize()`:** 这是进行精确 GPU 计时的一个重要函数。CPU 代码和 GPU 代码是异步运行的。当您在 GPU 上调用 `torch.matmul(a, b)` 时，CPU 会将操作排队，并可能在 GPU 实际完成计算之前立即移动到下一行代码（例如停止计时器）。`torch.cuda.synchronize()` 强制 CPU 等待所有先前排队的 GPU 任务完成，从而为我们提供准确的实际运行时间测量。

### 运行对比测试

打开您的终端，导航到保存 `benchmark.py` 的目录，然后运行它：

```bash
python benchmark.py
```

### 分析结果

您将看到类似以下内容的输出，尽管确切数字会因您具体的 CPU 和 GPU 型号而有很大差异。

```
--- Benchmarking on CPU ---
Matrix Size: 4096x4096
Iterations: 10
Total time: 11.8921 seconds
Average time per multiplication: 1189.2100 ms

--- Benchmarking on GPU ---
Matrix Size: 4096x4096
Iterations: 10
Total time: 0.0754 seconds
Average time per multiplication: 7.5400 ms
```

差异显著。在这个示例运行中，GPU 完成相同任务的速度比 CPU 快 150 多倍。这并非因为 CPU “不好”。它只是不适合这项特定的工作。CPU 的核心忙于顺序处理指令，而 GPU 的数千个核心则同时处理矩阵乘法的一小部分。



[交互图表：对比测试：CPU 与 GPU 矩阵乘法](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/practical-benchmarking-cpu-gpu#plot-dsmilz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "对比测试：CPU 与 GPU 矩阵乘法"
    },
    "xaxis": {
      "title": {
        "text": "硬件设备"
      }
    },
    "yaxis": {
      "title": {
        "text": "总时间（秒）"
      }
    },
    "plot_bgcolor": "#f8f9fa",
    "paper_bgcolor": "#f8f9fa"
  },
  "data": [
    {
      "x": [
        "CPU",
        "GPU"
      ],
      "y": [
        11.89,
        0.075
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#339af0",
          "#20c997"
        ]
      }
    }
  ]
}
```

</details>



> 10 次 4096x4096 大小矩阵乘法的执行时间。GPU 完成任务所需的时间只是 CPU 的一小部分。值越小越好。

这一实际结果是 GPU 成为深度学习 (deep learning)事实标准的核心原因。以数量级加速这些核心并行操作的能力，直接意味着将模型训练时间从数周缩短到数小时。随着我们继续，请记住这一根本的性能差距。它影响着我们做出的每一个决定，从选择云实例到设计多节点训练集群。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2023)
  Publisher: PyTorch Foundation
  PyTorch 深度学习框架的官方文档，涵盖张量操作、设备管理和 CUDA 集成，这些对于 CPU/GPU 基准测试至关重要。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，提供神经网络的全面理论背景，包括矩阵乘法在深度学习模型中的作用。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://books.google.com/books/about/Programming_Massively_Parallel_Processors.html?id=c738DQAAQBAJ) — David B. Kirk, Wen-mei W. Hwu (2017)
  Publisher: Morgan Kaufmann; DOI: [10.1016/C2016-0-03099-X](https://doi.org/10.1016/C2016-0-03099-X)
  这本书解释了 GPU 的架构和编程原理，深入了解了它们为何在高性能计算中擅长矩阵乘法等并行任务。

---

[上一节](08-%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F%E7%9A%84%E7%BD%91%E7%BB%9C%E8%80%83%E9%87%8F.md) · [下一节](../02-%E8%AE%BE%E8%AE%A1%E6%9C%AC%E5%9C%B0AI%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/01-%E8%AF%84%E4%BC%B0%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E9%9C%80%E6%B1%82.md)
