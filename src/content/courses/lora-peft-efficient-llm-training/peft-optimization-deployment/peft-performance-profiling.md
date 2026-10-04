---
course: "lora-peft-efficient-llm-training"
chapter: "peft-optimization-deployment"
lesson: "peft-performance-profiling"
sourceId: 4165
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-performance-profiling"
title: "PEFT 训练与推理的性能分析"
description: "分析 PEFT 工作流程中 GPU 利用率、内存使用和吞吐量的工具和方法。"
order: 5
plots: ["plots/4165-0.json", "plots/4165-1.json"]
sourceHash: "c4ebf9ba55ee9577d88233c13d06719ae275cff11e1629f53f85721315ce4035"
sourceCorrections: []
---

即使使用参数 (parameter)高效的方法进行微调 (fine-tuning)，也需要仔细关注性能，最大限度地提高效率并降低成本。简单地应用 LoRA 或 QLoRA 并不能保证资源获得最佳使用。性能分析提供所需的信息，以识别训练和推理 (inference)流程中的瓶颈，从而实现有针对性的优化。如果没有分析，尝试提高速度或减少内存使用通常依赖于猜测，这可能无效甚至适得其反。覆盖的主题包括分析 PEFT 工作流程性能特点的指标、工具和方法。

### 理解性能指标

有效的分析始于了解需要测量什么。相关的指标在训练阶段和推理 (inference)部署阶段通常有所不同。

**训练指标：**

- **GPU 利用率：** 衡量 GPU 计算单元的活跃程度。持续低利用率（例如，低于 70-80%）通常表明瓶颈在其他地方，例如数据加载或 CPU 处理。`nvidia-smi` 等工具提供实时视图。
- **GPU 内存使用：** 跟踪 GPU RAM 的消耗量。重要方面包括峰值内存使用（决定作业是否适合硬件）和平均使用量。PEFT 显著减少了*参数 (parameter)*内存，但激活和优化器状态（特别是使用 AdamW 等优化器时）仍占用大量内存。分析工具可以显示内存分配峰值。
- **训练吞吐量 (throughput)：** 量化 (quantization)训练速度，通常以每秒样本数、每秒 token 数或每秒步数衡量。更高的吞吐量通常意味着更快的训练收敛和更低的计算成本。这通常由训练脚本报告，或可从日志中获得。
  "\* **实际运行时间：** 训练周期或整个作业所花费的总时间。这受吞吐量、数据加载时间以及任何系统等待的影响。"
- **I/O 等待时间：** 等待从存储读取数据所花费的时间。大量的 I/O 等待可能导致 GPU 饥饿，从而降低利用率。系统级工具或框架分析器有时可以突出显示这一点。

**推理指标：**

- **延迟：** 处理单个推理请求（或一批）所需的时间。通常衡量从请求到达至响应生成的端到端时间。低延迟对于实时应用非常重要。
- **吞吐量：** 单位时间内处理的推理请求数量（例如，每秒请求数）。更高的吞吐量对于同时服务许多用户很重要。延迟和吞吐量之间通常存在权衡，尤其是在使用批处理时。
- **GPU 利用率（推理）：** 类似于训练，表明 GPU 在推理期间的使用效率。除非处理持续大量请求或大型批次，否则可能低于训练时的利用率。
- **GPU 内存使用（推理）：** 主要反映存储基础模型和活跃 PEFT 适配器所需的内存。对于 LoRA，合并权重 (weight)消除了在推理期间存储单独 A 和 B 矩阵的需要，与动态适配器加载相比，略微减少了内存使用。

### 性能分析工具

有多种工具可用于收集性能数据。选择合适的工具取决于所需的详细程度以及正在检查的系统特定方面。

- **`nvidia-smi` (NVIDIA 系统管理界面)：** 一个命令行工具，提供 GPU 利用率、内存使用、温度和功耗的实时监控。非常适合在运行时进行快速检查和基本监控。

  ```bash
  # 每秒更新一次 GPU 状态
  watch -n 1 nvidia-smi
  ```
- **NVIDIA Nsight Systems：** 一个系统级性能分析工具。它捕获 CPU 和 GPU 活动的详细时间线，包括 API 调用、内核执行和内存传输。它很适合识别 CPU、GPU 和系统内存之间的相互影响，并准确找出数据传输延迟或 CPU 密集型预处理等瓶颈。
- **NVIDIA Nsight Compute：** 一个 GPU 内核分析器。它提供 CUDA 内核的详细分析，显示有关指令吞吐量 (throughput)、内存访问模式和占用率的信息。当 Nsight Systems 识别出特定 GPU 内核是瓶颈时，它对高级优化很有用。
- **PyTorch 分析器 (`torch.profiler`)：** 直接集成到 PyTorch 中，此工具可分析训练或推理 (inference)脚本中的 CPU 和 GPU 操作。它可以跟踪操作员执行时间、GPU 内核启动和内存分配事件。结果可以在 TensorBoard 或 Chrome 的 `chrome://tracing` 工具中轻松查看。

  ```python
  import torch
  from torch.profiler import profile, record_function, ProfilerActivity

  # 假设模型、数据加载器、标准、优化器已定义

  # 用于分析特定代码块的上下文管理器
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
               record_shapes=True, 
               profile_memory=True) as prof:
      # 使用 record_function 标记代码的特定部分
      with record_function("data_loading"):
          # 模拟或获取数据批次
          inputs = get_next_batch() 

      with record_function("model_forward_backward"):
          outputs = model(inputs)
          loss = criterion(outputs, labels) # 假设标签可用
          loss.backward()

      with record_function("optimizer_step"):
          optimizer.step()
          optimizer.zero_grad()

  # 打印按 CUDA 时间排序的聚合统计数据
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

  # 打印按 CPU 时间排序的聚合统计数据
  print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

  # 可选地导出跟踪数据以进行详细的时间线可视化
  # prof.export_chrome_trace("peft_train_trace.json") 
  ```

  > 注意：`get_next_batch()` 和 `labels` 部分是数据处理逻辑的示例占位符。
- **TensorFlow 分析器 (`tf.profiler`)：** TensorFlow 内置的分析工具，提供与 PyTorch 分析器相似的功能。它与 TensorBoard 集成用于可视化，提供操作时间、GPU 利用率和潜在输入流程问题的信息。捕获配置文件通常涉及 `model.fit` 期间的回调或使用 `tf.profiler.experimental.start/stop`。
- **Python 分析器 (`cProfile`、`line_profiler`)：** 标准 Python 工具，可用于识别 CPU 密集型代码中的性能瓶颈，例如复杂的数据预处理或主模型执行之外的自定义逻辑。`line_profiler` 提供逐行计时信息，并要求修饰要分析的函数。

### 训练循环分析

PEFT 训练虽然参数 (parameter)较少，但仍涉及计算密集型步骤。分析有助于细分每次训练迭代中时间和资源的消耗情况。

1. **数据加载和预处理：** 使用框架分析器（如上述 PyTorch 示例）或 Python 分析器（`cProfile`、`line_profiler`）分析您的 `DataLoader` (PyTorch) 或 `tf.data` 流程 (TensorFlow)。需要关注：
   - 在时间线视图中，步骤开始时 GPU 利用率较低，这表明 GPU 正在等待数据。检查数据加载函数中 CPU 所花费的时间。
   - 如果预处理复杂且串行执行，则 CPU 使用率高。
   - 通过增加数据加载器工作进程数（PyTorch DataLoader 中的 `num_workers`）、启用 `pin_memory=True` (PyTorch) 以加快 CPU 到 GPU 的传输、使用异步数据加载模式、离线预处理数据或简化计算量大的转换来优化。
2. **前向传播：** 使用框架分析器分析模型前向传播的执行时间。LoRA 等 PEFT 方法在这里增加的计算开销很小，因为它们涉及小的矩阵乘法（在原始层的计算中添加 $BAx$ 计算）。QLoRA 可能会因为计算前的反量化 (quantization)操作而引入轻微的开销。检查分析器输出中模型 `forward` 方法和特定层（特别是那些使用适配器修改的层）中花费的时间。
3. **反向传播 (backpropagation)：** 反向传播计算梯度。虽然 PEFT 中更新的参数较少，但梯度仍需流经网络的冻结部分才能到达可训练适配器。分析 `loss.backward()`。由于梯度计算所需的存储激活，内存使用通常在此处达到峰值。框架分析器可以显示反向操作所消耗的时间和内存。
4. **优化器步进：** 分析 `optimizer.step()`。复杂性取决于优化器。标准 AdamW 需要为*所有*可训练参数存储一阶 ($m$) 和二阶 ($v$) 矩估计。即使 PEFT 的参数数量减少，这仍然可能很重要。AdamW 8 位或 QLoRA 分页优化器等内存高效优化器可以大幅减小此占用空间，这可以通过在此步骤中分析内存使用情况来验证（PyTorch 分析器中的 `profile_memory=True`）。

**可视化训练分析结果：**

TensorBoard 等工具，在接收 PyTorch 分析器 (`export_chrome_trace`) 或 TF 分析器的数据后，提供交互式时间线（通常在“Trace Viewer”下）。这些可视化清晰地显示了 CPU 和 GPU 上的操作序列，使得更容易发现空闲时间（GPU 活动中的空白）或耗时过长的操作。



![每个训练步骤的时间分解](plots/4165-0.json)



> 一个分解图，其中“数据加载”的 CPU 时间明显超过其 GPU 对应时间和其他阶段，表明存在数据加载瓶颈。GPU 时间主要集中在前向和反向传播中。

### 推理 (inference)性能分析

对于部署而言，推理速度和效率是主要考量。

1. **延迟测量：** 分析模型应用 PEFT 适配器后单次前向传播所需的时间。可以使用 `time.time()` 或框架特定的计时机制围绕推理调用进行简单测量，或使用框架分析器获取更多细节。如果包括必要的预处理/后处理步骤（如分词 (tokenization)和解码），则测量端到端延迟。使用代表性的输入长度进行测试，因为延迟通常随序列长度而变化。
2. **吞吐量 (throughput)测量：** 测试系统每秒可以处理多少请求。这通常涉及发送并发请求或批量输入。使用分析工具来监控负载下的 GPU 利用率和平均延迟。压力测试工具或自定义脚本可以模拟真实的流量模式。
3. **批处理影响：** 系统地评估推理延迟和吞吐量随批次大小增加而如何变化。通常，更大的批次会提高总吞吐量（每秒处理更多样本），直到 GPU 饱和为止，但它们通常会增加批次中任何单个请求的延迟。分析内存使用情况以确保批次符合 GPU 限制。找到满足应用程序延迟和吞吐量要求的最佳批次大小。
4. **适配器加载/切换：** 如果您的应用程序动态加载不同的 PEFT 适配器（例如，用于共享基础模型上的不同用户或任务），请分析从存储加载适配器权重 (weight)并将其集成到模型所需的时间。如果适配器很小且切换不频繁，则此开销可能可以忽略不计，否则可能会变得很重要。可以考虑将常用适配器缓存在内存中。
5. **合并 LoRA 与未合并 LoRA：** 分析动态应用 LoRA 适配器（即时计算 $W_0x + \alpha BAx$）与使用预合并权重（$W = W_0 + \alpha BA$，然后计算 $Wx$）进行推理时的延迟和内存使用。合并消除了额外的矩阵乘法 ($BAx$) 以及在运行时单独存储 $A$ 和 $B$ 的需要，可能略微降低延迟并简化推理代码路径。性能差异取决于硬件、批次大小和 LoRA 配置（`r`、`alpha`）。



![推理延迟与吞吐量（对数刻度批次大小）](plots/4165-1.json)



> 批次大小、每个请求的平均延迟和推理期间总吞吐量之间的示例关系。增加批次大小（对数刻度 X 轴）可以提高吞吐量，但也会增加每个请求的延迟。

### 分析结果解读与行动

分析数据只有在能带来可行建议和优化时才有价值。

- **GPU 利用率低：** 如果 `nvidia-smi` 或分析器时间线显示训练期间 GPU 有大量空闲时间：
  - 首先检查数据流程（CPU 时间、I/O 等待）。增加数据加载器工作进程，优化预处理逻辑，使用更快的存储，或预取数据。
  - 确保批次大小足够大，以有效利用 GPU 的计算核心，并在此与内存限制之间取得平衡。
  - 检查是否存在同步点或过多的 CPU-GPU 数据传输。
- **内存瓶颈（OOM 错误或高使用率）：**
  - 减少每个设备的批次大小。
  - 使用梯度累积来模拟更大批次大小的效果，同时降低峰值内存使用。
  - 采用内存高效的优化器（AdamW 8 位，使用 QLoRA 时由 `bitsandbytes` 等库提供的分页优化器）。
  - 直接应用 QLoRA，因为基础模型的 4 位量化 (quantization)会大幅减少其内存占用。
  - 使用分析器内存视图检查潜在的内存泄漏或碎片。
  - 仔细检查您的 PEFT 配置，确保只有预期的适配器参数 (parameter)（可能还有像层归一化 (normalization)这样重要的组件）是可训练的。正确冻结基础模型是基本要求。
- **计算瓶颈（GPU 利用率高，吞吐量 (throughput)低）：**
  - 如果尚未启用，请开启混合精度训练（例如，使用 `torch.amp.autocast` 或 `tf.keras.mixed_precision.Policy`），以利用 Tensor Cores 进行更快计算，同时减少内存带宽需求。
  - 如果分析器识别出的特定操作耗时过长，请检查是否存在替代实现，或者 Nsight Compute 是否显示了内核级效率低下。
  - 确保您使用的是最新且优化的库（CUDA、cuDNN、PyTorch/TensorFlow）。
  - 对于推理 (inference)，如果您的部署策略可行，请考虑合并 LoRA 权重 (weight)。
  - 尝试 PEFT 超参数 (hyperparameter)：过高的 LoRA 秩 `r` 可能会引入超出所需任务性能提升的计算量。如果计算受限，请分析不同秩的情况。
- **推理优化重点：**
  - 根据分析结果仔细调整推理批次大小，以根据应用程序要求平衡延迟和吞吐量。
  - 使用模型编译工具，如 `torch.compile` (PyTorch 2.0+)、TensorFlow XLA 或专用推理引擎（如 TensorRT），以融合操作、优化内核启动，并可能进一步量化。
  - 如果动态适配器切换缓慢，请考虑缓存已加载的适配器或使用为多租户推理和共享基础模型设计的服务器架构。

性能分析是一个迭代过程。根据您的发现实施优化，然后再次分析以衡量影响并识别下一个潜在瓶颈。系统地分析和处理这些性能特点对于在实际应用中有效且高效地部署 PEFT 微调 (fine-tuning)模型非常重要。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍QLoRA，一种内存高效的微调方法，利用4位量化和分页优化器，与内容中讨论的内存优化和性能特点直接相关。
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  Nsight Systems的官方指南，该系统是一个系统级分析器，用于详细的CPU-GPU活动时间线，对识别深层瓶颈至关重要。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR 2022); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  引入低秩适应（LoRA）的基础论文，为理解PEFT方法的性能和内存特性提供了基础。
