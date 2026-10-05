# PyTorch Profiler 性能分析

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/pytorch-profiler)

[返回章节目录](README.md) · [返回课程目录](../README.md)

了解模型在执行过程中时间与资源的分配情况，是进行优化的根本。在应用量化 (quantization)或剪枝等技术之前，你需要找出性能瓶颈所在。是CPU限制了性能？是GPU未充分利用？还是特定操作过慢？PyTorch Profiler (`torch.profiler`) 是回答这些问题的标准工具。

该分析器使你能够查看模型执行不同部分的时间和内存开销，包括CPU上的Python操作和GPU上的CUDA内核执行。它提供详细的视图，帮助你进行优化，确保你把精力放在那些能为推理 (inference)带来最大性能提升的方面。

### Profiler测量的内容

`torch.profiler` API 通过跟踪几个重要指标，提供模型执行的全面视图：

1. **运算符执行时间：** 测量单个PyTorch运算符在CPU和GPU（CUDA内核）上的执行时长。它区分“自身时间”（运算符自身代码中花费的时间，不包括对其他运算符的调用）和“总时间”（包括子调用中花费的时间）。
2. **内核启动和GPU利用率：** 跟踪CUDA内核的启动及其在GPU上的执行时间，帮助查看并行性并找出GPU可能空闲的时间段。
3. **内存使用（可选）：** 启用后，它会跟踪CPU和GPU设备上的内存分配和释放情况，帮助找出内存占用高或可能存在内存泄漏的操作。
4. **数据传输：** 通过CUDA运行时事件（如 `cudaMemcpy`）隐式显示，突出显示在主机（CPU）和设备（GPU）之间移动数据所花费的时间。
5. **运算符调用堆栈（可选）：** 启用后，它会记录导致每个分析操作的Python调用堆栈，使回溯耗时操作到源代码中的特定行变得更容易。

### Profiler基本用法

使用分析器最常见的方式是通过其上下文 (context)管理器接口。你将要分析的代码段包装在 `with torch.profiler.profile(...)` 块中。

```python
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

# 加载预训练模型（确保其处于评估模式以进行推理分析）
model = models.resnet18().cuda().eval()
inputs = torch.randn(16, 3, 224, 224).cuda() # GPU上的示例输入批次

# 基本分析上下文
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"): # 该代码块的可选标签
        model(inputs)

# 打印汇总统计信息
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出结果以进行更详细的分析
# prof.export_chrome_trace("resnet18_trace.json")
# prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
```

让我们分解 `profile()` 中使用的参数 (parameter)：

- `activities`: 一个列表，指定要分析的活动。常见选择是 `ProfilerActivity.CPU` 和 `ProfilerActivity.CUDA`。分析CUDA活动对于了解GPU性能很重要。
- `record_shapes`: 如果为 `True`，则记录被分析运算符的输入形状。这对于诊断与形状相关的性能问题有用，但会增加一些开销。
- `profile_memory`: 如果为 `True`，则启用内存分析（分配/释放）。开销较大。
- `with_stack`: 如果为 `True`，则记录Python调用堆栈。对于将运算符回溯到源代码很有用，但开销很大。
- `on_trace_ready`: 一个可调用对象（通常是 `torch.profiler.tensorboard_trace_handler`），用于处理结果导出，例如直接导出到TensorBoard。
- `schedule`: 控制长时间运行作业的分析持续时间。使用 `torch.profiler.schedule(wait, warmup, active, repeat)` 定义阶段：跳过初始 `wait` 步，执行 `warmup` 步（分析器活动但结果被丢弃），记录 `active` 步，并重复此循环 `repeat` 次。这对于排除初始化开销并专注于稳态性能很有用。

`record_function("label")` 上下文管理器将自定义标签添加到分析器输出中，使识别代码中特定逻辑块（如数据预处理、模型前向传播、后处理）变得更容易。

### 分析Profiler结果

分析器对象（示例中的 `prof`）提供了几种分析收集数据的方法：

#### 1. `key_averages()`

此方法返回运算符性能的汇总摘要，该摘要在分析窗口内取平均。调用 `.table()` 提供格式化的字符串输出。

```
# Example Output Snippet from prof.key_averages().table(...)
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Name                                         Self CPU %      Self CPU   CPU total %     CPU total      CUDA %      CUDA total    # Calls
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
aten::convolution                            0.14%     293.164us        17.21%       35.96ms       81.90%       35.91ms        20
aten::cudnn_convolution                      0.00%       0.000us        0.00%       0.000us       81.72%       35.83ms        20
aten::addmm                                  0.06%     117.880us         1.12%        2.34ms        4.97%        2.18ms         1
aten::mm                                     0.00%       0.000us         0.00%       0.000us        4.96%        2.18ms         1
aten::add_                                   0.13%     263.820us         0.51%        1.07ms        3.48%        1.53ms        21
aten::relu                                   0.12%     245.750us         0.28%     577.870us        1.91%     836.370us        16
aten::_native_batch_norm_legit_no_...        0.10%     215.530us         1.99%        4.15ms        1.85%     810.318us        20
aten::empty_strided                          1.76%        3.68ms         1.94%        4.05ms        0.00%       0.000us       140
aten::max_pool2d_with_indices                0.04%      79.630us         0.38%     788.380us        0.79%     346.077us         1
aten::copy_                                  0.06%     120.600us         0.06%     120.600us        0.00%       0.000us         2
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 208.96ms
Self CUDA time total: 44.08ms
```

- **名称：** PyTorch运算符的名称（例如，`aten::convolution`）。`aten` 是PyTorch原生C++运算符的命名空间。
- **自身CPU / CUDA时间：** 在CPU或GPU上，此运算符代码内部直接花费的时间，不包括其调用的函数所花费的时间。
- **总CPU / CUDA时间：** 在此运算符*及其*调用的任何函数中花费的时间（例如，`aten::convolution` 调用 `aten::cudnn_convolution`）。
- **# 调用次数：** 运算符被调用的次数。

你可以对表格进行排序（例如，`sort_by="cuda_time_total"`）并限制行数（`row_limit`），以便关注最耗时的操作。按输入形状（`group_by_input_shape=True`）或堆栈跟踪（`group_by_stack_n`）分组可以提供更多信息。像 `aten::convolution` 这样的运算符如果 `Self CUDA` 时间很高，表明底层的CUDA卷积内核花费了大量时间，这通常是预期的，但能确认GPU时间用在了何处。高 `Self CPU` 时间可能指向Python开销或CPU密集型计算。

#### 2. `export_chrome_trace()`

此方法将详细的时间线数据导出为JSON文件格式，该格式与Chrome的跟踪工具（`chrome://tracing`）或Perfetto UI（推荐）兼容。这种可视化对于理解模型的时间动态很有用。

> 查看跟踪：打开Google Chrome，导航到 `chrome://tracing`，然后点击“加载”，或者使用Perfetto UI：[ui.perfetto.dev](https://ui.perfetto.dev/)。

跟踪视图通常显示：

- **CPU线程：** 表示CPU线程的行，将运算符执行显示为时间线上的块。
- **GPU流：** 表示CUDA流的行（例如，`Stream 7`），显示在GPU上计划的内核执行和内存传输。

> CPU启动GPU内核的简化视图。Chrome跟踪提供详细的时间线视图，显示精确的开始/结束时间以及可能表示空闲期的间隔。

通过检查跟踪，你可以发现：

- **GPU空闲时间：** GPU流时间线上的间隔表示GPU正在等待的时间段。这可能是由于CPU启动内核过慢、数据加载效率低或同步点引起的。
- **CPU瓶颈：** CPU线程上的长块阻碍了后续GPU内核的启动。
- **数据传输开销：** `memcpy` 块显示数据移动所花费的时间。
- **内核持续时间：** GPU流上块的长度显示特定内核运行了多长时间。

#### 3. TensorBoard集成

使用 `torch.profiler.tensorboard_trace_handler` 可在TensorBoard中提供集成体验。

```python
import torch
import torchvision.models as models
from torch.profiler import profile, tensorboard_trace_handler

# 模型和输入设置（同前）
model = models.resnet18().cuda().eval()
inputs = torch.randn(16, 3, 224, 224).cuda()

log_dir = "./logs" # TensorBoard日志目录

with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
             profile_memory=True, # 可选地跟踪内存
             on_trace_ready=tensorboard_trace_handler(log_dir)) as prof:
    model(inputs)

print(f"Profiler results saved to {log_dir}. Run: tensorboard --logdir {log_dir}")
```

启动TensorBoard（`tensorboard --logdir ./logs`）并导航到“PyTorch Profiler”选项卡，提供多种交互式视图：

- **概览：** 步骤时间、运算符时间分布（CPU与GPU）以及工具检测到的潜在瓶颈的高级摘要。
- **运算符视图：** 类似于 `key_averages`，显示每个运算符的详细统计信息。允许过滤和搜索。
- **内核视图：** 专门关注GPU内核性能。
- **跟踪视图：** Chrome跟踪查看器的嵌入 (embedding)版本。
- **内存视图：** （如果 `profile_memory=True`）显示每个运算符的内存使用模式和分配情况。



[交互图表：运算符GPU时间分布（示例）](https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/pytorch-profiler#plot-u0icy6)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "运算符GPU时间分布（示例）",
    "xaxis": {
      "title": "运算符"
    },
    "yaxis": {
      "title": "CUDA总时间 (ms)"
    },
    "showlegend": false,
    "margin": {
      "l": 60,
      "r": 20,
      "t": 40,
      "b": 100
    }
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "aten::cudnn_convolution",
        "aten::mm",
        "aten::add_",
        "aten::relu",
        "aten::_native_batch_norm...",
        "aten::max_pool2d...",
        "Other"
      ],
      "y": [
        35.83,
        2.18,
        1.53,
        0.84,
        0.81,
        0.35,
        2.54
      ],
      "marker": {
        "color": [
          "#1c7ed6",
          "#4dabf7",
          "#74c0fc",
          "#a5d8ff",
          "#d0bfff",
          "#eebefa",
          "#adb5bd"
        ]
      }
    }
  ]
}
```

</details>



> 总GPU时间在不同运算符上的分布示例，源于分析器数据。在CNN中，卷积通常占用大部分时间。

### 识别常见瓶颈

分析器输出直接指向优化机会：

1. **CPU密集型执行：**
   - **症状：** `key_averages` 中总CPU时间高，Python开销大或某些运算符的自身CPU时间长。跟踪视图中GPU利用率低（大间隔）。
   - **可能原因：** 繁重的Python逻辑（循环、数据操作），过多的小操作产生开销，CPU上数据加载/预处理慢。
   - **潜在解决方案：** 优化Python代码，合并小操作，如果可行将预处理移至GPU，优化数据加载流程（`num_workers`，`pin_memory`）。
2. **GPU未充分利用：**
   - **症状：** 跟踪视图中GPU流中存在明显的间隔，意味着GPU经常空闲。CUDA内核花费的时间占总步长时间的比例总体较低。
   - **可能原因：** CPU瓶颈（见上文），低效的内核启动（许多小内核而非少量大内核），并行度不足（例如，批次大小小）。
   - **潜在解决方案：** 解决CPU瓶颈，增加批次大小（如果内存允许），尽可能使用融合内核，考虑模型架构调整。
3. **数据传输开销：**
   - **症状：** 跟踪视图中可见像 `cudaMemcpyDtoH`（设备到主机）或 `cudaMemcpyHtoD`（主机到设备）这样的操作花费了大量时间。
   - **可能原因：** 模型或数据流程中CPU和GPU内存之间频繁且不必要的传输。使用非固定内存进行传输。
   - **潜在解决方案：** 尽量减少数据移动。尽可能确保数据停留在GPU上。在 `DataLoader` 中使用 `pin_memory=True`，并在 `.to(device)` 调用中使用 `non_blocking=True` 以实现传输和计算的重叠。
4. **内存效率低下：**
   - **症状：** `profile_memory=True` 报告的峰值内存使用量高。执行期间出现 `OutOfMemoryError`。
   - **可能原因：** 较大的中间激活，存储不必要的张量，低效的运算符实现。
   - **潜在解决方案：** 推理 (inference)时使用 `torch.no_grad()`，删除不再需要的张量（`del tensor`），使用检查点技术（以计算换内存），应用模型优化技术如量化 (quantization)或剪枝（本章其他部分涵盖），减小批次大小。
5. **低效内核：**
   - **症状：** 特定CUDA内核（例如，自定义内核，甚至是 `aten::convolution` 这样的标准内核）在 `key_averages` 或跟踪视图中显示非常长的执行时间。
   - **可能原因：** 次优的内核实现，在存在专用内核（例如，通过cuDNN）时使用了通用内核，硬件限制。
   - **潜在解决方案：** 确保cuDNN等库已启用，调查替代运算符实现（如果可用），考虑编写自定义优化内核（第6章），或使用混合精度训练等技术（第3章），这些技术有时可以使用更快的Tensor Core内核。

### 高级用法和注意事项

- **自定义代码块：** 使用 `with torch.profiler.record_function("my_label"):` 向分析结果添加自定义注释，从而更容易将性能数据与代码的特定部分（例如，“data\_preprocessing”、“attention\_block”）关联起来。
- **分析调度：** 对于长时间运行的训练任务或复杂的推理 (inference)流程，使用 `torch.profiler.profile` 的 `schedule` 参数 (parameter)，以便在初始预热期后捕获特定迭代，避免生成过大的跟踪文件，并专注于稳态行为。
- **Profiler开销：** 请记住，分析会增加开销，特别是在启用内存或堆栈跟踪时。分析代表性的输入和模型状态，但不要在生产部署中持续启用分析器。进行初步检查时，使用更简单的分析设置（例如，仅CPU/CUDA活动，不带形状/内存/堆栈）以降低开销。
- **迭代改进：** 性能分析是一个迭代循环：分析 -> 识别瓶颈 -> 应用优化 -> 再次分析 -> 衡量改进。

通过系统地使用PyTorch Profiler，你将对模型的运行时行为获得必要的了解，以便明智地决定在哪里以及如何有效地应用优化技术，最终得到更快速、更高效的模型，为部署做好准备。

## 参考资料

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) — PyTorch Authors (2025)
  Publisher: PyTorch Foundation
  `torch.profiler` API 的官方文档，详细说明其功能、配置选项和与其他工具的集成。
- [PyTorch Profiler with TensorBoard](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — Shivam Raikundalia (2021)
  Publisher: PyTorch Foundation
  一个官方 PyTorch 教程，提供实际示例和使用 profiler（特别是与 TensorBoard 结合进行可视化）的分步指南。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA Nsight Systems 的官方用户指南，这是一个用于优化 NVIDIA GPU 上应用程序的系统级 profiler，提供对硬件利用率的了解。
- [Trace Viewer Guide](https://perfetto.dev/docs/) — Perfetto Authors (2024)
  Publisher: Perfetto Project
  Perfetto Trace Viewer 的使用指南，它与 PyTorch Profiler 导出的跟踪文件兼容，用于详细的时间线分析。

---

[上一节](03-%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D%E7%AD%96%E7%95%A5.md) · [下一节](05-%E9%80%9A%E8%BF%87%E5%A4%96%E9%83%A8%E5%BA%93%E4%BC%98%E5%8C%96%E7%AE%97%E5%AD%90.md)
