---
course: "practical-llm-quantization"
chapter: "evaluating-deploying-quantized-llms"
lesson: "benchmarking-quantized-llms"
sourceId: 4669
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-6-evaluating-deploying-quantized-llms/benchmarking-quantized-llms"
title: "衡量推理速度和内存占用"
description: "测量量化模型推理延迟、吞吐量和内存占用的方法。"
order: 2
plots: ["plots/4669-0.json", "plots/4669-1.json"]
sourceHash: "18f1fc13b794848c18f47c42a93d9858b844943093c413f0f6766909f2914b8c"
sourceCorrections: []
---

要量化 (quantization)从模型量化中获得的好处，需要仔细测量推理 (inference)速度和内存消耗。上一节讨论了评估模型*质量*（如困惑度或任务准确率），而此处我们着重衡量*效率*的提升，这往往是量化大型语言模型的主要原因。

### 衡量推理 (inference)速度

推理速度说明了模型处理输入和生成输出的速度。使用两个主要指标：延迟和吞吐量 (throughput)。

**延迟**
延迟是处理单个输入请求所需的时间。它通常以每词元 (token)毫秒数 (ms/token) 或每请求毫秒数 (ms/request) 来衡量。延迟越低越好，表示响应时间越快，这对于聊天机器人等交互式应用很重要。

为可靠地测量延迟：

1. **预热：** 在开始测量之前运行几次推理请求。这使系统能够将必要的组件加载到内存或缓存中，确保后续测量反映稳定状态下的性能。
2. **分离推理部分：** 仅测量模型前向传播的执行时间。排除预处理输入（如分词 (tokenization)）和后处理输出（如解码）所花费的时间，除非您正在对端到端应用程序响应时间进行基准测试。
3. **重复并取平均值：** 使用相同（或相似）的输入执行多次推理运行，并对时间进行平均。这可以减少系统变动的影响。在Python中使用`time.perf_counter()`进行高精度计时。

```python
import time

# 例子 - 替换为实际的模型推理调用
def run_inference(model, inputs):
    # 模拟模型处理时间
    time.sleep(0.1) 
    return "output"

# --- 预热阶段 ---
print("正在预热...")
for _ in range(5):
    _ = run_inference(model, sample_input)

# --- 测量阶段 ---
latencies = []
num_runs = 50
print(f"正在测量 {num_runs} 次运行的延迟...")
for i in range(num_runs):
    start_time = time.perf_counter()
    _ = run_inference(model, sample_input)
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000) # 转换为毫秒

avg_latency = sum(latencies) / len(latencies)
print(f"平均延迟: {avg_latency:.2f} 毫秒/请求")
```

**吞吐量**
吞吐量衡量模型在给定时间内可以处理的请求数量，通常表示为每秒请求数或每秒词元数。高吞吐量对于同时服务许多用户或离线处理大型数据集的应用是理想的。

吞吐量受批处理影响很大，批处理是指同时处理多个输入请求。通常，更大的批处理大小会在一定程度上提升吞吐量，但受限于可用的硬件资源（如GPU内存）。

为测量吞吐量：

1. **定义批处理大小：** 选择一个有代表性的批处理大小（或测试几个不同大小）。
2. **准备批处理输入：** 创建输入数据的批次。
3. **计时批处理推理：** 测量处理大量批次所需的时间。
4. **计算速率：** 将处理的请求总数除以总耗时。


$$
\text{吞吐量} = \frac{\text{处理的请求总数}}{\text{总耗时}}
$$


```python
import time

# 例子 - 替换为实际的批处理推理
def run_batched_inference(model, batch_inputs):
    # 模拟处理一个批次
    time.sleep(0.5) # 时间取决于批处理大小
    return ["output"] * len(batch_inputs)

batch_size = 8
num_batches = 20
total_requests = batch_size * num_batches
timings = []

print(f"正在测量批处理大小为 {batch_size} 的吞吐量...")
# 预热 (可选，但对批处理也推荐)

start_total_time = time.perf_counter()
for i in range(num_batches):
    # 假设 batch_input 已准备好
    batch_input = [sample_input] * batch_size 
    _ = run_batched_inference(model, batch_input)
end_total_time = time.perf_counter()

total_time_taken = end_total_time - start_total_time
throughput = total_requests / total_time_taken

print(f"在 {total_time_taken:.2f} 秒内处理了 {total_requests} 个请求。")
print(f"吞吐量: {throughput:.2f} 请求/秒")
```

模型量化 (quantization)通常会降低延迟并提升潜在吞吐量，因为在兼容硬件上，整数运算（如INT8或INT4）通常比浮点运算（FP16，FP32）快，且更小的数据类型能减少内存带宽需求。



![推理延迟对比](plots/4669-0.json)



> 每请求的平均延迟，展示了典型硬件上不同量化级别的表现。值越小表示单请求处理速度越快。

### 衡量内存占用

量化 (quantization)的主要好处通常是减少内存占用。这包含模型权重 (weight)存储和推理 (inference)时的激活内存。

**模型大小（存储）**
这是最直接的指标：只需检查磁盘上保存的模型文件大小即可。量化模型（如GGUF、GPTQ文件）将比FP16或FP32模型显著更小，大致与比特位宽的减少成比例（例如，INT4模型大小应约为FP16的1/4）。

**推理内存（峰值占用）**
这测量模型进行推理时消耗的最大RAM（CPU）或VRAM（GPU）量。这通常比磁盘大小更为重要，因为它决定了运行模型所需的硬件。

为测量峰值内存占用：

1. **GPU VRAM：** 使用`nvidia-smi`（适用于NVIDIA GPU）等工具或特定库函数（如PyTorch的`torch.cuda.max_memory_allocated()`）来监测推理期间的VRAM占用。使用典型输入运行推理并记录峰值。

   ```bash
   # 终端命令，用于在脚本运行时监测GPU内存
   watch -n 0.5 nvidia-smi
   ```
2. **CPU RAM：** 使用系统监测工具（如Linux上的`htop`、macOS上的Activity Monitor、Windows上的任务管理器）或Python库（如`psutil`）来追踪推理进程的内存占用。

   ```python
   import psutil
   import os
   import time
   import torch # 假设 PyTorch 环境
   ```

# 例子 - 替换为实际的模型加载和推理

````
def load_model():
    # 模拟加载一个大型模型
    time.sleep(2) 
    return "model_object"
    
def run_inference(model, inputs):
    # 模拟推理内存峰值
    _ = torch.randn(1024, 1024, device='cuda' if torch.cuda.is_available() else 'cpu') # 示例内存分配
    time.sleep(0.1)
    return "output"
    
process = psutil.Process(os.getpid())

# 测量模型加载前的内存
mem_before_load = process.memory_info().rss / (1024 * 1024) # MB
print(f"加载前内存: {mem_before_load:.2f} MB")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    vram_before_load = torch.cuda.memory_allocated() / (1024 * 1024) # MB
    print(f"加载前 VRAM: {vram_before_load:.2f} MB")

# 加载模型并测量加载后的内存
model = load_model()
mem_after_load = process.memory_info().rss / (1024 * 1024) # MB
print(f"加载后内存: {mem_after_load:.2f} MB")

if torch.cuda.is_available():
    vram_after_load = torch.cuda.memory_allocated() / (1024 * 1024) # MB
    print(f"加载后 VRAM: {vram_after_load:.2f} MB")
    
# 运行推理并测量推理期间的峰值内存
_ = run_inference(model, "sample_input")
mem_peak_inference = process.memory_info().rss / (1024 * 1024) # MB (运行后的快照)
print(f"推理运行后内存: {mem_peak_inference:.2f} MB")

if torch.cuda.is_available():
    peak_vram_inference = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
    print(f"推理期间峰值 VRAM: {peak_vram_inference:.2f} MB")

# 注意: 为了准确获得峰值 RAM，可能需要在 run_inference 调用期间在单独的线程/进程中进行监测，
# 因为这里的 'mem_peak_inference' 只是运行后的一个快照。
```
````

较低的比特精度减少了权重和激活所需的内存，使得更大的模型可以适应相同的硬件，或相同的模型能在算力 (compute)较弱的硬件上运行。



![推理期间 GPU VRAM 峰值占用](plots/4669-1.json)



> 针对不同量化级别，在示例推理任务中的GPU VRAM峰值消耗。值越小，模型就能在内存较小的GPU上运行。

### 可靠基准测试的注意事项

在比较量化 (quantization)模型与其基线版本或其他量化版本的性能时，一致性很重要：

- **硬件：** 始终在*相同*的硬件配置（GPU型号、CPU、RAM）上进行基准测试。不同硬件的性能差异很大。
- **软件：** 使用相同版本的库（PyTorch、TensorFlow、Transformers、Optimum、bitsandbytes）、CUDA驱动程序和操作系统。更新有时会影响性能。
- **输入数据：** 对于所有您打算直接比较的基准测试运行，请使用相同或统计学上相似的输入数据（序列长度、批处理大小）。更长的序列或更大的批次自然需要更多时间和内存。
- **测试条件：** 确保所有测试中系统处于相似的负载条件下。避免同时运行其他繁重进程。

通过在受控条件下仔细测量速度和内存占用，您可以准确量化量化带来的效率提升。这些测量结果，结合先前讨论的准确性评估，提供了做出明智决策所需的数据，以权衡相关利弊，并为您的特定部署场景选择最合适的量化模型。

## 参考资料

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — Szymon Migacz (2024)
  为PyTorch模型的优化和基准测试提供了详尽的指导，涵盖了衡量推理速度和内存消耗的通用性能方法。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了GPTQ，一种广泛用于LLM的训练后量化方法，详细阐述了其方法论，并展示了在推理速度和内存占用减少方面的全面基准测试。
- [PyTorch CUDA Semantics for Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management) — PyTorch Documentation (2024)
  阐明了PyTorch如何管理CUDA设备内存，并提供了监控显存使用情况的函数信息，例如`torch.cuda.max_memory_allocated`。
- [psutil documentation](https://psutil.readthedocs.io/en/latest/) — Giampaolo Rodola and contributors (2024)
  `psutil`库的官方文档，提供了以编程方式获取系统数据的方法，包括在不同操作系统上的进程内存使用情况。
