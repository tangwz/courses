---
course: "advanced-pytorch"
chapter: "deployment-performance-optimization"
lesson: "practice-profiling-quantization"
sourceId: 2755
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/practice-profiling-quantization"
title: "实践：模型性能分析与量化"
description: "使用性能分析工具找出性能瓶颈，并对模型应用训练后静态量化。"
order: 8
plots: []
sourceHash: "1986c04619ec260eedf05f0dbf2e444aa0ec8039d217de5dc32dc4841184b232"
sourceCorrections: []
---

提供一个动手练习，将性能分析和量化 (quantization)方法应用于标准 PyTorch 模型。目标是使用性能分析工具确定性能特征。随后，通过训练后静态量化（PTQ）减小模型大小并可能加快其推理 (inference)速度。此练习反映了准备模型部署时的常见流程。

我们假定您已安装 `torchvision` 并拥有可用的 PyTorch 环境。

### 环境和模型的准备

首先，我们需要导入所需的库并载入一个预训练 (pre-training)模型。我们将使用 `torchvision` 中的 ResNet18 作为示例模型。它足够复杂，可显示有意义的结果，但又足够小，可在此练习中快速运行。我们还需要一些模拟输入数据。

```python
import torch
import torchvision.models as models
import torch.quantization
import torch.profiler
import copy
import time
import os
import numpy as np

# 检查 CUDA 是否可用，如果不可用则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 载入预训练的 ResNet18 模型
original_model = models.resnet18(pretrained=True)
original_model.eval() # 将模型设置为评估模式
model_fp32 = copy.deepcopy(original_model).to(device)

# 创建与 ResNet18 预期输入形状匹配的模拟输入数据
# (批大小, 通道数, 高, 宽)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 保存模型并返回大小的函数
def get_model_size(model, file_path="temp_model.pt"):
    torch.save(model.state_dict(), file_path)
    size = os.path.getsize(file_path) / (1024 * 1024) # 大小，单位为 MB
    os.remove(file_path)
    return size
```

请务必使用 `model.eval()` 将模型设置为评估模式。这很重要，因为它会禁用 Dropout 等层，并使用运行统计数据对 BatchNorm 层进行归一化 (normalization)，这对于一致的推理 (inference)和量化 (quantization)非常必要。

### 原始浮点模型的性能分析

在优化之前，我们先建立一个基准。我们将使用 `torch.profiler.profile` 来分析原始 FP32 模型的推理 (inference)性能。性能分析工具会记录 CPU 和 GPU（如果可用）上不同操作的执行时间和内存消耗。

```python
# 对 FP32 模型进行推理性能分析
print("正在分析 FP32 模型...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA, # 仅当 CUDA 可用时
    ],
    record_shapes=True, # 可选：记录输入形状
    profile_memory=True, # 可选：分析内存使用情况
    with_stack=True # 可选：添加源代码上下文
) as prof:
    with torch.profiler.record_function("model_inference"): # 标记此代码块
        for _ in range(10): # 运行多次推理以获得稳定测量结果
            model_fp32(dummy_input)

# 打印按 self CPU 时间排序的性能分析结果
print("FP32 模型性能分析结果（按 self CPU 时间排序）：")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# 打印按 self CUDA 时间排序的性能分析结果（如果适用）
if device.type == 'cuda':
    print("\nFP32 模型性能分析结果（按 self CUDA 时间排序）：")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# 获取基准推理时间（多次运行的平均值）
start_time = time.time()
with torch.no_grad():
    for _ in range(50):
        model_fp32(dummy_input)
end_time = time.time()
fp32_inference_time = (end_time - start_time) / 50
print(f"\nFP32 平均推理时间: {fp32_inference_time:.6f} 秒")

# 获取基准模型大小
fp32_model_size = get_model_size(model_fp32)
print(f"FP32 模型大小: {fp32_model_size:.2f} MB")
```

查看性能分析工具的输出表格。查看 `Name` 列下消耗时间最多的操作（`self_cpu_time_total` 或 `self_cuda_time_total`）。对于 ResNet 等卷积网络，您通常会看到 `aten::conv2d`、`aten::batch_norm`、`aten::relu` 和 `aten::addmm`（用于线性层）占据了主要的执行时间。此分析印证了量化 (quantization)等优化工作可能带来最大益处的地方。

### 应用训练后静态量化 (quantization)（PTQ）

现在，我们将应用 PTQ，把 FP32 模型转换为量化的 INT8 版本。静态量化需要一个校准步骤，使用代表性数据来计算激活的量化参数 (parameter)（缩放因子和零点）。

**注意：** 对于 CPU 上的 PTQ，我们通常使用 'fbgemm' 后端。对于 ARM CPU（移动设备上常见），通常优选 'qnnpack'。如果使用 CUDA，量化支持较为有限，通常依赖于特定的硬件功能（如 Tensor Cores）以及特定的后端或库（如 TensorRT）。为简化起见，本例侧重于使用 'fbgemm' 进行 CPU 量化。

```python
# --- 训练后静态量化 ---
print("\n正在开始训练后静态量化...")

# 创建模型的副本用于量化并移至 CPU
# 量化通常首先在 CPU 上执行
quantized_model = copy.deepcopy(original_model)
quantized_model.eval()
quantized_model.cpu() # 将模型移至 CPU 以进行量化步骤

# 1. 模块融合：组合 Conv-BN-ReLU 序列以提升量化精度和性能
# 注意：融合列表可能需要根据具体的模型架构进行调整。
# 对于 ResNet，常见的融合包括 Conv-BN、Conv-BN-ReLU。
modules_to_fuse = []
for name, module in quantized_model.named_modules():
    if isinstance(module, models.resnet.Bottleneck) or isinstance(module, models.resnet.BasicBlock):
        # 寻找 (conv, bn, relu) 或 (conv, bn) 等序列
        # 这是一个简化示例；实际实现可能需要更复杂的模式匹配。
        seq = []
        for child_name, child_module in module.named_children():
            # 检查 Conv2d, BatchNorm2d, ReLU 模式
             # 简单模式：conv -> bn -> relu 或 conv -> bn
            if isinstance(child_module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU)):
                seq.append(f"{name}.{child_name}")
                if len(seq) >= 2: # 找到了至少 conv-bn
                    # 检查最后两个是否为 Conv-BN
                    is_conv_bn = isinstance(module.get_submodule(seq[-2].split('.')[-1]), torch.nn.Conv2d) and \
                                 isinstance(child_module, torch.nn.BatchNorm2d)
                    if is_conv_bn:
                        # 如果 BN 后面跟着 ReLU，可选地添加 ReLU
                        next_module_idx = list(module.named_children()).index((child_name, child_module)) + 1
                        if next_module_idx < len(list(module.named_children())):
                             next_child_name, next_child_module = list(module.named_children())[next_module_idx]
                             if isinstance(next_child_module, torch.nn.ReLU):
                                 modules_to_fuse.append(seq + [f"{name}.{next_child_name}"])
                             else:
                                 modules_to_fuse.append(seq.copy())
                        else:
                             modules_to_fuse.append(seq.copy())
                        seq = [] # 找到匹配后重置序列

            else: # 如果遇到不可融合的层，则中断序列
                seq = []

# 也考虑顶层的 conv1, bn1, relu
if hasattr(quantized_model, 'conv1') and hasattr(quantized_model, 'bn1') and hasattr(quantized_model, 'relu'):
     modules_to_fuse.append(['conv1', 'bn1', 'relu'])

print(f"要融合的模块数量: {len(modules_to_fuse)}")
# 应用融合
if modules_to_fuse:
    quantized_model = torch.quantization.fuse_modules(quantized_model, modules_to_fuse, inplace=True)
print("模块融合完成。")

# 2. 指定量化配置
# 对 x86 CPU 使用 'fbgemm'。对 ARM CPU 使用 'qnnpack'。
quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(f"量化配置设置为: {quantized_model.qconfig}")

# 3. 准备模型进行校准
# 插入观察器以收集激活统计信息
torch.quantization.prepare(quantized_model, inplace=True)
print("模型已准备好进行校准（观察器已插入）。")

# 4. 校准模型
# 在少量代表性数据集（校准数据）上运行推理
# 这里我们使用随机数据进行演示；在实际应用中，请使用验证集的一个子集。
print("正在运行校准...")
calibration_data = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(100)] # 使用约 100 个样本
with torch.no_grad():
    for input_data in calibration_data:
        quantized_model(input_data)
print("校准完成。")

# 5. 将模型转换为量化版本
# 将模块替换为量化对应项并使用收集到的统计信息
quantized_model = torch.quantization.convert(quantized_model, inplace=True)
print("模型已转换为量化版本 (INT8)。")

# 确保量化模型处于评估模式
quantized_model.eval()
```

让我们可视化简化的 PTQ 流程：

> 该过程涉及融合兼容层，通过插入观察器准备模型，使用样本数据进行校准，最后转换为量化格式。

### 评估量化 (quantization)模型

现在，我们来评估 INT8 量化模型在 CPU 上的性能，并将其与原始 FP32 模型进行比较。我们将测量推理 (inference)时间和模型大小。

````python
# 对 CPU 上的 INT8 量化模型进行性能分析
print("\n正在分析 INT8 量化模型 (CPU)...")
# 确保模拟输入数据在 CPU 上用于量化模型
dummy_input_cpu = dummy_input.cpu()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU], # 量化模型在此处在 CPU 上运行
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof_quant:
    with torch.profiler.record_function("quantized_model_inference"):
        # 重要：确保已为量化操作设置后端
        # 这通常是性能测量所必需的。
        torch.backends.quantized.engine = 'fbgemm'
        with torch.no_grad():
            for _ in range(10):
                quantized_model(dummy_input_cpu)

print("INT8 量化模型性能分析结果（按 self CPU 时间排序）：")
print(prof_quant.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# 测量 INT8 推理时间
start_time = time.time()
with torch.no_grad():
    for _ in range(50):
        quantized_model(dummy_input_cpu)
end_time = time.time()
int8_inference_time = (end_time - start_time) / 50
print(f"\nINT8 平均推理时间 (CPU): {int8_inference_time:.6f} 秒")

# 测量 INT8 模型大小
int8_model_size = get_model_size(quantized_model)
print(f"INT8 模型大小: {int8_model_size:.2f} MB")

# --- 比较 ---
print("\n--- 性能比较 ---")
speedup_factor = fp32_inference_time / int8_inference_time if device.type == 'cpu' else float('nan') # 仅直接比较 CPU 时间
size_reduction = fp32_model_size / int8_model_size

print(f"FP32 推理所用设备: {device}")
print(f"FP32 平均推理时间: {fp32_inference_time:.6f} 秒")
print(f"INT8 平均推理时间 (CPU): {int8_inference_time:.6f} 秒")
if device.type == 'cpu':
    print(f"CPU 推理加速比: {speedup_factor:.2f}x")
else:
    print("CPU 推理加速比: 不适用 (FP32 在 GPU 上运行)")

print(f"\nFP32 模型大小: {fp32_model_size:.2f} MB")
print(f"INT8 模型大小: {int8_model_size:.2f} MB")
print(f"模型大小缩减: {size_reduction:.2f}x")

# 可选：可视化比较结果
import json

chart_data = {
    "layout": {
        "title": "模型性能比较",
        "barmode": "group",
         "xaxis": {"title": "指标"},
         "yaxis": {"title": "数值"},
         "font": {"family": "sans-serif"}
    },
    "data": [
        {
            "type": "bar",
            "name": "推理时间 (秒)",
            "x": ["FP32", "INT8 (CPU)"],
            "y": [fp32_inference_time, int8_inference_time],
            "marker": {"color": "#4dabf7"} # blue
        },
         {
            "type": "bar",
            "name": "模型大小 (MB)",
            "x": ["FP32", "INT8 (CPU)"],
            "y": [fp32_model_size, int8_model_size],
            "marker": {"color": "#38d9a9"} # teal
        }
    ]
}
# 根据数据类型正确格式化 y 轴以增加清晰度
chart_data["layout"]["yaxis"] = {"title": "时间 (秒) / 大小 (MB)"}
chart_data["layout"]["yaxis2"] = {
    "title": "模型大小 (MB)",
    "overlaying": "y",
    "side": "right",
    "showgrid": False,
}
# 将条形图分配给不同的轴
chart_data["data"][0]["yaxis"] = "y1"
chart_data["data"][1]["yaxis"] = "y2"

print("\n性能图表数据:")
print(f"```plotly\n{json.dumps(chart_data)}\n```")
````

> 原始 FP32 模型与 INT8 量化模型之间平均推理时间和模型大小的比较。请注意，只有当 FP32 模型也在 CPU 上运行时，直接的加速比比较才有意义。

### 讨论

本次实践练习展示了使用性能分析和训练后静态量化 (quantization)来优化 PyTorch 模型的标准流程。

1. **性能分析：** 我们使用 `torch.profiler` 来确定原始 FP32 模型的性能特征。此步骤有助于理解计算时间的花费位置，并确认量化目标层（如卷积层）确实是重要的贡献者。
2. **量化：** 我们应用了 PTQ，其中包括融合模块、使用观察器准备模型、用样本数据校准，并将模型转换为 INT8。
3. **评估：** 将 INT8 模型与 FP32 基准进行比较通常会显示：
   - 模型大小减小：INT8 权重 (weight)和激活所需的存储空间显著减少（通常约 4 倍缩减）。
   - 更快的 CPU 推理 (inference)：INT8 操作可以在支持专用指令的 CPU 上更高效地执行，从而带来显著的加速。GPU 加速效果则在很大程度上取决于硬件支持和具体操作。
   - 潜在的精度权衡：尽管 PTQ 旨在最大程度地减少精度损失，但仍可能发生一些性能下降。在您的特定任务和验证数据集上评估量化模型以确保其仍满足要求，这一点很重要。如果精度显著下降，那么之前讨论过的量化感知训练 (QAT) 等技术可能就会有必要了。

"这个动手示例为应用这些优化技术奠定了基础。请记住，具体步骤（如融合列表）和结果可能因模型架构、所选量化后端以及用于推理的硬件而异。尝试不同的配置并评估精度是部署场景中的重要后续步骤。"

## 参考资料

- [Post Training Static Quantization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEib-cH-Kb-7p9O-puMsMEXIkBStzfYVRt7cFUik0f5f1mMiXsWgVV_L7XKoZhFLbT3xYDOyLbf7GrVM7xJ0meycUANKDBi10si7JnPMszNP1ksSIEEd93urtmWGgnBq3qxm5L5nYBTsi5-gnJPp5BmvzVT6eue9pxmuFt_Fc4-hQkea7o6) — Raghuraman Krishnamoorthi, Seth Weidman, Jerry Zhang (2019)
  Publisher: PyTorch
  PyTorch官方教程，详细介绍了训练后静态量化（PTQ）的步骤，包括模块融合、观察器、校准以及转换为INT8格式。
- [Torch Profiler](https://pytorch.org/docs/stable/profiler.html) — PyTorch Documentation (2024)
  《torch.profiler》的官方文档，解释了其用于性能分析的用法，包括活动、记录形状、内存分析以及结果解读。
