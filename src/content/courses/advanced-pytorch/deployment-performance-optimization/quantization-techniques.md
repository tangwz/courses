---
course: "advanced-pytorch"
chapter: "deployment-performance-optimization"
lesson: "quantization-techniques"
sourceId: 2734
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/quantization-techniques"
title: "模型量化技术"
description: "应用动态量化、静态训练后量化（PTQ）和量化感知训练（QAT）。"
order: 2
plots: []
sourceHash: "a34f12640b41add38c4dfb51d111a9fed3f362fb075bd32ae4105831ef574c03"
sourceCorrections: []
---

减少深度学习 (deep learning)模型的计算开销对于部署非常重要，尤其是在资源受限设备或对延迟敏感的应用上。模型量化 (quantization)是一种有效方法，通过将模型转换为使用低精度数值格式（通常是8位整数INT8），而不是训练期间使用的标准32位浮点(FP32)表示，来达成此目的。这种转换带来明显优势：

- **减小模型大小：** 精度降低意味着存储模型参数 (parameter)（权重 (weight)和偏置 (bias)）所需的内存减少，从FP32到INT8通常可以减少4倍。
- **更快的推理 (inference)：** 在许多硬件平台，特别是CPU和专用加速器（如NPU或DSP）上，整数算术运算通常比浮点运算快得多。这会降低推理延迟。
- **更低的功耗：** 更快的执行和更简单的算术运算通常会减少能耗，这对于移动和边缘设备很重要。

然而，量化并非没有代价。用更少位数表示值会引入近似误差，这可能降低模型精度。目标是尽量减少这种精度下降，同时尽可能提高性能收益。PyTorch提供了一个`torch.quantization`工具包来实现多种量化策略。

### 量化 (quantization)核心理念

本质上，量化涉及将一系列浮点值映射到较小范围的整数值。最常用方案是仿射量化，由两个参数 (parameter)定义：**比例因子** ($S$) 和 **零点** ($Z$)。比例因子是一个正浮点数，决定量化的步长；零点是一个整数，对应实数0.0。

从实数$r$ (FP32) 到其量化整数表示$q$ (例如INT8) 的映射由以下公式给出：


$$
q = \text{截断}(\text{取整}(r / S + Z))
$$


反向映射（反量化）从$q$ 返回到近似实数$r'$ 为：


$$
r' = (q - Z) \times S
$$


`round`操作将值四舍五入到最近的整数，`clamp`确保结果保持在目标整数类型的有效范围内（例如，有符号INT8为[-128, 127]）。比例因子$S$和零点$Z$是根据被量化的浮点值的范围（例如，在权重 (weight)或激活中观察到的最小值/最大值）来确定的。

量化可以应用于**逐张量**（对整个张量使用一个$S$和$Z$）或**逐通道**（对每个通道使用独立的$S$和$Z$值，通常沿卷积权重的输出通道轴）。逐通道量化通常为卷积层带来更高的精度，但会增加一些复杂性。

PyTorch支持三种主要模型量化方法：

### 1. 动态量化 (quantization)（训练后动态量化）

这通常是最简单的方法。

- **工作方式：** 权重 (weight)离线量化（转换为INT8并存储）。而激活在推理 (inference)期间“即时”量化。支持动态量化的算子（如`nn.Linear`、`nn.LSTM`）会动态量化激活，使用高效的INT8核进行计算，然后在传递给下一次操作前，将结果反量化回FP32。
- **优点：** 非常容易实现，无需改变模型定义或训练过程，并且不需要校准数据集。
- **缺点：** 激活的动态量化/反量化会引入运行时开销。性能提升通常不如静态量化明显，特别是对于计算量通常超过内存带宽的卷积网络。精度可能低于静态方法。
- **使用场景：** 很好的起点。特别适用于权重大小是瓶颈的模型，或LSTMs和Transformers等序列模型，其中全连接层通常主导计算。

以下是您可能对模型应用动态量化的方式：

```python
import torch
import torch.quantization
import torch.nn as nn

# 假设 'model_fp32' 是您已训练好的FP32模型
# 确保模型处于评估模式
model_fp32.eval()

# 指定要动态量化的层
# 通常侧重于 nn.Linear, nn.LSTM, nn.GRU
quantized_model = torch.quantization.quantize_dynamic(
    model=model_fp32,
    qconfig_spec={nn.Linear, nn.LSTM}, # 要量化的层类型集合
    dtype=torch.qint8                   # 目标数据类型
)

# 现在 'quantized_model' 可以用于推理
# input_fp32 = torch.randn(1, input_size) # 示例输入
# output = quantized_model(input_fp32)
```

### 2. 静态量化 (quantization)（训练后静态量化）

静态量化旨在通过尽可能在整数域中完成所有计算，来获得最大性能。

- **工作方式：** 权重 (weight)离线量化。非常重要的一点是，激活范围也通过一个称为**校准**的过程离线确定。您将训练或验证数据的代表性样本输入模型，特殊的“观察者”模块会追踪不同位置激活的分布（最小值/最大值）。这些统计数据用于计算激活的比例因子$S$和零点$Z$。在推理 (inference)期间，权重和激活都是INT8，可实现高效的基于整数的计算。插入`QuantStub`和`DeQuantStub`模块来处理FP32输入/输出与模型INT8量化核心之间的转换。
- **优点：** 有潜力带来最大的加速和内存节省，因为中间计算可以保留在INT8域中。通常比动态量化提供更高的精度。
- **缺点：** 需要代表性校准数据集。实现过程更复杂，通常需要模型修改（插入stub，合并模块）。
- **使用场景：** 适合卷积神经网络 (neural network)(CNN)以及部署在支持高效INT8硬件上的其他架构，目标是最大推理速度和最小占用空间。

静态量化工作流程通常包括以下步骤：

1. **准备模型：**

   - 合并操作：使用`torch.quantization.fuse_modules`尽可能将Conv+BatchNorm+ReLU等层合并为单个单元。这可以提高精度和性能。
   - 插入量化/反量化Stub：在模型输入处添加`QuantStub`，在输出前添加`DeQuantStub`，以管理FP32 <-> INT8转换。
   - 指定量化配置：定义要使用的量化方案（例如，x86平台的`fbgemm`，ARM平台的`qnnpack`）和观察者。
2. **校准：**

   - 将模型设置为评估模式（`model.eval()`）。
   - 将校准数据输入准备好的模型。观察者收集激活统计信息。
3. **转换：**

   - 使用`torch.quantization.convert`将校准后的模型转换为完全量化的INT8模型，替换模块为其量化版本，并存储计算出的比例因子和零点。

```python
import torch
import torch.quantization
import torch.nn as nn

# 假设 'model_fp32' 是您已训练好的FP32模型
model_fp32.eval()

# 1. 准备模型
# 添加 QuantStub 和 DeQuantStub （修改您的模型定义或对其进行封装）
class QuantizableModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = original_model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

model_to_quantize = QuantizableModel(model_fp32)
model_to_quantize.eval()

# 合并模块（Conv + ReLU 示例）
# 您通常需要遍历模型的层
# 示例：torch.quantization.fuse_modules(model_to_quantize.model, [['conv1', 'relu1']], inplace=True)

# 指定量化配置
# x86平台使用 'fbgemm'，ARM平台使用 'qnnpack'。为简单起见，使用 get_default_qconfig
model_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# 通过添加观察者来准备模型
prepared_model = torch.quantization.prepare(model_to_quantize, inplace=True)

# 2. 校准
# 将代表性数据输入准备好的模型
# 假设 'calibration_data_loader' 提供校准样本
print("正在运行校准...")
with torch.no_grad():
    for inputs, _ in calibration_data_loader:
        prepared_model(inputs)
print("校准完成。")

# 3. 转换
quantized_model = torch.quantization.convert(prepared_model, inplace=True)
quantized_model.eval()

# 'quantized_model' 现在已准备好进行INT8推理
# input_fp32 = torch.randn(1, 3, 224, 224) # 示例输入
# output = quantized_model(input_fp32)
```

### 3. 量化 (quantization)感知训练（QAT）

QAT在训练（或微调 (fine-tuning)）过程中模拟量化效果，让模型适应精度损失。

- **工作方式：** 在模型定义中插入“伪”量化模块（`torch.quantization.FakeQuantize`）。这些模块在正向传播过程中使用估计的量化参数 (parameter)模拟量化（量化-反量化）过程。梯度正常计算和反向传播 (backpropagation)，使模型权重 (weight)得以调整，从而最大程度地降低最终INT8转换对精度的影响。训练后，模型将转换为真正的INT8模型，这类似于静态量化过程，但使用在QAT期间学到的参数。
- **优点：** 通常能达到量化方法中的最高精度，常常非常接近原始FP32模型的精度。
- **缺点：** 需要对模型进行再训练或微调，这增加了训练阶段的复杂性和计算成本。
- **使用场景：** 当训练后方法（动态或静态）导致无法接受的精度下降，并且有可用于再训练的资源时采用。

QAT工作流程与静态量化类似，但与训练过程结合：

1. **为QAT准备模型：**

   - 像静态量化一样合并模块。
   - 定义QAT配置（例如，`torch.quantization.get_default_qat_qconfig('fbgemm')`）。
   - 使用`torch.quantization.prepare_qat`插入伪量化模块。
2. **训练或微调：**

   - 在伪量化模块激活的状态下训练模型。模型学习适应量化噪声的权重。确保模型以训练模式开始（`model.train()`）。
3. **转换：**

   - 训练后，将模型切换到评估模式（`model.eval()`）。
   - 使用`torch.quantization.convert`创建最终的INT8模型。

```python
import torch
import torch.quantization
import torch.nn as nn
import torch.optim as optim

# 假设 'model_fp32' 是您已训练好的FP32模型或架构
# QAT通常从预训练模型开始，或从头训练

# 1. 为QAT准备
# 首先适当合并模块（为简洁起见此处未展示）
model_to_train_qat = QuantizableModel(model_fp32) # 使用静态示例中的封装器
model_to_train_qat.train() # 设置为训练模式

# 定义QAT配置
model_to_train_qat.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# 通过插入伪量化模块来准备模型
prepared_model_qat = torch.quantization.prepare_qat(model_to_train_qat, inplace=True)

# 2. 训练或微调
optimizer = optim.SGD(prepared_model_qat.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs_qat = 3 # 示例：微调几个周期

print("开始QAT微调...")
for epoch in range(num_epochs_qat):
    prepared_model_qat.train() # 确保模型处于训练模式
    for inputs, labels in training_data_loader: # 使用您的训练数据
        optimizer.zero_grad()
        outputs = prepared_model_qat(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 如有需要，添加验证循环
    print(f"Epoch {epoch+1}/{num_epochs_qat}, 损失: {loss.item()}")
print("QAT微调完成。")


# 3. 转换为量化模型
prepared_model_qat.eval() # 在转换前设置为评估模式！
quantized_model_qat = torch.quantization.convert(prepared_model_qat, inplace=True)

# 'quantized_model_qat' 是最终可用于部署的INT8模型
```

### 选择合适的量化 (quantization)方法

选择合适的量化方法取决于您的具体限制和目标：

> 一份根据易用性、数据可用性、性能需求和精度容忍度等要求，选择PyTorch量化策略的决策指南。

### 实际考量

- **模块合并：** 在应用静态量化 (quantization)或QAT之前，使用`torch.quantization.fuse_modules`合并`Conv` + `BatchNorm` + `ReLU`等序列。这使得量化观察者能将组合操作作为一个整体处理，从而带来更好的数值精度并启用后端优化。
- **后端：** PyTorch对量化操作使用不同的后端（x86 CPU的`fbgemm`，ARM CPU的`qnnpack`）。请确保在配置期间选择适合目标硬件的后端。
- **算子支持：** 并非所有PyTorch算子都支持量化。请查看文档以了解支持的层和数据类型。如果模型某些部分使用了不支持的操作，您可能需要将其保留在FP32中（混合精度部署）。
- **调试：** 量化有时难以调试。仔细检查中间张量统计信息（`q_scale()`、`q_zero_point()`），并比较量化模型与FP32基线的精度。

模型量化是优化PyTorch模型以实现高效部署的重要步骤。通过理解动态、静态和量化感知训练方法之间的权衡，并仔细运用`torch.quantization`中提供的工具，您可以明显减少模型大小和延迟，同时为您的应用保持可接受的精度。

## 参考资料

- [PyTorch Model Quantization](https://pytorch.org/docs/stable/quantization.html) — PyTorch Authors (2019)
  PyTorch 中实现量化技术的官方指南，详细介绍了动态、静态和 QAT 工作流程。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://ieeexplore.ieee.org/document/8578613) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 870-879; DOI: [10.1109/CVPR.2018.00892](https://doi.org/10.1109/CVPR.2018.00892)
  介绍了训练后量化的核心概念，包括仿射量化、尺度和零点参数。
