# 权重量化 (INT8, INT4)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-27-model-compression-techniques/weight-quantization-int8-int4)

[返回章节目录](README.md) · [返回课程目录](../README.md)

权重 (weight)量化 (quantization)是一种主要技术，用于减少大型语言模型的内存占用，并通常加快其推理 (inference)速度。其基本思路是将模型通常用32位浮点数（FP32）存储的权重参数 (parameter)，转换为8位整数（INT8）甚至4位整数（INT4）等低精度整数格式表示。每个参数的比特宽度减少直接导致模型尺寸变小，并可能在支持低精度计算的硬件上实现更快的运算。

### 数值格式的理解

回顾一下，标准FP32格式提供宽泛的动态范围和高精度，这在敏感的训练过程中非常重要。然而，对于推理 (inference)来说，通常可以使用更少的比特位，而不会造成模型准确率的严重下降。

- **FP32 (单精度):** 标准格式。用1位表示符号，8位表示指数，23位表示尾数。提供大范围和高精度。
- **FP16 (半精度):** 用1位表示符号，5位表示指数，10位表示尾数。在兼容硬件（如NVIDIA Tensor Cores）上计算速度更快，与FP32相比内存使用减半，但范围有限，易受溢出/下溢影响（如第20章所述）。
- **BF16 (脑浮点):** 用1位表示符号，8位表示指数（与FP32相同），7位表示尾数。与FP16一样内存减半，但保持FP32的动态范围，以精度为代价减少溢出/下溢问题。在训练和推理中越来越常见。
- **INT8 (8位整数):** 用8位表示数值。显著减少内存（相对于FP32减少4倍），并可在具有专用INT8指令的硬件上实现大幅加速。范围和精度远低于浮点格式。
- **INT4 (4位整数):** 更进一步，每个权重 (weight)仅使用4位。相较于FP32可减少8倍内存，但由于范围和精度受到严格限制，在保持模型准确率方面带来更大挑战。

权重量化 (quantization)的核心难点在于将范围宽泛的FP32权重映射到INT8或INT4值的有限范围，同时最大限度地减少对模型性能非常重要的信息损失。

### 量化 (quantization)原理：浮点数到整数的映射

最常用的方法是仿射量化，它使用一个缩放因子 $S$（一个正浮点数）和一个零点 $Z$（一个整数，通常与 $x_{int}$ 类型相同），将浮点数值 $x_{float}$ 映射到整数值 $x_{int}$。关系如下：


$$
x_{float} \approx S \times (x_{int} - Z)
$$


反之，反量化将整数反向映射回近似浮点数：


$$
x_{dequant} = S \times (x_{int} - Z)
$$


- **缩放因子 ($S$):** 决定量化值之间的步长。它根据原始浮点数值的范围（$max(x_{float}) - min(x_{float})$）除以可用整数级别数（例如，INT8 为 $2^8 - 1$）来计算。
- **零点 ($Z$):** 确保浮点数值0.0能被整数准确表示。对于*对称量化*，范围 $[min(x_{float}), max(x_{float})]$ 对称地映射到零附近，零点 $Z$ 可能是隐式为零或固定值。对于*非对称量化*，$Z$ 会被调整以精确映射浮点零点，如果原始权重 (weight)不以零为中心，这可能更有益。$Z$ 通常是目标整数范围内的整数（例如，无符号INT8为0到255，有符号INT8为-128到127）。

缩放因子和零点可以以不同方式确定：

- **逐张量 (Per-Tensor):** 为整个权重张量（例如，线性层的权重矩阵）计算一个单独的 $S$ 和 $Z$。这很简单，但如果张量内部的数值范围差异很大，则效果可能不理想。
- **逐通道/逐轴 (Per-Channel / Per-Axis):** 为张量的切片计算独立的 $S$ 和 $Z$ 值，通常沿特定维度（例如，卷积层或线性层权重的输出通道维度）。这提供更细的粒度，并且通常比逐张量量化产生更好的准确率，特别是对于权重分布在不同通道或行/列之间差异显著的层。

> 权重张量的逐张量和逐通道量化方法的比较。逐通道量化为每个输出通道（本例中为行）使用不同的缩放因子/零点值。

### 训练后量化 (quantization) (PTQ)

PTQ是一种更简单的方法。您将一个已经用FP32训练好的模型，之后将其权重 (weight)转换为INT8等低精度格式。激活值在推理 (inference)过程中也可能被动态量化。

**流程：**

1. **训练 (Train):** 正常用FP32训练模型直至收敛。
2. **校准 (Calibrate):** 通过FP32模型输入少量有代表性的校准数据集（通常几百个样本就足够）。记录权重的范围（最小值/最大值），如果量化激活值，则记录模型中各个点的激活值范围。
3. **计算量化参数 (parameter) (Calculate Quantization Parameters):** 使用记录的范围来计算每个被量化的张量（或通道）的适当缩放因子 ($S$) 和零点 ($Z$) 值。
4. **转换权重 (Convert Weights):** 应用计算出的 $S$ 和 $Z$ 将FP32权重转换为INT8（或INT4）格式。存储这些量化后的权重及相应的 $S$/$Z$ 值。
5. **推理 (Inference):** 在推理过程中，权重以其整数格式加载。如果激活值也经过量化（动态或静态），它们会实时转换为INT8，或者在预先计算好后加载。如果硬件支持，计算（如矩阵乘法）使用整数运算执行，通常在应用偏置 (bias)或激活函数 (activation function)之前需要反量化回FP32（或中间累加器精度），或者使用直接处理量化操作的专用内核。

**示例 (PyTorch 权重)：**

```python
import torch
import torch.quantization

# 假设 'model' 是您训练好的 FP32 模型
model.eval()

# --- PTQ 静态量化示例 (权重 + 激活值) ---
# 注意：实际的 PTQ 涉及更多步骤，如操作合并和观察器放置

# 指定量化配置（例如，权重的对称 INT8 量化）
# 'fbgemm' 是 x86 常见的后端，'qnnpack' 是 ARM 的后端
qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model, inplace=False)
model_prepared.qconfig = qconfig

# 校准步骤（输入有代表性的数据）
# calibration_data_loader 提供校准样本
print("正在运行校准...")
with torch.no_grad():
    for input_batch, _ in calibration_data_loader:
        model_prepared(input_batch) # 前向传播以收集统计数据
print("校准完成。")

# 将模型转换为量化版本
model_quantized_static = torch.quantization.convert(
    model_prepared, inplace=False
)
print("模型已转换为静态量化版本。")

# --- PTQ 动态量化示例 (仅权重) ---
# 更简单：权重被量化，激活值实时量化

model_quantized_dynamic = torch.quantization.quantize_dynamic(
    model,  # 原始的 FP32 模型
    {torch.nn.Linear},  # 需要动态量化的层集合
    dtype=torch.qint8  # 权重的目标数据类型
)
print("模型已转换为动态量化版本。")

# 现在 'model_quantized_static' 或 'model_quantized_dynamic' 可用于
# 推理。
# 保存/加载这些模型需要专门处理
# 量化参数。

# 示例：检查模型大小减少情况
def print_model_size(mdl, label):
    torch.save(mdl.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    print(f"{label} 的大小: {size:.2f} MB")
    os.remove("temp.p")

# print_model_size(model, "FP32 模型")
# print_model_size(model_quantized_dynamic,
#                    "动态量化 INT8 模型")
# print_model_size(model_quantized_static,
#                    "静态量化 INT8 模型") # 通常最小
```

**PTQ 的优点：**

- 易于实现；不需要更改原始训练流程。
- 转换过程快。

**PTQ 的缺点：**

- 可能导致模型准确率明显下降，尤其是在量化到极低比特宽度（如INT4）或对于敏感模型而言。训练后引入的量化误差未得到补偿。
- 准确率对校准数据集的选择和大小敏感。

### 量化 (quantization)感知训练 (QAT)

QAT通过在微调 (fine-tuning)或训练过程中模拟量化效果来解决PTQ的准确率限制。这使得模型的权重 (weight)能够适应量化引入的精度损失。

**流程：**

1. **从预训练 (pre-training)模型开始 (Start with Pre-trained Model):** 从一个已收敛的FP32模型开始（或者从头开始训练，尽管微调更常见）。
2. **插入伪量化节点 (Insert Fake Quantization Nodes):** 通过在权重层之前和之后，以及可能在激活值之后插入“伪量化”（或量化/反量化）操作符来修改模型图。这些操作符在前向传播过程中模拟量化过程：它们将FP32值量化为目标整数格式（例如INT8），并立即将其反量化回FP32。
   - 前向传播: $x_{out} = dequantize(quantize(x_{in}, S, Z), S, Z)$
   - 反向传播 (backpropagation): 梯度使用直通估计器（STE）计算，本质上忽略不可微分的量化步骤，并像它是恒等函数一样传递梯度。
3. **微调 (Fine-tune):** 在这些伪量化节点激活的情况下，继续训练（微调）模型少量周期。优化器调整FP32权重，但在调整时“感知”到模拟量化引入的噪声和钳位效应。这有助于模型学习转换后更有效的权重。
4. **转换 (Convert):** 在QAT微调后，使用在QAT期间获得的（通常基于训练期间观察到的范围的移动平均值）学到的量化参数 (parameter)（$S$ 和 $Z$），将模型转换为真正的量化模型。

**示例 (PyTorch)：**

```python
import torch
import torch.quantization

# 假设 'model' 是您训练好的 FP32 模型
# 通常最好从一个收敛的 FP32 检查点开始 QAT

model.train() # QAT 需要训练模式

# 指定 QAT 配置
# 使用 get_default_qat_qconfig 获取适当的伪量化节点
qig_config = torch.quantization.get_default_qat_qconfig(
    'fbgemm') # 或 'qnnpack'
model_prepared_qat = torch.quantization.prepare_qat(model, inplace=False)
model_prepared_qat.qconfig = qig_config

# --- QAT 微调循环 ---
print("开始 QAT 微调...")
optimizer = torch.optim.Adam(
    model_prepared_qat.parameters(),
    lr=1e-5) # 使用较小的学习率
num_qat_epochs = 3 # 通常较短

for epoch in range(num_qat_epochs):
    for input_batch, target_batch in qat_training_data_loader:
        optimizer.zero_grad()
        output = model_prepared_qat(input_batch)
        loss = loss_function(output, target_batch) # 使用您的标准损失函数
        loss.backward() # 梯度通过伪量化节点经由 STE 传播
        optimizer.step()
    print(f"QAT 第 {epoch+1}/{num_qat_epochs} 周期完成。")
print("QAT 微调完成。")

# 将 QAT 模型转换为真正的量化模型
model_prepared_qat.eval() # 在转换前设置为评估模式
model_quantized_qat = torch.quantization.convert(model_prepared_qat, inplace=False)
print("模型已转换为 QAT 量化版本。")

# 此模型通常比 PTQ 具有更好的准确率，特别是对于 INT8/INT4
# print_model_size(model_quantized_qat, "QAT 量化 INT8 模型")
```

**QAT 的优点：**

- 通常比PTQ产生明显更好的准确率，通常接近原始FP32模型的性能，尤其是对于INT8。
- 对量化误差的抵抗力更强，因为模型在微调期间学习补偿。

**QAT 的缺点：**

- 实现更复杂，需要修改训练流程和额外的微调步骤。
- 由于微调阶段，增加了总体训练时间。

### 更低精度：INT4

将量化 (quantization)推向INT4甚至更低比特宽度（例如，三元或二元权重 (weight)）能提供最大的内存节省，但大幅增加了保持准确率的难度。

- **量化误差增大：** 仅有16个不同的值（对于INT4），可表示数字之间的差距大很多，导致更高的量化误差。
- **敏感性：** 在这些低比特级别下，模型对量化噪声敏感得多。
- **专用技术：** 标准的PTQ/QAT可能不足够。通常需要基于梯度的低秩量化（GPTQ）或涉及权重分组（分组量化）等高级技术来保持性能。这些方法可能将小块权重一起量化，使用共享或更复杂的量化参数 (parameter)。
- **硬件支持：** 尽管INT8支持在现代CPU和GPU/TPU中相对常见，但INT4或更低精度的有效硬件加速普及度较低，但正在出现（例如，NVIDIA Hopper架构支持FP8，其比特宽度介于INT4和INT8之间）。如果没有专用硬件内核，使用INT4可能无法带来加速。

`bitsandbytes` 等库在Hugging Face生态系统中被广泛用于对大型模型应用INT4量化（通常是NF4 - NormalFloat 4位等变体），这些库经常使用将量化与专用矩阵乘法内核相结合的技术。

### 权衡与实际考量

- **准确率与效率：** 这是核心权衡。PTQ速度快但可能会牺牲准确率。QAT能更好地保持准确率但需要更多精力。INT4/更低比特提供最大压缩，但如果不配合高级技术谨慎应用，则存在显著准确率损失的风险。
- **硬件依赖：** 量化 (quantization)带来的实际推理 (inference)加速很大程度上取决于底层硬件和软件堆栈。除非硬件具有高效的INT8矩阵乘法单元且框架使用优化过的内核（如NVIDIA GPU的cuDNN，或CPU上的特定指令），否则使用INT8权重 (weight)不会加速计算。
- **框架支持：** 深度学习 (deep learning)框架（PyTorch，TensorFlow）为PTQ和QAT都提供了工具。PyTorch有 `torch.quantization`，而TensorFlow通过TensorFlow Lite或模型优化工具包提供类似工具。Hugging Face的 `optimum` 和 `bitsandbytes` 等库专门为Transformer模型集成了量化功能。
- **层类型：** 量化在具有大型权重矩阵的层上最有效，例如线性（全连接）层和嵌入 (embedding)层，这些层在大型语言模型（LLM）参数 (parameter)数量中占据主导地位。对其他层（例如归一化 (normalization)层）的影响需要仔细考量。



[交互图表：典型量化权衡（示意性）](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-27-model-compression-techniques/weight-quantization-int8-int4#plot-1p8u2g6)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "FP32",
        "INT8 (PTQ)",
        "INT8 (QAT)",
        "INT4 (例如, GPTQ)"
      ],
      "y": [
        100,
        98.5,
        99.5,
        96.0
      ],
      "name": "相对准确率 (%)",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "bar",
      "x": [
        "FP32",
        "INT8 (PTQ)",
        "INT8 (QAT)",
        "INT4 (例如, GPTQ)"
      ],
      "y": [
        100,
        25,
        25,
        12.5
      ],
      "name": "相对尺寸 (%)",
      "marker": {
        "color": "#fab005"
      },
      "yaxis": "y2"
    }
  ],
  "layout": {
    "title": {
      "text": "典型量化权衡（示意性）",
      "x": 0.5,
      "xanchor": "center"
    },
    "yaxis": {
      "title": "相对准确率 (%)",
      "range": [
        90,
        101
      ]
    },
    "yaxis2": {
      "title": "相对模型尺寸 (%)",
      "overlaying": "y",
      "side": "right",
      "range": [
        0,
        105
      ]
    },
    "barmode": "group",
    "legend": {
      "x": 0.5,
      "y": -0.2,
      "xanchor": "center",
      "orientation": "h"
    },
    "margin": {
      "l": 50,
      "r": 50,
      "t": 30,
      "b": 40
    },
    "font": {
      "size": 10
    }
  }
}
```

</details>



> 模型准确率和尺寸之间不同权重量化方法的示意性权衡。实际结果根据模型、任务和使用的具体量化技术而明显不同。

权重量化，特别是通过PTQ或QAT进行的INT8量化，是一种广泛采用的技术，可使大型语言模型更适合部署。尽管INT4提供进一步压缩，但它通常需要更复杂的方法和仔细评估，以确保可接受的性能水平。选择正确的策略取决于模型尺寸、推理延迟和允许的准确率下降的具体要求。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了 QLoRA 和 NormalFloat (NF4) 数据类型，这是大型语言模型 4 位量化的一个重要方法，支持在消费级硬件上高效微调。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://doi.org/10.48550/arXiv.2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了 GPTQ，这是一种专为大型语言模型设计的高精度训练后量化方法，能够以最小的精度损失实现高效的 INT4 转换。
- [Quantization for Model Optimization](https://pytorch.org/docs/stable/quantization.html) — PyTorch Documentation (2019)
  PyTorch 官方文档，提供了实现训练后量化和量化感知训练的实用指南及 API。

---

[上一节](01-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E7%9A%84%E5%8A%A8%E5%9B%A0.md) · [下一节](03-%E6%BF%80%E6%B4%BB%E9%87%8F%E5%8C%96%E8%80%83%E9%87%8F.md)
