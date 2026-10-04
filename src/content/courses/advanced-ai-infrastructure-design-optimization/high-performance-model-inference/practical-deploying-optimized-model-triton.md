---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "high-performance-model-inference"
lesson: "practical-deploying-optimized-model-triton"
sourceId: 7011
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/practical-deploying-optimized-model-triton"
title: "实战操作：在Triton上部署优化模型"
description: "获取一个预训练模型，使用TensorRT对其进行优化，并将其部署到Triton推理服务器上，然后对其性能进行基准测试。"
order: 6
plots: ["plots/7011-0.json"]
sourceHash: "0c53f6c49799ad31dd99173c1f1c80315b165374f5aa7f8fdd917e59a74b7658"
sourceCorrections: []
---

优化方法将应用于真实模型。将一个标准的基于PyTorch的图像分类模型转换为高度优化的TensorRT引擎。基线版本和优化版本都将被部署到Triton推理 (inference)服务器上，并仔细测试其性能差异。此练习巩固了从训练产物到可投入生产的高性能推理服务的整个流程。

### 前提条件

要完成此练习，您需要一个安装了NVIDIA GPU、Docker和NVIDIA Container Toolkit的系统。这使得Docker容器能够访问GPU。所有模型优化和提供服务都将在容器化环境中执行，以确保可复现性。

首先，从NVIDIA NGC仓库拉取最新的Triton推理 (inference)服务器容器。此容器包含Triton、所有必要的CUDA库以及一个包含常用深度学习 (deep learning)框架的Python环境。

```bash
docker pull nvcr.io/nvidia/tritonserver:24.05-py3
```

> **注意：** 版本标签（例如，`24.05-py3`）会随时间变化。您可以在[Triton的NVIDIA NGC目录页面](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)上找到最新的可用标签。

### 步骤 1：准备基线模型和模型仓库

Triton从一个名为模型仓库的特殊结构目录中提供模型服务。每个模型都有其自己的子目录，其中包含模型文件和一个`config.pbtxt`文件，该文件告知Triton如何提供模型服务。

我们首先为PyTorch的基线ResNet-18模型创建一个模型仓库。

1. **创建目录结构：**

   ```bash
   mkdir -p triton_repo/resnet18_pytorch/1
   ```
2. **保存PyTorch模型：**
   创建一个名为`export_model.py`的Python脚本，用于下载预训练 (pre-training)的ResNet-18模型并以TorchScript格式保存，Triton可以直接执行此格式的模型。

   ```python
   # export_model.py
   import torch
   import torchvision.models as models

   # 加载预训练的ResNet-18模型
   model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
   model.eval()
   model.cuda() # 将模型移动到GPU

   # 创建一个示例输入张量
   # 形状必须与模型预期的一致：(批量大小, 通道数, 高度, 宽度)
   dummy_input = torch.randn(1, 3, 224, 224, device="cuda")

   # 使用TorchScript跟踪模型
   traced_model = torch.jit.trace(model, dummy_input)

   # 保存跟踪后的模型
   traced_model.save("triton_repo/resnet18_pytorch/1/model.pt")

   print("PyTorch模型已保存到 triton_repo/resnet18_pytorch/1/model.pt")
   ```

   运行脚本：`python export_model.py`。
3. **创建配置文件：**
   现在，在`triton_repo/resnet18_pytorch/`中创建`config.pbtxt`文件。此文件定义了模型的元数据。

   ```text
   # triton_repo/resnet18_pytorch/config.pbtxt
   name: "resnet18_pytorch"
   backend: "pytorch"
   max_batch_size: 64
   input [
     {
       name: "INPUT__0"
       data_type: TYPE_FP32
       dims: [ 3, 224, 224 ]
     }
   ]
   output [
     {
       name: "OUTPUT__0"
       data_type: TYPE_FP32
       dims: [ 1000 ]
     }
   ]
   instance_group [
     {
       count: 1
       kind: KIND_GPU
     }
   ]
   ```

您的模型仓库现在应具有以下结构：

```
triton_repo/
└── resnet18_pytorch/
    ├── 1/
    │   └── model.pt
    └── config.pbtxt
```

### 步骤 2：对基线PyTorch模型进行基准测试

模型仓库准备就绪后，启动Triton服务器并将其指向您的模型仓库。

```bash
docker run --rm --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v $(pwd)/triton_repo:/models \
nvcr.io/nvidia/tritonserver:24.05-py3 tritonserver --model-repository=/models
```

Triton将启动并加载`resnet18_pytorch`模型。为了对其进行基准测试，我们使用Triton的`perf_analyzer`工具，该工具也包含在容器中。打开一个新的终端并运行以下命令，以在新容器内（连接到服务器网络）执行`perf_analyzer`。

```bash
docker run --rm --net=host nvcr.io/nvidia/tritonserver:24.05-py3 \
perf_analyzer -m resnet18_pytorch --concurrency-range 1:16 -u localhost:8001
```

此命令使用从1到16的递增客户端并发级别测试模型。稍等片刻，您将看到一个汇总表。注意**吞吐量 (throughput)**（infer/sec）和**p99延迟**值。对于未优化的ResNet-18，您可能会看到类似以下的内容：

```text
***提示***
请求并发数: 16
  客户端:
    请求计数: 2174
    吞吐量: 271.5 infer/sec
    p99延迟: 62105 usec
```

这是我们的性能基线。让我们看看能将其提升多少。

### 步骤 3：使用TensorRT优化模型

现在我们将PyTorch模型转换为TensorRT引擎。此过程会应用多项优化，包括层融合、精度校准以及针对我们特定GPU的内核自动调优。

创建一个新的Python脚本`optimize_model.py`来执行转换。您需要首先安装`torch-tensorrt`库：`pip install torch-tensorrt`。

```python
# optimize_model.py
import torch
import torch_tensorrt

# 加载已保存的TorchScript模型
model = torch.jit.load("triton_repo/resnet18_pytorch/1/model.pt")
model.eval().cuda()

# 使用TensorRT编译模型
# 我们启用FP16精度以获得显著加速
trt_model = torch_tensorrt.compile(model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224), # 典型批量大小
            max_shape=(64, 3, 224, 224), # 与配置中的max_batch_size匹配
            dtype=torch.float32)
    ],
    enabled_precisions={torch.float16} # 启用FP16
)

# 保存TensorRT引擎
torch.jit.save(trt_model, "triton_repo/resnet18_trt/1/model.plan")

print("TensorRT引擎已保存到 triton_repo/resnet18_trt/1/model.plan")
```

在运行脚本之前，为新模型创建目录：`mkdir -p triton_repo/resnet18_trt/1`。然后，执行脚本：`python optimize_model.py`。

### 步骤 4：配置和部署TensorRT模型

优化后的模型需要自己的配置文件。主要区别是将`backend`更改为`tensorrt`，并将模型文件名为`model.plan`。

1. **创建TensorRT的`config.pbtxt`：**
   在`triton_repo/resnet18_trt/config.pbtxt`创建文件。

   ```text
   # triton_repo/resnet18_trt/config.pbtxt
   name: "resnet18_trt"
   backend: "tensorrt"
   max_batch_size: 64
   input [
     {
       name: "INPUT__0"
       data_type: TYPE_FP32
       dims: [ 3, 224, 224 ]
     }
   ]
   output [
     {
       name: "OUTPUT__0"
       data_type: TYPE_FP32
       dims: [ 1000 ]
     }
   ]
   instance_group [
     {
       count: 1
       kind: KIND_GPU
     }
   ]
   # 启用动态批处理以更好地利用GPU
   dynamic_batching {
     preferred_batch_size: [8, 16]
     max_queue_delay_microseconds: 100
   }
   ```

   请注意`dynamic_batching`块的添加。这个Triton功能将单个推理 (inference)请求组合成一个更大的批次，以更好地充分利用GPU，这是提高吞吐量 (throughput)的一种常见策略。
2. **验证模型仓库结构：**
   您的模型仓库现在应包含两个模型。

digraph G { rankdir=TB; node [shape=folder, style=rounded, fontname="Arial"]; edge [arrowhead=none]; "triton\_repo" -> {"resnet18\_pytorch" "resnet18\_trt"}; "resnet18\_pytorch" -> {"config.pbtxt\_py" "v1\_py"}; "config.pbtxt\_py" [label="config.pbtxt"]; "v1\_py" [label="1"]; "v1\_py" -> "model.pt"; "resnet18\_trt" -> {"config.pbtxt\_trt" "v1\_trt"}; "config.pbtxt\_trt" [label="config.pbtxt"]; "v1\_trt" [label="1"]; "v1\_trt" -> "model.plan"; }

````
```
> 部署仓库现在同时包含基线PyTorch模型和优化的TensorRT引擎，可以直接进行比较。
````

3. **重新启动Triton：**
如果您的Triton服务器（来自步骤2）仍在运行，请停止它（`Ctrl+C`）。然后，使用相同的命令重新启动。它现在将检测并加载`resnet18_pytorch`和`resnet18_trt`模型。

### 步骤 5：对优化模型进行基准测试和比较

最后，再次运行`perf_analyzer`，但这次针对新的`resnet18_trt`模型。

```bash
docker run --rm --net=host nvcr.io/nvidia/tritonserver:24.05-py3 \
perf_analyzer -m resnet18_trt --concurrency-range 1:16 -u localhost:8001
```

您应该会看到性能的显著提升。输出可能如下所示：

```text
***提示***
请求并发数: 16
  客户端:
    请求计数: 13010
    吞吐量: 1625.3 infer/sec
    p99延迟: 10015 usec
```

让我们将结果可视化。吞吐量 (throughput)大幅增加，p99延迟显著降低。



![性能比较：基线模型 vs. TensorRT模型](plots/7011-0.json)



> 原始PyTorch模型和TensorRT优化版本之间的吞吐量和p99延迟比较。延迟以毫秒（ms）为单位。

这个实操练习显示了一种标准且有效的生产模型部署流程。通过将标准框架模型转换为像TensorRT这样的专用推理 (inference)引擎，并使用像Triton这样的高级服务器提供服务，您可以获得数量级的性能提升。这个过程是构建经济高效且响应迅速的AI服务，以满足严格SLO要求的基础。

## 参考资料

- [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) — NVIDIA (2024)
  Publisher: NVIDIA
  Triton Inference Server的官方部署、配置和模型管理指南。
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA (2024)
  Publisher: NVIDIA
  使用TensorRT优化神经网络进行推理的全面指南。
- [Torch-TensorRT Documentation](https://pytorch.org/TensorRT/) — NVIDIA (2024)
  Publisher: NVIDIA
  关于使用torch-tensorrt库将PyTorch模型转换为TensorRT的详细信息。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  从系统层面阐述构建和部署机器学习应用的方法。
