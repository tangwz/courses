---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-training"
lesson: "gpu-acceleration-training"
sourceId: 5302
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/gpu-acceleration-training"
title: "训练的GPU加速"
description: "配置Docker以结合NVIDIA容器工具包，使用NVIDIA GPU加速模型训练。"
order: 5
plots: []
sourceHash: "94bbbc90b81feddc2a4cab78d8ec67a13e922562a0c1476bcf7dabecb1af8a14"
sourceCorrections: []
---

训练机器学习 (machine learning)模型，尤其是深度学习 (deep learning)模型，通常需要大量的计算工作。虽然CPU可以处理较小的任务，但大型数据集和复杂的架构需要图形处理单元（GPU）的并行处理能力，才能在合理的时间范围内完成训练。在容器中运行这些要求高的训练作业时，会引出一个具体问题：如何确保您的容器化训练脚本可以访问主机强大的GPU？默认情况下，标准的Docker容器与主机硬件是隔离的。

幸运的是，NVIDIA专门为此提供了一个解决方案：NVIDIA容器工具包。这个工具包扩展了Docker，使容器能够安全高效地访问安装在主机系统上的NVIDIA GPU。它管理容器、Docker运行时和主机NVIDIA驱动程序之间的协作。

### GPU加速的前提条件

在您能够将GPU用于Docker容器之前，您的主机系统必须满足以下几点要求：

1. **NVIDIA GPU：** 您需要在运行Docker容器的机器上安装一个或多个兼容的NVIDIA GPU。
2. **NVIDIA驱动程序：** 您的GPU和操作系统的合适NVIDIA驱动程序必须正确安装在*主机*系统上。容器将与这些主机驱动程序交互。
3. **Docker引擎：** 必须安装Docker的最新版本。
4. **NVIDIA容器工具包：** 您需要安装`nvidia-container-toolkit`软件包（或其前身，如`nvidia-docker2`，这取决于您的配置，尽管该工具包是现代标准）。安装说明因Linux发行版而异，可在NVIDIA文档网站上查阅。

### NVIDIA容器工具包的工作原理

NVIDIA容器工具包充当桥梁。当您指示Docker运行一个具有GPU访问权限的容器时，该工具包会拦截该命令。它会自动将主机系统中必需的NVIDIA驱动程序库和GPU设备文件挂载到容器的文件系统中。它还确保容器拥有与GPU硬件交互的正确权限。这样，在容器内运行的应用程序（如TensorFlow或PyTorch）就可以识别并使用GPU，就像直接在主机上运行一样，而不需要将完整的NVIDIA驱动程序安装在容器镜像本身*内部*。

> 主机组件、NVIDIA容器工具包与启用GPU的Docker容器之间的交互。

### 构建支持GPU的Docker镜像

虽然NVIDIA容器工具包处理运行时访问，但您的Docker镜像仍然需要必需的软件库来*使用*GPU，主要是CUDA工具包以及可能的cuDNN等库。实现这一点的最简单方法是使用NVIDIA或流行机器学习 (machine learning)框架提供的官方基础镜像：

- `nvidia/cuda:[version]-base-[os]` 或 `nvidia/cuda:[version]-cudnn[version]-devel-[os]`：它们提供了CUDA和cuDNN环境。
- `pytorch/pytorch:[version]-cuda[version]-cudnn[version]-runtime`：包含特定CUDA/cuDNN版本的官方PyTorch镜像。
- `tensorflow/tensorflow:[version]-gpu`：内置GPU支持的官方TensorFlow镜像。

在Dockerfile中选择其中一个作为基础镜像，可以大大简化设置，因为它们预先打包了所需GPU库的兼容版本。

以下是一个使用TensorFlow GPU基础镜像的简单Dockerfile片段：

```dockerfile
# 使用官方的TensorFlow GPU基础镜像
FROM tensorflow/tensorflow:2.10.0-gpu

# 设置工作目录
WORKDIR /app

# 首先复制requirements文件以利用层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制其余应用程序代码
COPY . .

# 默认运行训练脚本的命令
CMD ["python", "train.py"]
```

这个Dockerfile没有包含关于GPU本身的明确指令。基础镜像`tensorflow/tensorflow:2.10.0-gpu`处理了CUDA和cuDNN的包含。运行时配置将其连接到主机的GPU。

### 运行授予GPU访问权限的训练容器

当您运行容器时，要授予容器访问主机GPU的权限，您可以使用`docker run`命令的`--gpus`标志。这个由NVIDIA容器工具包集成管理的标志，告诉Docker应该将哪些GPU暴露给容器。

`--gpus`的常见使用模式：

- `--gpus all`：将所有可用的NVIDIA GPU暴露给容器。
- `--gpus device=0`：只暴露索引为0的GPU。
- `--gpus device=1,2`：暴露索引为1和2的GPU。
- `--gpus '"device=0,1"'`：多重特定GPU的另一种语法。
- `--gpus count=2`：暴露前2个可用的GPU。

以下是如何运行从上述Dockerfile构建的训练容器，并授予它访问所有可用GPU的权限：

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  my-gpu-training-image:latest \
  python train.py --data_dir /app/data --output_dir /app/output --epochs 50
```

此命令：

- 使用`docker run`运行容器。
- 使用`--gpus all`以启用对所有主机GPU的访问。
- 使用绑定挂载（如第3章所述）挂载本地的`./data`和`./output`目录，以提供输入数据并存储训练后的模型/日志。
- 指定镜像名称（`my-gpu-training-image:latest`）。
- 覆盖默认的`CMD`以将特定参数 (parameter)传递给训练脚本。

### 验证容器内部的GPU访问

一旦您的容器运行起来，您会想要确认它是否能够实际识别和使用GPU。

1. **使用`nvidia-smi`：** NVIDIA系统管理接口（`nvidia-smi`）实用工具通常包含在NVIDIA基础镜像中。您可以在容器内部运行它：

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

   如果成功，此命令将打印出熟悉的`nvidia-smi`输出，列出*容器*可见的GPU。
2. **框架特有的检查：** 您的机器学习 (machine learning)框架通常提供函数来检查GPU可用性。

   - **TensorFlow：**

     ```python
     import tensorflow as tf
     gpus = tf.config.list_physical_devices('GPU')
     print("可用GPU数量: ", len(gpus))
     if gpus:
         try:
             # 目前，内存增长在所有GPU上需要保持一致
             for gpu in gpus:
                 tf.config.experimental.set_memory_growth(gpu, True)
             print("内存增长已启用。")
         except RuntimeError as e:
             # 内存增长必须在GPU初始化之前设置
             print(e)
     ```
   - **PyTorch：**

     ```python
     import torch
     available = torch.cuda.is_available()
     count = torch.cuda.device_count()
     name = torch.cuda.get_device_name(0) if available else "N/A"
     print(f"CUDA 可用: {available}")
     print(f"设备数量: {count}")
     print(f"设备名称 (GPU 0): {name}")
     ```

   在您的容器中运行这些代码片段（例如，在交互式会话中或作为脚本启动的一部分），将确认框架是否检测到由Docker和NVIDIA容器工具包传递的GPU。

### 将GPU与Docker Compose配合使用

如果您正在使用Docker Compose管理您的训练环境（稍后会更详细地介绍），您可以使用`deploy`规范（在Compose规范版本3.8+和Docker Engine 19.03+中可用）在`docker-compose.yml`文件中请求GPU资源。

```yaml
version: '3.8'

services:
  training:
    build: .
    image: my-gpu-training-image:latest
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - NVIDIA_VISIBLE_DEVICES=all # 可选，通常由deploy部分处理
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 # 请求一个GPU
              # 或者指定能力：
              # capabilities: [gpu]
              # 或者特定设备ID：
              # device_ids: ['0']
    command: python train.py --data_dir /app/data --output_dir /app/output
```

此配置为`training`服务请求一个NVIDIA GPU。当您运行`docker compose up`时，Compose将指示Docker引擎通过NVIDIA容器工具包分配GPU资源。

通过使用NVIDIA容器工具包和合适的基础镜像，您可以将GPU加速集成到您的容器化机器学习 (machine learning)训练工作流中。这确保了您计算密集型训练作业的高效运行，同时受益于Docker提供的一致性和可移植性。

## 参考资料

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  提供将NVIDIA GPU与Docker及其他容器运行时集成的完整安装、配置和使用说明。
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  NVIDIA并行计算平台和编程模型的基础文档，对于开发和运行GPU加速应用程序是必要的。
- [GPU support](https://www.tensorflow.org/install/gpu) — TensorFlow Authors (2023)
  提供配置TensorFlow使用GPU的指导，包括安装要求和验证方法，这对于容器内的机器学习应用至关重要。
- [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) — PyTorch Authors (2017)
  Publisher: PyTorch Foundation
  详细说明了PyTorch如何与NVIDIA GPU和CUDA交互，涵盖设备管理和操作，这对优化PyTorch应用是必要的。
