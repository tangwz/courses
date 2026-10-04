---
course: "planning-optimizing-ai-infrastructure"
chapter: "leveraging-cloud-platforms-for-ai"
lesson: "practical-launching-gpu-cloud-instance"
sourceId: 7006
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-3-leveraging-cloud-platforms-for-ai/practical-launching-gpu-cloud-instance"
title: "动手实践：启动GPU云实例"
description: "一个动手实验，用于在主要云平台上配置、设置并连接一台支持GPU的虚拟机。"
order: 8
plots: []
sourceHash: "4760488307be19177c48c15bfa563ac0934b57b8dfbd0b454696f98a744ddd42"
sourceCorrections: []
---

云基础设施的原理通过直接操作最能体会。在这里，您将配置、设置并连接一台由GPU驱动的虚拟机，位于一个主要的云平台上。本次练习提供了一个可重复的基础操作流程，用于构建机器学习 (machine learning)开发和训练的远程环境。

虽然步骤以亚马逊网络服务（AWS）为例，但对于谷歌云平台（GCP）和微软Azure来说，过程是相似的。目标是弄清核心组成部分：选择实例、选取预配置的软件镜像、管理网络访问以及安全连接。

### 前提条件

开始前，请确保您已准备好以下各项：

- **一个活跃的云服务商账户：** 本实验使用AWS。您需要一个已启用计费功能的账户。
- **已安装并配置云服务商CLI：** 对于AWS，这是[AWS命令行界面](https://aws.amazon.com/cli/)。您应该通过运行`aws configure`配置好您的凭证。
- **一个SSH客户端：** 这通常在macOS和Linux上默认可用。Windows用户可以使用WSL（适用于Linux的Windows子系统）或PuTTY等客户端。
- **一个已生成的SSH密钥对：** 您将需要一个SSH密钥对来安全连接到您的实例。如果您没有，可以在下面的步骤中生成。

### 步骤1：选择实例和机器镜像

我们的第一个决定是选择虚拟机配置。如前所述，这涉及平衡性能与成本。对于本次练习，我们将使用一个AWS `g4dn.xlarge`实例。它配备NVIDIA T4 GPU，为通用机器学习 (machine learning)任务提供了一个良好的起点，且没有顶级训练加速器的高昂成本。

接下来，我们必须选择一个亚马逊机器镜像（AMI）。AMI是一个包含操作系统和预装软件的模板。为了节省大量设置时间，我们将使用官方的AWS深度学习 (deep learning)AMI。这些镜像预装了NVIDIA驱动程序、CUDA、cuDNN以及TensorFlow和PyTorch等主要的ML框架。

我们将寻找一个类似于`Deep Learning AMI GPU TensorFlow X.X.X (Ubuntu 20.04)`名称的AMI。您可以在AWS EC2控制台中或通过AWS CLI找到您所选区域的最新AMI ID。对于本例，我们将使用一个占位符AMI ID。您必须将`ami-0123456789abcdef0`替换为您所在区域的有效AMI ID。

### 步骤2：配置安全和访问

在启动实例之前，我们需要定义谁可以访问它。这通过安全组进行管理，安全组充当虚拟防火墙。对于本次实验，我们只需允许来自我们自己IP地址的入站SSH流量（端口22）。

我们还需要一个SSH密钥对来验证我们的连接。您可以通过AWS控制台或使用CLI创建新的密钥对。以下命令创建一个名为`ai-infra-key`的密钥对，并将私钥保存到名为`ai-infra-key.pem`的本地文件中。

```bash
aws ec2 create-key-pair --key-name ai-infra-key --query 'KeyMaterial' --output text > ai-infra-key.pem
```

**重要：** 保护好此`.pem`文件。这是访问您实例的唯一途径。您还需要更改其文件权限。

```bash
chmod 400 ai-infra-key.pem
```

### 步骤3：启动GPU实例

确定了实例类型、AMI和安全设置后，我们即可启动虚拟机。以下AWS CLI命令将所有这些配置打包为一个请求。

此命令执行多项操作：

- `--image-id`：指定深度学习 (deep learning)AMI。**请记住替换占位符。**
- `--instance-type`：将硬件设置为`g4dn.xlarge`。
- `--key-name`：关联我们刚创建的SSH密钥对。
- `--security-group-ids`：您应首先创建一个允许端口22访问的安全组，并在此处使用其ID（例如，`sg-012345abcdef`）。
- `--tag-specifications`：为实例分配一个描述性名称，以便后续查找。

```bash
# 首先，在您的区域（例如 us-east-1）中查找一个有效的深度学习AMI ID
# aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU TensorFlow*" --query 'Images[?CreationDate>`2023-01-01`].{ID:ImageId,Name:Name}' --region us-east-1

# 然后，创建一个安全组
# aws ec2 create-security-group --group-name my-gpu-sg --description "用于GPU实例的安全组"
# aws ec2 authorize-security-group-ingress --group-name my-gpu-sg --protocol tcp --port 22 --cidr YOUR_IP_ADDRESS/32

# 现在，使用您的AMI和安全组ID启动实例
aws ec2 run-instances \
    --image-id ami-0123456789abcdef0 \
    --instance-type g4dn.xlarge \
    --key-name ai-infra-key \
    --security-group-ids sg-012345abcdef \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AI-Infra-Lab-Instance}]'
```

运行此命令后，AWS将返回一个描述新实例的JSON对象。请记下输出中的`InstanceId`，因为您将需要它来管理实例。

### 步骤4：连接并验证设置

实例初始化需要几分钟。您可以使用`InstanceId`获取其公共IP地址。

```bash
# 将 i-012345abcdef 替换为您的实际 InstanceId
aws ec2 describe-instances --instance-ids i-012345abcdef --query 'Reservations[].Instances[].PublicIpAddress' --output text
```

获得IP地址后，使用SSH连接到实例。AWS上基于Ubuntu的AMI的默认用户名为`ubuntu`。

```bash
ssh -i "ai-infra-key.pem" ubuntu@YOUR_INSTANCE_PUBLIC_IP
```

如果连接成功，您将看到新云服务器的命令提示符。首要任务是验证GPU是否被识别且驱动程序正常运行。

运行NVIDIA系统管理界面（`nvidia-smi`）工具：

```bash
nvidia-smi
```

您应该会看到一个输出表格，详细列出NVIDIA驱动版本、CUDA版本以及有关所连接GPU的信息，在本例中是Tesla T4。

> 配置和验证云GPU实例的流程。

作为最终检查，我们运行一个简短的Python脚本，以确认PyTorch可以访问GPU。

```python
# 将此文件保存为 verify_gpu.py
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch 可以访问CUDA设备：{torch.cuda.get_device_name(0)}")

    # 创建一个张量并将其移动到GPU
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"张量已成功在GPU上创建：{x}")
    print(f"张量设备：{x.device}")
else:
    print("错误：PyTorch 无法找到支持CUDA的GPU。")
```

从您的SSH会话中运行此脚本：`python3 verify_gpu.py`。成功输出表明您的环境已完全准备就绪。

### 步骤5：终止实例（重要）

只要云资源运行，就会产生费用。“停止”的实例可能不产生计算费用，但其存储卷（EBS）仍会计费。为避免本次实验产生额外费用，您必须**终止**实例。

> **警告：** 终止是不可逆操作。实例本地存储上的所有数据将被永久删除。

使用您之前记下的`InstanceId`来终止实例。

```bash
# 将 i-012345abcdef 替换为您的实际 InstanceId
aws ec2 terminate-instances --instance-ids i-012345abcdef
```

此命令将安排实例终止。几分钟后，实例将被完全移除，计费也将停止。请务必在AWS管理控制台中再次检查，确保完成工作后资源已被终止。这个习惯是有效管理云成本最重要的一个方面。

## 参考资料

- [Amazon EC2 User Guide for Linux Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  关于管理 Amazon EC2 实例的指南，涵盖实例类型、AMI、安全和访问方法。
- [AWS Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  详细介绍 AWS 深度学习 AMI 提供的预配置机器学习环境的文档。
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  CUDA 工具包的官方文档，提供关于 GPU 计算和 `nvidia-smi` 工具的详细信息。
- [CUDA semantics - PyTorch documentation](https://pytorch.org/docs/stable/notes/cuda.html) — PyTorch Core Team (2025)
  Publisher: PyTorch
  解释 PyTorch 如何与 NVIDIA GPU 交互并管理 CUDA 操作的文档。
