---
course: "mlops-for-large-models-llmops"
chapter: "llm-deployment-serving-optimization"
lesson: "llm-packaging-containerization"
sourceId: 3166
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/llm-packaging-containerization"
title: "大型语言模型 (LLM) 的模型打包与容器化"
description: "使用容器化技术有效打包大型模型及其依赖项。"
order: 2
plots: []
sourceHash: "3a0408cd8ed115fcb61f7597342f69318e6f712dc32654bb618fc60c737689f0"
sourceCorrections: []
---

部署机器学习 (machine learning)模型需要将模型制品及其依赖项打包成可重现的格式。对于大型语言模型 (LLM)，此过程会带来显著的操作困难，因为模型权重 (weight)文件庞大，且其软件环境很复杂。在处理大小可达数十到数百GB的模型时，标准方法通常不够用。

### 打包 LLM 的难题

与较小的机器学习 (machine learning)模型不同，LLM 会带来特有的打包难题：

1. **庞大的制品体积**：模型检查点（权重 (weight)）通常对于 Git 等标准版本控制系统或简单的容器镜像层来说过大。一个拥有700亿参数 (parameter)的模型，即使采用半精度（FP16），也需要大约140GB的存储空间。
2. **复杂的依赖项**：LLM 依赖于特定版本的深度学习 (deep learning)框架（PyTorch、TensorFlow、JAX）、CUDA 库、优化过的核函数（如 FlashAttention），以及可能专门的硬件驱动。确保开发、测试和生产环境之间的一致性是很困难的。
3. **硬件关联性**：性能通常取决于特定的 GPU 架构和相关软件（CUDA 版本、cuDNN）。软件包需要考虑或适应目标硬件环境。
4. **安全**：模型权重代表重要的知识产权和投入。打包和分发必须考虑安全的处理和访问控制。

### 模型序列化与大型权重 (weight)文件处理

第一步是保存训练或微调 (fine-tuning)后的模型权重。尽管存在标准的框架方法（`torch.save`、`tf.saved_model.save`），但它们可能不适用于 LLM。

- **标准格式**：特定于框架的格式很常见，但有时可能依赖于 Python 的 `pickle` 模块，该模块存在已知的安全漏洞（可执行任意代码）。加载大型检查点也可能很慢。
- **Safetensors**：这种格式（`.safetensors`）在大型模型中很受欢迎。它在设计上强调安全性（不会执行任意代码），并可能加载更快，尤其在使用内存映射时。它以扁平二进制布局存储张量，并带有一个描述张量元数据的 JSON 头部。

考虑到其体积，直接将权重嵌入 (embedding)到版本控制系统或单个容器镜像层中通常是不实际的。常见的方法有：

- **模型拆分**：将大型检查点文件分成更小、更易于管理的分片。Hugging Face Transformers 等框架通常支持自动从分片检查点加载模型。
- **外部存储**：将模型制品存储在专门的对象存储（AWS S3、Google Cloud Storage、Azure Blob Storage）或制品库（如 Nexus 或 Artifactory）中。部署过程随后涉及在运行时或容器初始化期间将这些制品下载到推理 (inference)服务器。
- **大型文件的版本控制**：Git Large File Storage (LFS) 等工具可以提供帮助，但它们可能难以处理超大文件（100+ GB），并且需要仔细的基础设施管理。它们通常更适合管理数据集或较小的模型版本。

### 依赖管理

精确定义软件环境十分要紧。

- **需求文件**：使用 `requirements.txt`（针对 pip）或 `environment.yml`（针对 conda）来列出所有带有固定版本的 Python 依赖项。这包括深度学习 (deep learning)框架、像 `transformers`、`accelerate`、`bitsandbytes`（用于量化 (quantization)）等库。
- **系统依赖项**：LLM 通常需要特定的系统库，特别是 CUDA 工具包（编译器 `nvcc`、运行时库如 `libcudart.so`、`libcudnn.so`、`libnccl.so`）。这些依赖项最好通过容器化来管理。

### 使用 Docker 进行容器化

Docker 为复杂的 LLM 环境提供了必要的隔离性和可重现性。它将应用程序代码、模型（或获取模型的指令）、库和系统依赖项打包成一个独立单元：容器镜像。

#### 构建 LLM Docker 镜像

为 LLM 创建一个高效的 Dockerfile 涉及多方面考量：

1. **基础镜像选择**：从包含所需 GPU 驱动和 CUDA 工具包版本的官方基础镜像开始。NVIDIA 在 NGC（NVIDIA GPU Cloud）上提供了针对不同 CUDA 版本和深度学习 (deep learning)框架优化的容器镜像（例如 `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`）。选择正确的基础镜像可减少设置工作量，并确保与目标 GPU 硬件的兼容性。
2. **安装依赖项**：使用 `pip` 或 `conda` 安装需求文件中指定的 Python 包。确保安装过程高效且可重现。
3. **模型制品处理**：你有两种主要方法：

   - **将权重 (weight)打包到镜像中**：在构建过程中直接将模型权重复制到 Docker 镜像中（`COPY model_weights /app/model_weights`）。
     - *优点*：镜像独立完整，如果权重已存在，启动可能更快。
     - *缺点*：镜像体积非常大，构建和推拉（push/pull）时间较慢，灵活性较低（更新模型需要重新构建镜像）。容易达到容器仓库的存储限制。
   - **在运行时加载权重**：构建一个只包含代码和依赖项的较小镜像。配置容器入口点或初始化脚本，使容器启动时从外部存储（例如 S3）下载模型权重。
     - *优点*：镜像体积较小，构建/分发更快，模型更新更简便（只需指向新权重）。
     - *缺点*：冷启动时间较慢（需要下载），启动时需要网络访问制品存储，需要处理下载失败的情况。

   对于大多数大规模 LLM 部署，由于体积限制和操作灵活性，在运行时加载权重是更受青睐的方法。
4. **优化镜像体积和构建时间**：

   - **多阶段构建**：使用多阶段构建来将构建时依赖项（如编译器）与最终运行时环境分离。这会显著减少最终镜像的体积。
   - **层缓存**：组织你的 Dockerfile 以高效利用层缓存。将不常更改的命令（如安装系统包或基本依赖项）放在文件的靠前位置。
   - **减少层数**：在逻辑允许的情况下，合并多个 `RUN` 命令，以减少镜像层数。

#### Dockerfile 示例结构（运行时权重加载）

```dockerfile
# 阶段 1：构建阶段（如果自定义组件需要）
# FROM ... 作为构建器
# RUN ... 构建自定义内核或依赖项 ...

# 阶段 2：最终运行时阶段
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG OS_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${OS_VERSION}

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=off \
    TRANSFORMERS_CACHE=/app/.cache \
    HF_HOME=/app/.cache

# 安装系统依赖项
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# 复制必要文件（需求、应用程序代码）
WORKDIR /app
COPY requirements.txt .
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# 安装 Python 依赖项
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# （可选）从构建阶段复制预编译组件
# COPY --from=builder /path/to/built/artifact /app/

# 设置入口点/命令以运行推理服务器
# 如果权重未挂载，此脚本将处理权重下载
COPY scripts/start_server.sh .
ENTRYPOINT ["/app/start_server.sh"]

# 暴露推理端口
EXPOSE 8000
```

> 一个用于 LLM 推理 (inference)服务器的多阶段 Dockerfile 结构。它使用官方的 NVIDIA CUDA 基础镜像，安装依赖项，复制应用程序代码，并定义了一个入口点脚本。模型权重被假定由 `start_server.sh` 在运行时加载。

### 安全方面考量

在打包和容器化时，需要注意：

- **基础镜像漏洞**：使用 Trivy 或 Docker Scout 等工具定期扫描基础镜像，查找已知的安全漏洞。
- **制品访问控制**：如果从外部存储加载权重 (weight)，确保配置适当的权限（例如 S3 的 IAM 角色），以便只有授权服务才能访问模型制品。
- **秘密管理**：避免在 Dockerfile 或容器镜像中硬编码访问密钥或秘密信息。使用你的编排器（如 Kubernetes Secrets）或云提供商提供的安全秘密管理方案。

通过仔细管理模型序列化、依赖项和容器化方法，你可以为大型语言模型创建可重现、可移植且体积适中的部署单元，为在生产环境中高效提供服务创造条件。将权重打包到镜像中与在运行时加载它们的选择，很大程度上取决于你的具体基础设施、模型更新频率以及对冷启动延迟的容忍度。

## 参考资料

- [safetensors](https://github.com/huggingface/safetensors) — Hugging Face (2023)
  介绍 Safetensors 的设计和使用方法，这是一种专为大型深度学习模型安全高效序列化设计的格式。
- [NVIDIA CUDA Container Images](https://catalog.ngc.nvidia.com/containers) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  CUDA 和 cuDNN GPU 优化 Docker 基础镜像的官方来源，对高性能 LLM 服务至关重要。
- [Dockerfile best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Inc. (2024)
  Publisher: Docker Inc.
  创建高效、安全、可维护 Docker 镜像的官方指南，包括多阶段构建和层缓存。
- [MLOps Engineering at Scale](https://www.oreilly.com/library/view/mlops-engineering-at/9781098103138/) — Carl Osipov (2022)
  Publisher: O'Reilly Media
  提供构建和部署规模化 ML 系统的指南，包含模型打包、依赖管理和容器化的讨论。
