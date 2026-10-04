---
course: "docker-for-ml-projects"
chapter: "managing-ml-data-containers"
lesson: "cloud-storage-access"
sourceId: 5281
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/cloud-storage-access"
title: "从容器访问云存储"
description: "从机器学习容器内部安全地访问云存储服务的方法。"
order: 5
plots: []
sourceHash: "de60dcbad4bc78f04a0f74046b09785be8987183b9880ae0381f2505d15b10f3"
sourceCorrections: []
---

机器学习 (machine learning)工作流程中，高效的数据管理至关重要，特别是数据集和模型产物的处理。虽然像 Docker 卷和绑定挂载这样的本地存储方案可以管理与主机或 Docker 存储驱动程序直接关联的数据，但大量工作流程依赖于存储在云端的数据。Amazon S3、Google Cloud Storage (GCS) 和 Azure Blob Storage 等云服务因其可扩展性、持久性和易用性而被广泛应用。将这些服务与容器化应用程序结合是常见的需求。

从 Docker 容器内部访问云存储主要包括安全地提供凭据以及使用合适的云提供商 SDK 或工具。我们来看看常用的方法。

### 使用云提供商 SDK

最直接的方式是在您的 Python 脚本（或其他应用程序代码）中使用云厂商提供的官方软件开发工具包 (SDK)。常用的有：

- **AWS:** `boto3`
- **Google Cloud:** `google-cloud-storage`
- **Azure:** `azure-storage-blob`

您通常会在镜像构建过程中，通过 `requirements.txt` 或 `environment.yml` 文件在您的 Docker 镜像中安装这些库：

```dockerfile
# Dockerfile 安装 AWS SDK 的示例指令
RUN pip install boto3
```

SDK 安装完成后，您的 Python 代码就可以与存储服务交互（例如下载数据、上传模型）。主要的难点就是认证：容器内运行的 SDK 如何安全地获取所需权限？

### 认证方法

有几种方法可以为容器内运行的 SDK 提供凭据。

1. **环境变量：**
   这是一种简单直接的方法，常适用于本地开发或特定测试场景。在启动容器时，您可以使用 `docker run` 命令的 `-e` 标志，将凭据作为环境变量直接传递给容器。

   - **AWS:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (可选)
   - **Google Cloud:** 将 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量设置为服务账户密钥文件在*容器内的路径*（请看下一点）。
   - **Azure:** `AZURE_STORAGE_CONNECTION_STRING` 或单独的组件，如 `AZURE_STORAGE_ACCOUNT_NAME` 和 `AZURE_STORAGE_ACCOUNT_KEY`。

   ```bash
   # 示例：使用环境变量方式传递 AWS 凭据运行容器
   docker run -it \
     -e AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID \
     -e AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY \
     your-ml-image:latest \
     python train.py --data-bucket my-s3-bucket

   # 示例：运行指向 GCP 服务账户文件的容器
   # （需要先挂载文件，详见下文）
   docker run -it \
     -v /path/to/host/keyfile.json:/app/secrets/keyfile.json \
     -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/keyfile.json \
     your-ml-image:latest \
     python train.py --data-bucket my-gcs-bucket
   ```

   虽然简单，但将秘密直接嵌入 (embedding)环境变量可能带来安全风险，因为它们可能会被记录或检查。**切勿将凭据直接硬编码到 Dockerfile 中。**
2. **挂载凭据文件：**
   更安全的方法是使用卷或绑定挂载将主机上的凭据文件挂载到容器中。SDK 通常配置为自动检测这些标准位置的文件。

   - **AWS:** 将主机的 `~/.aws` 目录（包含 `credentials` 和 `config` 文件）挂载到 `/root/.aws`（或 `/home/user/.aws`，取决于容器内的用户）。
   - **Google Cloud:** 将服务账户密钥文件（JSON 格式）挂载到容器中，并将 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量设置为其在*容器内的路径*。
   - **Azure:** 像 Azure CLI 这样的工具通常将登录信息存储在 `~/.azure` 中，这也可以被挂载，尽管连接字符串或服务主体是常见的替代方案。

   ```bash
   # 示例：挂载 AWS 凭据
   docker run -it \
     -v ~/.aws:/root/.aws:ro \
     your-ml-image:latest \
     python train.py --data-bucket my-s3-bucket

   # 示例：挂载 GCP 服务账户文件
   docker run -it \
     -v /path/to/host/keyfile.json:/app/secrets/keyfile.json:ro \
     -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/keyfile.json \
     your-ml-image:latest \
     python train.py --data-bucket my-gcs-bucket
   ```

   建议对凭据文件使用只读 (`:ro`) 挂载。
3. **IAM 角色和实例元数据服务（推荐用于云部署）：**
   在云虚拟机（如 AWS EC2、Google Compute Engine、Azure Virtual Machines）或托管容器服务（ECS、EKS、GKE、AKS）上运行容器时，最安全且推荐的方法是利用与底层计算实例关联的身份和访问管理 (IAM) 角色或服务账户。

   - 云提供商的 SDK 通常设计为自动查询实例上可用的本地元数据服务。
   - 该元数据服务根据分配给实例的 IAM 角色提供临时、自动轮换的凭据。
   - 您的容器会继承其运行实例的权限，而无需向其传递任何明确的凭据文件或环境变量。

digraph G {
rankdir=LR;
node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];
edge [fontname="Helvetica", fontsize=9];

```
subgraph cluster_host {
    label = "云实例 (例如 EC2, GCE)";
    bgcolor="#e9ecef";
    style=filled;
    "实例元数据服务" [shape=cylinder, fillcolor="#ced4da"];
    "IAM 角色权限" [shape=note, fillcolor="#ced4da"];
    "实例元数据服务" -> "IAM 角色权限" [style=dotted, label=" 提供\n 临时\n 凭据"];
}

subgraph cluster_container {
    label = "Docker 容器";
    bgcolor="#a5d8ff";
    style=filled;
    "SDK (boto3 等)" [fillcolor="#74c0fc"];
    "ML 应用程序" [fillcolor="#74c0fc"];
    "ML 应用程序" -> "SDK (boto3 等)" [label=" 使用"];
}

"SDK (boto3 等)" -> "实例元数据服务" [label=" 1. 请求凭据"];
"实例元数据服务" -> "SDK (boto3 等)" [label=" 2. 返回临时凭据"];
"SDK (boto3 等)" -> "云存储 (S3, GCS 等)" [label=" 3. 访问存储\n (使用凭据)"];

"云存储 (S3, GCS 等)" [shape=cylinder, fillcolor="#96f2d7"];
```

}
\`\`\`

```
> 图示说明容器内的 SDK 如何通过实例元数据服务，依据主机的 IAM 角色获取临时凭据。

这种方法避免在容器设置中管理秘密文件或环境变量，显著提高安全状况，尤其在生产环境中。确保实例的 IAM 角色具有访问特定云存储资源所需的最低权限（最小特权原则），例如对特定 S3 存储桶的读取权限。
```

### 挂载云存储的工具

除了应用程序代码中的 SDK 访问，还有一些工具可以将云存储桶挂载为容器内的本地文件系统。例如：

- **s3fs-fuse:** 挂载 S3 存储桶。
- **gcsfuse:** 挂载 GCS 存储桶。
- **blobfuse:** 挂载 Azure Blob Storage 容器。

这些工具需要 FUSE (用户空间文件系统) 支持，通常需要在容器内安装和配置。对于需要标准文件系统访问的应用程序来说很方便，但与直接使用 SDK 相比，可能会引入性能开销，尤其对于涉及大量小文件或高延迟的操作。配置通常涉及使用与 SDK 类似的方法提供凭据。

### 选择合适的方法

- **本地开发：** 挂载凭据文件（`~/.aws`、服务账户密钥）或使用环境变量（需谨慎）可能比较方便。
- **云部署（虚拟机、托管容器）：** 强烈建议使用分配给计算实例的 IAM 角色或服务账户。这是最安全且易于管理的方法。
- **应用程序需求：** 如果您的应用程序高度依赖文件系统 API，基于 FUSE 的工具可能是一种选择，但需评估其性能影响。对于程序化访问，直接集成 SDK 通常更高效。

通过理解这些方法，您可以有效且安全地将容器化的机器学习 (machine learning)应用程序连接到云存储中的数据，实现可扩展的训练和推理 (inference)工作流程。请记住始终优先考虑安全性，遵循最小特权原则，并避免将秘密直接嵌入 (embedding)到您的镜像中。

## 参考资料

- [IAM roles for Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  解释了如何将IAM角色与EC2实例结合使用，通过实例元数据服务向包括Docker容器中的应用程序授予权限，从而通过避免明确的凭证管理来增强安全性。
- [Authenticating to Google Cloud](https://cloud.google.com/docs/authentication/how-adc-works) — Google Cloud (2025)
  Publisher: Google Cloud
  提供了Google Cloud的身份验证方法概述，包括服务账号和默认应用程序凭证，这些对于GCE/GKE上容器化应用程序的安全访问至关重要。
- [What are managed identities for Azure resources?](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/overview) — Microsoft Azure (2024)
  Publisher: Microsoft
  介绍了Azure的托管标识，这是一种安全的方式，使运行在Azure虚拟机或托管容器服务上的应用程序无需管理凭证即可向Azure服务进行身份验证。
- [Boto3 Documentation - S3 client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  提供了使用boto3 Python SDK与Amazon S3交互的详细API参考和示例，涵盖了从容器内下载和上传对象等操作。
