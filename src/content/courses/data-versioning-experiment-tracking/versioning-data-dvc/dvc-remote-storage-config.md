---
course: "data-versioning-experiment-tracking"
chapter: "versioning-data-dvc"
lesson: "dvc-remote-storage-config"
sourceId: 4070
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/dvc-remote-storage-config"
title: "将 DVC 连接到远程存储 (S3、GCS、Azure Blob)"
description: "配置 DVC 以使用 AWS S3、Google 云存储和 Azure Blob 存储等常用云存储方案。"
order: 6
plots: []
sourceHash: "b19bc6d08ecee604ce333b6b72be38d8ef4ea7e294f30a2693c6ce98753ff8e5"
sourceCorrections: []
---

虽然 `dvc add` 命令告诉 DVC 跟踪哪些文件或目录，且 `.dvc` 文件作为存储在 Git 中的指针，但实际的大数据文件需要在 Git 仓库之外有存放位置。这时就需要远程存储。DVC 适配多种存储后端，包括 AWS S3、Google 云存储 (GCS) 和 Azure Blob 存储等常用云服务商，以及网络驱动器或甚至本地机器上的其他目录。

可将 DVC 远程存储看作 DVC 管理的版本化数据内容的指定存放处。你的 Git 仓库存放着小的 `.dvc` 文件（元数据和指针），而远程存储则保存由哈希值确定的实际数据块。这种配置使得 Git 仓库保持小巧快速，同时又能访问与特定代码版本相关联的大型数据集。

### 设置远程存储位置

要让 DVC 知道数据推送和拉取的位置，请使用 `dvc remote add` 命令。基本语法如下：

```bash
dvc remote add <remote_name> <remote_url>
```

- `<remote_name>`: 这是你为远程存储配置选择的一个简短、易记的名称（例如，`my-s3`、`gcp-storage`、`azure-data`）。如果只有一个主要远程，通常会沿用 Git 中的做法使用 `origin`。
- `<remote_url>`: 这说明了存储的类型和地址。格式取决于存储服务商。

我们来看看如何为常见的云服务商配置远程。

#### AWS S3 (简单存储服务)

如果你使用 AWS S3，URL 通常形如 `s3://<bucket_name>/<optional_path>`。

例如，要配置一个名为 `my-ml-data-bucket` 的 S3 存储桶，并将数据存放在 `project-alpha/datasets` 路径下，你可以运行：

```bash
dvc remote add my_s3_storage s3://my-ml-data-bucket/project-alpha/datasets
```

你也可以将其设为默认远程，这意味着除非另行指定，`dvc push` 和 `dvc pull` 等命令将自动使用它。请使用 `-d` 或 `--default` 标志：

```bash
dvc remote add -d origin s3://my-ml-data-bucket/project-alpha/datasets
```

**认证：** DVC 采用标准的 AWS 凭证机制。它将按以下顺序自动查找凭证：

1. 环境变量（`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY` 等）。
2. 共享凭证文件（`~/.aws/credentials`）。
3. AWS 配置文件（`~/.aws/config`）。
4. 附加到 EC2 实例或 ECS 任务的 IAM 角色。

请确保你的环境已正确设置，可以使用 AWS CLI (`aws configure`) 等工具，或通过设置环境变量。

#### Google 云存储 (GCS)

对于 GCS，URL 格式为 `gs://<bucket_name>/<optional_path>`。

要配置一个名为 `my-gcp-project-bucket` 的 GCS 存储桶，并将数据存放在 `ml-experiments/data` 下，你可以运行：

```bash
dvc remote add gcs_data gs://my-gcp-project-bucket/ml-experiments/data
```

要将其设为默认：

```bash
dvc remote add -d gcs_default gs://my-gcp-project-bucket/ml-experiments/data
```

**认证：** DVC 通常依赖为 `gcloud` 命令行工具设置的认证，或通过 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量指定的服务账户凭证。

#### Azure Blob 存储

对于 Azure Blob 存储，URL 格式为 `azure://<container_name>/<optional_path>`。你需要有一个 Azure 存储账户并在其中拥有一个 Blob 容器。

假设你的存储账户中容器名为 `ml-datasets`，你可以这样配置它：

```bash
dvc remote add azure_data azure://ml-datasets/project-x/raw
```

并设为默认：

```bash
dvc remote add -d azure_default azure://ml-datasets/project-x/raw
```

**认证：** DVC 采用 Azure 的标准认证方式。这通常涉及通过 Azure CLI (`az login`) 登录，或设置 `AZURE_STORAGE_CONNECTION_STRING` 或 `AZURE_STORAGE_ACCOUNT_NAME` 和 `AZURE_STORAGE_ACCOUNT_KEY` 等环境变量。也支持使用托管标识或服务主体。

#### 其他存储类型

DVC 适配多种其他存储类型：

- **本地目录：** 适用于测试或在同一机器上存储数据（项目目录之外）。

dvc remote add local\_backup /path/to/external/dvc-storage
\`\`\`

- **SSH：** 将数据存储在可通过 SSH 访问的服务器上。

dvc remote add my\_server ssh://user@example.com:/home/user/dvc-storage
\`\`\`

- **HDFS、HTTP、WebDAV：** DVC 适配多种其他协议。有关具体的 URL 格式和认证信息，请查阅 DVC 文档。

### 管理远程设置

添加远程后，设置会保存在项目中的 `.dvc/config` 文件里。这是一个普通文本文件，内容大致如下：

```ini
[core]
    remote = origin
['remote "origin"']
    url = s3://my-ml-data-bucket/project-alpha/datasets
['remote "gcs_data"']
    url = gs://my-gcp-project-bucket/ml-experiments/data
```

由于此配置文件指明了你的数据存放位置，**将 `.dvc/config` 提交到 Git 仓库非常必要**。这能确保任何克隆你仓库的人都能设置其访问相同的远程存储（尽管他们需要自己的凭证），并使用 `dvc pull` 获取正确的数据版本。

你可以随时使用以下命令查看已设置的远程：

```bash
dvc remote list
```

如果你需要更改远程的设置（例如，更新凭证或调整某些选项），可以使用 `dvc remote modify`。例如，要为 `origin` 远程明确设置 AWS 凭证（尽管通常更推荐依赖环境变量）：

```bash
# 示例：设置特定配置文件（不常用，通常依赖环境变量）
dvc remote modify origin profile my_aws_profile

# 示例：明确设置 Azure 连接字符串
dvc remote modify azure_data connection_string "your_connection_string"
```

有关每种远程类型可用的具体选项，请查阅 DVC 文档。

要移除远程设置，请使用 `dvc remote remove <remote_name>`。

配置好远程存储后，你现在对整体情况已很清楚：Git 跟踪你的代码和小的 `.dvc` 指针文件，而 DVC 则管理大型数据文件，并通过 `dvc push` 和 `dvc pull` 命令在你的本地工作区和已设置的远程存储之间同步它们。这种分离对于在机器学习 (machine learning)项目中，配合代码版本控制来有效处理数据，是十分重要的。

## 参考资料

- [Remote Storage](https://dvc.org/doc/user-guide/data-management/remote-storage) — Iterative, Inc. (2024)
  解释DVC远程存储，并提供包括云服务商、SSH和本地存储在内的各种后端详细设置指南。
- [dvc remote](https://dvc.org/doc/command-reference/remote) — Iterative, Inc. (2024)
  `dvc remote`命令的官方文档，详细说明了其子命令如`add`、`modify`、`list`和`remove`，以及所有可用选项。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning-systems/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  全面概述了构建机器学习系统，包括数据管理、版本控制以及生产应用架构设计的意义。
