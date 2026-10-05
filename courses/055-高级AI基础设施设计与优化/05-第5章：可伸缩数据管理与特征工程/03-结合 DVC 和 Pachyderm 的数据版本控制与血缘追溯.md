# 结合 DVC 和 Pachyderm 的数据版本控制与血缘追溯

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-5-scalable-data-feature-engineering/data-versioning-lineage-dvc-pachyderm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 Git 解决了代码版本控制问题，但机器学习 (machine learning)生命周期引入了一个类似的问题：对数据、模型和中间产物进行版本控制，这些与代码本身同等重要。如果没有系统来管理这些资产，可复现性几乎不可能实现。如果底层数据集发生变化，对特定提交执行 `git checkout` 并不能保证能够重建模型。解决此可复现性问题需要建立稳健的数据版本控制和血缘追溯。数据版本控制和血缘追溯对于可审计和可靠的机器学习系统是基础，相关的工具和方法将被介绍。

数据版本控制提供了一种机制，用于跟踪和检索数据集的特定状态，类似于 Git 跟踪源代码的方式。数据血缘追溯更进一步，它会创建一个可审计的轨迹，将最终模型追溯回其来源：源代码的确切版本、使用到的数据集以及生成它所用的超参数 (parameter) (hyperparameter)。

### 代码的可复现性难题

在传统的软件项目中，构建过程是确定性的：相同的代码版本总是生成相同的二进制文件。在机器学习 (machine learning)中，“构建”过程（模型训练）依赖于三个要素：代码、数据和配置。

`Model = f(code, data, configuration)`

像 Git 这样的版本控制系统只管理 `code`，并在一定程度上管理 `configuration`（例如，在 YAML 文件中）。`data` 组件通常包含千兆字节或太兆字节的文件，位于此系统之外。如果 `data` 在未进行版本控制的情况下被修改，链接就会断裂。您将无法可靠地回答以下问题：

- “三个月前，用于训练生产模型的具体数据是什么？”
- “为什么今天训练的模型表现比上周的版本差，即使代码相同？”
- “我们能否向监管机构证明，我们的模型未在偏颇或未经批准的数据集上进行训练？”

为解决此问题，我们需要将版本控制原理引入数据和管道层面的工具。我们将考察两种流行且理念不同的工具：DVC (Data Version Control) 和 Pachyderm。

### DVC：扩展 Git 用于数据管理

DVC 旨在流畅集成到标准的 Git 工作流程中。它遵循一个简单但有效的方法：将大文件存储在 Git 之外，但在 Git 中跟踪指向它们的指针。这使得您可以使用熟悉的 `git checkout`、`git log` 和 `git push` 等命令来管理大型数据集，而不会使 Git 仓库变得臃肿。

#### DVC 如何工作

DVC 使用远程存储后端（例如，S3 存储桶、Google Cloud Storage 或 SSH 服务器）来保存实际数据。在您的 Git 仓库中，DVC 会创建小型元数据文件（以 `.dvc` 结尾）。这些是轻量级文本文件，包含数据 MD5 哈希值和文件在远程存储中位置的信息。

工作流程很简单：

1. **跟踪数据：** 您指示 DVC 跟踪一个数据文件或目录。

   ```bash
   # 将大型数据集目录添加到 DVC 跟踪
   dvc add data/raw_images

   # 这会创建 data/raw_images.dvc
   # 原始目录被添加到 .gitignore
   ```
2. **提交元数据：** 您将小的 `.dvc` 文件提交到 Git。

   ```bash
   # 暂存 .dvc 文件和 .gitignore 更新
   git add data/raw_images.dvc .gitignore

   # 提交指针文件
   git commit -m "feat: track initial raw image dataset"
   ```
3. **推送数据：** 您将实际数据推送到已配置的远程存储。

   ```bash
   # 这会将 raw_images.dvc 中引用的文件上传到您的 S3 存储桶
   dvc push
   ```

现在，另一位团队成员可以克隆 Git 仓库并运行 `dvc pull`，以下载与其 Git 提交相对应的正确数据版本。使用 `git checkout` 切换到之前的分支并运行 `dvc pull`，将把数据集恢复到该时间点上的确切状态。

#### 用于血缘追溯的 DVC 管道

除了简单的文件版本控制，DVC 还可以定义和执行多阶段的机器学习 (machine learning)管道。您可以使用 `dvc stage add` 或 `dvc run` 来指定每个步骤的依赖项、命令和输出。DVC 利用这些信息来构建整个工作流程的 有向无环图 (DAG)。

```bash
# 定义两步管道的示例
# 步骤 1：预处理数据
dvc stage add -n preprocess \
              -d src/preprocess.py -d data/raw_images \
              -o data/processed_features \
              python src/preprocess.py --in data/raw_images --out data/processed_features

# 步骤 2：训练模型
dvc stage add -n train \
              -d src/train.py -d data/processed_features \
              -p params.yaml:train \
              -o models/model.pkl \
              python src/train.py --in data/processed_features --out models/model.pkl
```

这会生成一个 `dvc.yaml` 文件，存储管道结构。DVC 现在可以可视化血缘关系，准确显示您的数据、代码和参数 (parameter)如何与最终模型关联。

> 一个 DVC 管道 DAG，演示了从原始数据和源代码到训练后的模型产物的血缘关系。DVC 会检测到任何依赖项（例如 `src/preprocess.py` 或 `data/raw_images`）的变化。

通过运行 `dvc repro`，DVC 将智能地只重新执行受更改影响的管道阶段，节省大量计算时间。

### Pachyderm：Kubernetes 原生数据管道

Pachyderm 提供不同的方法。它是一个基于 Kubernetes 构建的以数据为中心的平台，旨在自动化大规模数据转换和机器学习 (machine learning)管道。DVC 为开发者扩展了 Git 工作流程，而 Pachyderm 则提供了一个强大的集群级数据处理系统。

#### Pachyderm 如何工作

Pachyderm 的架构基于两个主要对象：

- **数据仓库：** 这些是数据的版本化存储位置，在理念上类似于 Git 仓库。您创建一个仓库并向其中“提交”数据。每次提交都是数据的一个不可变、版本化的快照。
- **管道：** 管道是订阅一个或多个数据仓库的 Kubernetes 作业。它由一个规范定义，该规范包含输入数据仓库、用于处理代码的 Docker 镜像以及要执行的命令。

最重要的特点是 **管道由新的数据提交自动触发**。当您向输入仓库提交新数据时，Pachyderm 会自动执行管道，处理新数据，并将结果作为新提交放置到输出仓库中。

这种以数据为中心的触发机制确保血缘关系自动捕获并生效。Pachyderm 中的每一份输出数据都可以追溯到确切的输入数据提交以及生成它的管道版本。这提供了集群中所有数据和管道的“全局”血缘关系。

> Pachyderm 中的数据驱动管道执行。对 `images` 仓库的一次提交会触发 `resize-images` 管道，该管道会在 `resized-images` 仓库中生成一个新的提交。这个输出提交，连同对 `labels` 仓库的提交，会触发 `train-model` 管道。

这种架构非常适合数据持续抵达且管道需要可靠、自动执行的生产环境。

### DVC 与 Pachyderm 的比较

在 DVC 和 Pachyderm 之间做出选择取决于您的团队工作流程、规模和基础设施。

| 方面 | DVC | Pachyderm |
| --- | --- | --- |
| **工作流程** | **以 Git 为中心。** 与现有 Git 工作流程集成。 | **以数据为中心。** 管道在数据提交时触发。 |
| **执行方式** | **命令式。** 用户运行诸如 `dvc repro` 之类的命令。 | **声明式。** 用户定义管道规范；执行自动进行。 |
| **运行环境** | **本地优先。** 在开发人员机器或 CI/CD 运行器上运行。 | **集群优先。** 作为平台在 Kubernetes 上运行。 |
| **编排** | **外部。** 依赖脚本、Makefile 或 CI/CD 系统。 | **内置。** 编排是该平台的核心功能。 |
| **最适合** | 优先考虑开发者友好、Git 集成体验的个人和团队，用于实验和项目。 | 在 Kubernetes 上构建集中式自动化机器学习 (machine learning)平台以处理生产工作负载的组织。 |

两种工具都解决了版本控制和血缘追溯问题，但它们从不同的架构角度进行处理。DVC 协助开发者，而 Pachyderm 则支持平台。在许多高级环境中，它们甚至可以一起使用。例如，数据科学家可能在本地使用 DVC 进行实验，一旦模型准备就绪，管道逻辑就会转换为 Pachyderm 管道以实现生产自动化。

通过引入这些工具，您将数据从瞬态的、非管理资产转变为机器学习系统中一个版本化、可审计的组成部分。这为可复现性、更轻松的调试以及企业级人工智能所需的治理提供了基础。

## 参考资料

- [Data Version Control (DVC) Documentation](https://dvc.org/doc) — Iterative, Inc. (2024)
  DVC的官方综合指南，涵盖数据、模型和管道版本控制。
- [Pachyderm Documentation](https://docs.pachyderm.com/) — Pachyderm, Inc. (2024)
  Publisher: Pachyderm, Inc.
  Pachyderm的官方综合指南，详细介绍了其Kubernetes原生数据版本控制和管道自动化。
- [MLOps: A Comprehensive Definition, Challenges and Best Practices](https://arxiv.org/abs/2305.13284) — Kowshik Thopalli, Rakshith Subramanyam, Pavan Turaga, Jayaraman J. Thiagarajan (2023)
  Journal: International Conference Machine Learning (ICML) 2023; DOI: [10.48550/arXiv.2305.13284](https://doi.org/10.48550/arXiv.2305.13284)
  强调了MLOps中数据版本控制和血缘对于可复现性和系统可靠性的重要性。

---

[上一节](02-%E5%AE%9E%E6%97%B6%E4%B8%8E%E6%89%B9%E5%A4%84%E7%90%86%E7%89%B9%E5%BE%81%E8%AE%A1%E7%AE%97.md) · [下一节](04-%E9%AB%98%E5%90%9E%E5%90%90%E9%87%8F%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%EF%BC%9ASpark%20%E4%B8%8E%20Ray.md)
