# 复现 DVC 流水线

来源：[原文](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/reproducing-dvc-pipelines)

[返回章节目录](README.md) · [返回课程目录](../README.md)

机器学习 (machine learning)工作流可以使用 `dvc stage add` 或通过直接编辑 `dvc.yaml` 定义为一系列相互关联的阶段。每个阶段都定义了依赖项（输入数据、代码脚本、参数 (parameter)）和输出（处理过的数据、模型、指标）。这种定义方式构建了一个有向无环图（DAG），其优势不只在于可视化，还在于自动化。DVC 提供了一个功能强大的命令 `dvc repro`，可以自动检测更改并重新运行流水线的必要部分，使所有内容保持最新。

### `dvc repro` 的作用

假设您修改了 `params.yaml` 文件中的一个参数 (parameter)，更新了数据清洗脚本，或者收到了新版本的原始数据集。手动重新运行所有后续步骤既繁琐又容易出错。您是否记得在清洗后运行特征工程步骤？您是否使用新特征重新训练了模型？

`dvc repro` 会自动处理这些。当您执行它时，DVC 会执行以下操作：

1. **检查依赖项：** 它检查 `dvc.yaml` 中定义的阶段。对于每个阶段，它会计算其依赖项（文件、目录、参数值、命令）的当前哈希值（例如 MD5）。
2. **比较哈希值：** 它将这些当前哈希值与上次成功执行阶段时记录在 `dvc.lock` 文件中的哈希值进行比较。
3. **识别过时阶段：** 如果某个阶段的任何依赖项哈希值已更改，DVC 会将该阶段及其所有依赖其输出的下游阶段标记 (token)为“过时”。
4. **执行过时阶段：** DVC 按照正确的拓扑顺序（遵循依赖项）重新运行与过时阶段关联的命令。
5. **更新锁定文件：** 阶段成功执行后，DVC 会使用其依赖项和输出的新哈希值更新 `dvc.lock` 文件。

这个过程确保您的流水线输出始终与输入、代码和参数的当前状态保持一致。

### `dvc repro` 的工作原理：内部机制

其奥妙在于 `dvc.yaml` 和 `dvc.lock` 之间的关联。

- `dvc.yaml`：定义了流水线的*结构*。它列出了阶段、它们的命令、依赖项 (`deps`)、参数 (parameter) (`params`) 和输出 (`outs`)。
- `dvc.lock`：记录了流水线上次成功运行时的*状态*。它存储了每个阶段所有依赖项和输出的精确哈希值，*即它们在阶段上次完成时的状态*。

当您运行 `dvc repro` 时，DVC 实际上会询问：“项目的当前状态（`dvc.yaml` 中定义的文件、参数）是否与 `dvc.lock` 中记录的状态匹配？” 如果不匹配，它会重新执行必要的命令来同步它们并更新 `dvc.lock`。

我们来可视化这个流程：

> 流程图说明了针对单个阶段执行 `dvc repro` 时的决策过程。

### 运行 `dvc repro`

执行此命令很简单。在您的终端中，导航到项目根目录（`.dvc` 所在的位置），然后运行：

```bash
dvc repro
```

DVC 将分析 `dvc.yaml` 中定义的流水线并执行任何过时的阶段。您将看到输出，指示正在检查哪些阶段以及正在运行哪些阶段。

### 复现特定阶段

有时，您可能只想复现流水线的特定部分。例如，也许您只更改了训练脚本 (`train.py`)，并且知道它只影响 `train` 阶段和随后的 `evaluate` 阶段。您可以让 DVC 针对特定阶段甚至特定输出文件进行操作。

要复现单个阶段及其所有下游内容：

```bash
# 复现 'train' 阶段以及所有依赖其输出的阶段
dvc repro train
```

要仅复现流水线直到特定文件生成为止：

```bash
# 通过运行其父阶段 ('train') 来确保 'model.pkl' 是最新的，
# 如果有任何上游阶段过时，则运行这些必要的上游阶段。
dvc repro model.pkl
```

这种有针对性的复现可以节省大量时间，尤其是在只修改了小部分的复杂流水线中。

### 使用 `--dry` 预览更改

在进行可能耗时的计算之前，您可以执行“空运行”。`--dry` 标志告诉 `dvc repro` 执行依赖项检查并报告 *将* 运行哪些阶段，但不会实际运行它们的命令。

```bash
dvc repro --dry
```

这有助于验证您的更改对流水线执行计划是否产生了预期的影响。

### 使用 `--force` 强制复现

有时，您可能希望强制重新运行某个阶段，即使 DVC 认为它是最新的。如果某个阶段具有您想再次捕获的非确定性行为，或者 DVC 未跟踪的外部因素（如软件库更新）可能会影响输出，则这可能是必要的。请谨慎使用 `--force` 标志。

要强制复现特定阶段：

```bash
# 强制运行 'featurize' 阶段，无论依赖项是否更改
dvc repro --force featurize
```

要强制复现整个流水线：

```bash
# 谨慎使用！
dvc repro --force
```

### 复现后提交更改

在 `dvc repro` 成功完成后，`dvc.lock` 文件将被更新以反映流水线的新状态。由于 `dvc.lock` 跟踪状态并确保可复现性，因此将此文件与您对代码、参数 (parameter) (`params.yaml`) 或流水线定义 (`dvc.yaml`) 所做的任何更改一起提交到 Git 中是必要的：

```bash
# 运行 'dvc repro' 后
git add dvc.lock params.yaml src/train.py dvc.yaml
git commit -m "Update training parameters and retrain model"

# 如果生成并跟踪了新的数据输出
dvc push
```

通过使用 `dvc repro`，您可以确保任何查看此 Git 提交的人都可以通过运行 `dvc pull` 接着运行 `dvc repro` 轻松复现完全相同的输出（尽管如果他们查看 `dvc.lock` 已更新的提交，`dvc repro` 通常会报告所有内容都是最新的）。此命令是您的 DVC 管理项目中实现自动化和可靠复现性的基本方式。在接下来的部分中，我们将看到如何将 MLflow 日志集成到这些可复现的 DVC 流水线中。

## 参考资料

- [Reproducing DVC pipelines](https://dvc.org/doc/command-reference/repro) — Iterative, Inc. (2024)
  详细说明`dvc repro`命令、其功能以及`dvc.yaml`和`dvc.lock`文件作用的官方文档。
- [Introducing MLOps: How to Go from Experimentation to Production](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/) — Mark Treveil, Nicolas Omont, Rishabh Bahri, Aurélien Géron, Saska Mojsilovic, Joshua Prud'homme, Alex Ratner (2022)
  Publisher: O'Reilly Media
  涵盖了MLOps的原则和实践，包括版本控制、可复现性以及管道自动化对于高效机器学习部署的必要性。
- [How DVC Works](https://dvc.org/doc/start/data-management/data-versioning) — Iterative, Inc. (2024)
  Publisher: Iterative, Inc.
  解释了DVC用于数据和模型版本控制、依赖跟踪以及管道执行的核心机制，为`dvc repro`提供了背景信息。

---

[上一节](04-%E6%9E%84%E5%BB%BA%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md) · [下一节](06-%E8%BF%BD%E8%B8%AA%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF%E6%8C%87%E6%A0%87.md)
