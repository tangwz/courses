---
course: "data-versioning-experiment-tracking"
chapter: "integrating-dvc-mlflow"
lesson: "creating-dvc-pipelines"
sourceId: 4100
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/creating-dvc-pipelines"
title: "构建 DVC 流水线"
description: "使用 'dvc run' 或 'dvc stage add' 定义并运行多阶段数据处理和训练流水线。"
order: 4
plots: []
sourceHash: "0089b08161cfdfd3267ec8487d32576a764967804cabef6610898c51b0b01f25"
sourceCorrections: []
---

虽然追踪单个数据文件和实验运行很有用，但大多数机器学习 (machine learning)项目都包含多个步骤：获取数据、清洗数据、转换特征、训练模型和评估模型。手动运行每个步骤并确保使用正确的输入和输出会变得繁琐且易出错。DVC 流水线提供了一种自动化管理这些多阶段工作流程的方式，使其可复现且更易于管理。

DVC 流水线定义了操作序列、它们的依赖项（如代码脚本和数据文件等输入）以及它们的输出。DVC 追踪这些关系，使其能够智能地判断当某些内容发生变化时，哪些阶段需要重新运行。

### 使用 `dvc stage add` 定义流水线阶段

定义流水线步骤的主要命令是 `dvc stage add`。它允许您将单个命令或脚本执行封装为工作流程中的一个“阶段”。DVC 会将阶段定义，包括其依赖项和输出，记录在一个名为 `dvc.yaml` 的文件中。

基本语法如下：

```bash
dvc stage add -n <stage_name> \
              -d <dependency1> [-d <dependency2> ...] \
              -o <output1> [-o <output2> ...] \
              -p <params_file>:<param_section> \
              <command_to_run>
```

我们来分解一下这些选项：

- `-n <stage_name>`: 为该阶段分配一个独特、易读的名称（例如，`prepare_data`、`train_model`）。这个名称由 DVC 在内部使用。
- `-d <dependency>`: 指定此阶段的输入依赖项。这可以是一个源代码文件（例如，`src/prepare.py`）、一个由 DVC 追踪的数据文件（例如，`data/raw/data.csv`），甚至是前一个阶段的另一个由 DVC 追踪的输出。DVC 监控依赖项；如果其中一个发生变化，该阶段需要重新运行。您可以指定多个 `-d` 标记 (token)。
- `-o <output>`: 指定此阶段生成的一个输出文件或目录。DVC 将开始追踪此输出。与依赖项类似，您可以指定多个 `-o` 标记。输出通常用作后续阶段的依赖项。
- `-p <params_file>:<param_section>`: 指定阶段使用的参数 (parameter)，这些参数通常存储在单独的配置文件中（通常是 `params.yaml`）。DVC 追踪这些参数的变化。如果相关参数发生变化，DVC 会知道该阶段可能会产生不同的结果，需要重新运行。有关参数追踪的更多信息见下文。
- `<command_to_run>`: 实际执行此阶段逻辑的 shell 命令。这通常涉及运行一个脚本，例如 `python src/prepare.py`。

**示例：数据准备阶段**

假设您有一个脚本 `src/prepare.py`，它接受原始数据 `data/raw/iris.csv`（已由 `dvc add data/raw/iris.csv` 追踪），并生成一个处理过的数据集 `data/prepared/train.csv`。您可能会这样定义这个阶段：

```bash
# 确保 data/prepared 目录存在
mkdir -p data/prepared

# 添加阶段
dvc stage add -n prepare \
              -d src/prepare.py \
              -d data/raw/iris.csv \
              -o data/prepared/train.csv \
              "python src/prepare.py --input data/raw/iris.csv --output data/prepared/train.csv"
```

运行此命令后，DVC 将创建或更新 `dvc.yaml` 文件，并添加如下条目：

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py --input data/raw/iris.csv --output data/prepared/train.csv
    deps:
    - src/prepare.py
    - data/raw/iris.csv
    outs:
    - data/prepared/train.csv
```

这个 YAML 条目清楚地定义了 `prepare` 阶段：要执行的命令、其代码和数据依赖项以及它生成的输出。DVC 还会为输出 (`data/prepared/train.csv.dvc`) 创建一个 `.dvc` 文件，以追踪其哈希值，这与 `dvc add` 的工作方式类似。请记得将 `dvc.yaml` 和新的 `.dvc` 文件添加到 Git：

```bash
git add dvc.yaml data/prepared/train.csv.dvc .gitignore
git commit -m "将数据准备阶段添加到 DVC 流水线"
```

### 构建多阶段流水线

真正的优势在于连接多个阶段。一个阶段的输出成为下一个阶段的依赖项。我们来添加一个依赖于已准备数据的特征工程阶段：

```bash
# 假设 src/featurize.py 存在
mkdir -p data/features

# 添加特征化阶段
dvc stage add -n featurize \
              -d src/featurize.py \
              -d data/prepared/train.csv \
              -o data/features/train.pkl \
              "python src/featurize.py --input data/prepared/train.csv --output data/features/train.pkl"

# 将更改添加到 Git
git add dvc.yaml data/features/train.pkl.dvc .gitignore
git commit -m "添加特征工程阶段"
```

现在，`dvc.yaml` 包含这两个阶段。DVC 明白 `featurize` 依赖于 `prepare` 的输出。我们可以将这个依赖图可视化。

> 一个展示代码、数据和阶段之间依赖关系的简单流水线。一个阶段的输出（如 `data/prepared/train.csv`）成为下一个阶段的输入。

您可以继续添加用于训练、评估等的阶段，构建一个完整的机器学习 (machine learning)工作流程图。

### 追踪参数 (parameter)

机器学习 (machine learning)实验通常涉及调整超参数 (hyperparameter)或更改配置设置。DVC 流水线允许您明确地追踪这些参数。

首先，创建一个参数文件，通常是 `params.yaml`：

```yaml
# params.yaml
prepare:
  split: 0.2 # 训练/测试集划分比例
  seed: 42

featurize:
  max_features: 100
  ngram_range: [1, 1]

train:
  n_estimators: 100
  min_split: 2
  seed: 42
```

现在，使用 `-p` 标记 (token)修改您的阶段定义，以声明该阶段依赖哪些参数。您可以使用 `<filename>:<section>.<parameter_name>` 语法引用参数，或者只使用 `<filename>:<section>` 来依赖该部分中的所有参数。

我们来修改 `featurize` 阶段定义，使其依赖于 `max_features` 参数：

```bash
# 使用 dvc stage add --force 覆盖现有阶段
dvc stage add -n featurize --force \
              -d src/featurize.py \
              -d data/prepared/train.csv \
              -p params.yaml:featurize.max_features \
              -o data/features/train.pkl \
              "python src/featurize.py --input data/prepared/train.csv --output data/features/train.pkl --max-features $(params.featurize.max_features)"
```

或者，依赖于整个 `featurize` 部分：

```bash
dvc stage add -n featurize --force \
              -d src/featurize.py \
              -d data/prepared/train.csv \
              -p params.yaml:featurize \
              -o data/features/train.pkl \
              "python src/featurize.py --input data/prepared/train.csv --output data/features/train.pkl --max-features $(params.featurize.max_features) --ngrams $(params.featurize.ngram_range)" # 假设脚本处理 ngram 参数
```

现在，`dvc.yaml` 文件将包含一个 `params` 部分：

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py --input data/raw/iris.csv --output data/prepared/train.csv
    deps:
    - src/prepare.py
    - data/raw/iris.csv
    outs:
    - data/prepared/train.csv
  featurize:
    cmd: python src/featurize.py --input data/prepared/train.csv --output data/features/train.pkl --max-features $(params.featurize.max_features) # ... 其他参数
    deps:
    - src/featurize.py
    - data/prepared/train.csv
    params: # 新增部分
    - featurize: # 对应 params.yaml 中的部分
      - max_features
      - ngram_range
    outs:
    - data/features/train.pkl
```

现在，如果您修改 `params.yaml` 中的 `max_features` 或 `ngram_range` 并提交更改，DVC 就会知道当您执行流水线时，`featurize` 阶段（以及任何依赖其输出的后续阶段）需要重新运行。这种对参数与代码和数据一同进行的明确追踪，对于实现可复现性非常重要。

不要忘记将 `params.yaml` 添加到 Git：

```bash
git add params.yaml dvc.yaml
git commit -m "为特征化阶段添加参数追踪"
```

通过在 `dvc.yaml` 中定义您的工作流程阶段、依赖项、输出和参数，您为项目创建了一个蓝图。这个蓝图使 DVC 能够自动化执行并确保一致性，我们将在下一节关于复现流水线的内容中详细介绍。这种结构化方法也为在每个自动化阶段中集成 MLflow 追踪提供了支持。

## 参考资料

- [DVC Documentation: Pipelines](https://dvc.org/doc/user-guide/pipelines) — Iterative.ai (2024)
  官方文档提供了定义、管理和复现 DVC 管道的全面指南和命令参考，包括对阶段、依赖项、输出和参数的详细解释。
- [Introducing MLOps: How to go from Model-centric to Data-centric AI](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGiSGu-I-oOaaHDgPmEkyr2AZeWbsjozSC9JEgBrFCa-Cbgms_vGToYb1Iq9Fb5ueNd04gGJFnGxJ54t-5GoHk7gv9mEZ0G9-epPWpU0oM19hMdGDqHZ799DCX79me6UcUWh4kWKJCM0ITsr) — Mark Treveil, Nicolas Omont, Aurélien Coillet, Chetan Parmar, John C. D. Jones, Larysa Surak, Simon Chapman, Paul Singman, Michael Blundell, Jean-Luc Chatelain (2020)
  Publisher: O'Reilly Media
  本书概述了 MLOps 实践，为 DVC 管道等工具如何帮助在生产环境中构建可复现和可维护的机器学习系统提供了背景。
