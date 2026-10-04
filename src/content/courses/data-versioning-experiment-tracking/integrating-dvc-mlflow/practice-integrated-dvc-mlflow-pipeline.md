---
course: "data-versioning-experiment-tracking"
chapter: "integrating-dvc-mlflow"
lesson: "practice-integrated-dvc-mlflow-pipeline"
sourceId: 4108
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/practice-integrated-dvc-mlflow-pipeline"
title: "动手实践：构建集成式流程"
description: "实践创建包含 MLflow 追踪功能的 DVC 流程，用于记录参数、指标和数据版本。"
order: 9
plots: []
sourceHash: "f1e26f83185b44a53db77f38b0b5fe3059b8f27b5cda720f23451f58f981287e"
sourceCorrections: []
---

一个实用的集成式流程展示了 DVC 和 MLflow 如何共同工作，以管理数据、追踪实验，并自动化流程执行，从而提升可复现性。

我们将创建一个简单的机器学习 (machine learning)流程，包含两个阶段：数据准备和模型训练。DVC 将管理数据工件并调度流程，而 MLflow 则会追踪训练阶段产生的参数 (parameter)、指标和模型。

### 前提条件

请确保您已安装并配置以下内容：

1. Python (3.7及更高版本)
2. Git
3. DVC (使用您选择的远程存储，运行 `pip install dvc[s3]` 或等效命令)
4. MLflow (`pip install mlflow scikit-learn pandas`)
5. 对 Git 命令有初步了解 (`git init`, `git add`, `git commit`)。
6. 熟悉前几章介绍的原理 (DVC 初始化、添加数据、MLflow 日志记录基本操作)。

### 项目设置

首先，让我们设置项目目录结构。

1. 为项目创建一个新目录并进入该目录：

   ```bash
   mkdir integrated-pipeline-example
   cd integrated-pipeline-example
   ```
2. 初始化 Git 和 DVC：

   ```bash
   git init
   dvc init
   ```

   这将创建 `.git` 和 `.dvc` 目录。请记住提交初始 DVC 配置文件：

   ```bash
   git add .dvc .dvcignore
   git commit -m "Initialize DVC"
   ```
3. 创建必要的子目录：

   ```bash
   mkdir data src models data/raw data/processed
   ```
4. 为代码和参数 (parameter)创建占位文件：

   ```bash
   touch src/prepare.py src/train.py params.yaml requirements.txt
   ```
5. 添加一个简单的原始数据集。对于本例，创建一个虚拟 CSV 文件 `data/raw/data.csv`：

   ```csv
   feature1,feature2,target
   1.0,2.1,0
   1.5,2.5,0
   1.8,2.9,0
   3.2,4.5,1
   3.5,5.1,1
   4.0,6.0,1
   0.8,1.9,0
   3.8,5.5,1
   ```
6. 将所有项目结构文件添加到 Git：

   ```bash
   git add data/raw/data.csv src/ params.yaml requirements.txt models/ .gitignore
   # (如果需要，请创建一个 .gitignore 文件，例如添加 __pycache__/, *.pyc 等)
   git commit -m "Initial project structure and raw data"
   ```

### 步骤 1：数据准备阶段 (DVC)

此阶段将获取原始数据，执行简单的准备步骤（如分割），并保存处理后的数据。DVC 将管理这些处理后的数据文件。

1. **编辑 `src/prepare.py`**：

   ```python
   # src/prepare.py
   import pandas as pd
   from sklearn.model_selection import train_test_split
   import os
   import yaml

   # 确保处理后的目录存在
   os.makedirs('data/processed', exist_ok=True)

   # 加载参数
   with open('params.yaml', 'r') as f:
       params = yaml.safe_load(f)

   split_ratio = params['prepare']['split']
   seed = params['base']['seed']

   # 加载原始数据
   raw_data_path = 'data/raw/data.csv'
   df = pd.read_csv(raw_data_path)

   # 分割数据
   train_df, test_df = train_test_split(df, test_size=split_ratio, random_state=seed)

   # 保存处理后的数据
   train_output_path = 'data/processed/train.csv'
   test_output_path = 'data/processed/test.csv'
   train_df.to_csv(train_output_path, index=False)
   test_df.to_csv(test_output_path, index=False)

   print(f"处理后的数据已保存：")
   print(f"- 训练集: {train_output_path}")
   print(f"- 测试集: {test_output_path}")
   ```
2. **编辑 `params.yaml`**：定义用于准备阶段和即将到来的训练阶段的参数 (parameter)。

   ```yaml
   # params.yaml
   base:
     seed: 42

   prepare:
     split: 0.3 # 测试集比例

   train:
     model_type: LogisticRegression
     solver: 'liblinear' # LogisticRegression 的示例参数
     C: 1.0          # 正则化强度
   ```
3. **定义 DVC 阶段**：使用 `dvc stage add` 在 `dvc.yaml` 中定义此准备步骤。此命令告诉 DVC 如何运行脚本、其依赖项是什么以及它产生哪些输出。

   ```bash
   dvc stage add -n prepare \
                 -p base.seed,prepare.split \
                 -d src/prepare.py -d data/raw/data.csv \
                 -o data/processed/train.csv -o data/processed/test.csv \
                 python src/prepare.py
   ```

   - `-n prepare`：将阶段命名为“prepare”。
   - `-p base.seed,prepare.split`：声明此阶段使用的来自 `params.yaml` 的参数。DVC 将追踪这些特定参数的更改。
   - `-d src/prepare.py -d data/raw/data.csv`：指定脚本和数据依赖项。如果这些发生更改，则需要重新运行此阶段。
   - `-o data/processed/train.csv -o data/processed/test.csv`：声明此阶段产生的输出。DVC 将开始追踪这些文件。
   - `python src/prepare.py`：此阶段要执行的命令。
4. **提交更改**：`dvc stage add` 命令会创建/更新 `dvc.yaml` 和 `dvc.lock`，并将输出文件 (`data/processed/*.csv`) 添加到 DVC 追踪（为它们创建 `.dvc` 文件）。将这些更改提交到 Git。

   ```bash
   git add dvc.yaml dvc.lock data/processed/.gitignore src/prepare.py params.yaml
   git commit -m "Add DVC stage: prepare data"
   ```

   *注意*：DVC 会自动将输出路径 (`data/processed/train.csv`, `data/processed/test.csv`) 添加到 `.gitignore`（在 `data/processed/.gitignore` 中），这样 Git 就会忽略大型数据文件本身。

如果您已配置远程存储，可以选择将 DVC 追踪的数据推送到远程存储 (`dvc push`)。

### 步骤 2：模型训练阶段 (DVC + MLflow)

此阶段使用处理后的数据训练模型，使用 MLflow 记录实验，并保存模型和指标。DVC 管理依赖项（处理后的数据、脚本、参数 (parameter)）和输出（模型文件、指标文件）。

1. **编辑 `src/train.py`**：此脚本现在包含了 MLflow 日志记录功能。

   ```python
   # src/train.py
   import pandas as pd
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   import mlflow
   import mlflow.sklearn
   import os
   import yaml
   import pickle
   import json

   # 加载参数
   with open('params.yaml', 'r') as f:
       params = yaml.safe_load(f)

   # --- MLflow 设置 ---
   # 可选：如果使用远程服务器，设置追踪 URI
   # mlflow.set_tracking_uri("http://...")
   mlflow.set_experiment("Simple Classification")

   # 启动 MLflow 运行
   with mlflow.start_run():
       seed = params['base']['seed']
       model_params = params['train']

       # --- 记录参数 ---
       mlflow.log_param("seed", seed)
       mlflow.log_params(model_params) # 记录所有训练参数

       # --- 加载数据 ---
       train_data_path = 'data/processed/train.csv'
       test_data_path = 'data/processed/test.csv'
       train_df = pd.read_csv(train_data_path)
       test_df = pd.read_csv(test_data_path)

       X_train = train_df[['feature1', 'feature2']]
       y_train = train_df['target']
       X_test = test_df[['feature1', 'feature2']]
       y_test = test_df['target']

       # --- 记录 DVC 数据信息（示例） ---
       # 将输入数据的路径或哈希值记录为标签
       # 这需要解析 dvc.lock 或使用 dvc api，此处已简化
       mlflow.set_tag("train_data_path", train_data_path)
       mlflow.set_tag("test_data_path", test_data_path)
       # 一种方法是获取 dvc.lock 中的哈希值

       # --- 训练模型 ---
       # 使用 params.yaml 中定义的逻辑回归模型
       if model_params['model_type'] == 'LogisticRegression':
           model = LogisticRegression(
               solver=model_params['solver'],
               C=model_params['C'],
               random_state=seed
           )
       else:
           raise ValueError(f"Unsupported model type: {model_params['model_type']}")

       model.fit(X_train, y_train)

       # --- 评估模型 ---
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)

       # --- 记录指标 ---
       mlflow.log_metric("accuracy", accuracy)
       print(f"模型准确率: {accuracy:.4f}")

       # --- 保存和记录模型 ---
       os.makedirs('models', exist_ok=True)
       model_output_path = 'models/model.pkl'
       with open(model_output_path, 'wb') as f:
           pickle.dump(model, f)

       # 使用 MLflow 的 scikit-learn 集成记录模型
       mlflow.sklearn.log_model(model, "sklearn-model")
       print(f"模型已保存到: {model_output_path}")
       print(f"模型已记录到 MLflow 运行: {mlflow.active_run().info.run_id}")

       # --- 保存指标文件（用于 DVC 追踪） ---
       metrics_output_path = 'metrics.json'
       metrics_data = {'accuracy': accuracy}
       with open(metrics_output_path, 'w') as f:
           json.dump(metrics_data, f, indent=4)
       print(f"指标已保存到: {metrics_output_path}")

   print("MLflow 运行完成。")
   ```
2. **定义 DVC 阶段**：将训练阶段添加到 `dvc.yaml`。它依赖于“prepare”阶段的输出、训练脚本和相关参数。它生成模型文件和指标文件。

   ```bash
   dvc stage add -n train \
                 -p base.seed,train \
                 -d src/train.py -d data/processed/train.csv -d data/processed/test.csv \
                 -o models/model.pkl \
                 -m metrics.json \
                 python src/train.py
   ```

   - `-n train`：将阶段命名为“train”。
   - `-p base.seed,train`：追踪 general seed 和 `params.yaml` 中 `train` 部分下的所有参数。
   - `-d ...`：指定依赖项：脚本 (`src/train.py`) 和上一阶段的输出 (`data/processed/*.csv`)。
   - `-o models/model.pkl`：将模型文件声明为 DVC 追踪的输出。
   - `-m metrics.json`：将 `metrics.json` 声明为指标文件。DVC 可以从此类文件解析和显示指标。
   - `python src/train.py`：要执行的命令。
3. **提交更改**：提交更新后的 `dvc.yaml`、`dvc.lock`、新的训练脚本、DVC 追踪的模型占位符和指标文件定义。

   ```bash
   git add dvc.yaml dvc.lock src/train.py models/.gitignore metrics.json
   # DVC 创建 models/.gitignore 来使 Git 忽略实际的模型文件
   git commit -m "Add DVC stage: train model with MLflow logging"
   ```

### 运行和复现流程

现在我们已在 `dvc.yaml` 中定义了一个两阶段流程。

1. **执行流程**：使用 `dvc repro` 从头到尾运行整个流程。DVC 会检查依赖项并按需执行阶段。

   ```bash
   dvc repro
   ```

   您将看到来自 `prepare.py` 和 `train.py` 的输出，包括 MLflow 日志消息和最终准确率。DVC 将首先运行“prepare”，然后运行“train”。
2. **检查状态**：

   - `git status`：应显示 `dvc.lock` 和 `metrics.json` 可能已更改（因为流程已运行）。提交这些更新。

     ```bash
     git add dvc.lock metrics.json
     git commit -m "Run integrated pipeline"
     ```
   - `dvc status`：应显示流程已是最新状态。
   - `dvc metrics show`：显示从 `metrics.json` 追踪的指标。
3. **检查 MLflow 用户界面**：启动 MLflow 用户界面以查看追踪到的实验运行：

   ```bash
   mlflow ui
   ```

   在浏览器中导航到 `http://localhost:5000`（或您配置的地址）。找到“Simple Classification”实验。您应该看到一个记录了以下内容的运行：

   - **参数 (parameter)**：`seed`、`model_type`、`solver`、`C`。
   - **标签**：`train_data_path`、`test_data_path`。
   - **指标**：`accuracy`。
   - **工件**：保存的 `sklearn-model`（包括 `model.pkl`、`conda.yaml`、`python_env.yaml`、`requirements.txt`）。
   > 该图表显示了 Git、DVC 追踪的工件/阶段以及 MLflow 实验追踪在项目结构中的关系。

### 流程迭代

现在，让我们看看当我们进行更改时，这种集成设置的能力。

1. **修改参数 (parameter)**：编辑 `params.yaml` 并更改一个训练参数，例如正则化 (regularization)强度 `C`：

   ```yaml
   # params.yaml
   base:
     seed: 42

   prepare:
     split: 0.3

   train:
     model_type: LogisticRegression
     solver: 'liblinear'
     C: 0.1 # 从 1.0 更改为 0.1
   ```
2. **复现**：再次运行 `dvc repro`。

   ```bash
   dvc repro
   ```

   注意 DVC 检测到 `params.yaml` 中影响“train”阶段的更改。它会跳过“prepare”阶段（因为其依赖项没有改变），只重新运行“train”阶段。训练脚本再次执行，以 `C=0.1` 的参数向 MLflow 记录一个*新的运行*。
3. **验证**：

   - 将更新后的 `dvc.lock` 和 `metrics.json` 提交到 Git。

     ```bash
     git add dvc.lock metrics.json params.yaml
     git commit -m "Experiment: Change C to 0.1"
     ```
   - 再次检查 MLflow 用户界面。您将在“Simple Classification”实验中看到第二次运行。您可以选择两次运行并使用“比较”功能，查看参数 (`C`) 的差异以及产生的 `accuracy`。

这个动手实践说明了 DVC 流程如何根据依赖项（代码、数据、参数）的变化自动化执行流程，同时 MLflow 捕获每次执行的具体信息（参数、指标、工件）。通过将 DVC 元数据（`dvc.yaml`、`dvc.lock`）与您的代码和参数一起提交到 Git，您就创建了一个完全版本化和可复现的机器学习 (machine learning)流程。任何有权访问您的 Git 仓库和 DVC 远程存储的人，都可以检出特定的提交，并复现您的精确流程和结果。

## 参考资料

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — Databricks (2024)
  MLflow的官方文档，涵盖实验跟踪、模型注册、项目和部署。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本系统性介绍构建和部署机器学习应用的图书，涵盖数据管理、实验和MLOps等方面。
- [Reproducibility in Machine Learning: Challenges and Solutions](https://proceedings.mlr.press/v151/pineau21a.html) — Joelle Pineau, Koustuv Sinha, Michael Smith, Marc-Alexandre Côté, Vincent Drouin, Andrew L. Maas, Christopher J. Maddison, Benjamin Recht, Aaron van den Oord (2021)
  Journal: Proceedings of Machine Learning Research; Publisher: PMLR; Volume: 151; Pages: 111-124; DOI: [10.48550/arXiv.2010.04689](https://doi.org/10.48550/arXiv.2010.04689)
  一篇学术论文，讨论了机器学习研究中实现可重复性的挑战，并提出了潜在的解决方案和最佳实践。
