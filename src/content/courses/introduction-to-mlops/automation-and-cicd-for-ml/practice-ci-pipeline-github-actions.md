---
course: "introduction-to-mlops"
chapter: "automation-and-cicd-for-ml"
lesson: "practice-ci-pipeline-github-actions"
sourceId: 7244
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-4-automation-and-cicd-for-ml/practice-ci-pipeline-github-actions"
title: "动手实践：使用 GitHub Actions 创建简单的 CI 流水线"
description: "按照分步指南，使用 GitHub Actions 为机器学习项目构建简单的持续集成 (CI) 流水线。"
order: 7
plots: []
sourceHash: "7a6e030e96e3bc822d392a83c71879c38e83f53dbf370c6c44d7a53dbe5136b3"
sourceCorrections: []
---

构建一个实现持续集成原则的简单 CI 流水线。该流水线使用 GitHub Actions（直接集成在 GitHub 中的自动化工具），在代码发生变更时自动测试机器学习 (machine learning)代码。

通过此动手练习，你将设置一个工作流来检查代码质量并运行基础测试，从而确保项目保持稳定可靠。

### 练习准备条件

若要进行此练习，你需要一个 GitHub 账号和一个新的空仓库。我们将在该仓库中创建一个虽然小巧但完整的机器学习 (machine learning)项目。

首先，在项目的根目录下创建以下三个文件：

1. **数据文件 (`data.csv`)**：一小部分示例数据。在本例中，我们可以使用著名的鸢尾花 (Iris) 数据集中的几行内容。

   ```csv
   sepal_length,sepal_width,petal_length,petal_width,species
   5.1,3.5,1.4,0.2,setosa
   4.9,3.0,1.4,0.2,setosa
   6.2,2.9,4.3,1.3,versicolor
   5.9,3.0,5.1,1.8,virginica
   ```
2. **Python 训练脚本 (`train.py`)**：该脚本将使用我们的数据训练一个简单的模型。请注意，它并没有复杂的逻辑，其主要目的是作为一段可运行的代码，供 CI 流水线进行检查。

   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

   def run_training():
       """
       一个用于加载数据和训练模型的简单函数。
       """
       # 加载数据集
       df = pd.read_csv('data.csv')

       # 定义特征和目标变量
       X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
       y = df['species']

       # 拆分数据
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
       )

       # 初始化并训练模型
       model = LogisticRegression(max_iter=200)
       model.fit(X_train, y_train)

       print("训练成功完成。")

   if __name__ == '__main__':
       run_training()
   ```
3. **依赖文件 (`requirements.txt`)**：此文件列出了项目所需的 Python 库。我们的 CI 流水线将使用此文件来创建一个一致的环境。其中包含了用于代码规范检查的 `flake8` 和用于测试的 `pytest`。

   ```text
   pandas
   scikit-learn
   flake8
   pytest
   ```

将这三个文件提交到 GitHub 仓库的 `main` 分支。项目结构就绪后，我们现在可以定义自动化流程了。

### 理解 GitHub Actions 工作流

GitHub Actions 工作流是在 YAML 文件中定义的自动化流程。你将这些工作流文件存储在仓库中一个名为 `.github/workflows/` 的特殊目录下。当仓库中发生特定事件（如推送代码）时，GitHub 就会触发相应的工作流。

我们的目标是创建一个执行三个主要任务的工作流：

1. 设置一个包含 Python 及所需库的干净环境。
2. 对代码进行规范检查，以查看是否存在格式问题。
3. 运行一个简单的测试，确保训练脚本能够正常运行。

### 创建你的第一个 CI 工作流

在你的仓库中，创建一个名为 `.github` 的新目录，并在其内部再创建一个名为 `workflows` 的目录。在 `.github/workflows/` 中，创建一个名为 `ci-pipeline.yml` 的新文件。

将以下 YAML 内容复制到 `ci-pipeline.yml` 中。随后我们将详细拆解每个部分的功能。

```yaml
# 工作流的描述性名称
name: Basic ML Code CI

# 当代码推送到 main 分支时触发工作流
on:
  push:
    branches: [ "main" ]

# 定义要运行的作业
jobs:
  test-and-lint:
    # 使用最新版本的 Ubuntu 作为运行环境
    runs-on: ubuntu-latest

    # 定义作业中的步骤序列
    steps:
      # 步骤 1：检出仓库代码
      - name: Check out repository code
        uses: actions/checkout@v4

      # 步骤 2：设置特定版本的 Python
      - name: Set up Python 3.9
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      # 步骤 3：从 requirements.txt 安装项目依赖
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # 步骤 4：使用 flake8 检查代码规范和格式问题
      - name: Lint with flake8
        run: |
          # 如果存在 Python 语法错误或未定义的名称，则停止构建
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          # exit-zero 将所有错误视为警告。GitHub 编辑器宽度为 127 字符
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      # 步骤 5：运行基础测试以确保脚本可执行
      - name: Test with pytest
        run: |
          pytest
```

#### 工作流文件拆解

- **`name`**：这是一个易于理解的名称，会显示在 GitHub 仓库的 “Actions” 选项卡中。
- **`on: push: branches: [ "main" ]`**：这是触发器。它告知 GitHub 每当有人向 `main` 分支推送提交时，就运行此工作流。
- **`jobs`**：工作流由一个或多个作业组成。我们的工作流包含一个名为 `test-and-lint` 的作业。
- **`runs-on: ubuntu-latest`**：这指定了作业将在 GitHub 托管的新虚拟机上运行，使用的是最新版本的 Ubuntu 操作系统。
- **`steps`**：这是最核心的部分。它定义了作业将要执行的一系列任务。
  - **`uses: actions/checkout@v4`**：这是一个预构建的动作，用于将仓库代码副本下载到运行器上，以便后续步骤可以访问你的文件。
  - **`uses: actions/setup-python@v5`**：另一个预构建的动作，用于安装特定版本的 Python，在本例中为 3.9 版本。
  - **`run: pip install -r requirements.txt`**：`run` 命令用于执行命令行指令。在这里，我们使用 `pip` 安装 `requirements.txt` 文件中列出的所有库。
  - **`run: flake8 . --count ...`**：此步骤运行 `flake8` 代码检查器。检查器在不实际运行代码的情况下分析代码是否存在潜在错误和风格问题。这是维护代码质量的标准做法。
  - **`run: pytest`**：最后一步执行 `pytest` 测试框架。我们还没有创建任何测试，所以现在来创建一个。

### 添加一个简单的测试

为了让 CI 流水线具有实际意义，它需要一些测试对象。让我们创建一个非常基础的测试，确认我们的 `train.py` 脚本可以被导入并运行而不会崩溃。

在项目的根目录下创建一个名为 `test_script.py` 的新文件：

```python
import pytest
from train import run_training

def test_training_runs():
    """
    测试训练函数是否能在不抛出异常的情况下运行。
    """
    try:
        run_training()
    except Exception as e:
        pytest.fail(f"run_training() 抛出了异常：{e}")
```

对于 CI 检查来说，这个测试简单且有效。它导入 `run_training` 函数并调用它。如果运行期间出现任何错误，就会触发 `pytest.fail()`，导致流水线中的测试步骤失败。

### 可视化流水线

我们定义的工作流遵循清晰的线性步骤序列。这个过程确保了在任何代码被认定为“合格”之前，它都经过了代码检出、环境构建，并且通过了质量与功能检查。

> 持续集成工作流中的自动化步骤序列。该流程始于代码推送，随后进行环境设置、代码检查和测试。

### 运行工作流并查看结果

你现在可以准备查看流水线的运行情况了。

1. 将 `ci-pipeline.yml` 和 `test_script.py` 文件添加、提交并推送到你的 GitHub 仓库。
2. 在 GitHub 网站上进入你的仓库，点击 “Actions” 选项卡。
3. 你会看到列出的 “Basic ML Code CI” 工作流。GitHub 自动检测到了新的工作流文件，并因为你向 `main` 分支推送了代码而触发了一次运行。

点击该工作流运行记录以查看详情。你可以展开 `test-and-lint` 作业，查看每个步骤的日志输出。如果所有步骤都成功完成，你会在它们旁边看到一个绿色对勾。如果某个步骤失败（例如 `flake8` 发现了语法错误或 `pytest` 测试未通过），该步骤将被标记 (token)为红色的 “X”，且整个工作流运行将被标记为失败。

你现在已经成功构建了一个基础的 CI 流水线。这种简单的自动化为你的项目增加了一层安全保障和质量控制。它充当了自动化的守护者，确保对主分支的每次更改都经过自动审核，让你能够专注于开发新功能。这是构建可靠且可维护的机器学习 (machine learning)系统的基本做法。

## 参考资料

- [GitHub Actions Documentation](https://docs.github.com/en/actions) — GitHub Docs (2024)
  Publisher: GitHub
  GitHub Actions官方指南，涵盖工作流语法、事件触发器和运行器环境。
- Introducing MLOps: How to go from Model to Production — Mark Treveil, Nicolas Omont, Clement Stenac, Kenji Lefevre, Du Phan, Joachim Zentici, Adrien Lavoillotte, Makoto Miyazaki, Lynn Heidmann (2020)
  Publisher: O'Reilly Media
  关于MLOps实践的资源，包括将CI/CD集成到机器学习工作流中。
- [Continuous Integration](https://martinfowler.com/articles/continuousIntegration.html) — Martin Fowler (2024)
  一篇定义持续集成原则和实践的文章，为CI管道的理论基础提供支持。
