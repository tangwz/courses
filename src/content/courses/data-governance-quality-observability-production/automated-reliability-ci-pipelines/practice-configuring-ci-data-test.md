---
course: "data-governance-quality-observability-production"
chapter: "automated-reliability-ci-pipelines"
lesson: "practice-configuring-ci-data-test"
sourceId: 8120
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-5-automated-reliability-ci-pipelines/practice-configuring-ci-data-test"
title: "实践：配置CI数据测试"
description: "动手设置运行数据质量测试的CI管道步骤。"
order: 5
plots: []
sourceHash: "0136ed9833681b2231feaf6e314a44b5cc394148e0dd4ddb8ce4c0b81c7e2139"
sourceCorrections: []
---

自动化数据质量检查将验证从手动、易错的任务转变为确定的软件工程流程。为达成此目的，构建了一个持续集成工作流，每次代码修改时都会执行数据断言。目标是建立一个严格的质量门限：如果数据完整性测试失败，管道必须拒绝代码更改，防止有缺陷的逻辑或损坏的数据定义部署到生产环境。

### 定义失败条件

为配置持续集成（CI）测试，我们首先需要一个可执行脚本，它会向编排服务器返回特定信号。像GitHub Actions、Jenkins或GitLab CI这样的CI系统依赖退出码来判断任务状态。以退出码`0`结束的进程表示成功，而任何非零整数则表示失败。

我们将使用`pytest`作为测试运行器，因为它有强大的断言处理能力，并在数据工程中被普遍采用。下面是一个用于验证暂存数据集的测试脚本。该脚本假设我们正在检查一个pandas DataFrame，这是中小批量验证的常见模式。

创建名为`tests/test_data_quality.py`的文件：

```python
import pytest
import pandas as pd
import sys

# 模拟数据加载函数
def load_staging_data():
    # 在实际场景中，这将连接到数据仓库或读取S3存储桶
    return pd.DataFrame({
        'transaction_id': [101, 102, 103, 104],
        'amount': [50.00, 120.50, 9.99, -5.00],  # 负值是一个异常
        'currency': ['USD', 'USD', 'EUR', None]  # 空值是一个异常
    })

@pytest.fixture
def dataset():
    return load_staging_data()

def test_transaction_completeness(dataset):
    """断言关键字段永远不为空。"""
    null_currencies = dataset['currency'].isnull().sum()
    assert null_currencies == 0, f"发现 {null_currencies} 个缺少货币代码的交易。"

def test_transaction_validity(dataset):
    """断言交易金额为正数。"""
    negative_amounts = dataset[dataset['amount'] < 0]
    count = len(negative_amounts)
    assert count == 0, f"发现 {count} 个金额为负数的交易。"
```

在此脚本中，`assert`语句充当断路器。如果条件判断为`False`，`pytest`会抛出`AssertionError`并以失败的退出码终止进程。

### 构建管道配置

测试逻辑定义好后，我们必须配置CI环境以自动执行此脚本。我们将使用一个声明式配置文件，这是现代DevOps平台中的标准做法。本例使用GitHub Actions的语法，但其逻辑适用于其他提供商。

配置必须定义三个主要阶段：

1. **触发器：** 启动管道的事件（例如，拉取请求）。
2. **环境设置：** 为容器配置Python和所需库。
3. **执行：** 运行测试脚本并获取结果。

创建名为`.github/workflows/data_quality_gate.yml`的文件：

```yaml
name: 生产数据质量门限

on:
  pull_request:
    branches: [ "main" ]

jobs:
  validate-schema-and-logic:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas pytest

    - name: Execute Data Quality Suite
      run: |
        pytest tests/test_data_quality.py -v
```

此配置确保没有代码合并到`main`分支，除非通过`test_transaction_completeness`和`test_transaction_validity`检查。如果`pytest`命令失败，CI系统会禁用界面中的合并按钮，从而执行代码中定义的治理策略。

### 管道执行流程

了解事件序列对于调试CI故障是必不可少的。该过程不仅仅是运行一个脚本；它是在创建模拟生产环境的隔离环境，以验证假设。

> 该图展示了从本地提交到CI质量门限的流程。核心决策点完全依赖于测试运行器提供的退出码。

### 注入环境变量

在上述例子中，数据是硬编码的。然而，生产测试通常需要连接到实时数据库或安全的云存储。你不能将凭据（密码、API密钥）提交到代码仓库。

为安全处理此事，CI平台提供了“秘密”管理功能。这些是运行时注入到运行器中的加密环境变量。

为更新数据库连接的工作流，你需要修改YAML配置的`env`部分：

```yaml
    - name: Execute Data Quality Suite
      env:
        DB_HOST: ${{ secrets.PROD_DB_HOST }}
        DB_USER: ${{ secrets.DATA_ENG_USER }}
        DB_PASS: ${{ secrets.DATA_ENG_PASS }}
      run: |
        pytest tests/test_data_integration.py -v
```

在你的Python代码中，你使用`os`模块访问这些值：

```python
import os
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS')
    )
```

### 分析测试结果

管道的可靠性在数学上是二元的。测试运行后系统$S$的状态可以表示为：

$S = \begin{cases} \text{可部署} & \text{如果 } \sum_{i=1}^{n} E_i = 0 \\ \text{已阻止} & \text{如果 } \sum_{i=1}^{n} E_i > 0 \end{cases}$

其中$E_i$表示第$i$个测试用例的错误计数。

当管道失败时，CI日志会提供堆栈跟踪。在我们的`pytest`示例中，`test_transaction_validity`中的失败会精确输出哪些行违反了约定（例如，“发现1个金额为负数的交易”）。这种即时反馈机制让工程师能在不良数据污染下游分析表之前修复数据问题或调整架构预期。通过将数据问题视为代码缺陷，我们将数据治理与标准软件交付速度对齐 (alignment)。

## 参考资料

- [pytest documentation](https://docs.pytest.org/) — pytest Developers (2024)
  pytest的官方文档，提供了关于编写和运行测试、fixture和断言处理的完整指南，与本节中使用的Python测试框架直接相关。
- [GitHub Actions Documentation](https://docs.github.com/en/actions) — GitHub Docs (2024)
  GitHub Actions的官方指南，涵盖了工作流语法、事件触发器、环境设置和秘密管理，这些都是讨论的CI管道配置中的关键组成部分。
- [Fundamentals of Data Engineering: Planning and Building Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108251/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  本书提供了构建强大数据系统的现代视角，包括数据质量、数据治理以及将CI/CD等软件工程实践应用于数据管道的章节。
