---
course: "evaluating-synthetic-data-quality"
chapter: "foundations-synthetic-data-evaluation"
lesson: "setting-up-evaluation-environment"
sourceId: 5639
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-1-foundations-synthetic-data-evaluation/setting-up-evaluation-environment"
title: "建立评估环境"
description: "为数据评估任务准备好带有必要库的Python环境。"
order: 5
plots: []
sourceHash: "f4ce0fee982676963180d603a66aab70955454621221ba9fd7e3e34ea61424f5"
sourceCorrections: []
---

为了有效地实施本课程中介绍的高级评估技术，一个配置良好的Python环境必不可少。此设置可确保您拥有数据处理、统计检验、机器学习 (machine learning)模型训练、隐私评估和可视化所需的工具。这些功能对于评估保真度-效用-隐私维度以及应对合成数据评估中固有的挑战是主要能力。

我们强烈建议使用虚拟环境来管理项目依赖。这种做法可隔离项目需求，避免不同项目之间的冲突，并确保可复现性。您可以使用Python内置的`venv`模块或流行包管理器（如`conda`）创建虚拟环境。

使用`venv`：

```bash
# 创建一个名为 'synth_eval_env' 的虚拟环境
python -m venv synth_eval_env

# 激活环境
# 在macOS/Linux上：
source synth_eval_env/bin/activate
# 在Windows上：
.\synth_eval_env\Scripts\activate
```

使用`conda`：

```bash
# 使用Python创建一个名为 'synth_eval_env' 的conda环境
conda create --name synth_eval_env python=3.9  # 或您偏好的Python版本

# 激活环境
conda activate synth_eval_env
```

环境激活后，即可安装所需库。

### 核心数据科学库

这些库构成了Python中大多数数据分析和机器学习 (machine learning)任务的基础：

- **NumPy:** 数值计算的基本包。它支持大型多维数组和矩阵，并附带一系列用于操作这些数组的数学函数。对于数值比较和数据表示操作至关重要。
- **Pandas:** 提供高性能、易于使用的数据结构（主要是`DataFrame`）和数据分析工具。对于加载、清洗、转换和分析表格数据不可或缺，这在合成数据评估中很常见。
- **Scikit-learn:** 一个全面的机器学习库。我们将广泛使用它进行：
  - 训练分类和回归模型（用于TSTR/TRTS效用评估）。
  - 数据预处理（缩放、编码）。
  - 计算性能指标（准确率、F1分数、AUC）。
  - 实现基本的成员推断攻击（MIAs）。
- **Matplotlib & Seaborn:** 数据可视化库。Matplotlib为绘图提供了基础，而Seaborn在此基础上创建更具信息性和吸引力的统计图形。对分布进行视觉检查（直方图、密度图、散点图）通常是保真度评估的第一步。

### 专用合成数据评估库

核心库提供通用工具，而专用软件包可简化合成数据的评估过程。在这方面，一个重要的库是`SDMetrics`。

- **SDMetrics:** `SDMetrics`是专门为评估表格合成数据而开发的，提供了广泛的指标，涵盖统计保真度、机器学习 (machine learning)效用和某些隐私方面。它允许系统地比较真实数据集和合成数据集，并生成全面的质量报告。在本课程中，我们将使用`SDMetrics`进行多次实践练习。

其他库可能与数据模态相关（例如，用于图像的`pytorch-fid`，用于文本的NLP库如`nltk`或`transformers`），但`SDMetrics`为表格数据提供了坚实支持，这是常见的侧重点。

### 安装

您可以在已激活的虚拟环境中使用`pip`安装这些库：

```bash
pip install numpy pandas scikit-learn matplotlib seaborn sdmetrics
```

如果您使用`conda`，您可能更喜欢通过`conda`通道安装兼容版本，尽管`pip`通常在`conda`环境中也能工作：

```bash
conda install numpy pandas scikit-learn matplotlib seaborn
pip install sdmetrics # SDMetrics通常通过pip安装
```

*注意：始终参考官方文档以获取最新的安装说明和具体的版本要求。*

### 验证您的设置

为确认所需库已正确安装，您可以运行一个简单的Python脚本或使用交互式Python会话：

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import sdmetrics

print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
print(f"Scikit-learn 版本: {sklearn.__version__}")
print(f"SDMetrics 版本: {sdmetrics.__version__}")

# 基本测试：创建一个简单的pandas DataFrame
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df = pd.DataFrame(data)
print("\n示例 DataFrame:")
print(df)

# 基本绘图测试（可选，可能需要GUI后端）
# try:
#     sns.histplot(df['col1'])
#     plt.title("测试图")
#     plt.show()
#     print("\n绘图库似乎功能正常。")
# except Exception as e:
#     print(f"\n绘图测试跳过或失败: {e}")

print("\n环境设置似乎成功！")
```

执行此代码应打印已安装库的版本，并在无错误的情况下确认基本功能。

### 项目组织

由于评估任务可能变得复杂，涉及多个数据集、模型和指标，保持有组织的项目结构是有益的。考虑采用如下布局：

```
synth_eval_project/
├── synth_eval_env/           # 您的虚拟环境（如果使用venv）
├── data/
│   ├── real_data.csv
│   └── synthetic_data_model_A.csv
│   └── synthetic_data_model_B.csv
├── notebooks/                # 用于探索和分析的Jupyter Notebook
│   └── 1_统计保真度检查.ipynb
│   └── 2_机器学习效用评估.ipynb
├── scripts/                  # 用于自动化流程的Python脚本
│   └── run_evaluation.py
│   └── privacy_attacks.py
├── results/                  # 存储评估输出、报告、图表
│   ├── quality_report_model_A.json
│   └── model_comparison_plots.png
└── README.md                 # 项目说明，设置指南
```

这种结构将数据、代码（用于探索的Notebooks，用于自动化的脚本）和结果分开，使您的工作更易于管理、复现和共享。

环境配置完成并对项目组织有了基本理解后，您现在已准备好实施后续章节中详细介绍的统计保真度检查、机器学习 (machine learning)效用评估和隐私评估。此设置提供了严格分析合成数据质量的依据。

## 参考资料

- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python 内置模块 `venv` 的官方文档，用于创建轻量级虚拟环境。
- [Conda documentation](https://docs.conda.io/en/latest/) — Anaconda, Inc. (2024)
  Publisher: Anaconda, Inc.
  Conda 的官方文档，Conda 是一个跨平台的包和环境管理器。
- [pandas documentation](https://pandas.pydata.org/docs/) — The pandas development team (2025)
  pandas 库的官方文档，该库在 Python 中对数据进行操作和分析至关重要。
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html) — scikit-learn developers (2024)
  scikit-learn 的官方文档，这是一个提供分类、回归、聚类等功能的机器学习库。
- [SDMetrics documentation](https://docs.sdv.dev/sdmetrics/) — DataCebo (2024)
  Publisher: DataCebo
  SDMetrics 的官方文档，该库专用于评估表格合成数据的质量。
