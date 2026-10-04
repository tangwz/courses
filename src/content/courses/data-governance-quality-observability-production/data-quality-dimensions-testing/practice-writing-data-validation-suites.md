---
course: "data-governance-quality-observability-production"
chapter: "data-quality-dimensions-testing"
lesson: "practice-writing-data-validation-suites"
sourceId: 8090
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-2-data-quality-dimensions-testing/practice-writing-data-validation-suites"
title: "实践：编写数据验证套件"
description: "动手练习，创建全面的数据验证套件。"
order: 5
plots: ["plots/8090-0.json"]
sourceHash: "119fed85e4aa144c92f9f6a48b5ceae07f341eb4878a230d5b2f8c3502eb88a0"
sourceCorrections: []
---

编写独立的断言对临时调试有帮助，但生产环境需要一种结构化的方法。数据验证套件将各项检查汇总到一个统一的执行引擎中。此引擎读取配置文件，将逻辑应用于数据集，并生成一份数据健康状况的综合报告。

在本节中，我们将使用Python构建一个轻量级验证框架。此练习展示了Great Expectations或Soda等企业工具背后的运行机制。通过从头开始构建它，您将明白如何将验证逻辑与数据处理代码解耦。

### 验证套件的架构

验证套件由三个独立部分组成：

1. **契约（配置）：** 对数据*应有*样式的声明式定义。
2. **验证器（引擎）：** 接受数据和契约并执行必需逻辑的代码。
3. **报告（产物）：** 详细说明哪些测试通过、哪些失败以及这些失败的严重程度的结构化输出。

分离这些部分让您无需修改底层管道代码即可更新数据质量规则。

> 数据和元数据在验证架构中的流动。引擎充当静态规则和动态数据之间的接口。

### 定义验证契约

在ETL脚本中硬编码断言（例如，`assert df['id'].notnull().all()`）使其难以管理。相反，我们在字典或JSON文件中定义预期。这充当了一个策略即代码层。

设想一个用户交易数据集。我们预期`transaction_id`是唯一的，`amount`是正数，`currency`是有效的ISO代码。

```python
validation_config = [
    {
        "column": "transaction_id",
        "test": "is_unique",
        "severity": "block"
    },
    {
        "column": "amount",
        "test": "is_positive",
        "severity": "block"
    },
    {
        "column": "user_age",
        "test": "in_range",
        "params": {"min": 18, "max": 120},
        "severity": "warn"
    }
]
```

请注意`severity`字段。并非所有数据故障都需高度关注。缺少的`transaction_id`会中断管道（阻止），而不寻常的`user_age`可能只需要调查（警告）。

### 实现验证器引擎

引擎读取配置并将字符串标识符（如“is\_unique”）映射到实际的Python函数。我们将使用Pandas进行此实现，尽管该模式同样适用于PySpark或SQL。

```python
import pandas as pd

class DataValidator:
    def __init__(self, df):
        self.df = df
        self.results = []

    def check_is_unique(self, col, severity):
        """断言列不包含重复项。"""
        is_valid = self.df[col].is_unique
        self._log_result("is_unique", col, is_valid, severity)

    def check_is_positive(self, col, severity):
        """断言数值大于0。"""
        # 首先检查类型是否为数值型以避免错误
        if pd.api.types.is_numeric_dtype(self.df[col]):
            is_valid = (self.df[col] > 0).all()
        else:
            is_valid = False
        self._log_result("is_positive", col, is_valid, severity)

    def check_in_range(self, col, severity, min_val, max_val):
        """断言值落在特定范围内。"""
        is_valid = self.df[col].between(min_val, max_val).all()
        self._log_result("in_range", col, is_valid, severity)

    def _log_result(self, test_name, col, status, severity):
        self.results.append({
            "test": test_name,
            "column": col,
            "status": "PASS" if status else "FAIL",
            "severity": severity
        })

    def run_suite(self, config):
        for rule in config:
            test_type = rule["test"]
            col = rule["column"]
            severity = rule["severity"]
            params = rule.get("params", {})

            # 动态分派
            if test_type == "is_unique":
                self.check_is_unique(col, severity)
            elif test_type == "is_positive":
                self.check_is_positive(col, severity)
            elif test_type == "in_range":
                self.check_in_range(col, severity, params["min"], params["max"])
        
        return pd.DataFrame(self.results)
```

### 运行套件和解析输出

定义类后，我们使用包含示例数据的DataFrame对其进行实例化。在生产管道中，此DataFrame将是加载前紧随其后的提取或转换步骤的结果。

```python
# 示例数据
data = pd.DataFrame({
    "transaction_id": [101, 102, 103, 101], # 重复的101
    "amount": [50.0, -10.0, 100.0, 25.0],   # 负值
    "user_age": [25, 30, 150, 40]           # 150超出范围
})

# 执行
validator = DataValidator(data)
report = validator.run_suite(validation_config)

print(report)
```

输出提供了数据质量的明确状态：

| 测试 | 列 | 状态 | 严重性 |
| --- | --- | --- | --- |
| is\_unique | transaction\_id | FAIL | block |
| is\_positive | amount | FAIL | block |
| in\_range | user\_age | FAIL | warn |

### 统计集成

基于规则的检查（如空值检查）是确定性的。但是，正如统计分析部分所讨论的，我们通常需要根据分布进行验证。我们可以扩展`DataValidator`以包含统计断言。

例如，确保平均交易值$\mu$与历史常态保持一致。如果历史平均值为\50$，并且我们允许$\$10$的偏差$\sigma\$，我们的断言逻辑变为：

$40 \le \mu_{current} \le 60$

我们可以将其添加到我们的验证器方法中：

```python
    def check_mean_within_range(self, col, severity, min_mean, max_mean):
        current_mean = self.df[col].mean()
        is_valid = min_mean <= current_mean <= max_mean
        self._log_result("mean_within_range", col, is_valid, severity)
```

这让套件能够发现细微的数据漂移问题，即数据格式正确（有效浮点数），但业务逻辑存在偏差（例如，货币转换错误导致所有值下降90%）。

### 可视化测试结果

在持续集成（CI）或计划管道中运行这些套件时，可视化有助于快速识别批次数据的健康状况。下面是一个套件如何可视化100项跨不同类别的检查结果的示意。



![按严重性划分的验证套件结果](plots/8090-0.json)



> 测试执行的摘要，显示通过与失败的测试，并按严重性分类。

### 实施质量门禁

编写验证套件的最后一步是确定退出代码。套件应以编程方式决定管道是否继续运行。

1. **筛选失败项：** 选择报告中`status == 'FAIL'`的所有行。
2. **检查严重性：** 如果任何失败项的`severity == 'block'`，则引发异常。
3. **警报：** 如果存在失败但仅为`severity == 'warn'`，则发送通知（例如，到Slack或PagerDuty），但允许管道继续运行。

```python
def enforce_gate(report):
    failed_blocks = report[
        (report['status'] == 'FAIL') & 
        (report['severity'] == 'block')
    ]
    
    if not failed_blocks.empty:
        error_msg = f"Blocking quality checks failed: {failed_blocks['test'].tolist()}"
        raise ValueError(error_msg)
        
    print("Quality gate passed.")

# 这将根据我们的示例数据引发一个ValueError
# enforce_gate(report) 
```

通过将验证逻辑封装到套件中，您将数据质量从手动、被动的过程转变为自动化、主动的工程实践。这种结构构成了我们将在下一章审视的可观测性系统的基础。

## 参考资料

- [Data Contracts: Define, Test & Monitor Data Quality](https://docs.soda.io/soda-cl/contract-language.html) — Soda (2025)
  Publisher: Soda
  将数据契约描述为数据质量的“策略即代码”方法，直接对应于验证套件中的“契约（配置）”组件。
- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis and Matt Housley (2022)
  Publisher: O'Reilly Media
  在现代数据工程实践和系统设计的背景下，全面介绍了数据质量、治理和可观察性。
- [Data quality assessment: A survey](https://dl.acm.org/doi/10.1145/2967118) — Carlo Batini and Cinzia Scannapieco (2016)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 49; Pages: 1-52; DOI: [10.1145/2967118](https://doi.org/10.1145/2967118)
  全面回顾了各种数据质量维度、评估技术和相关框架，为数据验证提供了学术背景。
