---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "financial-operations-governance-ai"
lesson: "automating-cost-anomaly-detection"
sourceId: 7033
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-6-financial-operations-governance-ai/automating-cost-anomaly-detection"
title: "自动化成本异常检测"
description: "设置自动警报和仪表盘，以快速识别并响应云支出中的意外激增。"
order: 5
plots: ["plots/7033-0.json"]
sourceHash: "8082f7dfd9a3fc307fcae934b665b0b67ffae38c1cc532bea9eaba51779e02e4"
sourceCorrections: []
---

虽然手动检查云账单仪表盘能提供支出历史记录，但这主要是一种被动方式。当周报中发现成本超支时，由配置错误的训练任务或失控的推理 (inference)循环造成的财务损失已经形成。对于AI平台，资源消耗可能从零飙升至每小时数千美元，你需要一个充当自动财务监督员的系统，实时发现意外支出。

自动化成本异常检测将管理方式从周期性手动审计转变为持续的、程序化的流程。其旨在识别与既定基准显著偏离的支出模式，使你能在问题升级为重大预算事件之前进行调查和纠正。

### 定义成本异常

异常不仅仅是账单上的高数字；它是与预期模式的偏差。在机器学习 (machine learning)工作负载的背景下，这些模式可能很复杂。GPU集群5000美元的费用对于每周一次的大模型训练可能正常，但对于日常数据预处理任务来说则异常高。因此，有效的检测系统应结合具体情况来分析支出。

AI基础设施中的异常通常属于以下类型：

- **幅度异常：** 服务或标记 (token)资源绝对成本的突然大幅增长。这是最普遍的异常类型，通常由扩缩策略中的错误或用户配置了过大资源引起。
- **持续时间异常：** 作业或流程运行时间明显长于其历史平均水平。例如，进入无限循环的模型训练脚本或因文件格式错误而停滞的数据管道，都可能悄悄地累积成本。
- **效率异常：** 工作负载单位经济效益的变化。例如，端点每服务1000次推理 (inference)的成本突然翻倍，这表明新部署的模型版本可能存在性能退化。
- **新奇异常：** 出现来自以前从未用过的新服务、实例类型或区域的成本。这可能预示未经授权的试验或部署脚本中的配置错误。

下图展示了一个典型的幅度异常，其中每日支出远超既定的季节性模式。



![每日机器学习平台支出与异常情况](plots/7033-0.json)



> 10月26日的飙升明显偏离了正常的平日支出模式，从而触发了警报。

### 构建异常检测流程

虽然云提供商提供内置的异常检测服务（如AWS成本异常检测或Azure成本管理警报），但它们通常提供有限的自定义功能。对于复杂的AI平台，自定义方案能够更好地控制检测逻辑、数据源和警报集成。

自定义检测系统通常包含几个不同阶段：

> 此图显示了构建自定义成本异常检测流程的常见模式。

1. **数据摄取：** 本系统以细粒度账单数据为基础。你应该配置你的云提供商，将详细报告（例如AWS成本和使用报告(CUR)或Google Cloud的详细账单导出）导出到存储服务，如Amazon S3或Google Cloud Storage。确保资源标记 (token)得到执行，以便按项目、团队或实验对成本进行分组。
2. **处理与分析：** 计划函数（例如AWS Lambda）会定期触发，解析新的账单文件。该函数按所需维度（例如按项目、按服务）聚合成本，并将其存储为时间序列数据。这里是检测逻辑所在之处。移动平均等简单方法是一个不错的起点。算法在一个滚动窗口（例如过去14天）内计算基准和标准差，并标记任何超出特定阈值（例如高于平均值3个标准差）的新数据点。
3. **警报与通知：** 当处理逻辑检测到异常时，它会触发警报。有效的警报不只是问题通知；它是一份可操作的报告。它应包含异常范围（例如“项目'X-Research'”）、负责的服务（例如“Amazon SageMaker”）、预期成本范围、实际成本，以及指向相关云控制台仪表盘的深层链接，以便及时调查。

### 实施检测逻辑

系统的主要部分是其统计模型。虽然简单的静态阈值（`如果成本 > 1000美元则发出警报`）容易失效且易产生误报，但时间序列预测方法更具弹性。

一个常见起点是使用移动平均线来平滑短期波动并建立动态基准。例如，对于时间 $t$ 的给定成本指标 $c_t$，7天移动平均线 $MA_t$ 为：


$$
MA_t = \frac{1}{7} \sum_{i=0}^{6} c_{t-i}
$$


如果当前成本 $c_t$ 超过移动平均线，且超出幅度达到同一窗口内标准差 ($\sigma$) 的某个倍数，则可能触发警报：


$$
c_t > MA_t + (k \times \sigma_t)
$$


这里，$k$ 是一个可调的灵敏度参数 (parameter)。$k=3$ 是一个常见的起始值。

对于具有明显周度模式（例如工作日训练成本较高，周末较低）的工作负载，考虑季节性的更高级模型，如季节性ARIMA或Facebook Prophet，可以提供更准确的基准并减少误报。

这里是一个简化的Python代码片段，用于实现移动平均检查：

```python
def check_for_anomaly(cost_history: list[float], window_size: int = 7, k: float = 3.0) -> tuple[bool, str]:
    """
    根据移动窗口检查最新的成本数据点是否存在异常。

    参数：
        cost_history: 成本列表，从旧到新排列。
        window_size: 用于形成移动平均线的近期数据点数量。
        k: 阈值的标准差倍数。

    返回：
        一个元组，包含一个布尔值，指示是否发现异常
        和一条描述性消息。
    """
    if len(cost_history) < window_size + 1:
        return False, "数据不足，无法进行分析。"

    window = cost_history[-(window_size + 1):-1]
    latest_cost = cost_history[-1]

    mean = sum(window) / len(window)
    variance = sum([(x - mean) ** 2 for x in window]) / len(window)
    std_dev = variance ** 0.5
    
    upper_bound = mean + (k * std_dev)

    if latest_cost > upper_bound:
        message = (
            f"发现异常！成本为 ${latest_cost:.2f} "
            f"超出阈值 ${upper_bound:.2f}。"
        )
        return True, message
    
    return False, f"成本 ${latest_cost:.2f} 在正常范围内。"

# 特定项目的每日成本示例
project_alpha_costs = [120, 125, 118, 130, 122, 115, 128, 450]
is_anomaly, msg = check_for_anomaly(project_alpha_costs)
print(msg)
# 输出：发现异常！成本 $450.00 超出阈值 $143.19。
```

通过自动化异常检测，你可以将成本管理从被动的、事后审查转变为主动的、实时的管理机制。这营造了一个财务可持续的环境，工程团队可以快速改进，确信有安全网能在大额错误失控前及时发现并纠正。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: CRC Press
  这本全面的开源教科书涵盖了各种时间序列预测方法，包括ARIMA和指数平滑等模型，这些模型对于建立动态基线和检测时间序列数据中的异常至关重要。
- [Time-series Anomaly Detection Service at Microsoft](https://www.vldb.org/pvldb/vol12/p1832-jiang.pdf) — Bin Jiang, Jingge Jia, Zhengping Ma, Jian Li, and Xiaola Lin (2019)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 12; Pages: 1832–1843; DOI: [10.14778/3352063.3352136](https://doi.org/10.14778/3352063.3352136)
  本文描述了一个工业级的时间序列异常检测系统，详细介绍了其架构和算法在实际应用中的情况，适用于构建定制的成本异常检测管道。
- [AWS Cost and Usage Reports (AWS CUR)](https://docs.aws.amazon.com/cur/latest/userguide/what-is-cur.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  官方文档，解释了如何配置和使用详细的AWS成本和使用报告，这些报告是定制云成本异常检测系统的基础数据源。
