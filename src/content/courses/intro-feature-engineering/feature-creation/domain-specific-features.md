---
course: "intro-feature-engineering"
chapter: "feature-creation"
lesson: "domain-specific-features"
sourceId: 1382
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-5-feature-creation/domain-specific-features"
title: "领域特征工程"
description: "阐述专业知识在创建有意义、针对特定问题的特征方面的作用。"
order: 6
plots: []
sourceHash: "e2d979d562a9243e7ec94c5526fcf96fad4bf8f2eee611c7bf5f33ada82a21c1"
sourceCorrections: []
---

虽然生成交互项（$f_1 \times f_2$）或多项式特征（$f_1^2$）等技术提供了系统化的方法来分析数据中的联系，但它们在操作时通常不理解变量背后的*含义*。最有效且效果出色的特征常常不仅来自自动化转换，更源于对问题本身的专业背景的透彻理解。特定领域特征工程解决了这一需求。

"可以将其视为运用生成数据的过程中的专家知识。如果您在预测房价，那么了解`total_square_footage`通常比单独的`living_room_sqft`和`bedroom_sqft`更有信息量，这就是专业知识。如果您在分析工业机器的传感器数据，那么了解`temperature_increase`突然升高并伴随`vibration_spike`的同步出现通常预示着故障，这就是专业知识。算法最终可能识别出这些关联，尤其是在有足够数据的情况下，但基于这种理解明确地设计特征可以显著加快学习并提高模型准确性。"

### 运用专业知识

特定领域的特征源于对数据来源专业背景的细节、规则和固有联系的理解。为什么它如此有效？

1. **捕捉隐性知识：** 专业知识有助于将隐性理解转化为显性特征。例如，在金融领域，计算债务收入比（DTI）$DTI = \frac{每月总债务支付}{每月总收入}$ 比单独查看债务和收入数据更能概括借款人的财务负担。
   "2. **代表过程：** 特征可以被设计成反映已知的物理、生物或经济过程。例如，在化学反应模型中，反应物之比可能是一个重要特征。"
2. **提高可解释性：** 与仅依赖抽象交互项或主成分的模型相比，使用专业相关特征（如`平均购买价值`或`身体质量指数`）构建的模型通常更容易让利益相关者理解和信任。
3. **简化模型任务：** 精心设计的特征可以直接代表模型可能难以学习的某个概念。例如，与其强迫模型学习经度、纬度与市中心距离之间的复杂关系，不如直接设计一个`distance_to_city_center`特征。

### 各专业领域的例子

让我们看看专业知识如何转化为具体特征：

- **电子商务/零售：**
  - 原始数据：`用户ID`、`购买时间戳`、`商品价格`、`类别`。
  - 专业特征：`上次购买以来的天数`、`平均购买间隔时间`、`客户生命周期价值`（CLV，可能通过单独的模型或公式计算）、`最常购买类别`、`购物篮大小`（每笔交易的商品数量）、`会话时长`。
- **金融/银行：**
  - 原始数据：`账户ID`、`交易金额`、`时间戳`、`商户代码`、`客户收入`、`客户债务`。
  - 专业特征：`过去7天的交易频率`、`平均交易价值`、`是否国际交易`、`一天中的时段类别`（例如，“深夜”）、`债务收入比`、`信用使用率`。
- **医疗健康：**
  - 原始数据：`患者ID`、`心率`、`收缩压`、`舒张压`、`体温`、`体重_kg`、`身高_m`。
  - 专业特征：`身体质量指数`（$BMI = \frac{体重_{公斤}}{身高_{米}^2}$）、`脉压`（$收缩压 - 舒张压$）、`平均动脉压`（$MAP \approx 舒张压 + \frac{1}{3} 脉压$）、源自生命体征组合的风险评分（如SOFA或MEWS评分的组成部分）。
- **制造/物联网：**
  - 原始数据：`传感器ID`、`时间戳`、`温度`、`压力`、`振动_x`、`振动_y`。
  - 专业特征：`温度变化率`、`总振动幅度`（$\sqrt{振动_x^2 + 振动_y^2}$）、`1分钟压力滚动标准差`、`运行状态`（源自传感器模式）。

此图说明了如何运用专业知识组合原始数据点来创建信息量更大的特征：

> 运用专业公式，根据原始测量值得出身体质量指数（BMI）、平均动脉压（MAP）和购买间隔时间等特征。

### 融入专业见解

您如何获取并运用这些知识？

- **与专家交流：** 与理解业务或科学专业背景的人密切合作。询问他们关注哪些指标、哪些关系是重要的以及流程如何运作。
- **学习专业知识：** 阅读与您问题相关的文献、内部文档或行业标准。
- **有针对性地进行数据分析（EDA）：** 进行EDA不仅是为了寻找一般关联，更是为了专门验证或确认专业知识所建议的联系。提出类似这样的问题：“数据是否支持X和Y之间已知的反向关系？”或者“这组用户是否与市场部门描述的‘高价值客户’画像相符？”
- **迭代优化：** 特征工程很少是一次性过程。根据初步理解创建特征，构建模型，分析其误差（特别是专业知识表明它*本应*表现得更好的地方），并相应地优化您的特征。

### 实现

实现特定领域的特征通常涉及数据处理，常使用Pandas等库。例如，计算BMI非常直接：

```python
# 假设 'df' 是您的 Pandas DataFrame
df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)
```

计算时间差需要日期时间处理：

```python
import pandas as pd

# 确保时间戳是 datetime 对象
# 示例 DataFrame 设置（请替换为您的实际数据加载）
data = {'user_id': [1, 1, 2, 1, 2],
        'purchase_timestamp': ['2023-01-10 10:00:00', '2023-01-15 12:30:00', '2023-01-12 08:00:00', '2023-01-05 09:00:00', '2023-01-20 15:00:00']}
df = pd.DataFrame(data)
df['purchase_timestamp'] = pd.to_datetime(df['purchase_timestamp'])

# 按用户和时间排序以计算上次购买以来的时间
df = df.sort_values(by=['user_id', 'purchase_timestamp'])
# 计算同一用户连续购买之间的时间差
df['time_since_last_purchase'] = df.groupby('user_id')['purchase_timestamp'].diff()

# 将 timedelta 转换为数值单位，例如天
# .dt.days 从 timedelta 中提取完整的天数
df['days_since_last_purchase'] = df['time_since_last_purchase'].dt.days

# 显示结果（可选）
# print(df)
#    user_id  purchase_timestamp time_since_last_purchase  days_since_last_purchase
# 3        1 2023-01-05 09:00:00                      NaT                       NaN
# 0        1 2023-01-10 10:00:00          5 days 01:00:00                       5.0
# 1        1 2023-01-15 12:30:00          5 days 02:30:00                       5.0
# 2        2 2023-01-12 08:00:00                      NaT                       NaN
# 4        2 2023-01-20 15:00:00          8 days 07:00:00                       8.0
```

尽管实现可能使用Pandas等标准工具，但定义特征的*逻辑*直接来源于您对专业背景的理解。这些新创建的特征随后可以进行其他章节中讨论的缩放、编码或选择技术处理。

最终，将自动化特征创建技术与周密、专业驱动的特征工程结合起来，通常能产生最有效和准确的机器学习 (machine learning)模型。这需要批判性思维，有时还需要创造力，将纯粹机械的数据转换提升到理解数据的背景层面。

## 参考资料

- [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQESDSHUiYHjaQ56Et5k71fMhff1b1yZRoPVDdeW7s2IpUmYCVpe_avRPiXPT2-xY-wy-KkkRcbRkr-hmu7K2yHMduzTcrIE4uIUC8kqlZJ1INR2C6WfRA==) — Max Kuhn, Kjell Johnson (2019)
  Publisher: CRC Press; DOI: [10.1201/9781315108212](https://doi.org/10.1201/9781315108212)
  这本权威书籍提供了特征工程的实用方法，侧重于领域专业知识如何指导有效特征的创建。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这是一本广泛使用的实践指南，涵盖了整个机器学习流程，包括结合领域背景创建特征的最佳实践。
- [An Introduction to Data Science](https://us.sagepub.com/en-us/nam/an-introduction-to-data-science/book253303) — Jeffrey S. Saltz, Jeffrey M. Stanton (2017)
  Publisher: SAGE Publications, Inc.; Pages: 288
  这本教科书介绍了数据科学的核心概念和工作流程，强调了理解问题领域在数据准备和特征构建中的基础作用。
