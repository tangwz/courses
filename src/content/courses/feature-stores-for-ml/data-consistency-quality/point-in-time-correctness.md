---
course: "feature-stores-for-ml"
chapter: "data-consistency-quality"
lesson: "point-in-time-correctness"
sourceId: 3380
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-3-data-consistency-quality/point-in-time-correctness"
title: "训练数据的时间点准确性"
description: "实现机制以确保模型训练时准确查找历史特征。"
order: 2
plots: []
sourceHash: "883d61fd9ed375395cdd44d08b4faad68c158c0f72f7a3ba2babf03477be75b0"
sourceCorrections: []
---

生成准确的训练数据是构建可靠机器学习 (machine learning)模型的基本要求。一个不明显但很重要的难题源于数据的时间特性：与实体相关的特征会随时间变化。在构建训练数据集时未能考虑这一时间维度，会导致“数据泄露”或“时间旅行”，即预测事件发生时不可用的信息无意中影响模型训练过程。本节的重心是实现“时间点准确”的特征查找，这是一种必要的机制，用于阻止此类泄露并确保训练数据准确反映预测“本应”发生时的信息状态。

### 数据泄露的风险

设想你正在构建一个模型来预测客户流失。你的训练数据包含客户属性和行为特征，并配有指示客户是否在特定时间范围内流失的标签。一个简单方法可能是将每个客户的当前特征值与其历史流失标签进行关联。

一位客户在6月15日流失。在6月10日，他们的`login_frequency_last_7_days`特征值很低，这可能表明不满意。然而，在6月20日（流失之后），他们的账户可能被标记 (token)为不活跃，`login_frequency_last_7_days`被设置为零或某个其他反映他们流失后状态的值。如果将6月20日的特征值与6月15日的流失标签用于训练，模型将从流失事件发生“之前”不可用的信息（流失后状态）中学习。这会引入数据泄露，导致训练期间的性能估计过于乐观，并在生产环境中泛化能力差。

时间点准确性确保，当为在时间 $t_{event}$ 发生的事件生成训练样本时，我们只使用在 $t_{event}$ 时刻或之前已知特征值。

### 时间戳是核心

实现时间点准确性取决于细致的时间戳管理：

1. **特征事件时间戳：** 离线存储中每次记录的特征值更新“必须”关联一个准确的时间戳，表示该值何时生效或在源系统中被观察到（通常称为“事件时间”）。
2. **标签事件时间戳：** 训练数据集中的每个标签或预测目标（例如，流失事件、欺诈检测、购买）也必须有一个精确的时间戳，指示事件发生的时间或何时应进行预测。

在不同数据源和管道中维护准确、一致的时间戳是一个重要的工程难题，涉及到时区标准化和处理迟到数据等考量。

### 离线特征存储的作用

离线存储专门设计用于支持时间点查找。与仅存储每个特征“最新”值的简单数据库表不同，离线存储必须保留特征值的“历史”。这通常通过将特征数据与实体标识符、有效时间间隔或事件时间戳一起存储来实现。

例如，`user_id = 123`的特征`user_purchase_count`可能这样存储：

| 用户ID | 特征名称 | 值 | 事件时间戳 |
| --- | --- | --- | --- |
| 123 | user\_purchase\_count | 5 | 2023-01-10T10:00:00Z |
| 123 | user\_purchase\_count | 6 | 2023-02-15T14:30:00Z |
| 123 | user\_purchase\_count | 7 | 2023-03-20T09:00:00Z |
| ... | ... | ... | ... |

这种结构允许我们查询`user_id = 123`的`user_purchase_count`在过去任何特定时间点的值。

### “AS-OF”连接：查询历史数据

用于为给定事件时间戳正确检索特征的机制被称为\*\*“AS-OF”连接**或**时间连接\*\*。给定一个包含实体及其对应事件时间戳的数据集（“标签数据集”或“主干”），特征存储会执行一个连接操作，为标签数据集中的每一行查找在该行特定时间戳时有效的一个或多个特征值。

对于给定的实体 $e$ 和事件时间戳 $t_{event}$，查询会找到特征值 $v$，使得其时间戳 $t_{feature}$ 是小于或等于 $t_{event}$ 的最新时间戳：

$v = \text{f的值}(f) \text{ 其中 } t_{feature}(f) = \max \{ t \mid t \le t_{event} \text{ 且 f的实体}(f) = e \}$

特征存储SDK通常抽象化这种复杂性。用户无需编写复杂的时间SQL查询，通常只需提供一个包含实体和时间戳的DataFrame，特征存储库就会处理针对离线存储的正确时间连接逻辑。

```python
# 使用特征存储SDK的例子
# 'spine_df' 包含实体ID和事件时间戳
# 例如，列：['用户ID', '事件时间戳', '标签']

spine_df = pd.DataFrame({
    'user_id': [123, 456, 123],
    'event_timestamp': pd.to_datetime([
        '2023-02-20T12:00:00Z',
        '2023-03-01T08:00:00Z',
        '2023-03-25T10:00:00Z'
    ]),
    'label': [0, 1, 1] # 示例标签
})

# 请求在每个事件时间戳时的特征
training_data = fs.get_historical_features(
    entity_dataframe=spine_df,
    features=[
        'user_features:user_purchase_count',
        'user_features:login_frequency_last_7_days'
    ],
    timestamp_key='event_timestamp' # 指定包含事件时间的列
)

# 'training_data' 现在包含原始的 spine_df 列
# 加上每行的时间点准确特征值。
# 对于 user_id=123, event_timestamp='2023-02-20T12:00:00Z',
# user_purchase_count 将是 6 (来自上表)。
# 对于 user_id=123, event_timestamp='2023-03-25T10:00:00Z',
# user_purchase_count 将是 7。
```

### 可视化时间点连接

以下图表阐释了这一原理。我们有`user_id=123`在不同时间发生的特征更新。我们也有该用户的两个标签事件。“AS-OF”连接选择时间戳在标签事件时间戳“之前或等于”该时间戳的最新特征值。

> 此图说明了在特定时间（T\_L1，T\_L2）发生的标签事件如何与该时间点或之前可用的最新特征值进行关联。事件1使用T2的特征值，事件2使用T3的值。

### 实现考量与挑战

- **离线存储格式：** 底层存储系统（例如，Delta Lake、Apache Iceberg、Hudi或传统数据仓库）必须高效支持查询历史数据和执行时间连接。基于时间的分区策略通常对性能很重要。
- **计算成本：** 对大型数据集执行“AS-OF”连接可能计算密集。优化离线存储布局和查询执行很重要。
- **时间戳精度和时区：** 确保涉及特征生成和标签创建的所有系统在时间戳精度和时区处理上保持一致。
- **迟到数据：** 如何处理在对应事件时间戳之后才到达的特征数据？策略可能包括重新处理受影响的时间窗口，或根据应用程序的容忍度接受潜在的微小不准确性。双时间模型（追踪事件时间和处理时间）可以明确解决此问题，但会增加复杂性。

确保时间点准确性不仅仅是一个技术细节；它对于使用特征存储数据训练的机器学习 (machine learning)模型的有效性很重要。通过细致管理时间戳并运用特征存储离线组件的“AS-OF”连接功能，你可以防止数据泄露并大幅减少训练-服务偏差的主要来源，最终带来更可靠、更值得信赖的机器学习系统。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  全面概述了特征商店的作用，包括如何确保时间点正确性、管理数据一致性以及在生产ML系统中防止数据泄漏。
- [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml) — Martin Zinkevich (2017)
  Journal: Google Developers; Publisher: Google
  为构建可靠的ML系统提供了重要指导，其中明确提及数据泄漏以及使用正确、及时数据进行训练的至关重要性。
- [Temporal Data and the Relational Model](https://www.elsevier.com/books/temporal-data-and-the-relational-model/date/978-1-55860-855-9) — C.J. Date, Hugh Darwen, Nikos A. Lorentzos (2002)
  Publisher: Morgan Kaufmann
  一本关于时态数据库的基础学术著作，详细阐述了管理时变数据的原理和机制，包括时态连接和时间点查询的理论基础。
- [Feast Documentation: Retrieving Historical Features](https://docs.feast.dev/how-to-guides/feature-retrieval) — Feast Project Contributors (2024)
  Journal: Feast Project (Open Source); Publisher: Feast Project (Open Source)
  开源特征商店Feast的官方文档，展示了如何使用其SDK检索用于训练数据的历史、时间点正确特征，与提供的代码示例直接相关。
