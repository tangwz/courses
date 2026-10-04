---
course: "feature-stores-for-ml"
chapter: "data-consistency-quality"
lesson: "diagnosing-mitigating-skew"
sourceId: 3378
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-3-data-consistency-quality/diagnosing-mitigating-skew"
title: "诊断与缓解在线/离线偏差"
description: "用于识别、测量和解决训练与服务特征之间不一致的方法。"
order: 1
plots: ["plots/3378-0.json"]
sourceHash: "feef6e5e88d2a8552cb44fea7bb50397f9f447de03389b4b4dda334f5ec9c2fe"
sourceCorrections: []
---

模型训练（离线）时生成的特征值与实时预测（在线）时生成的特征值之间存在差异，这被称为在线/离线偏差。这种不一致性，正式表示为分布差异 $P_{train}(X) \neq P_{serve}(X)$，会大幅降低生产模型的性能。一个在特定特征上训练的模型，在服务路径中遇到以不同方式生成的特征时，其行为可能会变得不可预测，即使原始数据看起来相似。了解这种偏差的来源并实施检测和缓解方案，对于可靠运行机器学习 (machine learning)系统非常重要。

### 在线/离线偏差的常见原因

偏差通常并非源于单一的灾难性故障，而是通过训练和服务数据管道之间不明显的差异逐渐出现。识别潜在原因是对其进行预防的第一步。

1. **独立实现逻辑：** 这可能是最常见的原因。特征工程逻辑通常为批处理训练管道（例如，使用Spark、Pandas处理历史数据）和在线服务（例如，使用通常以不同语言或框架编写的低延迟服务来访问实时数据）独立实现。即使实现细节、库版本或边缘情况处理上的微小差异，也可能导致特征值出现差异。
2. **数据访问的时间差异：** 训练数据基于历史快照生成，而服务数据则使用最新的可用信息。如果原始数据分布随时间自然漂移（数据随时间漂移），服务时生成的特征将与较旧的训练特征本质上有所不同。此外，对“现在”的定义或时间窗口计算方式上的细微变化，也可能导致不一致，特别是对于时间敏感的聚合。
3. **不同数据源：** 训练管道可能从数据湖或数据仓库（例如，S3中的Parquet文件、BigQuery中的表）读取数据，而在线路径可能从生产数据库、缓存或事件流（例如，DynamoDB、Redis、Kafka）访问数据。这些源之间在数据新鲜度、模式强制或数据清洗流程上的差异都可能引入偏差。
4. **数据类型和精度不匹配：** 不同系统如何处理浮点精度、整数类型或字符串编码，可能引入微小但累积的误差，这些误差会表现为偏差。例如，在训练期间使用Python/Pandas中的`float64`进行的计算，可能与在性能要求高的在线服务中使用`float32`或定点表示的相同计算产生略微不同的结果。
5. **连接逻辑和时间安排：** 特征通常需要连接多个数据源（例如，用户画像数据与活动数据）。这些连接的确切时间和逻辑可能有所不同。批处理管道可能基于大型数据集上的固定时间窗口执行连接，而在线系统可能基于请求时间执行即时查询。在两个路径中，对某一源中缺失数据的处理方式也可能不同（例如，填充空值与默认值）。
6. **错误和边缘情况：** 在线或离线管道中简单的编程错误或未处理的边缘情况（如除以零、空值传播）显然会导致不同的特征值。

### 检测偏差：测量与监控

检测偏差需要系统性的比较和监控。仅仅假设一致性不足以满足生产系统的要求。

#### 统计分布分析

核心思路是比较训练期间生成的特征值与服务期间观测到的特征值的统计特性。

1. **收集特征日志：** 记录在线服务系统为具有代表性的预测请求样本生成的特征值。同时存储相关元数据，如时间戳和实体ID。
2. **生成训练特征：** 确保您能够访问用于训练当前部署模型版本的准确特征值。
3. **计算汇总统计量：** 计算来自两个来源（训练数据和服务日志）的每个特征的描述性统计量。这包括均值、中位数、标准差、方差、最小值、最大值和分位数（例如，第1、5、25、75、95、99百分位）。这些统计量的显著差异是偏差的有力指标。
4. **统计检验：** 运用统计检验来量化 (quantization)分布之间的差异。
   - **Kolmogorov-Smirnov (K-S) 检验：** 一种非参数 (parameter)检验，用于比较两个样本的累积分布函数（CDF）。它对位置、尺度和形状的差异敏感。较小的p值表明分布存在显著差异。
   - **群体稳定性指数（PSI）：** 常用于信用风险建模，但在此也适用。它衡量一个变量在两个群体（训练与服务）之间分布的变化程度。它涉及对变量进行分箱，并比较每个箱中观测值的百分比。
     $PSI = \sum (\%_{服务} - \%_{训练}) \times \ln \left( \frac{\%_{服务}}{\%_{训练}} \right)$
     常见的解释阈值为：PSI < 0.1（无显著偏移），0.1 <= PSI < 0.25（中度偏移，需调查），PSI >= 0.25（显著偏移，需采取行动）。
   - **卡方检验：** 适用于比较分类特征的分布。

#### 可视化比较

可视化提供了一种直观的方式来发现差异。并排或叠加绘制训练和服务日志中特征值的直方图或密度图，可以显示形状、集中趋势或分散度上的不一致。



![分布比较：特征 'avg_purchase_value_7d'](plots/3378-0.json)



> 特征 'avg\_purchase\_value\_7d' 的概率密度直方图比较，显示服务分布与训练分布相比有明显向更高值偏移。

#### 直接值比较（影子模式）

如果可行，可以将在线特征生成逻辑部署到“影子模式”中，它会处理实时请求，但不会直接提供结果。记录生成的特征。同时，使用批处理管道为*相同*的实体和时间戳计算特征。直接比较这些值可以准确找出差异。这需要细致的时间同步和实体匹配。

#### 监控与告警

将偏差检测集成到您的MLOps监控栈中。定期计算重要特征的分布统计量和PSI。设置自动化告警，在指标超过预设阈值时通知团队，表示可能存在需要调查的偏差。

### 缓解偏差的方案

预防通常比补救更有效。缓解措施着重于统一特征生成过程并实施严格的验证。

1. **统一特征逻辑：** 这是最有效的方案。使用可相同执行的库或框架定义特征转换，无论是在批处理（例如，Spark、Pandas）还是在线（例如，低延迟服务、流处理引擎如Flink/Kafka Streams）环境中。特征平台本身通常通过提供SDK或API来促进这一点，这些SDK或API抽象了底层计算引擎但使用相同的转换定义。目标是*一次性*编写特征逻辑并在各处重用。
2. **数据源一致性：** 尽可能确保在线和离线管道都访问*相同*的原始数据源。如果独立的源不可避免（例如，日志流与数据仓库），则实施对账流程并密切监控源数据一致性。使用版本化数据集进行训练以确保可复现性。
3. **即时校正连接：** 仔细实现连接逻辑，特别是对于时间相关特征。确保批处理训练期间执行的连接，准确反映每个事件发生时*本应可用*的信息。特征平台通常提供即时校正查询的内置功能，应一致使用。
4. **严格测试：**
   - **单元测试：** 使用已知输入和预期输出测试单个转换函数，覆盖边缘情况。
   - **集成测试：** 创建测试场景，使用相同的输入数据，在批处理和在线执行路径中运行*相同*的特征定义代码。比较输出是否相等（在浮点数可接受的精度容差范围内）。
5. **模式强制和数据验证：** 在*两个*管道的数据摄取点应用严格的模式验证和数据质量检查。拒绝或标记 (token)不符合预期的数据。这可以防止由意外输入数据格式或值引起的下游错误。Great Expectations 或 Deequ 等工具可以在此集成。
6. **标准化环境：** 尽可能保持训练和服务环境之间的库版本、配置和依赖项一致。容器化（例如，Docker）有助于强制一致性。
7. **定期审计和反馈：** 不要仅仅依赖初始设置。安排定期审计，明确重新运行偏差检测分析。使用监控告警作为反馈，触发对特征管道或定义的调查和改进。

通过系统地解决在线/离线偏差的潜在原因并实施检测和缓解机制，可以显著提高生产环境中机器学习 (machine learning)模型的可靠性和性能一致性。这种一致性是成熟MLOps实践和精心设计的特征平台实现的一个标志。

## 参考资料

- [TFX: A TensorFlow-Based Production Machine Learning Platform](https://doi.org/10.1145/3097983.3098021) — Denis Baylor, Eric Breck, Heng-Tze Cheng, Noah Fiedel, Chuan Yu Foo, Zakaria Haque, Salem Haykal, Mustafa Ispir, Vihan Jain, Levent Koc, Chiu-Yuen Koo, Lukasz Lew, Clemens Mewald, Akshay Naresh Modi, Neoklis Polyzotis, Sukriti Ramesh, Sudip Roy, Steven Euijong Whang, Martin Wicke, Jarek Wilkiewicz, Xin Zhang, Martin A. Zinkevich (2017)
  Journal: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM (Association for Computing Machinery); Pages: 2099-2107; DOI: [10.1145/3097983.3098021](https://doi.org/10.1145/3097983.3098021)
  介绍了TensorFlow Extended (TFX)，详细说明了其组件，例如TensorFlow Data Validation (TFDV)，该组件通过统计分析提供工具来检测数据异常和训练-服务偏差。
- [Machine Learning Design Patterns](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZWOoP-hvsH7iLeUkV4fxfO8xmhYV8l5P2-fKtVvYM_DQ6OPAxVv-VMDORk_f3IdFRy2Dr--DV8A5elRJTfe2VhmMG8qUrFkv5et-RcxgyuA2IS_542BuUxkGLwbGURU0uGSasgDFJX_ntJviDhvhDKJuAvQzDxPByM8BUTf-BrTz3iLyxXqup8QY3s8SmYfOZbug-G2Q3_KpWYQqh9PVln1dMtLt4ndrhrjWm6qfo4rFrOmE5NJppLqXwLfWuyq5jx0FMHbE=) — Valliappa Lakshmanan, Sara Robinson, Michael Munn (2020)
  Publisher: O'Reilly Media
  本书提供了构建可靠机器学习系统的实用设计模式，包括特征工程一致性、数据验证和缓解训练-服务偏差的策略。
- [Hidden Technical Debt in Machine Learning Systems](https://neurips.cc/paper/2015/hash/86df7dcfd896fcaf26747a82c6981ad2-Abstract.html) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS Proceedings; Volume: 28
  这篇被广泛引用的论文指出了机器学习系统特有的各种形式的技术债务，其中许多可能导致在线/离线偏差，例如数据依赖性和模型更新周期。
