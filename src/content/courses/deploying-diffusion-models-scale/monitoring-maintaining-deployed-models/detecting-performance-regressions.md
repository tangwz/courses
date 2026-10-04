---
course: "deploying-diffusion-models-scale"
chapter: "monitoring-maintaining-deployed-models"
lesson: "detecting-performance-regressions"
sourceId: 5463
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-5-monitoring-maintaining-deployed-models/detecting-performance-regressions"
title: "检测性能下降"
description: "用于识别更新后运行缓慢或错误率增加的策略。"
order: 4
plots: ["plots/5463-0.json"]
sourceHash: "7bae7da27d2721912328d1f9823abc46c9cb914d3cfa3cbbafb911e543f277f3"
sourceCorrections: []
---

从部署的扩散模型中收集的监控数据，例如生成延迟（$L_{gen}$）、请求吞吐量 (throughput)（$T_{req}$）、错误率和GPU利用率（$U_{gpu}$），提供了识别何时出现性能问题所需的信息。性能下降，起初通常不明显，但如果置之不理，可能会严重影响用户体验和运营成本。及时发现这些下降很必要，尤其是在部署新模型版本、更新基础设施或更改配置之后。

性能下降表现为多种方式：

- **延迟增加：** 用户遇到图像生成时间变慢（$L_{gen}$ 上升）。
- **吞吐量下降：** 在相同资源下，系统处理的并发请求减少（$T_{req}$ 下降）。
- **错误率增加：** 更多生成请求失败，或API端点返回更多错误。
- **成本上升：** 每生成一张图像的计算成本增加，可能是由于资源使用效率低下或处理时间延长。
- **质量下降：** 生成的图像连贯性变差，美观度降低，或未能有效遵循提示（这通常是最难自动发现的）。

### 发现性能下降的策略

发现性能下降涉及将当前性能与基线或预期行为进行比较。可以使用多种方法：

#### 基于阈值的预警

最直接的方法是为您的主要指标设置预设阈值。例如，您可以配置在以下情况发出预警：

- 95%分位延迟（$P_{95}(L_{gen})$）超过某个值（例如，8秒）。
- 平均吞吐量 (throughput)（$Avg(T_{req})$）低于可接受的最低水平。
- API错误率超过某个百分比（例如，1%）。

虽然使用Prometheus Alertmanager或云服务商报警工具实现起来很简单，但静态阈值可能不稳定。它们可能在自然高峰期触发误报，或未能发现低于绝对限制的渐进式下降。动态阈值会根据历史模式（例如一天中的时间、一周中的某天）进行调整，提供了一些改进，但仍需要仔细调整。

#### 统计过程控制 (SPC)

SPC技术，借鉴自制造业质量控制，提供了一种更具统计学依据的方式来发现变化。像累积和 (CUSUM) 图表或指数加权移动平均 (EWMA) 图表这样的方法随时间跟踪指标，并在偏离预期过程均值或方差的统计显著性时发出预警。

例如，用于$L_{gen}$的EWMA图表会给近期观测值更大的权重 (weight)。公式可能如下所示：

$EWMA_t = \lambda \cdot L_{gen, t} + (1 - \lambda) \cdot EWMA_{t-1}$

其中，$L_{gen, t}$ 是时间 $t$ 的延迟，$EWMA_{t-1}$ 是前一个EWMA值，以及 $\lambda$ ($0 < \lambda \le 1$) 是一个平滑因子。如果 $EWMA_t$ 超出控制限（例如，与历史均值相差 $\pm 3$ 个标准差），则可以触发预警。与简单的阈值方法相比，SPC更擅长发现较小、持续的变化。

#### 时间序列异常发现

这种方法使用专门设计的算法来发现时间序列数据中的异常模式。这些方法包括统计方法（例如，比较滚动窗口统计数据）到机器学习 (machine learning)模型（例如，Isolation Forests、Autoencoders、Facebook Prophet）。异常发现系统可以自动学习季节性和趋势，使其在应对正常工作负载变化时更有效。

考虑监控 $P_{95}(L_{gen})$ 指标。异常发现系统可以在部署后识别出突然、持续的跳跃，即使新的延迟水平未超出预设的静态阈值。



![P95 生成延迟异常发现](plots/5463-0.json)



> 在10:25之后不久，P95延迟突然增加被发现，这可能表示近期更改引入了性能下降。

#### 用于性能下降发现的部署策略

如第6章中进一步讨论的，像**金丝雀发布**和**A/B测试**这样的部署模式本身对于发现性能下降很有用。通过将一小部分流量路由到新模型版本（金丝雀）或在两个版本之间拆分流量（A/B测试），您可以直接比较新旧版本之间在相同条件下的性能指标（$L_{gen}$，$T_{req}$，错误率，质量指标）。如果新版本表现出明显更差的性能，则可以在影响大多数用户之前自动停止或回滚发布。

> 性能下降发现涉及将当前或金丝雀指标与既定基线进行比较。

#### 例行基准测试

建立一套标准化的提示和生成参数 (parameter)，代表典型使用模式。定期（例如，每晚或每周）对已部署的模型API运行此基准测试套件。存储由此产生的性能指标（$L_{gen}$，成本）以及可能的客观质量分数。长期跟踪这些基准测试结果，提供了一个稳定的基线，以发现渐进的性能漂移或特定更改引入的性能下降。

### 处理质量下降

发现生成图像*质量*的变化比监控延迟或错误更具挑战性。

- **自动化质量指标：** 定期使用一组固定的评估提示生成图像，并根据参考数据集或之前的模型输出计算FID (Fréchet Inception Distance) 或 CLIP Score 等指标。虽然有用，但这些指标并不总是与人类感知完全一致，并且计算成本可能很高。如果这些指标显著下降，则配置预警。
- **监控输出嵌入 (embedding)：** 为生成的图像样本生成嵌入（例如，使用CLIP）。随时间监控这些嵌入的分布。分布的突然变化可能表示模型正在生成不同*类型*的图像，可能是由于性能下降或意外的行为改变。可以使用例如监控随时间分布质心之间距离的方法。
- **人工反馈循环：** 纳入机制，允许用户对他们认为低质量的输出进行评分或标记 (token)。定期分析此反馈。为了内部验证，建立评审流程，团队成员定期评估生成的样本，尤其是在模型更新之后。

### 将发现与行动结合

发现只有在它能促进行动时才有用。将您的性能下降发现机制与预警系统（PagerDuty、Slack、电子邮件）集成，以通知负责团队。对于严重的性能下降，尤其是在金丝雀部署或A/B测试期间发现的下降，考虑实施自动化回滚程序，以快速恢复到最后一个已知的稳定版本，最大程度地减少用户影响。

通过系统地监控性能和实施发现策略，您可以维护扩散模型部署的健康、效率和质量，确保其持续可靠地提供价值。

## 参考资料

- [Site Reliability Engineering: How Google Runs Production Systems](https://sre.google/sre-book/table-of-contents/) — Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy (2017)
  Publisher: O'Reilly Media
  一本关于监控、警报和运营实践的指南，包括设置阈值和管理金丝雀发布等部署。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper_files/paper/2017/file/8a1d69470764724a0384a67303e4b71f-Paper.pdf) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, Andreas Maier, Sepp Hochreiter, and Michael Widmann (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2017); Publisher: NeurIPS; Volume: 30; Pages: 6626-6636; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  引入了Fréchet Inception Distance (FID) 度量，这是评估生成图像质量的常用指标。
