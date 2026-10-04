# 运用可解释性方法（SHAP、LIME）进行诊断

来源：[原文](https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-3-granular-performance-monitoring/explainability-for-diagnostics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当整体性能指标下降或针对特定数据片段发出警报时，仅了解 *发生了什么* 是不够的。有效的诊断需要理解模型 *为何* 表现不同。模型可解释性技术（传统上在开发阶段使用）在此类情况下在生产监控和诊断中发挥重要作用。LIME（局部可解释模型无关解释）和 SHAP（Shapley 可加性解释）等方法提供实例层面的信息，补充了指标跟踪，并有助于查明性能下降或意外行为的根本原因。

监控不同数据切片上的性能有助于识别问题 *在哪里*，而可解释性工具则有助于诊断这些问题 *为何* 发生。它们从关联性出发，提供有关模型如何使用输入特征得出特定预测的信息，尤其是在预测错误的实例或被标记 (token)为异常的实例上。

### LIME 用于快速诊断

LIME 的工作原理是，在感兴趣的特定实例周围，局部近似任何复杂黑箱模型的行为。它会轻微扰动输入实例，观察模型预测的变化，然后将一个更简单、可解释的模型（如加权线性回归）拟合到这些扰动数据上。该局部模型的系数或特征重要性将作为原始预测的解释。

在生产诊断中，LIME 的用途包括：

1. **调试单个错误：** 当用户报告某个错误预测，或监控系统标记 (token)出高影响错误时，可以在该特定实例上运行 LIME。它有助于识别是哪些特征导致预测出现错误结果。例如，某个特征异常高的值是否导致了贷款申请被拒？
2. **理解边缘情况行为：** 模型可能在罕见或分布外输入上表现不佳。将 LIME 应用于这些边缘情况，有助于了解模型是否在不合理地外推，或依赖于训练过程中学到的虚假关联。

LIME 的一个主要优点是其模型无关性。它几乎可以应用于任何分类或回归模型，而无需访问模型内部结构。然而，解释有时可能不稳定，这意味着输入实例的微小变化可能导致截然不同的解释。这需要仔细解读，并可能需要生成多个解释以提高稳定性。

### SHAP 用于更深入、一致的分析

SHAP 提供了一种更具理论基础的方法，该方法基于 Shapley 值，Shapley 值是合作博弈论中的一种值。它为特定预测的每个特征分配一个重要性值，表示其对将预测从基线（例如，训练集上的平均预测）推开的贡献。SHAP 值的属性包括局部准确性（特征贡献的总和等于预测减去基线）和一致性（如果模型更改为 *更* 依赖某个特征，则该特征的重要性不会下降）。

SHAP 为诊断提供了多项优势：

1. **一致的特征贡献：** SHAP 值提供了一种统一的特征重要性衡量标准，它在数学上严谨，并且在不同模型类型之间保持一致（尽管针对树模型等特定模型存在优化实现）。
2. **全局和群体分析：** 虽然 LIME 主要关注局部，但为单个实例计算的 SHAP 值可以进行汇总。这使得群体分析变得有效：
   - **比较数据片段：** 计算表现不佳的数据片段与表现良好数据片段或总体数据的聚合 SHAP 值（例如，每个特征的平均绝对 SHAP 值）。这可以突出显示在有问题群体中，哪些特征对预测的驱动方式不同。
   - **追踪特征重要性漂移：** 随时间监控主要特征的 SHAP 值分布。某个特征重要性的显著变化可能表示数据关系漂移，即特征与目标之间的关系已改变，甚至在准确性等整体指标大幅下降之前。
   - **错误分析：** 专门针对预测错误的实例分析 SHAP 值。错误中特征贡献是否存在普遍模式？也许某个特定特征总是将预测推向错误方向，针对某种输入类型。



[交互图表：平均绝对SHAP值比较：数据片段 vs 总体](https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-3-granular-performance-monitoring/explainability-for-diagnostics#plot-1cwwkfb)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "平均绝对SHAP值比较：数据片段 vs 总体",
    "xaxis": {
      "title": "平均绝对SHAP值"
    },
    "yaxis": {
      "title": "特征",
      "categoryorder": "total ascending"
    },
    "margin": {
      "l": 150,
      "r": 20,
      "t": 50,
      "b": 50
    },
    "barmode": "group",
    "legend": {
      "title": {
        "text": "数据来源"
      }
    }
  },
  "data": [
    {
      "type": "bar",
      "name": "问题数据片段",
      "y": [
        "Feature A",
        "Feature B",
        "Feature C",
        "Feature D"
      ],
      "x": [
        0.45,
        0.15,
        0.3,
        0.05
      ],
      "orientation": "h",
      "marker": {
        "color": "#fa5252"
      }
    },
    {
      "type": "bar",
      "name": "总体数据",
      "y": [
        "Feature A",
        "Feature B",
        "Feature C",
        "Feature D"
      ],
      "x": [
        0.3,
        0.25,
        0.28,
        0.12
      ],
      "orientation": "h",
      "marker": {
        "color": "#339af0"
      }
    }
  ]
}
```

</details>



> 比较有问题的数据片段与总体数据之间的平均特征重要性（平均绝对 SHAP 值）。特征 A 在问题片段中的重要性显著更高，而特征 B 和 D 的重要性更低，这表明该片段内可能存在与特征 A 相关的行为变化或数据问题。

### 将可解释性整合到监控工作流中

在生产环境中有效应用这些方法需要周密的规划：

- **计算开销：** 计算解释，尤其是 SHAP，可能耗费资源。在高吞吐量 (throughput)系统中，为每个预测生成解释通常不切行。应对方法包括：
  - **抽样：** 为具有统计代表性的预测样本生成解释。
  - **基于触发的解释：** 仅当监控警报指示潜在问题（例如，性能下降、检测到漂移、某个片段中错误率高）时才计算解释。
  - **离线分析：** 在批量作业中对已记录的预测数据运行解释，而不是在实时预测路径中。
  - **优化实现：** 使用 `SHAP` 等库，它们为特定模型类型（例如，用于基于树的集成模型的 `TreeExplainer`）提供优化的解释器。
- **存储和可视化：** SHAP 值或 LIME 系数需要与其他监控数据一起存储。时序数据库可以随时间追踪聚合的 SHAP 值。专用仪表板可以可视化单个实例的解释或群体比较，就像上面显示的汇总图一样。
- **可操作性：** 目标是诊断并采取行动。解释应融入前面描述的根本原因分析过程。错误中某个特征的重要性较高是否表示该特征存在数据质量问题？特征重要性发生变化是否表明需要使用更新的数据重新训练模型？

可解释性方法并非万能。它们提供了模型行为的模型，而解读需要领域知识和批判性思维。然而，当周全地整合到监控系统中时，LIME 和 SHAP 会成为强大的诊断工具，使团队能够观察性能变化，并理解和解决其根本原因。

## 参考资料

- [“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍LIME，这是一种模型无关的局部解释方法，通过简化的模型来近似复杂模型的行为。
- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) — Scott Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; DOI: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
  提出SHAP值，这是一种博弈论方法，用于统一各种解释方法，提供一致的特征贡献。
- [SHAP (SHapley Additive exPlanations) Documentation](https://shap.readthedocs.io/en/latest/) — Scott Lundberg and SHAP Contributors (2024)
  SHAP Python库的官方文档，包含示例和优化解释器。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  Publisher: Leanpub
  一本全面的在线书籍，涵盖各种可解释性方法，包括LIME和SHAP，并提供理论和实践考量。

---

[上一节](05-%E6%80%A7%E8%83%BD%E4%B8%8B%E9%99%8D%E7%9A%84%E6%A0%B9%E6%9C%AC%E5%8E%9F%E5%9B%A0%E5%88%86%E6%9E%90.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%80%9F%E5%8A%A9%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%80%A7%E8%AF%8A%E6%96%AD%E6%80%A7%E8%83%BD%E9%97%AE%E9%A2%98.md)
