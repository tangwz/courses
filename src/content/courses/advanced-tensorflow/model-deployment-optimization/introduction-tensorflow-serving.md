---
course: "advanced-tensorflow"
chapter: "model-deployment-optimization"
lesson: "introduction-tensorflow-serving"
sourceId: 2920
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/introduction-tensorflow-serving"
title: "TensorFlow Serving 简介"
description: "了解 TensorFlow Serving，这是一个灵活、高性能的机器学习模型服务系统。"
order: 2
plots: []
sourceHash: "833ad1d2a581513cf64528ca85ef451c6c13aee4f0446a9749e0e4f3675b6434"
sourceCorrections: []
---

在您训练并导出模型（通常使用 SavedModel 格式）后，接下来重要的一步是使其可用于提供预测服务。虽然您可以构建一个自定义应用程序（例如使用 Flask 或 FastAPI）来加载 SavedModel 并公开 API 端点，但这种方法通常缺乏高要求的生产环境所需的稳定性、性能和生命周期管理功能。这正是 TensorFlow Serving 旨在解决的问题。

TensorFlow Serving 是一个专用的高性能服务系统，专为生产环境中的机器学习 (machine learning)模型而设计。将其视为一个独立的服务器应用程序，而不仅仅是一个库，它针对推理 (inference)进行了优化。它接收您训练好的模型（以 SavedModels 形式打包），并通过定义清晰的 API（通常是 REST 或 gRPC）使其在网络上可供访问。

### 为何使用专用服务系统？

部署模型不仅仅是加载文件和运行 `model.predict()`。生产系统常常面临以下要求：

1. **高吞吐量 (throughput)和低延迟：** 同时快速地向许多用户提供预测。
2. **模型版本管理：** 在不停机的情况下部署更新的模型（金丝雀发布、回滚）。
3. **多模型支持：** 从同一基础设施提供不同模型或模型类型的服务。
4. **资源管理：** 高效利用硬件（CPU、GPU、TPU）。
5. **生命周期管理：** 动态加载、卸载和管理模型。

TensorFlow Serving 旨在有效处理这些难题。它为模型生命周期管理提供了开箱即用的解决方案，使您能够部署新版本、在不同版本之间运行 A/B 测试，或同时服务多个不同模型，同时保持高性能。

### TensorFlow Serving 的核心抽象概念

虽然后续章节将介绍实际用法，但了解基本架构会很有帮助。TensorFlow Serving 采用了一些抽象概念：

- **Servables（可服务项）：** 这是 TensorFlow Serving 管理的基本对象。通常，一个 Servable 代表一个已训练模型的特定版本，可用于推理 (inference)。它也可以是查找表或计算所需的其他数据。
- **Sources（源）：** 发现并提供 Servables 的插件。常见的源会监视文件系统路径以获取新的 SavedModel 目录（每个代表一个模型版本）。
- **Loaders（加载器）：** 负责加载 Servable 的数据，包括分配必要资源（如 GPU 内存）。它们知道如何解释特定的模型格式（如 SavedModel）。
- **Managers（管理器）：** 处理 Servables 的完整生命周期，包括根据源定义的策略进行加载（通过加载器）、服务和卸载。它们协调模型版本之间的切换。
- **Core（核心）：** 管理生命周期并公开 API（gRPC、REST）以供客户端发送推理请求的中心组件。

> TensorFlow Serving 的基本架构，展示了客户端请求如何流经 API 到管理器，管理器使用加载器和源从受管理模型版本（Servables）提供预测服务。

### 优势

使用 TensorFlow Serving 带来多种优势：

- **性能：** 高度优化的 C++ 服务器，专为低延迟和高吞吐量 (throughput)设计。支持 GPU 等硬件加速器。自动实现请求批处理等优化。
- **灵活性：** 轻松同时管理多个模型或同一模型的多个版本。支持金丝雀部署和回滚。
- **生产就绪性：** 为生产环境中的稳定性和可靠性而构建。
- **可扩展性：** 设计为可扩展，以支持新的模型类型、文件系统或部署场景。
- **标准化：** 提供标准推理 (inference) API（REST 和 gRPC），简化客户端集成。依赖 SavedModel 格式，与 TensorFlow 生态系统良好集成。

总的来说，TensorFlow Serving 在您训练好的 TensorFlow 模型与需要大规模使用其预测的应用程序之间，提供了基础设施连接。以下章节将演示如何准备模型并使用这个强大的系统进行部署。

## 参考资料

- [TensorFlow Serving](https://www.tensorflow.org/tfx/serving) — TensorFlow Developers (2024)
  官方文档提供了关于TensorFlow Serving的设置、配置和使用的全面信息，包括其架构、API以及部署的最佳实践。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本构建和部署机器学习系统的综合指南，其中包含关于模型服务挑战和解决方案的章节，包括使用TensorFlow Serving等专用系统。
