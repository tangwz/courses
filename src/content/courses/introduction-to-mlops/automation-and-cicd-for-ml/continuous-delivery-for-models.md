---
course: "introduction-to-mlops"
chapter: "automation-and-cicd-for-ml"
lesson: "continuous-delivery-for-models"
sourceId: 7233
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-4-automation-and-cicd-for-ml/continuous-delivery-for-models"
title: "模型持续交付 (CD)"
description: "了解持续交付如何自动化将经过训练和验证的机器学习模型发布到生产环境的过程。"
order: 3
plots: []
sourceHash: "59f22f611720439cd0ee7737bbfdf72c3e66403a6a176a4b2447e2f70a5df272"
sourceCorrections: []
---

模型持续交付 (CD) 是一种致力于安全、高效地将机器学习 (machine learning)模型工件交付到生产环境的实践。它将整个发布流程自动化，确保经过全面测试、验证并通过自动化检查的模型能够可靠地部署。这种方法扩展了持续集成 (CI) 的原则，CI 专注于验证单个组件，而 CD 则专注于自动化发布经过验证的完整模型。

机器学习的持续交付是指将训练好并经过验证的模型自动化发布到生产环境的实践。主要目标是让部署成为一种低风险、高频率且可预测的活动。需要将其与持续*部署*区分开来，在持续部署中，每个通过所有自动化测试的更改都会自动发布给用户。在许多机器学习系统中，CD 包含最后的各种人工审核步骤，让操作人员有机会在全面推广前检查模型的预期业务影响。

### 机器学习 (machine learning)持续交付的特殊性

在传统的软件工程中，CD 流水线通常处理编译后的代码。对于机器学习，交付的“工件”更为复杂。它不仅仅是代码，而是一个完整的预测服务。

一个典型的机器学习工件包包括：

- **模型文件：** 序列化后的训练模型（例如 `model.pkl` 或 `saved_model.pb` 文件）。
- **服务代码：** 一个应用程序，通常是像 Flask 或 FastAPI 这样的轻量级 Web 框架，用于加载模型并通过 API 接口公开。
- **依赖项：** 列出所有必要库及其特定版本的文件（例如 `requirements.txt`），以确保环境完全可复现。
- **容器化文件：** 一个配置文件，如 `Dockerfile`，定义了如何将上述所有组件构建为一个便携且独立的单元。

这个包是 CI 或持续训练 (CT) 流程的输出，也是 CD 流水线的输入。

### 模型 CD 流水线的构成

机器学习 (machine learning)模型的自动化 CD 流水线由几个不同的阶段组成，每个阶段都在增加新模型应对生产流量的信心。如果任何阶段失败，流水线就会停止，防止缺陷模型被部署。

> 机器学习模型持续交付流水线示意图。

让我们看看图中显示的每个步骤。

#### 1. 打包模型

流水线的首要任务是将模型工件和所有相关组件打包成一个单一且不可变的单元。行业的标准做法是使用 **Docker 容器**。容器将模型、预测代码和所有系统依赖项打包在一起，创建一个轻量、隔离的环境。这保证了模型在测试、预发布和生产环境中的运行方式完全一致，解决了常见的“在我的机器上能运行”的问题。

#### 2. 部署到测试环境

打包完成后，容器会自动部署到**测试环境**（Staging Environment）。这是一个预生产环境，旨在完全模拟真实的生产系统。在此处部署可以在不影响实际用户的情况下，在真实设置中进行最终测试。

#### 3. 在测试环境中运行高级测试

在测试环境中执行的测试比 CI 期间运行的单元测试和数据验证测试更全面。它们侧重于模型作为服务的运营和性能方面。

- **集成测试：** 验证模型的 API 是否与应用程序的其他部分（如数据存储或面向用户的前端）正确交互。
- **负载测试：** 对模型服务施加模拟流量，以衡量其在压力下的性能。目标是回答以下问题：它每秒能处理多少请求？平均预测延迟是多少？性能是否随负载增加而下降？
- **影子测试：** 一种强大的技术，将新模型以“影子模式”部署。它接收一份真实生产流量的副本，并记录其预测结果，但不发送给用户。这让你可以将新模型的性能和预测结果与当前部署的模型在真实数据上进行比较，且不承担任何风险。

#### 4. 人工审批环节

如果测试环境中的所有自动化测试都通过了，流水线通常会暂停以等待**人工审批**。这是一个计划内的检查点，由机器学习工程师或产品经理等相关方审查测试结果。他们在给出最终许可前，会检查模型的性能指标、在影子模式下的表现以及潜在的业务影响。这一人工参与步骤是一种安全措施，在自动化速度与监管需求之间取得平衡。

#### 5. 发布到生产环境

获得最终批准后，CD 系统执行最后一步：将模型发布到生产环境。这个过程也可以非常先进。团队通常不会一次性更换旧模型，而是使用逐步发布策略，例如：

- **金丝雀发布：** 将一小部分用户流量（例如 5%）引导至新模型并严密监控。如果表现良好，则逐渐增加流量，直到其处理 100% 的流量。
- **蓝绿部署：** 维护两个相同的生产环境（“蓝色”和“绿色”）。新模型部署在不活动的环境（例如绿色）中，一旦确认运行正常，流量就会从蓝色环境切换到绿色环境。

这些发布策略降低了风险，并在发现问题时提供了一种快速回滚的方法。通过将验证后的模型到上线服务的路径自动化，持续交付使机器学习部署成为一种常规、可靠的流程，而不是一种压力巨大的紧急任务。

## 参考资料

- [Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation](https://www.oreilly.com/library/view/continuous-delivery/9780321670250/) — Jez Humble, David Farley (2022)
  Publisher: Addison-Wesley Professional
  这本基础性著作确立了软件工程中持续交付的核心原则和实践，为机器学习模型的适配提供了概念。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2020)
  这份白皮书详细介绍了MLOps的最佳实践，着重于如何在Google Cloud上为机器学习系统实现持续集成、持续交付和持续训练，对实际实现细节高度相关。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning-systems/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书提供了设计和部署ML系统的见解，包括模型服务、金丝雀和蓝绿部署模式，以及适用于模型持续交付的各种测试策略的讨论。
