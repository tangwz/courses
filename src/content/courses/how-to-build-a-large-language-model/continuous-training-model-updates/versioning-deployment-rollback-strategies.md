---
course: "how-to-build-a-large-language-model"
chapter: "continuous-training-model-updates"
lesson: "versioning-deployment-rollback-strategies"
sourceId: 6112
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-30-continuous-training-model-updates/versioning-deployment-rollback-strategies"
title: "版本管理、部署与回滚策略"
description: "实施持续训练模型的版本管理、更新部署以及必要时回滚的实践。"
order: 6
plots: []
sourceHash: "e234105f2f482b4b769e57a32083ba871ace1f9deb22680ca63a2dcb172f4025"
sourceCorrections: []
---

管理持续更新的大语言模型 (LLM)生命周期需要严谨的工程实践，特别是在新版本如何追踪、部署以及可能回退方面。如果没有健全的策略，将更新后的模型引入生产环境可能导致性能下降、异常行为或服务中断。本节阐述了适用于持续训练中的大语言模型的版本管理、部署和回滚的实用方法。

### 模型版本管理

有效的版本管理对于追踪模型演进、确保可复现性并实现安全回滚非常重要。仅仅保存模型权重 (weight)是不够的；版本管理必须包含所有相关成果物和元数据。

**版本管理方案：** 采用语义化版本管理（SemVer - `MAJOR.MINOR.PATCH`）提供了一种实用的结构：

- **MAJOR（主版本，例如2.0.0）：** 在发生重大架构变化、不兼容的API变更，或在显著不同数据集上重新训练导致模型能力或行为从根本上改变时增加。
- **MINOR（次版本，例如1.1.0）：** 在进行实质性更新时增加，例如在新的数据切片上持续预训练 (pre-training)，使用大量新指令数据进行新的SFT阶段，或进行重大超参数 (parameter) (hyperparameter)调整以提升性能但保持核心能力的向后兼容性。
- **PATCH（修订版本，例如1.0.1）：** 在进行小错误修复、微调 (fine-tuning)数据的小幅调整或不会显著改变模型核心行为或性能特征的小优化时增加。

**成果物和元数据追踪：** 每个版本标签应与以下内容相关联：

1. **模型权重：** 模型实际的参数。
2. **分词 (tokenization)器 (tokenizer)配置：** 包括词汇文件和任何特定配置（`tokenizer.json`、`vocab.txt`等）。使用不兼容的分词器可能导致静默故障或性能下降。
3. **模型配置：** 详细说明架构、隐藏层大小、层数、激活函数 (activation function)等的文件（`config.json`）。
4. **训练元数据：** 使用的数据集来源和版本、重要的超参数、训练脚本的提交哈希、在标准基准上的评估指标，以及可能的特定硬件/软件环境。

诸如Git大文件存储（LFS）之类的工具可以在Git仓库中管理大型权重文件，而机器学习 (machine learning)实验追踪平台（例如MLflow、Weights & Biases）旨在系统地记录成果物和元数据。

```python
# 示例：使用Hugging Face Transformers保存版本化的模型组件
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设模型和分词器已加载并经过训练/更新
model_version = "1.1.0"
model_save_path = f"./llm_model_v{model_version}"
tokenizer_save_path = f"./llm_tokenizer_v{model_version}"

# 保存模型权重和配置
model.save_pretrained(model_save_path)

# 保存分词器文件
tokenizer.save_pretrained(tokenizer_save_path)

# 通常，您还会将元数据（数据集信息、提交哈希、指标）
# 与这些成果物一起记录到实验追踪系统。
print(f"模型保存至：{model_save_path}")
print(f"分词器保存至：{tokenizer_save_path}")

# 稍后，加载特定版本
# loaded_model = AutoModelForCausalLM.from_pretrained(model_save_path)
# loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
```

### 部署策略

部署数千兆字节或数太字节的模型需要仔细规划，以最大程度地减少停机时间和风险。常见策略包括：

**蓝绿部署：** 维护两个相同的生产环境：“蓝色”（当前线上版本）和“绿色”（新版本）。一旦“绿色”环境测试完毕并准备就绪，负载均衡器将所有流量从“蓝色”环境重定向到“绿色”环境。

> 蓝绿部署：活动流量导向蓝色环境。绿色环境存放新版本，准备切换。

- **优点：** 瞬间流量切换，切换时零停机，回滚简单（只需将流量切换回蓝色环境）。
- **缺点：** 需要两倍的基础设施资源，可能成本较高。在切换前彻底测试绿色环境很重要。

**金丝雀发布：** 逐渐将小部分流量路由到新的模型版本（“金丝雀”版本）。密切监控性能和错误指标。如果金丝雀版本表现良好，逐渐增加流量百分比，直到100%的流量路由到新版本。

> 金丝雀发布：一小部分流量路由到新版本（金丝雀），而大多数用户仍停留在稳定版本。

"\* **优点：** 限制潜在问题的影响范围，支持测试和性能比较，渐进式发布降低风险。"

- **缺点：** 更复杂的基础设施和监控设置，发布阶段可能出现用户体验不一致，发布过程较慢。

**影子部署：** 将新模型版本与当前版本并行部署。将线上流量路由到当前版本，但也将请求镜像或“影子”到新版本。比较影子模型的输出和性能，而不影响用户。

- **优点：** 在实际生产负载下测试新模型的最安全方式，可以比较其行为与当前版本而不影响用户。
- **缺点：** 需要大量额外的计算资源来运行两个模型，设置镜像和比较逻辑的复杂性较高。

策略的选择取决于风险承受能力、资源可用性以及模型更新的性质。对于大语言模型 (LLM)而言，由于难以察觉的细微退步可能难以发现，金丝雀或影子部署尽管复杂，但常被选择。

### 回滚策略

即使经过仔细测试和部署，新模型版本在生产中仍可能出现未预料的问题（例如，延迟增加、错误率升高、有害生成模式、在特定用户群体上表现不佳）。明确定义的回滚策略是必需的。

**回滚规划：**

1. **准备就绪：** 确保以前的稳定版本（模型成果物、分词 (tokenization)器 (tokenizer)、配置）在成果物存储中随时可用。
2. **机制：** 回滚机制与部署策略紧密关联：
   - **蓝绿部署：** 通过将流量重定向回之前活动的（蓝色）环境，实现简单的回滚。
   - **金丝雀发布：** 将金丝雀流量百分比降至0%，并可能移除金丝雀部署。
   - **影子部署：** 只需移除影子部署；未影响用户流量。
3. **触发条件：** 定义清晰的回滚触发条件。这可以根据监控性能指标（KPI）或健康度指标自动化：
   - 推理 (inference)延迟显著增加。
   - 与模型输出相关的应用层面错误率升高。
   - 用户参与度指标下降（如适用）。
   - 关键评估分数（例如，毒性、对齐 (alignment)指标）超过预设阈值。
   - 基于用户投诉或定性审查的手动触发。
4. **回滚后分析：** 回滚后，进行彻底的根本原因分析，以理解新版本为何在生产中失败，然后才能尝试重新部署。

**自动化回滚示例：**

```python
# 自动化回滚的监控循环
import time
import random # 模拟指标检查

CURRENT_MODEL_VERSION = "1.0.1"
CANARY_MODEL_VERSION = "1.1.0"
CANARY_TRAFFIC_PERCENT = 10 # 从10%开始

MAX_ERROR_RATE_THRESHOLD = 0.05 # 5% 错误率
MAX_LATENCY_MS_THRESHOLD = 500 # 500毫秒 p95 延迟

def get_canary_metrics():
  # 实际上，应查询您的监控系统（Prometheus、Datadog等）
  # 获取金丝雀部署的特定指标。
  simulated_error_rate = random.uniform(0.01, 0.07)
  simulated_latency = random.uniform(300, 600)
  print(
      f"金丝雀指标 - 错误率: {simulated_error_rate:.3f}, "
      f"延迟: {simulated_latency:.0f}毫秒"
  )
  return {"error_rate": simulated_error_rate, "latency_p95": simulated_latency}

def set_traffic_split(canary_percent):
  # 实际上，应与您的负载均衡器/服务网格API交互
  # （例如，Istio、Nginx、云负载均衡器）
  global CANARY_TRAFFIC_PERCENT
  CANARY_TRAFFIC_PERCENT = canary_percent
  print(f"--- 将金丝雀流量设置为 {canary_percent}% ---")

def rollback_deployment():
  print("!!! 正在回滚金丝雀部署 !!!")
  set_traffic_split(0)
  # 在此处添加步骤，以可能缩减/移除金丝雀基础设施
  print(
      f"--- 回滚完成。流量已恢复到 {CURRENT_MODEL_VERSION} ---"
  )


# 主监控循环
while CANARY_TRAFFIC_PERCENT > 0:
  time.sleep(60) # 每分钟检查一次
  metrics = get_canary_metrics()

  if metrics["error_rate"] > MAX_ERROR_RATE_THRESHOLD or \
     metrics["latency_p95"] > MAX_LATENCY_MS_THRESHOLD:
    rollback_deployment()
    # 退出循环或触发警报以进行手动调查
    break
  else:
    # （可选）如果指标稳定，则逐渐增加流量
    # if CANARY_TRAFFIC_PERCENT < 100:
    #    set_traffic_split(min(CANARY_TRAFFIC_PERCENT + 10, 100))
    print("金丝雀指标稳定。")

if CANARY_TRAFFIC_PERCENT > 0 : # 如果循环在没有回滚的情况下完成
    print(
        f"金丝雀部署 {CANARY_MODEL_VERSION} 看起来稳定，"
        f"流量在 {CANARY_TRAFFIC_PERCENT}%。考虑全面发布。"
    )
```

实施版本管理、部署和回滚策略，将持续的模型更新从高风险的工作转变为可管理的工程过程。这些实践对于维护在动态生产环境中运行的大语言模型 (LLM)的可靠性和性能非常重要。

## 参考资料

- [Semantic Versioning 2.0.0](https://semver.org/) — Tom Preston-Werner (2013)
  语义化版本控制的官方规范，为跟踪软件（和模型）演进提供了一个广泛采用的框架，并根据兼容性和功能变化制定了明确的版本号增量规则。
- [Machine Learning Engineering](https://www.oreilly.com/library/view/machine-learning-engineering/9781492067850/) — Valliappa Lakshmanan, Sara Robinson, Michael Hyttinen (2020)
  Publisher: O'Reilly Media
  一本关于在生产环境中构建、部署和维护机器学习系统的综合指南，涵盖了 MLOps 实践、模型版本控制、部署模式（蓝绿、金丝雀）和监控等主题。
- [MLflow: An Open Platform for the Machine Learning Lifecycle](https://www.cidrdb.org/cidr2020/papers/p11-zaharia-cidr20-paper.pdf) — Matei Zaharia, Andy Konwinski, Patrick Wendell, Burak Yavuz, Tathagata Das, Joe Spisak, Jeremy Chow, Virginia Adams, Michael Armbrust, Sameer Agarwal, Xiangrui Meng, Aaron Davidson, Ali Ghodsi (2020)
  Journal: Proceedings of the 10th Biennial Conference on Innovative Data Systems Research (CIDR '20); Publisher: CIDR; Pages: Article No. 11; DOI: [10.54412/cidr2020-11](https://doi.org/10.54412/cidr2020-11)
  介绍 MLflow，一个开源平台，旨在管理机器学习生命周期，包括实验跟踪、模型打包和可重现部署，这与版本控制和工件管理相关。
