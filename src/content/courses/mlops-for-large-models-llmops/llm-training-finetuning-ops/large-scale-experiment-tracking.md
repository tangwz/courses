---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "large-scale-experiment-tracking"
sourceId: 3151
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/large-scale-experiment-tracking"
title: "大规模运行的实验跟踪"
description: "针对LLM训练和微调的规模，调整实验跟踪工具和方法。"
order: 6
plots: []
sourceHash: "a6c29f43df99c83e1c8a4413184896ae5b0e17d602b3a32831f49be94b3cc4e2"
sourceCorrections: []
---

训练大型语言模型是一个迭代过程，通常涉及数十甚至数百次实验运行，以找到最佳架构、超参数 (parameter) (hyperparameter)和训练配置。鉴于这些运行会消耗大量的计算资源（可能数百个GPU，持续数天或数周），并生成数TB的数据（检查点、日志），因此细致的实验跟踪变得必不可少。面对LLM训练和微调 (fine-tuning)的庞大规模和复杂性时，用于小型模型的标准跟踪方法往往不足。

使实验跟踪适应大规模运行需要解决几个增加的挑战：产生的指标和工件量庞大、实验持续时间长、分布式环境的复杂性，以及在这种复杂性中对可重现性的重要要求。未能有效地跟踪实验会导致计算周期浪费、难以调试如发散或性能瓶颈之类的问题，以及无法可靠地比较不同的方法。

### 要跟踪什么：指标

虽然像损失和评估分数这样的标准指标仍然很重要，但大型模型的跟踪必须包含更广泛的信息，以反映该过程的分布式特性和资源密集度。

- **超参数 (parameter) (hyperparameter)和配置：** 这不仅仅是学习率和批量大小。您必须细致地记录：

  - **模型超参数：** 层数、隐藏维度、注意力头等。
  - **优化参数：** 学习率调度、优化器（AdamW等）、权重 (weight)衰减、梯度裁剪阈值。
  - **数据集详情：** 数据集标识符、版本、预处理步骤、分词 (tokenization)配置。
  - **分布式训练设置：** 数据并行度（`world_size`）、张量并行度、流水线并行度、每个GPU的微批量大小、梯度累积步数。
  - **框架配置：** DeepSpeed（例如，ZeRO优化阶段、CPU/NVMe卸载标志）或Megatron-LM等框架的特定设置。
  - **PEFT配置（如果适用）：** 所用技术（LoRA、适配器等）、秩、alpha、目标模块、丢弃率。
  - **环境详情：** 代码版本（Git提交哈希）、库版本（PyTorch、Transformers、DeepSpeed）、底层硬件配置（GPU类型、数量）。
- **系统资源指标：** 理解资源消耗对于优化和调试瓶颈非常重要。跟踪以下指标：

  - **GPU利用率：** 每个GPU的利用率百分比以及GPU之间的平均利用率。
  - **GPU内存使用：** 每个GPU分配和保留的内存。高内存使用可能表示潜在的OOM错误或低效配置。
  - **CPU利用率：** 每个节点的总CPU使用率。
  - **网络带宽：** 节点之间的数据传输速率，对于流水线并行和梯度同步尤其重要。高延迟或低带宽会严重限制训练。
  - **磁盘I/O：** 读/写速率，与数据加载和检查点保存特别相关。
- **模型训练指标：** 足够频繁地跟踪指标以观察趋势，但不要过于频繁以至于使日志系统不堪重负。

  - **损失：** 训练损失（每批次或N步平均）、验证损失（定期）。
  - **学习率：** 实际使用的学习率（对于复杂的调度器尤其重要）。
  - **梯度范数：** 裁剪前梯度的总体范数。突增或爆炸可能表示不稳定。$L_2$范数是常见的：$||\nabla L||_2 = \sqrt{\sum_{i} (\frac{\partial L}{\partial w_i})^2}$
  - **吞吐量 (throughput)：** 每秒处理的训练样本或token数。这是一个衡量整体训练效率的重要指标。
- **评估指标：** 在保留数据集上定期跟踪：

  - 特定任务指标（准确率、F1、BLEU、ROUGE等）。
  - 困惑度（用于语言建模）。
- **工件：** 存储或链接到重要的输出：

  - **模型检查点：** 定期保存检查点并记录其存储位置（例如，云存储中的路径）。
  - **配置文件：** 用于启动运行的精确配置文件。
  - **日志：** 完整的训练日志，可能单独存储以便详细调试。
  - **样本输出：** 偶尔使用模型在不同训练阶段生成样本文本。

### 解决与规模相关的跟踪挑战

标准实验跟踪工具在LLM训练的负载下可能会遇到困难。以下是常见的挑战和缓解策略：

- **高数据量：** 在数百个GPU上每一步记录指标会生成海量数据。

  - **解决方案：聚合和采样：** 减少详细指标的记录频率（例如，每10或100步）。以较低频率（例如，每分钟）记录系统指标。在记录之前，聚合分布式工作器之间的指标（例如，数据并行秩的平均损失）。
  - **解决方案：结构化日志：** 对日志使用结构化格式（如JSON），使其更容易解析和查询。
  - **解决方案：专用平台：** 使用为规模而构建的实验跟踪平台（例如，MLflow、Weights & Biases、Comet ML、ClearML、Neptune.ai）。这些平台提供后端存储、高效API以及设计用于处理大量运行和指标的用户界面。
- **分布式复杂性：** 协调数百个在多台机器上运行的进程的日志并非易事。

  - **解决方案：集中式跟踪服务：** 使用带有中央服务器的跟踪平台，所有工作器都可以将它们的日志和指标发送到那里。
  - **解决方案：基于秩的日志记录：** 指定一个单独的秩（通常是秩0）来记录聚合或主要指标（如总体损失、评估分数）。其他秩可能只记录其节点或GPU特有的系统指标，或者只记录错误。许多分布式训练框架提供实用程序或集成来促进这一点。例如，您可以封装日志调用：

    ```python
    # 使用分布式库 'dist' 的示例
    # 和跟踪库 'tracker'

    import my_distributed_lib as dist
    import my_tracker_lib as tracker

    # 初始化跟踪器（例如，tracker.init(project="llm-training")）

    if dist.get_rank() == 0:
        # 只记录一次超参数
        tracker.log_params(hyperparameters)
        tracker.log_config(distributed_config)

    # 在训练循环内
    loss = calculate_loss()
    aggregated_loss = dist.average(loss) # 计算所有秩上的平均损失

    if dist.get_rank() == 0:
        tracker.log_metric("train_loss", aggregated_loss, step=global_step)
        if global_step % log_interval == 0:
             # 记录其他秩0特有的指标，如学习率
             tracker.log_metric("learning_rate", optimizer.get_lr(), step=global_step)

    # 如果需要，记录特定秩的指标（例如GPU温度）——可能频率较低
    if global_step % system_log_interval == 0:
         gpu_temp = get_gpu_temperature()
         tracker.log_metric(f"gpu_temp_rank_{dist.get_rank()}", gpu_temp, step=global_step)
    ```
- **长时间运行和容错性：** 运行数周的实验容易受到硬件故障或瞬时问题的影响。

  - **解决方案：与检查点集成：** 确保当作业从检查点恢复时，实验跟踪也能正确恢复，将新日志与原始运行实例关联起来。跟踪平台通常提供一个运行ID，可以在重启时持久化和重复使用。
  - **解决方案：实时监控：** 使用跟踪平台提供的仪表板实时监控进度，从而及早发现如发散或训练停滞等问题。
- **可重现性：** 鉴于众多的配置参数 (parameter)和依赖项，重现特定的运行可能很困难。

  - **解决方案：全面日志记录：** 记录所有内容：代码版本（Git哈希）、精确的配置文件、库版本（通过`requirements.txt`或容器镜像哈希捕获）、数据集标识符以及硬件设置。
  - **解决方案：容器化：** 将整个训练环境（包括依赖项）打包到容器镜像中（例如Docker）。记录用于运行的容器镜像标签或摘要。

### 分析大规模实验

细致跟踪的价值在于能够有效地分析和比较运行。实验跟踪平台提供强大的可视化工具：

- **指标比较：** 绘制多个运行中的训练损失、验证困惑度或系统利用率曲线，有助于识别不同超参数 (parameter) (hyperparameter)或配置的影响。

```plotly
{"data":[{"type":"scatter","mode":"lines","name":"运行A (学习率=1e-4)","x":[0,100,200,300,400,500],"y":[3.5,2.1,1.8,1.6,1.5,1.45],"line":{"color":"#4263eb"}},{"type":"scatter","mode":"lines","name":"运行B (学习率=3e-4)","x":[0,100,200,300,400,500],"y":[3.6,1.9,1.5,1.3,1.2,1.18],"line":{"color":"#12b886"}},{"type":"scatter","mode":"lines","name":"运行C (学习率=1e-5)","x":[0,100,200,300,400,500],"y":[3.4,2.8,2.5,2.3,2.2,2.15],"line":{"color":"#f76707"}}],"layout":{"title":"训练损失比较","xaxis":{"title":"训练步数"},{"title":"Loss"},"template":"plotly_white"}}
```

> 比较不同学习率的运行的训练损失曲线。运行B最初收敛最快，但运行A后期可能提供更高的稳定性。运行C收敛速度慢得多。

- **超参数重要性：** 工具通常提供平行坐标图或参数重要性分析等可视化功能，以帮助理解哪些超参数对结果指标影响最大。
- **资源使用分析：** 比较采用不同并行策略（例如，ZeRO Stage 2 对比 Stage 3）的运行的GPU内存使用或网络带宽，可以显示性能瓶颈或效率提升。
- **工件浏览：** 轻松访问和比较不同运行期间生成的配置文件、日志甚至样本输出。

总之，大型语言模型的实验跟踪不仅仅是日志记录；它是一种管理复杂性的系统方法。它需要仔细规划要跟踪的内容，选择能够处理该规模的合适工具，将跟踪集成到分布式训练工作流中，并使用收集到的数据进行有见地的分析。这种系统方法对于高效迭代、有效调试以及最终在大型模型的开发和微调 (fine-tuning)中取得成功非常重要。

## 参考资料

- [Weights & Biases Documentation: Tracking Large Language Models](https://docs.wandb.ai/guides) — Weights & Biases (Accessed 2025)
  Publisher: Weights & Biases
  官方文档，为大型语言模型实验跟踪提供了实用指南和最佳实践，涵盖了指标、工件和系统健康状况。
- [DeepSpeed: System Optimizations for Large-Scale Model Training](https://arxiv.org/abs/2008.01666) — Samyam Rajbhandari, Cong Guo, Jeff Rasley, Shaden Smith, Yuxiong He (2020)
  Journal: arXiv preprint arXiv:2008.01666; DOI: [10.48550/arXiv.2008.01666](https://doi.org/10.48550/arXiv.2008.01666)
  介绍了DeepSpeed，一个分布式训练框架，其优化（如ZeRO）需要详细跟踪内容中讨论的分布式配置和资源利用。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于MLOps系统设计的综合性书籍，其中包含专门讨论实验跟踪、基础设施和管理大规模机器学习模型开发复杂性的章节。
- [Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Anna Stooke, Erin Cooke, Scott Clark, Allie Schmidt, Aditya Ramesh, Andy Jones, Chris McMahon, Ambrose Slone, Chris Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; Pages: 1877-1901; DOI: [10.55989/nips.2020.01633](https://doi.org/10.55989/nips.2020.01633)
  详细介绍了GPT-3的架构和训练过程，展示了LLM训练的巨大规模以及为此类模型跟踪大量超参数和配置的重要性。
