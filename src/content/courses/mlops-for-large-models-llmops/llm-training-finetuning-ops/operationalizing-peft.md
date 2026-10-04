---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "operationalizing-peft"
sourceId: 3148
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/operationalizing-peft"
title: "参数高效微调（PEFT）的实施"
description: "将LoRA等PEFT方法整合到MLOps流程中，实现模型的高效适应。"
order: 5
plots: ["plots/3148-0.json"]
sourceHash: "bb310b2106b07a725ab3eb851c6825966d9c794946915670bec3e82bb40f5878"
sourceCorrections: []
---

大型语言模型（LLM）的全面微调 (fine-tuning)可以提供高度定制化，但它需要大量的计算资源和时间，这与模型预训练 (pre-training)面临的挑战类似。参数 (parameter)高效微调（PEFT）方法提供了一个很有益的替代方案，它只更新模型参数的一小部分，从而大幅减少计算需求、内存占用和训练时间。低秩适应（LoRA）、Adapter模块和前缀微调等技术使机构能够以更高的效率调整大型基础模型，以适应特定任务。实施PEFT包括将这些技术整合到现有的MLOps工作流程中，管理其特有的产物，并调整部署策略。

### 将PEFT整合到MLOps流程中

成功实施PEFT意味着不应仅将其视为一种建模方法，而应作为MLOps流程中的一个标准组成部分。这需要调整以下几个阶段：

1. **配置管理：** PEFT引入了新的超参数 (parameter) (hyperparameter)，必须与标准训练配置一同管理。对于LoRA，这包括秩（$r$）、缩放因子（$\alpha$）、dropout概率，以及基础模型中目标模块（例如，注意力查询层和值层）的指定。使用YAML或JSON等格式系统地存储这些配置，并在实验管理系统中细致地记录它们。

   ```yaml
   # PEFT配置示例 (LoRA)
   base_model: "meta-llama/Llama-2-7b-hf"
   dataset: "task_specific_data_v1.jsonl"
   output_dir: "./results/llama2-7b-lora-task1"

   training_args:
     per_device_train_batch_size: 4
     gradient_accumulation_steps: 8
     num_train_epochs: 3
     learning_rate: 3e-4
     logging_steps: 10

   peft_config:
     peft_type: "LORA"
     r: 16            # LoRA秩
     lora_alpha: 32   # LoRA缩放因子
     lora_dropout: 0.05
     target_modules:  # 应用LoRA的模块
       - "q_proj"
       - "v_proj"
   ```
2. **代码实现：** Hugging Face的`peft`等库简化了各种PEFT方法的应用。集成通常涉及加载基础模型，然后应用PEFT配置来包装目标层。

   ```python
   # 使用Hugging Face `peft`的简化示例
   from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
   from peft import LoraConfig, get_peft_model, TaskType
   import datasets

   # 加载基础模型和分词器
   model_name = "meta-llama/Llama-2-7b-hf"
   model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") # 对大型模型使用device_map
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   # 加载并预处理数据集（省略细节）
   # tokenized_datasets = datasets.load_dataset(...)

   # 定义LoRA配置
   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"], # 示例目标模块
       lora_dropout=0.05,
       bias="none",
       task_type=TaskType.CAUSAL_LM
   )

   # 将PEFT应用于基础模型
   peft_model = get_peft_model(model, lora_config)
   peft_model.print_trainable_parameters()
   # 输出可能显示：可训练参数：4,194,304 || 所有参数：6,742,609,920 || 可训练百分比：0.0622

   # 定义训练参数（标准）
   training_args = TrainingArguments(
       output_dir="./lora-finetune-output",
       per_device_train_batch_size: 4,
       num_train_epochs: 1,
       # ... 其他参数
   )

   # 使用PEFT模型初始化Trainer
   trainer = Trainer(
       model=peft_model, # 训练PEFT模型
       args=training_args,
       # train_dataset=tokenized_datasets["train"],
       # ... 其他Trainer参数
   )

   # 开始微调
   # trainer.train()

   # 只保存适配器权重
   # peft_model.save_pretrained("./my-lora-adapters")
   ```

   这里的运行差异在于，`trainer.train()`只更新PEFT引入的少量参数，与全面微调 (fine-tuning)相比，大幅减少了计算和内存需求。
3. **产物管理：** PEFT引入了一种不同类型的模型产物：少量已训练的适配器权重 (weight)，而非完整的模型检查点。您的MLOps系统必须有效处理这些产物：

   - **存储：** 适配器权重通常很小（兆字节），相比于多吉字节的模型检查点，这简化了存储。
   - **版本控制：** 必须对适配器权重进行版本控制，并明确地将其链接到其训练所用的特定版本基础模型。在推理 (inference)时使用不兼容的基础模型将导致错误或性能不佳。请在模型注册表或产物跟踪系统中存储这种关联。

### PEFT的实验跟踪

实验跟踪平台（如MLflow、Weights & Biases、Comet ML）需要为PEFT进行略微调 (fine-tuning)整：

- **记录PEFT超参数 (parameter) (hyperparameter)：** 确保每次运行都记录`r`、`lora_alpha`、`target_modules`或适配器维度等参数。
- **跟踪可训练参数：** 明确记录可训练参数的数量或百分比，以量化 (quantization)效率提升。
- **性能与效率：** 根据可训练参数数量或微调持续时间，分析任务特定的性能指标，以了解不同PEFT配置的权衡。



![可训练参数比较](plots/3148-0.json)



> 针对一个70亿参数模型，使用不同微调方法时的大致可训练参数数量比较。PEFT方法大幅减少了参数数量。

### 部署考量

部署经过PEFT微调 (fine-tuning)的模型需要仔细考量：

1. **权重 (weight)合并（离线）：** 在部署之前，您可以将已训练的适配器权重合并到基础模型权重中。这会创建一个标准模型检查点，可以无需修改地使用现有推理 (inference)基础设施提供服务。缺点是失去了灵活性；部署后无法轻松更换适配器。

   ```python
   # 示例：合并LoRA权重（使用peft库）
   # from peft import AutoPeftModelForCausalLM

   # 加载PEFT模型（包含基础模型和适配器）
   # model = AutoPeftModelForCausalLM.from_pretrained("./my-lora-adapters", device_map="auto")

   # 合并权重
   # merged_model = model.merge_and_unload()

   # 保存合并后的模型（现在是一个标准Transformer模型）
   # merged_model.save_pretrained("./merged-llama2-lora-task1")
   # tokenizer.save_pretrained("./merged-llama2-lora-task1")
   ```
2. **动态适配器（在线）：** 将基础模型加载到推理服务器中，并在运行时动态应用PEFT适配器权重。这提供了更大的灵活性，允许使用相同的基础模型实例提供多个适配器（例如，用于不同任务）的服务，通过按需加载不同的适配器权重。然而，这可能需要修改推理服务器代码来处理适配器的加载和应用，可能会给初始请求或适配器切换增加少量延迟。Text Generation Inference（TGI）或vLLM等框架正在越来越多地增加对动态适配器加载的支持。

选择这两种方法取决于运行需求。如果每次部署只需要一个专用模型，则合并更简单。动态加载更适用于多任务服务或频繁的适配器更新。版本控制在两种情况下都仍然很重要，以确保正确的基础模型和适配器（或合并模型）一起部署。

### 自动化PEFT工作流程

PEFT非常适合在MLOps的CI/CD流程中进行自动化。一个典型的自动化工作流程可能如下所示：

> 一个自动化流程，用于实施PEFT。触发器启动此过程，加载配置，准备数据，加载基础模型，然后应用并训练PEFT，生成的适配器经过评估和注册（链接到其基础模型），最后使用合并或动态加载方式部署。

这种自动化确保模型能够高效地更新或专门化以适应新任务或数据集，同时将人工干预降到最低，并利用PEFT在成本和速度方面的优势。

### 挑战与良好实践

虽然PEFT简化了微调 (fine-tuning)，但请考量以下几点：

- **技术选择：** 最适合的PEFT方法（LoRA、Adapters、Prompt Tuning等）可能取决于具体任务。通常需要进行实验。
- **超参数 (parameter) (hyperparameter)敏感性：** PEFT方法有其自身的超参数（如LoRA秩`r`），需要调整以获得最佳性能。
- **评估：** 必须在任务特定指标上进行严格评估，以确保PEFT模型符合质量要求。不要假设效率可以直接转化为与全面微调相当的性能。
- **基础模型兼容性：** 始终跟踪适配器训练所使用的基础模型版本。升级基础模型可能需要重新训练适配器。
- **组合性：** 尽管一些研究在探讨适配器的组合使用，但管理同时应用的多个PEFT修改之间的复杂相互作用可能具有挑战性。

通过将PEFT技术周全地整合到MLOps实践中，正确管理配置和产物，并选择合适的部署策略，团队能够有效地为各种应用定制大型模型，同时管理计算成本和运行复杂性。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），一种用于大型语言模型的参数高效方法，减少了可训练参数和计算成本。
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  介绍了适配器模块，一种向预训练模型添加小型任务特定神经网络层的方法，可以在不修改原始模型权重的情况下进行高效微调。
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Xiang Lisa Li, Percy Liang (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2101.00190](https://doi.org/10.48550/arXiv.2101.00190)
  提出了Prefix-Tuning，一种PEFT方法，通过优化一系列连续的任务特定向量（前缀）来条件化大型语言模型。
- [PEFT Library Documentation](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，提供了实现各种参数高效微调技术的指南和API参考。
