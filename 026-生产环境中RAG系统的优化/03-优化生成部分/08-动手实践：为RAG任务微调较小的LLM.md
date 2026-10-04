# 动手实践：为RAG任务微调较小的LLM

来源：[原文](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-3-optimizing-rag-generation/hands-on-finetune-small-llm-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本次实践练习将指导您专门针对RAG相关任务微调 (fine-tuning)一个较小的LLM（大语言模型 (LLM)）。微调较小的模型在性能、成本效益和速度之间提供了令人满意的平衡，使其成为生产环境中有吸引力的选择。通过根据RAG管道中您特定的生成需求调整模型，通常可以比使用更大的通用模型获得更好的结果，尤其是在与检索到的上下文 (context)保持事实一致性方面。

我们的目标是获取一个预训练 (pre-training)的较小LLM，并对其进行调整，使其能够根据提供的上下文和用户查询生成答案，这是RAG系统中的一个常见任务。我们将侧重使用参数 (parameter)高效微调（PEFT）技术，例如低秩适应（LoRA），以使此过程在计算上可行且高效。

### 前提条件

开始之前，请确保您拥有以下内容：

1. **Python环境**：Python 3.8或更高版本。
2. **库**：
   - `transformers`：用于访问预训练 (pre-training)模型和分词 (tokenization)器 (tokenizer)。
   - `datasets`：用于处理和准备数据。
   - `peft`：用于应用LoRA或其他PEFT方法。
   - `accelerate`：简化PyTorch训练在各种硬件上的运行。
   - `torch`：PyTorch库。
   - `evaluate` 和 `rouge_score`：用于评估指标。
     您可以使用pip安装这些库：

pip install transformers datasets peft accelerate torch evaluate rouge\_score
\`\`\`
3. **基础模型**：我们将使用Hugging Face上一个相对较小但功能齐全的模型，例如`t5-small`。它非常适用于条件生成任务。

4. **数据集**：本次练习需要一个数据集，其中每个例子包含一个上下文 (context)、一个问题以及一个从上下文中得出的真实答案。您可以调整SQuAD等问答数据集的子集，或创建一个小型自定义数据集。格式应为：

   ```json
   [
       {
       "context": "Kuala Lumpur is the capital city of Malaysia, known for its iconic Petronas Twin Towers. It is also a major economic and cultural hub in Southeast Asia.",
       "question": "What is the capital of Malaysia?",
       "answer": "Kuala Lumpur is the capital city of Malaysia."
   },
   {
       "context": "The Petronas Twin Towers were once the tallest buildings and remain an architectural marvel in Kuala Lumpur, Malaysia. They were completed in 1998.",
       "question": "When were the Petronas Twin Towers completed?",
       "answer": "The Petronas Twin Towers were completed in 1998."
   },
   // ... more examples
   ]
   ```

### 步骤1：模型和任务选择

我们已选择`t5-small`作为基础模型。T5是一个编码器-解码器模型，在无监督和有监督任务的多任务混合上进行了预训练 (pre-training)，使其能够灵活应对各种生成任务。任务是：给定一个`context`和一个`question`，生成一个在`context`中有事实依据的`answer`。我们需要适当地格式化T5模型的输入，通常是通过在输入字符串前添加任务特定指令，例如“answer the question based on the context:”。

### 步骤2：准备微调 (fine-tuning)数据集

数据集准备是一个重要步骤。您的微调数据质量直接影响专用LLM的性能。

1. **加载您的数据集**：如果您有如上所述的JSON文件，可以使用`datasets`库加载它。

   ```python
   from datasets import load_dataset

   # Assuming your data is in 'rag_finetuning_data.jsonl'
   # For demonstration, let's create a dummy dataset
   data_files = {"train": "path_to_your_train_data.jsonl", "validation": "path_to_your_validation_data.jsonl"}
   # raw_datasets = load_dataset("json", data_files=data_files)
   # For this example, let's use a small dummy dataset
   # In a real scenario, you'd load your actual dataset
   from datasets import Dataset
   dummy_data = {
       "train": Dataset.from_list([
           {"id": "1", "context": "Kuala Lumpur is the capital city of Malaysia, known for its iconic Petronas Twin Towers.", "question": "What is the capital of Malaysia?", "answer": "Kuala Lumpur is the capital city of Malaysia."},
           {"id": "2", "context": "The Petronas Twin Towers were once the tallest buildings and remain an architectural marvel in Kuala Lumpur, Malaysia. They were completed in 1998.", "question": "When were the Petronas Twin Towers completed?", "answer": "The Petronas Twin Towers were completed in 1998."}
       ]),
       "validation": Dataset.from_list([
           {"id": "3", "context": "Langkawi is an archipelago of 99 islands in the Andaman Sea, off the west coast of Malaysia.", "question": "Where is Langkawi located?", "answer": "Langkawi is located off the west coast of Malaysia."}
       ])
   }
   raw_datasets = dummy_data
   ```
2. **分词 (tokenization)和格式化**：我们需要对输入（上下文 (context)和问题）和目标（答案）进行分词。对于T5，常见做法是拼接上下文和问题，通常会带有一个前缀。

   ```python
   from transformers import AutoTokenizer

   model_checkpoint = "t5-small"
   tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

   max_input_length = 512  # Adjust based on your context lengths
   max_target_length = 64   # Adjust based on your answer lengths
   prefix = "answer the question based on the context: "

   def preprocess_function(examples):
       inputs = [prefix + "question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
       model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

       # Tokenize targets
       with tokenizer.as_target_tokenizer():
           labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True, padding="max_length")

       model_inputs["labels"] = labels["input_ids"]
       # Ensure -100 is used for padding tokens in labels so they are ignored in the loss function
       for i, label_ids in enumerate(model_inputs["labels"]):
           model_inputs["labels"][i] = [lid if lid != tokenizer.pad_token_id else -100 for lid in label_ids]

       return model_inputs

   tokenized_datasets = {
       "train": raw_datasets["train"].map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names),
       "validation": raw_datasets["validation"].map(preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names)
   }
   ```

   `prefix`帮助模型理解任务。拼接“question: ”和“context: ”明确地为模型标记 (token)了这些部分。对标签进行分词时，`tokenizer.as_target_tokenizer()`用于序列到序列模型。填充的标签标记被设为-100，以便损失函数 (loss function)忽略它们。

### 步骤3：使用PEFT（LoRA）进行微调 (fine-tuning)

即使是“小型”LLM的完全微调也可能资源密集。参数 (parameter)高效微调（PEFT）方法，如LoRA，允许我们只微调模型参数的一小部分，从而大幅减少计算需求和存储。

1. **加载基础模型**：

   ```python
   from transformers import AutoModelForSeq2SeqLM

   model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
   ```
2. **配置LoRA**：

   ```python
   from peft import LoraConfig, get_peft_model, TaskType

   lora_config = LoraConfig(
       r=16,  # Rank of the update matrices. Higher rank means more parameters.
       lora_alpha=32,  # Alpha scaling factor.
       target_modules=["q", "v"],  # Apply LoRA to query and value weights in attention
       lora_dropout=0.05,
       bias="none",
       task_type=TaskType.SEQ_2_SEQ_LM # For sequence-to-sequence models like T5
   )

   peft_model = get_peft_model(model, lora_config)
   peft_model.print_trainable_parameters()
   # Example output: trainable params: XXXX || all params: YYYYYY || trainable%: Z.ZZZZ
   ```

   此配置指定LoRA将应用于注意力层中的查询（`q`）和值（`v`）投影矩阵。`print_trainable_parameters`方法将展示LoRA如何大幅减少需要更新的参数数量。
3. **设置训练参数和训练器**：

   ```python
   from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

   data_collator = DataCollatorForSeq2Seq(
       tokenizer,
       model=peft_model,
       label_pad_token_id=-100, # Important for ignoring padding in loss calculation
       pad_to_multiple_of=8 # Optional: for TPU efficiency
   )

   output_dir = "t5-small-rag-finetuned-lora"

   training_args = Seq2SeqTrainingArguments(
       output_dir=output_dir,
       per_device_train_batch_size=4, # Adjust based on your GPU memory
       per_device_eval_batch_size=4,  # Adjust based on your GPU memory
       gradient_accumulation_steps=4, # Effective batch size = 4 * 4 = 16
       learning_rate=1e-4, # Common learning rate for LoRA
       num_train_epochs=3,   # Adjust as needed
       logging_dir=f"{output_dir}/logs",
       logging_steps=10,
       evaluation_strategy="epoch", # Evaluate at the end of each epoch
       save_strategy="epoch",       # Save model at the end of each epoch
       load_best_model_at_end=True,
       metric_for_best_model="eval_loss", # Or a ROUGE score if you implement compute_metrics
       predict_with_generate=True, # Necessary for generating text during evaluation
       fp16=True, # Enable mixed-precision training if GPU supports it
   )

   trainer = Seq2SeqTrainer(
       model=peft_model,
       args=training_args,
       train_dataset=tokenized_datasets["train"],
       eval_dataset=tokenized_datasets["validation"],
       tokenizer=tokenizer,
       data_collator=data_collator,
   )
   ```
4. **开始训练**：

trainer.train()
\`\`\`
这将启动微调过程。请监控训练和验证损失。

5. **保存PEFT适配器**：训练后，只需保存LoRA适配器权重 (weight)，它们非常小。

   ```python
   peft_model.save_pretrained(f"{output_dir}/best_lora_adapter")
   # You can also save the tokenizer for convenience
   tokenizer.save_pretrained(f"{output_dir}/best_lora_adapter")
   ```

### 步骤4：评估

评估微调 (fine-tuning)后的模型非常重要。我们需要评估其根据所提供的上下文 (context)生成准确且相关联答案的能力。

1. **为训练器定义`compute_metrics`函数（可选但建议）**：您可以使用ROUGE等指标，它衡量生成答案与参考答案之间的重叠度。

   ```python
   import numpy as np
   import evaluate

   rouge_metric = evaluate.load("rouge")

   def compute_metrics(eval_preds):
       preds, labels = eval_preds
       if isinstance(preds, tuple):
           preds = preds[0]

       # Decode generated summaries, handling -100 pads
       decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

       # Replace -100 in the labels as we can't decode them.
       labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
       decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

       # ROUGE expects a newline after each sentence
       decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
       decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

       result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

       # Extract specific ROUGE scores
       result = {key: value * 100 for key, value in result.items()}
       prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
       result["gen_len"] = np.mean(prediction_lens)
       return result

   # Re-initialize trainer with compute_metrics if you define it
   # trainer = Seq2SeqTrainer(
   #     ...
   #     compute_metrics=compute_metrics,
   #     ...
   # )
   # Then call trainer.evaluate() or it will be called during training if evaluation_strategy is set.
   ```

   如果您将`compute_metrics`整合到`Seq2SeqTrainer`中，它将在评估阶段自动计算这些指标。对于本次演练，我们侧重于`eval_loss`来选择最佳模型。
2. **定性评估**：对生成结果进行定性检查。

   - 加载微调后的适配器：

     ```python
     from peft import PeftModel

     # Load the base model
     base_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
     # Load the LoRA adapter
     fine_tuned_model = PeftModel.from_pretrained(base_model, f"{output_dir}/best_lora_adapter")
     fine_tuned_model = fine_tuned_model.to("cuda" if torch.cuda.is_available() else "cpu") # Move to device
     fine_tuned_model.eval() # Set to evaluation mode

     # If you saved the tokenizer along with the adapter:
     # fine_tuned_tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}/best_lora_adapter")
     # else, use the original tokenizer
     fine_tuned_tokenizer = tokenizer
     ```
   - 为一些测试示例生成答案：

     ```python
     test_context = "The Atacama Desert is a desert plateau in South America covering a 1,600 km strip of land on the Pacific coast, west of the Andes Mountains. It is the driest nonpolar desert."
     test_question = "What is the Atacama Desert?"

     input_text = f"{prefix}question: {test_question} context: {test_context}"
     input_ids = fine_tuned_tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).input_ids.to(fine_tuned_model.device)

     outputs = fine_tuned_model.generate(input_ids, max_length=max_target_length, num_beams=4, early_stopping=True)
     generated_answer = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)

     print(f"Context: {test_context}")
     print(f"Question: {test_question}")
     print(f"Generated Answer: {generated_answer}")
     ```

   检查答案是否：

   - 根据上下文事实正确。
   - 与问题相关联。
   - 流畅且连贯。
   - 避免幻觉 (hallucination)（即不引入上下文中不存在的信息）。
3. **比较**：将这些输出与以下内容进行比较：

   - 基础`t5-small`模型（未经微调）。
   - 真实答案。

   如果您有标注测试集，简单的定量比较可能涉及ROUGE分数：

   
   
   [交互图表：微调对答案质量的影响（模拟ROUGE-L）](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-3-optimizing-rag-generation/hands-on-finetune-small-llm-rag#plot-1q3kf8a)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "layout": {
       "title": {
         "text": "微调对答案质量的影响（模拟ROUGE-L）"
       },
       "xaxis": {
         "title": {
           "text": "模型版本"
         }
       },
       "yaxis": {
         "title": {
           "text": "ROUGE-L分数"
         },
         "range": [
           0,
           1
         ]
       }
     },
     "data": [
       {
         "x": [
           "T5-小型（基础版）",
           "T5-小型（RAG LoRA微调版）"
         ],
         "y": [
           0.32,
           0.58
         ],
         "type": "bar",
         "name": "ROUGE-L",
         "marker": {
           "color": [
             "#748ffc",
             "#40c057"
           ]
         }
       }
     ]
   }
   ```
   
   </details>
   
   

   > 模拟的ROUGE-L分数显示了RAG专用微调后的潜在提升。实际结果将根据数据和训练情况有所不同。

### 步骤5：集成与后续考量

对微调 (fine-tuning)后的较小LLM满意后：

1. **集成**：此模型（基础模型 + LoRA适配器）现在可以替换RAG管道中的生成器组件。进行预测时，加载基础模型，然后应用训练好的LoRA权重 (weight)。
2. **效率**：您很可能获得了一个模型，与更大的通用LLM相比，它在推理 (inference)时更快，所需的计算能力更少，同时可能为您的特定RAG任务提供更好、更上下文 (context)感知的生成。
3. **迭代改进**：
   - **数据增强**：如果性能不理想，考虑增加微调数据集，加入更多样化的例子或针对特定故障模式的例子。
   - **超参数 (parameter) (hyperparameter)调整**：尝试LoRA配置（`r`、`lora_alpha`）、学习率以及其他训练参数。
   - **任务适应**：如果您的RAG系统需要不同的生成风格（例如，摘要与直接问答），您可以对单独的适配器进行微调，或在混合任务数据集上微调单个适配器。
4. **灾难性遗忘**：请注意，微调有时可能导致模型“遗忘”其部分通用能力。如果您的RAG任务需要广泛知识同时遵守上下文，请确保您的微调数据和评估涵盖这一点。与完全微调相比，LoRA由于基础模型权重是冻结的，因此固有一些程度地缓解了这种情况。

本次实践练习演示了一种优化RAG系统生成组件的技术。通过使用PEFT方法微调像T5-small这样的较小LLM，您可以创建根据您的生产需求量身定制的专用、高效且有效的生成器，从而带来更高质量的输出和更好的资源使用。这种方法是构建可维护RAG解决方案的一个步骤。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种参数高效的微调技术，可显著减少大型语言模型的可训练参数数量，使微调更易于实现。
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了应用LoRA等参数高效微调方法的实践指南和API参考。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474
  介绍了检索增强生成（RAG）框架的基础论文，该框架结合了预训练参数和非参数记忆，用于开放域问答。
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) — Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2020)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 21; Pages: 1-67
  介绍了T5（Text-to-Text Transfer Transformer）模型，该模型将所有NLP问题视为文本到文本任务，并作为各种生成任务的通用基础模型。

---

[上一节](07-%E7%94%9F%E6%88%90%E5%86%85%E5%AE%B9%E8%B4%A8%E9%87%8F%E7%9A%84%E7%94%9F%E4%BA%A7%E8%AF%84%E4%BC%B0.md) · [下一节](../04-%E7%AB%AF%E5%88%B0%E7%AB%AF%20RAG%20%E7%B3%BB%E7%BB%9F%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/01-%E5%88%86%E6%9E%90%E4%B8%8E%E9%99%8D%E4%BD%8ERAG%E7%B3%BB%E7%BB%9F%E5%BB%B6%E8%BF%9F.md)
