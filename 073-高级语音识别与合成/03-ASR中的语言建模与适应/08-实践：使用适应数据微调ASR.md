# 实践：使用适应数据微调ASR

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-3-language-modeling-adaptation-asr/practice-asr-adaptation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

调整预训练 (pre-training)的自动语音识别（ASR）模型以适应特定情况，例如新的说话人或不同的声学环境，通常需要使用相对少量特定目标数据。这个过程，常被称为微调 (fine-tuning)或领域适应，对于在部署条件可能不同于初始模型训练所用的大型通用数据集的场景中，充分提升ASR表现非常有益。

正如本章前面所说，说话人、口音、背景噪声和录音通道的变化会明显降低通用ASR模型的准确性。适应技术旨在通过调整模型参数 (parameter)来减少这种不匹配，这些参数使用代表目标情况的数据。虽然i-vector提取或特征空间最大似然线性回归（fMLLR）等方法在混合系统中很常见，但对端到端深度神经网络 (neural network)模型参数进行微调已成为一种常用且有效的方法。

### 场景与目的

设想你有一个目前表现最好的ASR模型，它在数千小时的各类语音数据上进行了预训练 (pre-training)（我们称之为`base_model`）。你的目标是提升其对特定用户“Alex”的表现，Alex的口音独特；或者将其部署在特定环境，比如嘈杂的呼叫中心。你已经收集到少量（例如1-2小时）Alex的语音或在目标呼叫中心录制的音频数据，以及准确的转录。这就是我们的适应数据集（$D_{adapt}$）。

我们的目标是使用 $D_{adapt}$ 创建一个新模型（`adapted_model`），该模型在Alex的语音（或呼叫中心环境）上的词错误率（WER）相较于`base_model`表现更低。

### 微调 (fine-tuning)方法

微调是指继续`base_model`的训练过程，但只使用适应数据集 $D_{adapt}$。其核心思路是温和地调整模型的参数 (parameter)，使其更符合适应数据中的特征，而不剧烈改变预训练 (pre-training)期间学到的通用知识。

成功进行微调的考量因素包括：

1. **学习率：** 这可能是最重要的超参数 (hyperparameter)。微调所用的学习率应远*低于*原始预训练时使用的学习率（例如，小10到100倍）。通常范围可能是 $1e-5$ 到 $5e-7$，具体取决于模型架构和原始学习率计划。低学习率可避免模型根据小规模适应数据集进行大的、可能具有破坏性的更新，从而保留了从大型预训练语料库中学到的有用信息，并防止灾难性遗忘。
2. **数据量：** 尽管适应可以在数据有限的情况下进行，但通常随着适应样本的增多，表现会得到提升。即使是30分钟到一小时的数据也能带来明显提升，但如果能获得数小时的数据，通常会更好。
3. **微调层：** 你有几种选择：

   - **完全微调：** 更新所有模型参数（编码器、解码器等）。这通常有效，但需要仔细选择学习率。
   - **部分微调：** 冻结一些层（例如，编码器的较低层，它们可能捕获更通用的声学特征），只更新上层或特定组件（如解码器或预测网络，在RNN-T中）的权重 (weight)。这有时会更快、更稳定，尤其是在适应数据非常有限的情况下。
   - **适配器模块：** 在预训练模型中插入小型、新的“适配器”层，只训练这些适配器。这使得原始权重保持冻结，可能使适应更高效和模块化。为简化本次实践，我们将侧重于完全微调。
4. **训练时长：** 微调通常比预训练需要少得多的训练周期或步骤。通常，在适应数据集上进行1-5个周期就足够了。监测 $D_{adapt}$ 的小验证集上的表现，以确定何时停止。

### 实现步骤

尽管具体代码很大程度上取决于你使用的工具包（如ESPnet、NeMo、SpeechBrain或Hugging Face Transformers），大致的工作流程保持一致：

1. **加载预训练 (pre-training)模型：** 实例化你的ASR模型架构，并从预训练的`base_model`检查点加载权重 (weight)。

   ```python
   # 使用工具包的伪代码
   import asr_toolkit

   # 加载模型架构和预训练权重
   config = asr_toolkit.load_config('path/to/base_model_config.yaml')
   model = asr_toolkit.models.ASRModel(config)
   model.load_state_dict(asr_toolkit.load_checkpoint('path/to/base_model.pth'))

   # 确保模型处于训练模式
   model.train()
   ```
2. **准备适应数据：** 为你的适应数据集（$D_{adapt}$）创建数据加载器，如果可能，将其分成训练集和验证集。确保预处理（特征提取、分词 (tokenization)）与`base_model`的训练设置匹配。

   ```python
   # 伪代码
   adapt_train_loader = asr_toolkit.create_dataloader(
       manifest_path='path/to/adapt_train_manifest.json',
       config=config,
       batch_size=16 # 使用合理的批次大小
   )
   adapt_val_loader = asr_toolkit.create_dataloader(
       manifest_path='path/to/adapt_val_manifest.json',
       config=config,
       batch_size=16
   )
   ```
3. **配置优化器：** 设置一个优化器（例如AdamW），并使用非常低的学习率。

   ```python
   # 伪代码
   import torch.optim as optim

   optimizer = optim.AdamW(model.parameters(), lr=5e-6) # 学习率示例
   ```
4. **微调 (fine-tuning)循环：** 针对适应训练数据，迭代少量训练周期。执行前向传播，计算损失（例如CTC损失、Transducer损失、交叉熵损失，具体取决于模型），进行反向传播 (backpropagation)，并更新权重。

   ```python
   # 伪代码
   num_epochs = 3 # 示例：微调3个周期

   for epoch in range(num_epochs):
       model.train()
       for batch in adapt_train_loader:
           audio_signals, audio_lengths, transcripts, transcript_lengths = batch

           optimizer.zero_grad()

           # 前向传播
           log_probs, encoded_lengths = model.forward(
               input_signal=audio_signals,
               input_signal_length=audio_lengths
           )

           # 计算损失（具体取决于模型类型）
           loss = model.calculate_loss(
               log_probs=log_probs,
               encoded_lengths=encoded_lengths,
               targets=transcripts,
               target_lengths=transcript_lengths
           )

           loss.backward()
           optimizer.step()

       # 验证（可选，但推荐）
       model.eval()
       with torch.no_grad():
           # 在 adapt_val_loader 上计算 WER 或损失
           val_wer = evaluate_model(model, adapt_val_loader)
           print(f"Epoch {epoch+1}, Validation WER: {val_wer:.2f}")

   # 保存适应后的模型
   asr_toolkit.save_checkpoint(model, 'path/to/adapted_model.pth')
   ```

   *注意：`evaluate_model` 是工具包评估功能的占位符。*
5. **评估：** 微调后，严格评估表现差异。使用`base_model`和`adapted_model`，解码来自目标场景的独立测试集（例如，Alex的未见话语或呼叫中心的音频）。计算两者的WER。

### 预期结果

你应该会看到，在使用`adapted_model`时，目标场景测试集上的WER相较于`base_model`有所降低。提升的程度取决于适应数据的数量和质量、预训练 (pre-training)和适应场景之间的相似性，以及微调 (fine-tuning)的超参数 (parameter) (hyperparameter)。



[交互图表：适应后WER的提升](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-3-language-modeling-adaptation-asr/practice-asr-adaptation#plot-8v59ib)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "基础模型",
        "适应模型"
      ],
      "y": [
        25.3,
        14.8
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#4c6ef5",
          "#20c997"
        ]
      },
      "name": "词错误率"
    }
  ],
  "layout": {
    "title": "适应后WER的提升",
    "yaxis": {
      "title": "词错误率 (%)"
    },
    "xaxis": {
      "title": "模型"
    },
    "bargap": 0.3
  }
}
```

</details>



> 在目标场景测试集上，使用适应数据微调前后的词错误率对比。

### 进一步考量

- **多任务适应：** 如果同时适应多个说话人或情况，你可以在微调 (fine-tuning)期间考虑多任务学习设置。
- **资源限制：** 如果微调的计算资源有限，适配器模块或仅微调最终层等技术可能更可行。
- **工具包：** 请查阅你所选语音处理工具包（ESPnet、NeMo等）的文档，获取模型微调和适应的具体脚本和方案。它们通常提供即用型示例。

“这个实践练习说明了一种基本技术，用于根据特定需求调整ASR系统。通过仔细应用微调和相关的适应数据，你可以在目标部署场景中显著提高识别准确率，从而弥合通用模型与特定需求之间的差距。”

## 参考资料

- [Efficient Transfer Learning for End-to-End Speech Recognition](https://doi.org/10.21437/Interspeech.2020-2139) — Raghava Krishna, S. R. K. Mahata (2020)
  Journal: Interspeech 2020; Publisher: International Speech Communication Association (ISCA); Pages: 3775-3779; DOI: [10.21437/Interspeech.2020-2139](https://doi.org/10.21437/Interspeech.2020-2139)
  讨论端到端ASR的迁移学习的各个方面，包括微调策略和超参数考量，与本节内容直接相关。
- [Speech and Language Processing: An Introduction to Computational Linguistics and Natural Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Stanford University
  一本全面的教科书，涵盖语音识别的基本概念，包括自适应技术以及从混合系统到端到端系统的演变，提供必要的理论背景。
- [Fine-tuning a pre-trained model for ASR](https://huggingface.co/docs/transformers/tasks/asr_finetuning) — Hugging Face (2024)
  Publisher: Hugging Face
  提供使用Hugging Face Transformers库微调预训练ASR模型的实用、分步指南和代码示例，与实现相关。

---

[上一节](07-%E5%A4%9A%E8%AF%AD%E8%A8%80%E5%92%8C%E8%B7%A8%E8%AF%AD%E8%A8%80%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB.md) · [下一节](../04-%E9%AB%98%E7%BA%A7%E6%96%87%E6%9C%AC%E5%88%B0%E8%AF%AD%E9%9F%B3%E5%90%88%E6%88%90/01-%E8%87%AA%E5%9B%9E%E5%BD%92%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B%20%28Tacotron%2C%20Transformer%20TTS%29.md)
