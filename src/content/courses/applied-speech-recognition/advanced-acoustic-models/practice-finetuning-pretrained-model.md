---
course: "applied-speech-recognition"
chapter: "advanced-acoustic-models"
lesson: "practice-finetuning-pretrained-model"
sourceId: 7147
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models/practice-finetuning-pretrained-model"
title: "实践：微调预训练ASR模型"
description: "关于在特定语音数据集上使用Hugging Face库微调预训练模型的动手操作环节。"
order: 7
plots: []
sourceHash: "68d93250840d277d85066f3da32570dde34f94bc7b6ee557e6f3d5a126d77fb3"
sourceCorrections: []
---

强大的架构，如Transformer，以及Wav2Vec 2.0等预训练 (pre-training)模型具有出色的性能。然而，从零开始训练这些大型模型需要巨大的计算资源和海量数据集。一种更实用且高效的方法是微调 (fine-tuning)预训练模型。此过程使已在数千小时语音上训练的通用模型适应您的特定应用范围或数据集，通常只需少量数据和计算资源即可获得出色表现。

在本动手操作部分，您将使用Hugging Face的`transformers`和`datasets`库，在一个示例数据集上微调预训练的Wav2Vec 2.0模型。此工作流程代表着构建现代ASR系统的标准且高效的办法。

### 环境配置

首先，请确保您已安装所需库。我们将使用`datasets`处理数据加载，`transformers`用于模型和训练流程，`torchaudio`用于音频操作，以及`jiwer`计算词错误率（WER）。

```bash
pip install datasets transformers[torch] torchaudio jiwer
```

> **硬件提示**
>
> 微调 (fine-tuning)，即使是小规模的，也是计算密集型任务。尽管此示例可在CPU上运行几步，但要在合理的时间内完成训练，需要GPU。如果您使用的是Google Colab等基于云的笔记本环境，请务必选择GPU运行时。

### 1. 加载数据集和处理器

我们将从加载数据集和预训练 (pre-training)模型的“处理器”开始。Hugging Face中的处理器是一个便利的对象，它将特征提取器（用于音频）和分词 (tokenization)器 (tokenizer)（用于文本）打包在一起。

对于此示例，我们将使用LibriSpeech数据集的一个小型预处理子集，它在Hugging Face Hub上易于获取。我们还将加载`facebook/wav2vec2-base-960h`的处理器，这是一个热门的Wav2Vec 2.0模型，在960小时的英语语音上预训练。

```python
from datasets import load_dataset
from transformers import AutoProcessor

# 从 Hugging Face Hub 加载一个小型示例数据集
librispeech_sample = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# 加载包含特征提取器和分词器的处理器
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

`processor` 对象现在包含两个主要组件：

- `processor.feature_extractor`: 将原始音频波形转换为模型期望的输入格式（例如，对数梅尔频谱图或原始特征）。
- `processor.tokenizer`: 将目标文本转录内容转换为整数ID序列。

### 2. 微调 (fine-tuning)工作流程

使预训练 (pre-training)模型适应新数据集包含几个不同步骤。模型的基础部分，即学习了通用声学表示的部分，通常被冻结或以非常低的学习率训练。一个新的随机初始化分类头放置在基础模型顶部，用于将声学特征映射到我们新数据集词汇中的字符。然后整个模型在我们的特定数据上进行训练。

> 微调过程通过替换其最终层并在新的特定数据集上进行训练，使其适应通用基础模型。

### 3. 数据预处理

ASR模型需要音频输入和文本标签采用特定格式。音频必须具有一致的采样率，并且文本必须经过分词 (tokenization)。我们的预处理函数将为数据集中的每个例子处理此项。

我们使用的Wav2Vec 2.0模型在16 kHz采样率的音频上预训练 (pre-training)。我们的数据集可能包含不同采样率的音频，因此我们必须先对其进行重采样。

```python
import torch
from datasets import Audio

# 模型期望16kHz采样率的音频
SAMPLING_RATE = 16_000

# 如果需要，对数据集中的音频进行重采样
librispeech_sample = librispeech_sample.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# 定义预处理函数
def prepare_dataset(batch):
    # 提取音频数组
    audio = batch["audio"]

    # 使用处理器从音频数组生成输入值
    # 并从文本转录生成标签
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])

    # input_values 是模型的特征
    # labels 是分词后的文本
    return batch
```

现在，我们使用`.map()`方法将此函数应用于整个数据集。这种方式非常高效，因为它以批次方式处理数据。

```python
# 将预处理函数应用于所有示例
processed_dataset = librispeech_sample.map(
    prepare_dataset, 
    remove_columns=librispeech_sample.column_names
)
```

此步骤后，`processed_dataset`包含两列：`input_values`（声学模型的数值特征）和`labels`（CTC损失函数 (loss function)的整数ID）。

### 4. 配置训练器

Hugging Face `Trainer`是一个强大的类，它抽象了训练循环的复杂性。要使用它，我们需要配置三项内容：一个数据收集器、一组训练参数 (parameter)和一个评估指标。

#### 数据收集器

数据收集器接收单个数据点的列表并将其分组为批次。对于使用CTC的ASR，我们需要一个特殊的数据收集器，它将每个批次中的`input_values`和`labels`填充到最大长度。`input_values`用零填充，而`labels`用-100填充，这是CTC损失函数 (loss function)忽略的值。

```python
from transformers import DataCollatorCTCWithPadding

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
```

#### 评估指标

为了在训练期间监测模型的表现，我们需要一个计算评估指标的函数。ASR的标准指标是词错误率（WER）。我们将使用`jiwer`库来计算它。模型输出token ID，因此我们的函数必须首先将这些ID解码回文本，然后与参考标签进行比较。

```python
import numpy as np
import jiwer

def compute_metrics(pred):
    # 将预测的ID解码为文本
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    # 标签已经是正确格式，但我们需要将-100替换为
    # pad_token_id，以便正确解码。
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)

    # 计算WER
    wer = jiwer.wer(label_str, pred_str)

    return {"wer": wer}
```

#### 训练参数和模型

接下来，我们加载预训练 (pre-training)模型本身。我们使用`AutoModelForCTC`，因为我们的模型将使用CTC损失进行训练。我们还定义了`TrainingArguments`，它指定了学习率、批次大小和训练周期数等超参数 (hyperparameter)。

```python
from transformers import AutoModelForCTC, TrainingArguments, Trainer

# 加载预训练模型
model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./wav2vec2-base-librispeech-demo",
    group_by_length=True,
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,  # 使用混合精度训练以提高速度
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
)
```

### 5. 运行微调 (fine-tuning)过程

所有组件准备就绪后，我们现在可以实例化`Trainer`并开始微调过程。`Trainer`将处理整个循环：向模型馈送批次、计算损失、执行反向传播 (backpropagation)、更新权重 (weight)，并定期评估验证集上的表现。

```python
# 实例化训练器
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset, # 在实际情况中，这将是您的训练集
    eval_dataset=processed_dataset,  # 这将是您的验证集
    tokenizer=processor.feature_extractor,
)

# 开始训练
trainer.train()
```

训练现在将开始。您将看到进度条和日志，显示训练损失和评估集上的WER。此过程结束后，`output_dir` 中的模型将适应您的数据集，能够比原始预训练 (pre-training)模型更准确地转录此类特定数据。

### 6. 使用微调 (fine-tuning)模型进行推理 (inference)

训练完成后，您可以轻松使用微调后的模型进行转录。最简单的方法是使用`pipeline`抽象，它为您处理所有预处理和后处理步骤。

```python
from transformers import pipeline
import torchaudio

# 使用 pipeline 加载微调模型
asr_pipeline = pipeline("automatic-speech-recognition", model="./wav2vec2-base-librispeech-demo")

# 加载示例音频文件（请确保它是16kHz单声道WAV文件）
# 对于此示例，我们将直接从数据集中获取一个音频数组
sample_audio = librispeech_sample[0]["audio"]["array"]

# 获取转录内容
transcription = asr_pipeline(sample_audio)
print(transcription)
```

这将输出模型的音频文件转录内容。此动手操作练习展示了构建高性能ASR系统的完整现代工作流程。通过从强大的预训练 (pre-training)模型开始，与从头开始训练相比，您可以在自定义数据集上以最少的代码和资源获得出色结果。

## 参考资料

- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11477](https://doi.org/10.48550/arXiv.2006.11477)
  介绍了Wav2Vec 2.0的模型结构及其自监督预训练方法，这是本微调部分的核心模型。
- [Hugging Face Transformers Library Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  `transformers`库的官方文档，用于实现微调流程、模型加载和`Trainer`设置。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出Transformer架构，它是Wav2Vec 2.0等先进声学模型的核心组件。
- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://dl.acm.org/doi/10.1145/1150447.1150493) — Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning (ICML); Publisher: ACM; Pages: 369-376; DOI: [10.1145/1150447.1150493](https://doi.org/10.1145/1150447.1150493)
  阐述了连接时间分类（CTC）算法，该算法是用于语音识别Wav2Vec 2.0微调的损失函数。
