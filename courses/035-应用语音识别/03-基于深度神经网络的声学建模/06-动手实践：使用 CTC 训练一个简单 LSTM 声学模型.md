# 动手实践：使用 CTC 训练一个简单 LSTM 声学模型

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-3-acoustic-modeling-deep-learning/practice-training-lstm-ctc-model)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本次实践练习将利用 LSTM 和联结主义时间分类 (CTC) 损失，构建并训练声学模型。它将指导您完成构建一个简单但功能完备的端到端语音识别系统，并使用 PyTorch 实现。我们将使用一个小型、易于管理的数据集，以便专注于实现细节，而不受漫长训练时间的困扰。

我们的目标是创建一个模型，该模型将梅尔频谱图作为输入，并输出一个字符概率序列，然后 CTC 损失函数 (loss function)使用这些概率序列来计算梯度并训练网络。

### 1. 项目设置与数据准备

首先，请确保您已安装所需的库。我们将主要使用 `torch` 和 `torchaudio` 进行建模和数据处理，使用 `librosa` 进行特征提取。

```bash
pip install torch torchaudio librosa
```

对于本次练习，我们假设您有一个预处理过的数据集，包含音频文件和一个对应的元数据文件（例如 `.csv` 或 `.json`），该文件将每个音频文件映射到其转录文本。接下来，我们定义一个 PyTorch `Dataset` 来处理数据的加载、处理和分词 (tokenization)。

首要一步是定义我们的字母表。模型的输出层必须为每个可能的字符设置一个节点，以及一个用于 CTC 所需的特殊 `blank` 标记 (token)的额外节点。

```python
# 一个简单的英文字符集
# 空白标记通常在索引 0
char_map_str = """
' 0
<SPACE> 1
a 2
b 3
c 4
d 5
e 6
f 7
g 8
h 9
i 10
j 11
k 12
l 13
m 14
n 15
o 16
p 17
q 18
r 19
s 20
t 21
u 22
v 23
w 24
x 25
y 26
z 27
"""
# 在实际项目中，您会根据训练数据的转录文本生成此内容。
```

接下来，我们为 `DataLoader` 创建一个自定义的 `collate_fn`。由于每个音频片段及其转录文本的长度不同，我们无法简单地将它们堆叠成一个批次。此函数将把批次中的每个序列填充到最长序列的长度，并且它还会记录原始的、未填充的长度。CTC 损失函数 (loss function)需要这些原始长度才能正常工作。

```python
# 在您的数据加载脚本中
import torch
import torchaudio

def collate_fn(batch):
    # 一个批次包含一个元组列表：(频谱图, 标签, 输入长度, 标签长度)
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_lengths = [item[2] for item in batch]
    label_lengths = [item[3] for item in batch]

    # 填充频谱图和标签
    padded_spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return padded_spectrograms, padded_labels, torch.tensor(input_lengths), torch.tensor(label_lengths)
```

这个 `collate_fn` 是在 PyTorch 中处理变长序列数据时的标准模式，对于批处理我们的语音数据很必要。

### 2. LSTM-CTC 模型架构

我们的声学模型将采用一种直接的架构。它将包含几个双向 LSTM 层，之后是一个全连接线性层。LSTM 负责学习语音特征中的时间模式，而最终的线性层将 LSTM 的输出投影到每个时间步的字符词汇表 (vocabulary)上的概率分布。

让我们在 PyTorch 中定义这个模型。

```python
import torch.nn as nn

class LSTMAcousticModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_class, n_layers, dropout):
        super(LSTMAcousticModel, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 分类层
        # 输出为 2 * n_hidden，因为 LSTM 是双向的
        self.classifier = nn.Linear(n_hidden * 2, n_class)

    def forward(self, x):
        # x 是输入频谱图：(批次, 时间, 特征)
        lstm_out, _ = self.lstm(x)

        # 将 LSTM 输出通过分类器
        # 输出为 (批次, 时间, 类别数)
        output = self.classifier(lstm_out)

        # CTC 损失需要 log_softmax
        # CTC 损失期望时间维度在前
        return nn.functional.log_softmax(output, dim=2).permute(1, 0, 2)
```

下图呈现了数据流经我们模型的路径。输入频谱图由双向 LSTM 处理，生成的隐藏状态被传递到线性层，该线性层生成 CTC 所需的字符概率。

> 简单基于 LSTM 的声学模型内的数据流。代码中最后的 `permute` 操作会重新排列张量维度，以符合 PyTorch 的 `CTCLoss` 所期望的格式。

### 3. 训练循环

模型和数据加载器准备就绪后，我们可以编写主要的训练函数。此函数将安排将数据馈送给模型、计算损失以及更新模型权重 (weight)的过程。

此过程的核心是 `torch.nn.CTCLoss`。它需要四个参数 (parameter)：

1. `log_probs`: 我们模型输出的对数概率，形状为 `(时间, 批次, 类别)`。
2. `targets`: 批次中所有字符标签连接在一起的 1D 张量。
3. `input_lengths`: 包含批次中每个频谱图原始长度的张量。
4. `target_lengths`: 包含批次中每个转录文本原始长度的张量。

这是一个用于单个训练周期的简化函数。

```python
def train_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        # 将数据移动到选定的设备（例如 GPU）
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        # 输出形状为 (时间, 批次, 类别数)
        output = model(spectrograms)

        # 计算 CTC 损失
        loss = criterion(output, labels, input_lengths, label_lengths)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss
```

要运行训练，您需要实例化模型、损失函数 (loss function)和优化器，然后在循环中调用 `train_epoch` 函数。

```python
# 超参数
n_features = 128   # 梅尔频谱图中的特征数量
n_hidden = 256
n_class = 28       # 词汇表中的字符数量 + 1（用于空白标记）
n_layers = 2
dropout = 0.2
learning_rate = 1e-4
epochs = 10

# 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAcousticModel(n_features, n_hidden, n_class, n_layers, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CTCLoss(blank=0).to(device) # blank=0 假定空白标记在索引 0

# 仅用于演示的虚拟训练加载器和验证加载器
# 实际中，它们会是 torch.utils.data.DataLoader 实例
# train_loader = ...
# valid_loader = ...

# 训练循环
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
    # 通常您也会在此处运行验证循环
    # valid_loss = evaluate(...)
```

### 4. 查看结果

随着训练的进行，您应该看到 CTC 损失下降，这表明模型正在学习将音频特征映射到正确的字符序列。绘制每个周期的训练和验证损失是一种标准方法，用于监控此过程并检查过拟合 (overfitting)。良好的训练过程会显示两个损失值稳步下降。



[交互图表：训练和验证损失](https://apxml.com/zh/courses/applied-speech-recognition/chapter-3-acoustic-modeling-deep-learning/practice-training-lstm-ctc-model#plot-17e0dyb)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "训练和验证损失",
    "xaxis": {
      "title": "周期"
    },
    "yaxis": {
      "title": "CTC 损失"
    }
  },
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        5.6,
        4.1,
        3.2,
        2.5,
        2.1,
        1.8,
        1.6,
        1.4,
        1.3,
        1.2
      ],
      "name": "训练损失",
      "type": "scatter",
      "mode": "lines+markers",
      "marker": {
        "color": "#339af0"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        5.8,
        4.5,
        3.7,
        3.1,
        2.8,
        2.6,
        2.5,
        2.4,
        2.35,
        2.3
      ],
      "name": "验证损失",
      "type": "scatter",
      "mode": "lines+markers",
      "marker": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> LSTM-CTC 模型的训练曲线示例。训练损失和验证损失之间的差距表明模型可能开始过拟合，这是一个常见问题，可以通过增加数据、正则化 (regularization)或数据增强来解决。

训练结束后，您可以使用简单的贪心解码器来转录新的音频文件。这包括将频谱图通过模型，在每个时间步获取输出概率的 `argmax` 以得到最可能的字符，然后折叠重复字符并删除空白标记 (token)以生成最终文本。

本次动手实践提供了一个完整（尽管简单）的声学模型训练流程。您已成功构建了一个能从音频特征学习转录语音的系统。尽管这个模型是一个很好的开始，但其性能可以通过更先进的架构和解码技术显著提高，我们将在后续章节中介绍这些内容。

## 参考资料

- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://dl.acm.org/doi/10.1145/1143844.1143891) — Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning (ICML 2006); Publisher: ACM; Pages: 369-376; DOI: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
  介绍了连接时序分类（CTC）算法，一种训练循环神经网络转录未分段序列的方法，为端到端语音识别奠定了基础。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  一本全面的教科书，涵盖了语音识别的基础知识，包括声学建模、循环神经网络（如LSTM）以及CTC在现代自动语音识别系统中的应用。
- [CTCLoss - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  PyTorch中连接时序分类（CTC）损失函数实现的官方文档，详细说明了其参数和预期的输入/输出格式。
- [torchaudio: Speech and Audio Processing - Torchaudio 2.3 documentation](https://pytorch.org/audio/stable/index.html) — Torchaudio Developers (2024)
  Publisher: PyTorch Foundation
  torchaudio的官方文档，这是一个与PyTorch集成的音频和语音处理库，与语音识别任务中的特征提取和数据集准备相关。

---

[上一节](05-%E5%AE%9E%E7%8E%B0%E5%9F%BA%E4%BA%8ECTC%E7%9A%84ASR%E6%A8%A1%E5%9E%8B.md) · [下一节](../04-%E9%AB%98%E7%BA%A7%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%9E%B6%E6%9E%84/01-%E7%94%A8%E4%BA%8E%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
