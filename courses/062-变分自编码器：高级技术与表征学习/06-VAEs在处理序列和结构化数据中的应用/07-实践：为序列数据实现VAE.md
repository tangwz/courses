# 实践：为序列数据实现VAE

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-6-vaes-sequential-structured-data/practice-vaes-sequential-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实现一个变分自编码器（VAE）来对序列数据进行建模和生成。构建一个循环变分自编码器（RVAE）进行字符级文本生成的步骤将被概述。变分自编码器可以适配序列和结构化数据。这个练习将强化对如何在VAE架构中使用循环神经网络 (neural network)（RNN）来捕捉时间依赖性的理解。

我们的目标是训练一个RVAE，使其能够学习文本序列的压缩表示，然后使用这个表示来生成新的、可信的文本。

### 任务：字符级文本生成

我们将进行字符级文本生成。这意味着我们的模型将学习根据前面的字符预测序列中的下一个字符。虽然词级模型也很普遍，但字符级模型在词汇管理方面更容易配置，并且可以生成新颖的词或风格。

### 数据集和预处理

1. **语料库选择**:
   选择一个文本语料库。为了学习，一个大小适中、内容连贯的文本效果不错。例子包括：

   - 莎士比亚的十四行诗或戏剧。
   - 古腾堡计划中的一部小说。
   - 一篇短新闻文章的集合。
     在本指南中，我们假设你有一个名为`corpus.txt`的纯文本文件。
2. **词汇表 (vocabulary)创建**:
   首先，我们需要确定我们的词汇表，在这种情况下，它就是语料库中独有字符的集合。

   ```python
   # 示例性的Python伪代码
   text = open('corpus.txt', 'r').read()
   chars = sorted(list(set(text)))
   char_to_int = {ch: i for i, ch in enumerate(chars)}
   int_to_char = {i: ch for i, ch in enumerate(chars)}
   vocab_size = len(chars)
   ```
3. **创建输入-输出序列**:
   我们需要将原始文本转换为RVAE可以处理的序列。我们将使用滑动窗口方法。对于给定的`sequence_length`，我们将创建输入序列和相应的目标序列（通常是输入序列向后平移一个字符的结果）。

   ```python
   # 示例性的Python伪代码
   sequence_length = 50 # 示例长度
   data_X = [] # 输入序列
   data_y = [] # 目标序列（用于解码器重构）

   for i in range(0, len(text) - sequence_length, 1):
       seq_in = text[i:i + sequence_length]
       seq_out = text[i + 1:i + sequence_length + 1] # 用于重构的目标
       data_X.append([char_to_int[char] for char in seq_in])
       # 对于RVAE，解码器将尝试重构seq_in，
       # 或者如果以seq_in和z为条件，则生成seq_out
       # 为了简单起见，这里假设解码器旨在重构seq_in
       data_y.append([char_to_int[char] for char in seq_in]) # 如果是那样设计的话，也可以是seq_out

   num_sequences = len(data_X)
   ```

   注意：对于旨在从潜在变量 $z$ 生成数据的“经典”RVAE，解码器通常会重构输入序列 `seq_in`。RNN结构本身处理序列预测。
4. **数据格式化**:
   输入数据需要为RNN适当塑形，通常是`(num_sequences, sequence_length, feature_dim)`。对于字符级模型，`feature_dim`通常是1（如果直接将整数输入送入嵌入 (embedding)层）或`vocab_size`（如果使用独热编码输入）。如果整数输入没有首先馈送到嵌入层，我们还会对其进行归一化 (normalization)。

   ```python
   # 示例性的Python伪代码
   # 假设使用嵌入层，因此输入是(num_sequences, sequence_length)
   # X = np.reshape(data_X, (num_sequences, sequence_length))
   # y = np.reshape(data_y, (num_sequences, sequence_length))
   # PyTorch/TensorFlow将处理批处理
   ```

### 循环变分自编码器（RVAE）模型架构

我们的RVAE将由一个基于RNN的编码器、一个潜在空间采样机制和一个基于RNN的解码器组成。

#### 1. 编码器

编码器的作用是将输入序列 $x$ 映射到近似后验分布 $q(z|x)$ 的参数 (parameter)，我们假定这个分布是高斯分布 $\mathcal{N}(\mu_z, \text{diag}(\sigma_z^2))$。

- **输入**：字符嵌入 (embedding)或独热向量 (vector)的序列。
- **RNN层**：一个LSTM或GRU层处理输入序列。RNN的最终隐藏状态（或所有隐藏状态的摘要）捕获序列信息。

  ```python
  # 编码器的PyTorch伪代码
  # self.embedding = nn.Embedding(vocab_size, embedding_dim)
  # self.encoder_rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
  # self.fc_mu = nn.Linear(hidden_dim, latent_dim)
  # self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

  # def encode(self, x_sequence):
  #     embedded = self.embedding(x_sequence) # (批次, 序列长度, 嵌入维度)
  #     _, (h_n, _) = self.encoder_rnn(embedded) # h_n 是 (1, 批次, 隐藏维度)
  #     h_n_last_layer = h_n.squeeze(0) # (批次, 隐藏维度)
  #     mu = self.fc_mu(h_n_last_layer)
  #     logvar = self.fc_logvar(h_n_last_layer)
  #     return mu, logvar
  ```

  LSTM中的`h_n`包含批次中每个序列的最终隐藏状态。

#### 2. 潜在变量采样

这是标准的VAE过程：
$z = \mu_z + \sigma_z \odot \epsilon, \quad \text{式中 } \epsilon \sim \mathcal{N}(0, I)$
且 $\sigma_z = \exp(0.5 \cdot \log \sigma_z^2)$. 这是重参数化技巧。

```python
# 重参数化的PyTorch伪代码
# def reparameterize(self, mu, logvar):
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn_like(std)
#     return mu + eps * std
```

#### 3. 解码器

解码器从潜在空间接收样本 $z$，旨在重构原始输入序列（或者，如果 $z$ 是从先验分布 $p(z)$ 中采样的，则生成新序列）。

- **以 $z$ 为条件**：潜在变量 $z$ 需要为生成过程提供信息。一种常用方法是使用 $z$ 来初始化解码器RNN的隐藏状态。另一种方法是，可以在每个时间步将 $z$ 连接到RNN的输入。
- **自回归 (autoregressive)生成**：解码器RNN一次生成一个字符序列。
  - 嵌入层将输入字符（或序列开始标记 (token)）转换为向量。
  - RNN（LSTM/GRU）接收嵌入字符和先前的隐藏状态，以产生一个输出和一个新的隐藏状态。
  - 一个全连接层（通常称为输出层）将RNN输出映射到词汇表 (vocabulary)上的概率分布（使用softmax），以预测下一个字符。
- **教师强迫**：在训练期间，通常会使用“教师强迫”。这意味着在每个时间步 $t$，我们不是将模型在步 $t-1$ 生成的字符作为输入，而是馈送输入序列中实际的真实字符。这有助于稳定训练。

```python
# 解码器的PyTorch伪代码
# self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
# self.decoder_rnn_cell = nn.LSTMCell(embedding_dim + latent_dim, hidden_dim) # 示例：连接z
# # 或者：self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim) 用于h0, c0初始化
# self.fc_out = nn.Linear(hidden_dim, vocab_size)

# def decode(self, z, target_sequence, teacher_forcing_ratio=0.5):
#     batch_size = z.size(0)
#     seq_len = target_sequence.size(1)
#     
#     # 初始化隐藏状态（例如，来自z或零）
#     # hx = self.latent_to_hidden(z) # (批次, 隐藏维度)
#     # cx = self.latent_to_hidden(z) # (批次, 隐藏维度)
#     # 或者如果连接z：
#     hx = torch.zeros(batch_size, hidden_dim).to(z.device)
#     cx = torch.zeros(batch_size, hidden_dim).to(z.device)
#
#     # 起始标记（例如，<SOS>字符的嵌入，或目标序列的第一个字符）
#     current_input_char_idx = target_sequence[:, 0] # 示例：使用目标的第一个字符
#     outputs = []
#
#     for t in range(seq_len):
#         embedded_char = self.decoder_embedding(current_input_char_idx) # (批次, 嵌入维度)
#         
#         # 选项1：将z与每个输入连接
#         rnn_input = torch.cat((embedded_char, z), dim=1) # (批次, 嵌入维度 + 潜在维度)
#         hx, cx = self.decoder_rnn_cell(rnn_input, (hx, cx))
#
#         # 选项2：使用z初始化hx, cx（在循环前完成）
#         # hx, cx = self.decoder_rnn_cell(embedded_char, (hx, cx))
#
#         output_logits_t = self.fc_out(hx) # (批次, 词汇表大小)
#         outputs.append(output_logits_t)
#
#         use_teacher_force = random.random() < teacher_forcing_ratio
#         if use_teacher_force and t < seq_len -1:
#             current_input_char_idx = target_sequence[:, t+1]
#         else:
#             _, top_idx = output_logits_t.topk(1)
#             current_input_char_idx = top_idx.squeeze(1).detach() # 使用模型自己的预测
#
#     return torch.stack(outputs, dim=1) # (批次, 序列长度, 词汇表大小)
```

### 损失函数 (loss function)

RVAE损失函数是标准的VAE ELBO，但重构项现在是序列元素的总和。


$$
\mathcal{L}_{RVAE}(x, \hat{x}, \mu_z, \log \sigma_z^2) = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|x) || p(z))
$$


1. **重构损失（$\mathcal{L}_{recon}$）**：对于字符级生成，这通常是序列中每个位置的预测字符分布与实际目标字符之间的交叉熵损失的总和。

   
   $$
   \mathcal{L}_{recon} = -\sum_{t=1}^{T} \log p(x_t | x_{<t}, z)
   $$
   

   在实际中，你会使用你所用框架的`CrossEntropyLoss`函数，并将其应用于序列维度。请确保logits和目标的形状正确（例如，logits：`(batch_size * seq_len, vocab_size)`，targets：`(batch_size * seq_len)`）。
2. **KL散度（$D_{KL}$）**：近似后验 $q(z|x)$ 与先验 $p(z)$（通常是 $\mathcal{N}(0, I)$）之间的KL散度。

   D\_{KL}(q(z|x) || p(z)) = -0.5 \sum\_{j=1}^{\text{latent\_dim}} (1 + \log(\sigma\_{z\_j}^2) - \mu\_{z\_j}^2 - \sigma\_{z\_j}^2)

   $\beta$项来自$\beta$-VAE，可用于控制对解耦或重构质量的侧重。对于标准VAE，$\beta=1$。

### 训练循环

训练循环包括：

1. 获取一批序列。
2. 将它们通过编码器以获得 $\mu_z$ 和 $\log \sigma_z^2$。
3. 使用重参数 (parameter)化技巧采样 $z$。
4. 将 $z$ 和目标序列（用于教师强迫）传递给解码器，以获得重构序列（logits）。
5. 计算重构损失和KL散度。
6. 将它们求和以获得总损失。
7. 反向传播 (backpropagation)并更新模型参数。

```python
# 示例性训练步骤伪代码
# rvae_model = RVAE(...)
# optimizer = Adam(rvae_model.parameters(), lr=1e-3)
#
# for epoch in range(num_epochs):
#     for batch_sequences_x, batch_sequences_y in data_loader:
#         optimizer.zero_grad()
#
#         mu, logvar = rvae_model.encode(batch_sequences_x)
#         z = rvae_model.reparameterize(mu, logvar)
#         decoded_logits = rvae_model.decode(z, batch_sequences_y) # 用于教师强迫的batch_sequences_y
#
#         # 重构损失
#         # 为CrossEntropyLoss重塑形状: (批次 * 序列长度, 词汇表大小) 和 (批次 * 序列长度)
#         recon_loss = criterion_recon(
#             decoded_logits.view(-1, vocab_size),
#             batch_sequences_y.view(-1)
#         )
#
#         # KL散度
#         kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         kl_div = kl_div / batch_sequences_x.size(0) # 按批次平均
#
#         loss = recon_loss + beta * kl_div
#         loss.backward()
#         optimizer.step()
#
#     # 记录损失，定期生成样本
```

### 生成文本（推断）

模型训练完成后，你可以生成新的文本序列：

1. 从先验分布 $p(z) = \mathcal{N}(0, I)$ 中采样一个潜在向量 (vector) $z_{new}$。
2. 向解码器提供 $z_{new}$ 和一个序列开始（SOS）标记 (token)（或一个短的种子序列）。
3. 逐字符自回归 (autoregressive)生成序列：
   - 解码器预测下一个字符的分布。
   - 从这个分布中采样一个字符（例如，使用`torch.multinomial`或取`argmax`）。
   - 将采样的字符作为下一个时间步的输入。
   - 重复此过程，直到生成期望长度或一个序列结束（EOS）标记。

```python
# 示例性生成伪代码
# def generate_sequence(rvae_model, z_sample, start_token_idx, max_len=100):
#     rvae_model.eval()
#     generated_sequence_indices = [start_token_idx]
#     current_input_char_idx = torch.tensor([[start_token_idx]], device=device) # 批次大小为1
#
#     # 初始化解码器隐藏状态（来自z_sample或如果z被连接则为零）
#     # hx, cx = ... initialized based on z_sample
#
#     with torch.no_grad():
#         for _ in range(max_len - 1):
#             embedded_char = rvae_model.decoder_embedding(current_input_char_idx)
#             # rnn_input = torch.cat((embedded_char.squeeze(1), z_sample), dim=1) if concatenating z
#             # hx, cx = rvae_model.decoder_rnn_cell(rnn_input, (hx, cx))
#             # output_logits = rvae_model.fc_out(hx)
#             
#             # 简化：假设模型中有一个decode_step函数
#             output_logits, hx, cx = rvae_model.decode_step(current_input_char_idx, z_sample, hx, cx)
#
#             # 采样下一个字符（可以添加温度以增加多样性）
#             # probabilities = F.softmax(output_logits / temperature, dim=-1)
#             # next_char_idx = torch.multinomial(probabilities, 1)
#             _, next_char_idx = output_logits.topk(1, dim=-1)
#
#             generated_sequence_indices.append(next_char_idx.item())
#             current_input_char_idx = next_char_idx
#
#             # 如果 next_char_idx.item() == eos_token_idx: break
#
#     return "".join([int_to_char[idx] for idx in generated_sequence_indices])
#
# z_prior = torch.randn(1, latent_dim).to(device)
# generated_text = generate_sequence(rvae_model, z_prior, char_to_int['A'])
# print(generated_text)
```

### 实际考量与后续步骤

- **KL退火**：RVAE的一个常见问题，特别是那些带有强大自回归 (autoregressive)解码器（如LSTM）的，是“后验崩溃”或“KL消失”。KL散度项 $D_{KL}(q(z|x) || p(z))$ 会迅速趋近于零，这意味着潜在变量 $z$ 被解码器忽视，解码器完全依赖其自回归特性。为了缓解这种情况，可以在训练期间逐渐增加KL项的权重 (weight)（KL退火）。从$\beta=0$开始，在一定数量的epoch或训练步骤中缓慢将其增加到最终值（例如1）。
  
  $$
  \beta_t = \min(1.0, \text{当前步骤} / \text{退火持续步数})
  $$
  
- **教师强迫与计划采样**：虽然教师强迫有助于训练稳定性，但它在训练（总是看到正确输入）和推断（看到模型自身可能错误的预测）之间产生了差异。计划采样是一种技术，你在训练期间逐渐从使用真实输入切换到使用模型自身的预测作为解码器输入。
- **嵌入 (embedding)维度、隐藏单元、层数**：尝试嵌入层、RNN隐藏单元和RNN层数的大小。更深或更宽的模型可以捕获更复杂的模式，但更难训练，也更容易过拟合 (overfitting)。
- **梯度裁剪**：RNN可能遭受梯度爆炸。在训练期间应用梯度裁剪通常是必要的稳定技术。
- **评估**：
  - **困惑度**：语言模型的一个常用指标。较低的困惑度表明模型对测试集的“惊讶程度”较低。
  - **定性评估**：阅读生成的样本。它们有意义吗？它们多样吗？它们是否捕捉到训练语料库的风格？
  - **潜在空间分析**：如果你在具有已知属性（例如文本中的情感）的序列上训练RVAE，你可以尝试查看这些属性是否编码在潜在空间 $z$ 中。对不同序列的 $z$ 向量 (vector)进行插值，并观察生成的输出是否平滑过渡。

本实践指导为序列数据实现VAE提供了一个蓝图。RVAE是一个基础模型，并且存在许多扩展和变体，例如那些结合注意力机制 (attention mechanism)（我们已在本章前面讨论过）以更有效地处理长距离依赖的模型。尝试这些组件，观察它们的影响，并在应对更复杂的序列建模任务时参考研究论文以获得更高级的技术。

## 参考资料

- [Auto-Encoding Variational Bayes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGAqu7hGkpLEZC7nWY2XqeOIz8R5Un3RqONo7z05yZxNUWNnWP2xd8gBfRq7BxmvQbrzURwjOpcH8jgsf3wmrAuBQtLFebDJi09BZGfH4y9ZGiXyHGCPLKHFA==) — Diederik P Kingma, Max Welling (2014)
  Journal: 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自动编码器（VAE）和重参数化技巧，它们是本节的基础。
- [Generating Sentences from a Continuous Space](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH3WVBFnTH5Xe5KS3i_5WoASnA87tm1KU52GX_zzpeRiLEd0n9eY9nNxYqhmo_YtxUMbOThf3_2zwkQ9xBvfDCHA0VoIGB3rS2P-ie8SQXEWmJ2XnWANLKn_jp4m_-zsvj5jTE=) — Samuel Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio (2016)
  Journal: Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning; Pages: 10–21; DOI: [10.18653/v1/N16-1085](https://doi.org/10.18653/v1/N16-1085)
  这是一项将VAE应用于使用LSTM的自然语言生成的基础性工作，讨论了诸如后验坍缩等挑战以及KL退火的重要性。
- [Deep Learning](https://deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖变分自动编码器（VAE）、循环神经网络（RNN、LSTM、GRU）以及梯度裁剪和序列建模方法等实际训练方面的综合性教材。

---

[上一节](06-%E7%8A%B6%E6%80%81%E7%A9%BA%E9%97%B4%E6%A8%A1%E5%9E%8B%E4%B8%8E%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%9A%84%E5%85%B3%E8%81%94.md) · [下一节](../07-VAE%E7%9A%84%E8%BF%9B%E9%98%B6%E8%AE%AE%E9%A2%98%E4%B8%8E%E5%BA%94%E7%94%A8%E6%8B%93%E5%B1%95/01-%E4%BD%BF%E7%94%A8%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%EF%BC%88VAE%EF%BC%89%E7%9A%84%E5%8D%8A%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0.md)
