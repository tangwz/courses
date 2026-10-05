# 自然语言处理中的 VAE

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-6-vaes-sequential-structured-data/vaes-natural-language-processing)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自然语言处理 (NLP) 是应用 VAE 的一个主要方面，特别是考虑到它们擅长对序列数据建模，而序列数据是文本的一个主要特点。虽然变分自编码器 (VAEs) 通常应用于独立数据样本，但文本引入了复杂性，例如时间依赖性（词序非常重要）以及生成连贯长篇文本的需求。VAEs 提供了一个强大的框架来学习句子和文档的概率分布，使得我们不仅能够生成新文本，还能够理解并操作其底层语义结构。

核心而言，为 NLP 设计的 VAE 目标是学习输入文本序列 $x$ 的压缩、连续的潜表征 $z$。从这个潜空间中，VAE 的解码器随后试图重构原始文本或生成新的、相似的序列。

### 文本 VAE 的架构

文本的序列性质需要能够处理有序信息的编码器和解码器。通常，这包括：

1. **输入表征**：文本首先被分词 (tokenization)成词序列或子词 (subword)单元。每个词元 (token)随后被映射到一个密集向量 (vector) (dense vector)表征，通常称为词嵌入 (embedding)（例如 Word2Vec、GloVe，或端到端学习的嵌入）。因此，输入句子 $x = (w_1, w_2, ..., w_T)$ 变为嵌入向量序列 $(e_1, e_2, ..., e_T)$。
2. **编码器网络 $q_\phi(z|x)$**:

   - 循环神经网络 (neural network) (RNN)，例如长短期记忆 (LSTM) 或门控循环单元 (GRU)，处理嵌入序列。RNN 的最终隐藏状态（或其隐藏状态的组合）随后用于参数 (parameter)化潜变量 $z$ 的分布。
   - 或者，Transformer 编码器及其自注意力 (self-attention)机制 (attention mechanism)，对于许多 NLP 任务而言，可以比传统 RNN 更有效地捕获长距离依赖。
   - 编码器输出高斯分布的参数（均值 $\mu_\phi(x)$ 和对数方差 $\log \sigma^2_\phi(x)$），并从中通过重参数化技巧采样 $z$：$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。
3. **解码器网络 $p_\theta(x|z)$**:

   - 解码器通常是自回归 (autoregressive)模型，也常是 RNN 或 Transformer。它将潜向量 $z$ 作为初始隐藏状态或在每一步作为条件输入。
   - 它逐词元生成输出序列。在每个时间步 $t$，它根据 $z$ 和先前生成的词元 $w_{<t}$ 预测下一个词元 $w_t$ 的概率分布：$p_\theta(w_t | z, w_{<t})$。
   - 训练期间，通常使用“教师强制”，即将真实的前一个词元作为输入来预测当前词元。推理 (inference)期间，模型馈送其自身先前生成的词元。

> 变分自编码器在自然语言处理任务中的一般架构。

### VAEs 在 NLP 中的应用

VAEs 已在多种 NLP 任务中得到应用，主要凭借其生成能力和所学习到的潜空间的语义属性。

#### 1. 文本生成

最直接的应用是生成新文本。通过从先验分布 $p(z)$（通常是标准正态分布 $\mathcal{N}(0, I)$）中采样 $z$，并将其通过训练好的解码器，VAE 可以生成新句子甚至段落。生成文本的质量取决于 VAE 对底层数据分布的学习情况。具有强大自回归 (autoregressive)解码器的模型，例如那些结合了 Transformer 或带有注意力的 LSTM 的模型，倾向于生成更连贯、更流畅的文本。

#### 2. 可控文本生成

基于条件 VAE (CVAE) 框架（在第 3 章讨论），VAEs 可以适应于可控文本生成。通过向编码器和解码器提供一个额外的条件变量 $c$（例如主题、情感、风格），我们可以引导生成过程。编码器变为 $q_\phi(z|x, c)$，解码器变为 $p_\theta(x|z, c)$。例如：

- **情感修改**：给定输入句子，生成一个具有积极情感的句子，或翻转现有句子的情感。
- **主题条件生成**：生成关于特定主题的文本。
- **风格迁移**：分离文本中的内容与风格，使得能够在保留含义的同时以不同作者的风格重写文本。

#### 3. 潜空间操作与插值

VAEs 所学习到的潜空间 $Z$ 的连续性质使得能够进行有趣的操作：

- **句子插值**：给定两个句子 $x_1$ 和 $x_2$，它们分别编码为 $z_1$ 和 $z_2$，可以在潜空间中对 $z_1$ 和 $z_2$ 进行插值（例如 $z_{interp} = \alpha z_1 + (1-\alpha)z_2$），并解码 $z_{interp}$ 以生成在含义或风格上平滑过渡的句子。
- **语义相似性**：含义相似的句子预期会映射到潜空间中相邻的点。该属性可用于诸如释义检测或语义搜索的任务。
- **类比**：如果潜空间结构良好，像 $z(\text{"king"}) - z(\text{"man"}) + z(\text{"woman"})$ 这样的向量 (vector)运算可能会解码为语义上接近“queen”的东西，尽管对于句子级 VAE 而言，实现这种结构水平比词嵌入 (embedding)更具挑战性。

#### 4. 抽象式文本摘要

VAEs 可以被训练用于将长文档编码为一个潜向量 $z$，然后将此 $z$ 解码为更短的抽象式摘要。其理念是 $z$ 捕获了文档的本质或显著信息。

#### 5. 对话生成

在对话式人工智能中，VAEs 可以对给定对话上下文 (context)的可能响应分布进行建模。潜变量 $z$ 可以捕获不同可能回复的可变性和意图，从而产生与确定性模型相比更多样化且不那么通用化的响应。

### 挑战与缓解策略

训练文本 VAEs 并非没有困难，其中许多与第 2 章讨论的一般 VAE 训练挑战相呼应。

#### 1. KL 散度消失 (后验坍塌)

这是文本 VAE 中的一个普遍问题。ELBO 目标中的 KL 散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 在训练期间可能变得非常小（接近零）。这意味着近似后验 $q_\phi(z|x)$ 坍塌到先验 $p(z)$，即潜变量 $z$ 实际上不携带关于输入 $x$ 的任何信息。强大的自回归 (autoregressive)解码器（例如 LSTM、Transformer）学会忽略 $z$，并主要基于自身的序列动态和先前生成的词元 (token)来对 $p_\theta(x)$ 进行建模，有效地将 VAE 简化为简单的语言模型。

**缓解策略：**

- **KL 退火**：从 KL 项的小权重 (weight)（或零权重）开始，并在训练期间逐渐增加。这使得解码器在 KL 项强制 $z$ 趋向先验之前，首先能够有效地从 $z$ 进行重构。

  
  
  [交互图表：KL 退火调度示例](https://apxml.com/zh/courses/vae-representation-learning/chapter-6-vaes-sequential-structured-data/vaes-natural-language-processing#plot-d7c4h1)
  
  <details>
  <summary>图表数据（JSON）</summary>
  
  ```json
  {
    "data": [
      {
        "x": [
          0,
          500,
          1000,
          1500,
          2000,
          2500,
          3000,
          3500,
          4000,
          4500,
          5000
        ],
        "y": [
          0,
          0.002,
          0.01,
          0.05,
          0.1,
          0.2,
          0.4,
          0.6,
          0.8,
          0.9,
          1.0
        ],
        "type": "scatter",
        "mode": "lines+markers",
        "name": "KL 权重 (β)",
        "marker": {
          "color": "#339af0"
        },
        "line": {
          "color": "#339af0"
        }
      }
    ],
    "layout": {
      "title": "KL 退火调度示例",
      "xaxis": {
        "title": "训练步数"
      },
      "yaxis": {
        "title": "KL 权重 (β)",
        "range": [
          0,
          1.1
        ]
      },
      "height": 350,
      "width": 500,
      "font": {
        "family": "Arial"
      }
    }
  }
  ```
  
  </details>
  
  

  > KL 退火调度的一个示例，其中 KL 散度项的权重 $\beta$ 逐渐增加。
- **自由比特**：为 KL 散度引入一个裕量 $\lambda$，仅当其对于 $z$ 的每个维度低于此阈值时才对其进行惩罚。KL 项的目标变为 $\sum_i \max(\lambda, D_{KL}(q_\phi(z_i|x) || p(z_i)))$。
- **更强的解码器与更弱的编码器**：这有些反直觉，但有时一个非常富有表现力的解码器与一个表达能力较弱的编码器配对，可以强制更多地依赖 $z$。
- **推断网络的积极训练**：在训练早期，比解码器 $p_\theta(x|z)$ 更频繁地或以更高的学习率更新编码器 $q_\phi(z|x)$。
- **词元丢弃/词元掩码**：在解码期间，随机地用 `<UNK>` 词元替换一些真实输入词元或掩码它们。这强制解码器更多地依赖 $z$ 来重构缺失信息。
- **辅助目标**：添加其他依赖于 $z$ 的损失，例如在标签可用时的分类损失（与第 7 章讨论的 CVAE 或半监督学习 (supervised learning) (semi-supervised learning)结合）。
- **使用更具表现力的近似后验**，如第 4 章所讨论（例如 $q_\phi(z|x)$ 的归一化 (normalization)流）。

#### 2. 曝光偏差

在使用教师强制进行训练期间，解码器总是馈送真实的前一个词元。然而，在推理 (inference)期间，它消耗其自身的（可能错误的）预测。训练和推理条件之间的这种差异被称为曝光偏差，可能导致错误累积和生成质量下降。诸如计划采样或使用强化学习 (reinforcement learning)进行训练的技术可以帮助缓解这一点。

#### 3. 生成文本的评估

量化 (quantization)生成文本的“好坏”是众所周知的困难。诸如 BLEU、ROUGE（衡量与参考文本的 n-gram 重叠）或困惑度（语言模型预测样本的准确程度）等标准指标提供了一些见解，但通常与人类对流畅性、连贯性和创造力的判断相关性较差。人工评估仍然必不可少，但成本高昂且耗时。

### 注意力机制 (attention mechanism)的作用

如章节目标中所强调，注意力机制由 Transformer 推广，但也可用于 RNN，在增强 NLP 用途的 VAE 方面发挥重要作用。

- **在编码器中**：Transformer 编码器内的自注意力 (self-attention)使得模型在形成随后映射到 $z$ 的句子表征时，能够权衡输入序列中不同词语的重要性。这有助于更有效地捕获长距离依赖和上下文 (context)。
- **在解码器中**：注意力可以用于使得解码器选择性地聚焦于潜向量 (vector) $z$ 的不同部分（如果 $z$ 本身是结构化的或多部分的），或输入序列的相关部分（在条件设置中，或如果使用从编码器到解码器的直接跳跃连接，尽管这在纯粹从 $z$ 生成文本的 VAE 中较不常见）。更常见的是，Transformer 解码器内的自注意力有助于其在生成序列时保持上下文。

### 自然语言处理中的 VAE：发展的重要一步

VAEs 提供了一种有原则的概率方法来对文本建模，学习可用于生成和表征学习的平滑潜空间。尽管它们面临 KL 散度消失等挑战，但持续的研究仍在不断改进架构和训练技术。例如，通过将 VAEs 与自回归 (autoregressive)流结合，或使用更具结构的潜空间（例如第 3 章中的分层 VAE），研究人员正在拓宽可实现目标的界限。理解自然语言处理中的 VAE 为研究这些更高级的文本生成模型提供了扎实的前提。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）框架、重参数化技巧和ELBO目标函数，为包括自然语言处理在内的所有VAE应用奠定了基础。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，该架构大量依赖自注意力机制，并已成为现代NLP VAE中编码器和解码器组件的标准。

---

[上一节](03-%E5%9B%BEVAEs%E7%94%A8%E4%BA%8E%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0.md) · [下一节](05-%E8%A7%86%E9%A2%91%E4%B8%8E%E5%8A%A8%E6%80%81%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E6%97%B6%E5%BA%8F%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.md)
