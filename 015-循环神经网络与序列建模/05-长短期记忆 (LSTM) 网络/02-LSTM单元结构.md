# LSTM单元结构

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/lstm-cell-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

简单的循环神经网络 (neural network) (RNN)在学习长序列中的依赖关系时面临很大困难，主要原因是梯度消失和梯度爆炸。网络从较早时间步传递相关信息的能力受到影响。长短期记忆（LSTM）网络正是为了解决这一局限而开发，通过引入更复杂的单元结构，使其能够长时间保持记忆。

LSTM网络的主要组成部分是**LSTM单元**。它用一个由门控和专用单元状态组成的复杂系统，替代了标准RNN单元中的简单变换。这种结构使得网络能够随时间选择性地添加、删除或保留信息。

LSTM单元有两个主要组成部分，实现了这种受控的信息流动：

1. **单元状态 ($C_t$):** 这是LSTM的一个突出特性。你可以将其视为一条内部记忆轨道，水平贯穿一系列单元。信息可以在这条状态线上流动，只发生轻微的线性交互。这种结构使得信息（以及训练时的梯度）更容易在多个时间步中持续存在，而不会明显衰减。
2. **门控:** 它们是神经网络层（通常使用Sigmoid $\sigma$激活函数 (activation function)），用于调节信息进出单元状态的流动。因为Sigmoid函数的输出值介于0和1之间，这些门控就像过滤器一样发挥作用：接近0的值表示“只让很少信息通过”，而接近1的值表示“让大部分信息通过”。一个LSTM单元包含三个主要门控：
   - **遗忘门:** 决定从上一时间步的单元状态 ($C_{t-1}$) 中丢弃哪些信息。
   - **输入门:** 决定将当前输入 ($x_t$) 和上一时间步的隐藏状态 ($h_{t-1}$) 中的哪些新信息存储到单元状态中。
   - **输出门:** 控制当前单元状态 ($C_t$) 的哪些部分应作为当前时间步的隐藏状态 ($h_t$) 传递出去。

我们来描绘这些组成部分如何在一个时间步 $t$ 的单个LSTM单元中相互作用：

> 数据流和组成部分在一个时间步 $t$ 的单个LSTM单元中。它接收当前输入 $x_t$、上一隐藏状态 $h_{t-1}$ 和上一单元状态 $C_{t-1}$。它计算新的单元状态 $C_t$ 和新的隐藏状态 $h_t$。圆圈表示运算（Sigmoid $\sigma$、双曲正切 $\tanh$、元素级乘法 $\times$、元素级加法 $+$）。

LSTM单元内的信息处理过程如下：

1. **遗忘门 ($f_t$):** 单元首先决定从上一单元状态 $C_{t-1}$ 中丢弃哪些信息。它查看上一隐藏状态 $h_{t-1}$ 和当前输入 $x_t$。这些数据通过一个Sigmoid函数 $\sigma$。输出 $f_t$ 包含 $C_{t-1}$ 中每个数值介于0到1之间的值。1表示“完全保留”，而0表示“完全丢弃”。
   $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
2. **输入门 ($i_t$) 和候选值 ($\tilde{C}_t$):** 接下来，单元决定要将哪些新信息存储到单元状态中。这包含两个步骤：

   - 一个Sigmoid层，称为“输入门层”，接收 $h_{t-1}$ 和 $x_t$，并决定更新哪些值 ($i_t$)。
   - 一个 $\tanh$ 层接收 $h_{t-1}$ 和 $x_t$，并生成一个新候选值向量 (vector) $\tilde{C}_t$，这些值有可能被添加到状态中。
     $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
     $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$
3. **单元状态更新 ($C_t$):** 现在，旧的单元状态 $C_{t-1}$ 被更新为新的单元状态 $C_t$。上一状态 $C_{t-1}$ 与遗忘向量 $f_t$ 进行元素级乘法 ($\odot$)，从而遗忘选定的部分。接着，将元素级乘法 $i_t \odot \tilde{C}_t$ 的结果（新信息，根据我们决定更新的程度进行缩放）相加。
   $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
4. **输出门 ($o_t$) 和隐藏状态 ($h_t$):** 最后，单元决定输出，即隐藏状态 $h_t$。这个输出是单元状态的过滤版本。

   - 首先，一个Sigmoid层接收 $h_{t-1}$ 和 $x_t$，以决定单元状态的哪些部分应该作为输出 ($o_t$)。
   - 然后，新计算出的单元状态 $C_t$ 通过 $\tanh$（将值压缩到-1和1之间）。
   - 这个 $\tanh(C_t)$ 与输出门的激活值 $o_t$ 进行元素级乘法 ($\odot$)。这会产生隐藏状态 $h_t$。
     $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
     $h_t = o_t \odot \tanh(C_t)$

在上述方程中，$W_f, W_i, W_C, W_o$ 代表权重 (weight)矩阵，$b_f, b_i, b_C, b_o$ 是偏置 (bias)向量，它们在训练过程中被学习。符号 $[h_{t-1}, x_t]$ 通常表示这两个向量的连接。

这种门控结构，特别是独立的单元状态 $C_t$，它仅通过门控进行轻微的加法和乘法运算，这是LSTM比简单RNN更擅长捕获长距离依赖关系的原因。信息可以在许多时间步中保持，门控学习控制哪些信息是相关的，通过提供更直接的梯度传播路径，有效地缓解了梯度消失问题。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）网络架构的开创性论文，详细阐述其核心组件并解决梯度消失问题。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教材，涵盖深度学习的理论基础和实践方面，包括对循环神经网络和LSTM的详细解释。
- [Recurrent Neural Networks and LSTMs](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFl_exwjd2ZInsrQD3woTEy2mnqJBCrx8UzVk-5XuogWkEZAXsuGTIDeAcyR_X505BLFS3GOoWFewWrHshCtmPrbEj5iVcFB3DsRtFat3dG96cXBErg4MCyiBWW9doRdcwNl1ncfDYT_wSdkwsWwlTGKRVc2op9OYW8jYTnqrcyH5TjcP_iudE=) — Stanford University CS224N Course Staff (2019)
  来自一所顶尖大学课程的详细讲义，对循环神经网络和LSTM提供易于理解而又严谨的解释，包括其架构和工作原理。

---

[上一节](01-%E9%80%9A%E8%BF%87%E9%97%A8%E6%8E%A7%E5%BA%94%E5%AF%B9%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%B1%80%E9%99%90.md) · [下一节](03-%E9%81%97%E5%BF%98%E9%97%A8.md)
