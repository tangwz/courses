# 自回归波形模型（WaveNet, WaveRNN）

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation/autoregressive-waveform-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

传统声码器主要在频域操作并依赖信号处理启发法，而神经网络 (neural network)声码的首批重要进展则源于那些能在时域中逐样本直接生成原始音频波形的模型。这些被称为**自回归 (autoregressive)波形模型**。

传统声码器主要在频域工作并依赖于信号处理的启发式方法，但神经声码领域最初的重大进展来自于直接在时域中逐样本生成原始音频波形的模型。这些模型被称为**自回归波形模型**。这些模型的主要构想是简洁但计算量大的：根据所有先前生成的样本，特别是基于上游TTS声学模型提供的条件声学特征（如梅尔频谱图），来预测下一个音频样本。当前样本 $x_t$ 的概率分布是基于过去的样本 $x_1, ..., x_{t-1}$ 和条件输入 $c$ 进行建模的。


$$
p(x_t | x_1, ..., x_{t-1}, c)
$$


这种顺序依赖使模型能够捕获音频波形中固有的复杂长时序结构，从而带来非常高的保真度和自然度。然而，一次生成一个音频样本，尤其是在常见的采样率（例如16kHz、24kHz甚至48kHz）下，使得推断本身速度较慢。

### WaveNet：开创者

由DeepMind推出的WaveNet是深度学习 (deep learning)在原始音频生成方面具有里程碑意义的成就。它在TTS方面取得了当前最佳的成果，在自然度上明显胜过现有参数 (parameter)和拼接系统。

WaveNet架构中的改进应对了高分辨率音频信号中长距离依赖建模的难题：

1. **因果卷积：** 为保持自回归 (autoregressive)特性（即仅使用过去样本 $x_{<t}$ 来预测 $x_t$），WaveNet使用因果卷积。在标准的1D卷积中，时间 $t$ 的输出可能依赖于时间 $t-k, ..., t, ..., t+k$ 的输入。因果卷积确保时间 $t$ 的输出仅依赖于时间 $t, t-1, t-2, ...$ 的输入。这通常通过在输入序列的一侧进行适当填充，并确保卷积滤波器不会“预见未来”来实现。
2. **空洞卷积：** 使用标准因果卷积对数千个过去样本（例如数百毫秒）的依赖进行建模，将需要非常深的网络或非常大的滤波器，这使得计算上不可行。WaveNet采用*空洞卷积*。在这些层中，滤波器通过跳过某些步长（即空洞率）的输入值，应用于比其长度更大的输入区域。通过堆叠具有指数级增长空洞率（例如1、2、4、8、...、512）的层，网络能够以相对较少的层实现非常大的*感受野*（时间步 $t$ 的输出所依赖的过去时间范围），有效地捕获跨不同时间尺度的依赖关系。

> 堆叠空洞因果卷积使感受野呈指数级增长，使模型能够有效捕获音频信号中的长距离依赖。每层处理输入时都带有增加的间隙（空洞）。

3. **门控激活单元：** 受LSTMs和GRUs启发，WaveNet在其残差块中使用了门控激活机制：

   
   $$
   z = \tanh(W_{f,k} * x) \odot \sigma(W_{g,k} * x)
   $$
   

   这里，$*$ 表示卷积，$k$ 是层索引，$W_{f,k}$ 和 $W_{g,k}$ 是滤波器权重 (weight)，$\sigma$ 是sigmoid函数，$\odot$ 是逐元素乘法。这种门控机制使网络能够更有效地控制信息流。
4. **残差连接和跳跃连接：** 为方便此类深度网络的训练，WaveNet同时采用残差连接（将一个块的输入加到其输出上）和跳跃连接（将不同块的输出求和以形成最终预测），类似于ResNet架构。
5. **条件化：** WaveNet需要根据TTS模型提供的声学特征（例如梅尔频谱图）进行条件化。这通过*局部条件化*来实现，其中条件特征 $c$（上采样以匹配音频分辨率）影响门控机制：

   
   $$
   z = \tanh(W_{f,k} * x + V_{f,k} * c) \odot \sigma(W_{g,k} * x + V_{g,k} * c)
   $$
   

   其中 $V_{f,k}$ 和 $V_{g,k}$ 是应用于条件输入 $c$ 的学习得到的线性投影。*全局条件化*（例如说话人身份向量 (vector)）也可以类似地添加，以影响整个语音。
6. **输出层：** 最终输出层预测下一个样本 $x_t$ 的概率分布。由于原始音频样本通常表示为16位整数（65536个可能值），对所有值进行简单的softmax计算成本很高。WaveNet最初使用8位mu律压扩变换（$\mu$-律编码）将可能值的数量减少到256个，然后是一个softmax层。后来的研究考察了使用混合密度网络（MDNs），具体来说是逻辑斯蒂分布的混合，以更有效地直接建模连续波形或16位离散值。

尽管WaveNet生成的音频质量极高，但其逐样本生成过程使得推断速度极慢，通常在标准硬件上远低于实时速度，这限制了它在对延迟敏感的应用中的实际使用。

### WaveRNN：速度优化

WaveRNN专门为解决WaveNet缓慢的推断速度而开发，同时保留了自回归 (autoregressive)建模的优点。WaveRNN没有完全依赖计算密集型空洞卷积，而是采用循环神经网络 (neural network)（RNN），通常是GRU或LSTM，来建模顺序依赖。

WaveRNN的核心更新方程大致如下：


$$
o_t, h_t = \text{RNN}( [x_{t-1}, c_t], h_{t-1} )
$$


$$
P(x_t) = \text{OutputLayer}(o_t)
$$


其中 $x_{t-1}$ 是前一个音频样本，$c_t$ 是对应的条件特征帧，$h_{t-1}$ 是RNN的前一个隐藏状态，$o_t$ 是RNN的输出，$h_t$ 是更新后的隐藏状态。`OutputLayer`（通常由全连接层组成）然后预测当前样本 $x_t$ 的概率分布，通常使用类似于WaveNet的mu律量化 (quantization)分箱上的softmax。

WaveRNN的重要方面和优化措施包含：

1. **紧凑的RNN单元：** 相较于WaveNet中大型卷积堆叠，使用更简单的RNN单元（如GRU）使每步计算速度可能更快。
2. **子尺度或帧级处理：** 不是一次预测一个样本，一些WaveRNN变体同时预测多个样本，或在音频的更粗“子尺度”上操作，明显加快了生成速度。例如，顺序预测样本的粗粒度位和细粒度位。
3. **稀疏化和剪枝：** WaveRNN模型常利用权重 (weight)剪枝技术以在RNN权重矩阵中强制实现稀疏性。这使高度优化的稀疏矩阵乘法例程得以使用，带来在CPU等硬件上的显著加速，而不会明显影响音频质量。
4. **直接16位预测：** 一些变体不再使用mu律量化，并直接预测16位样本，通常通过顺序预测粗粒度位和细粒度位来实现（例如，先预测高8位，然后根据高位条件化低8位的预测）。

WaveRNN及其优化变体实现了比原始WaveNet快得多的生成速度，使在CPU和移动设备上进行实时合成成为可能，尽管与最佳WaveNet实现相比，通常在可达到的最高音频保真度上略有折衷。

WaveNet和WaveRNN都代表了神经网络声码领域的开创性工作，表明自回归建模原始音频波形能够产生高质量结果。它们的主要局限性——顺序生成过程，促成了接下来要讨论的并行波形生成模型的发展。

## 参考资料

- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu (2016)
  Journal: arXiv preprint arXiv:1609.03499; DOI: [10.48550/arXiv.1609.03499](https://doi.org/10.48550/arXiv.1609.03499)
  介绍WaveNet架构的基础论文，包括因果卷积和空洞卷积，用于高保真原始音频生成。
- [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435) — Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron van den Oord, Sander Dieleman, Koray Kavukcuoglu (2018)
  Journal: International Conference on Machine Learning (ICML); Pages: 2410-2419; DOI: [10.48550/arXiv.1802.08435](https://doi.org/10.48550/arXiv.1802.08435)
  介绍WaveRNN的原始论文，这是一种用于神经音频合成的高效自回归模型，重点在于速度优化。
- [A Survey of Text-to-Speech Synthesis Based on Deep Neural Networks](https://ieeexplore.ieee.org/document/8682122) — Heiga Zen, Andrew Senior, Mike Schuster (2019)
  Journal: ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP); Publisher: IEEE; Pages: 7655-7659; DOI: [10.1109/ICASSP.2019.8682122](https://doi.org/10.1109/ICASSP.2019.8682122)
  一篇关于基于深度神经网络的语音合成的综合综述，提供了WaveNet等神经声码器的背景和讨论。

---

[上一节](01-%E4%BC%A0%E7%BB%9F%E5%A3%B0%E7%A0%81%E5%99%A8%E7%9A%84%E4%B8%8D%E8%B6%B3%E4%B9%8B%E5%A4%84.md) · [下一节](03-%E5%9F%BA%E4%BA%8E%E6%B5%81%E7%9A%84%E5%A3%B0%E7%A0%81%E5%99%A8%20%28WaveGlow%2C%20FloWaveNet%29.md)
