---
course: "speech-recognition-synthesis-asr-tts"
chapter: "neural-vocoders-waveform-generation"
lesson: "traditional-vocoder-limitations"
sourceId: 3191
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation/traditional-vocoder-limitations"
title: "传统声码器的不足之处"
description: "分析经典信号处理声码方法的伪影和质量不足。"
order: 1
plots: ["plots/3191-0.json"]
sourceHash: "ae7a91a1bd611eb808b41c45dcaa3920df81427c979a387343064906f3d6c2dd"
sourceCorrections: []
---

在基于深度学习 (deep learning)的方法出现之前，声码技术主要依赖于信号处理技术。这些方法获取声学特征表示，通常是幅度谱图（例如由TTS前端生成的梅尔谱图），并试图重建时域音频波形。

一个典型的例子是**Griffin-Lim算法 (GLA)**。这些传统方法面临的核心难题是**相位重建问题**。TTS声学模型通常只预测谱图的*幅度*部分，舍弃了*相位*信息。尽管幅度捕捉了大部分频谱内容，但相位对于准确重建波形的时域结构和感知质量是必要的。

试想短时傅里叶变换 (STFT)，它将信号段转换为不同频率的幅度和相位成分。逆转此过程（逆STFT或ISTFT）需要两者。仅给定幅度 $|X(t, f)|$，我们如何找出正确的相位 $\phi(t, f)$ 以重建原始信号 $x(n)$？


$$
x(n) \xrightarrow{\text{STFT}} X(t, f) = |X(t, f)| e^{j \phi(t, f)}
$$


$$
\hat{X}(t, f) = |X_{\text{目标}}(t, f)| e^{j \hat{\phi}(t, f)} \xrightarrow{\text{ISTFT}} \hat{x}(n)
$$


Griffin-Lim算法迭代处理这个问题。它以目标幅度谱图和相位的初始猜测（通常是随机噪声或零相位）开始。随后它在两个步骤之间交替进行：

1. **时域合成：** 将目标幅度与当前相位估计结合，并应用逆STFT (ISTFT) 以获得时域信号。
2. **频域精修：** 对合成信号应用STFT。保留此STFT结果的*相位*，但将其*幅度*替换为原始目标幅度谱图。

此过程重复进行，旨在找出STFT幅度与目标匹配的信号，同时满足STFT/ISTFT过程中固有的一致性限制。其潜在假设是，强制一致性将隐含地引导相位趋向于一个合理的估计。

然而，此过程存在明显的缺点：

- **次优相位估计：** 迭代过程不保证找出原始相位或感知上最佳的相位。仅从幅度进行相位重建从根本上是一个病态问题，这意味着多种相位配置可以对应相同的幅度谱图，但导致不同的感知结果。GLA通常收敛于局部最优解，该解并非理想。
- **可闻伪影：** 最明显的不足之处是合成音频的质量。使用Griffin-Lim等方法生成的波形经常出现特征性伪影。这些包括：
  - **嗡嗡声或机器人般的声音：** 缺乏自然流畅性。
  - **金属音色：** 不自然的共振或频谱特征。
  - **沉闷的音质：** 清晰度损失，尤其在高频区域。
  - **“相位感”声音：** 源于跨频率区间和时间帧的不一致或不正确的相位关系所导致的感知失真。
- **收敛缓慢：** Griffin-Lim需要多次迭代（通常50-100次或更多）才能收敛，这可能计算密集，特别是对于长音频序列，阻碍实时应用。

考虑谱图的差异。尽管GLA声码后的谱图可能与目标*幅度*谱图高度匹配，但其潜在的相位结构导致感知质量下降。



![谱图幅度比较](plots/3191-0.json)



> 说明：尽管Griffin-Lim重建的幅度（中间）可能与目标幅度（左侧）高度匹配，但由于相位估计次优，所得音频质量相比源自原始波形（右侧，其幅度可能略有差异但相位正确）的音频有所下降。

这些不足之处促成了神经网络 (neural network)声码器的发展。通过直接从声学特征学习到波形的复杂映射，或通过建模音频样本的条件分布，神经网络可以隐式或显式地学习正确的相位关系，从而产生明显更自然、更高保真度的合成语音。我们将在后续章节中考察这些先进技术。

## 参考资料

- [A Survey of Neural Vocoders](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/survey-of-neural-vocoders/C6B500C547849D945536413247B2C19C) — Kun Song, Yi-Chiao Wu, Peng-Hsiang Chen, Yu-Hua Chen, Shih-Min Chao, and Chi-Chun Lee (2021)
  Journal: APSIPA Transactions on Signal and Information Processing; Publisher: Cambridge University Press; Volume: 10; Pages: e8; DOI: [10.1017/ATSIP.2021.5](https://doi.org/10.1017/ATSIP.2021.5)
  这篇综述论文讨论了声码器的演变，详细阐述了Griffin-Lim等传统方法的局限性以及向神经方法转变的原因。
