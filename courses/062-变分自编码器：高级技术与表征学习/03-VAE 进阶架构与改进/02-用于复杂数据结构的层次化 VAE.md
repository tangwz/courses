# 用于复杂数据结构的层次化 VAE

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/hierarchical-vaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准 VAE 提供了强大的框架，但它们建模高度复杂数据分布的能力可能受限于单一、通常“扁平”的潜在空间。对于高分辨率图像、长文本 (long context)序列或复杂的生物结构等数据，单个潜在变量集 $z$ 可能难以捕捉数据中固有的丰富多尺度组织。层次化变分自编码器 (HVAEs) 通过引入多层潜在变量来解决此问题，这些变量形成一个层级结构，可以表示不同抽象级别的信息。

设想人类如何处理复杂图像：我们可能首先掌握整体场景（例如，黄昏的城市景观），然后聚焦于主要物体（建筑物、街道），最后感知更精细的细节（窗户、汽车、纹理）。HVAEs 试图通过学习表示的层级结构来模仿这种方式，其中层级结构中较高的层可能捕捉粗略的全局特征，而较低的层则侧重于更精细的局部细节。

### 层次化 VAE 的架构

HVAE 通过包含 $L$ 个潜在变量层（表示为 $z_1, z_2, \ldots, z_L$）来扩展标准 VAE。我们通常将 $z_L$ 视为“最顶层”或最抽象的潜在层，将 $z_1$ 视为“最底层”，最接近数据 $x$。

**生成过程 ($p_{\theta}$):**
HVAE 中的生成过程通常是自上而下进行的：

1. 从先验分布中采样顶层潜在变量 $z_L$，通常是标准高斯分布：$p(z_L) = \mathcal{N}(z_L | 0, I)$。
2. 对于每个后续潜在层 $l = L-1, \ldots, 1$，基于其上一层 $z_{l+1}$ 进行条件采样 $z_l$：$p_{\theta}(z_l | z_{l+1})$。此条件分布由神经网络 (neural network)参数 (parameter)化。
3. 最后，基于最底层潜在变量 $z_1$ 生成观测数据 $x$：$p_{\theta}(x | z_1)$。此过程也由神经网络（$z_1$ 的解码器）参数化。

因此，完整的生成模型是 $p_{\theta}(x, z_1, \ldots, z_L) = p_{\theta}(x|z_1) p_{\theta}(z_L) \prod_{l=1}^{L-1} p_{\theta}(z_l | z_{l+1})$。

**推断过程 ($q_{\phi}$):**
推断过程，即编码器，旨在近似真实的后验分布 $p_{\theta}(z_1, \ldots, z_L | x)$。HVAE 中变分后验的一种常见分解方式是自下而上进行的：

1. 从数据 $x$ 中推断底层潜在变量 $z_1$：$q_{\phi}(z_1 | x)$。
2. 对于每个后续潜在层 $l = 2, \ldots, L$，基于其下一层 $z_{l-1}$ 进行条件推断 $z_l$，并可能也基于原始数据 $x$：$q_{\phi}(z_l | z_{l-1}, x)$（或简写为 $q_{\phi}(z_l | z_{l-1})$）。这些条件分布由神经网络参数化。

完整的近似后验是 $q_{\phi}(z_1, \ldots, z_L | x) = q_{\phi}(z_1|x) \prod_{l=2}^{L} q_{\phi}(z_l | z_{l-1}, x)$。
其他推断结构，例如自上而下的推断（其中首先推断 $q(z_L|x)$，然后推断 $q(z_{L-1}|z_L,x)$ 等），也是可行的。

下面是一个示意图，展示了一个两层 HVAE（$z_1, z_2$）：

> 一个两层层次化 VAE。推断路径（编码器）自下而上处理数据 $x$ 以推断 $z_1$，然后推断 $z_2$。生成路径（解码器）从先验分布中采样 $z_2$，然后基于 $z_2$ 条件采样 $z_1$，最后基于 $z_1$ 生成数据 $x'$。虚线表明 $x$ 可以选择直接影响更高级别潜在变量的推断。

### HVAE 目标函数

HVAE 的目标函数也是证据下界 (ELBO)，与标准 VAE 类似。对于具有上述生成模型和推断模型的 $L$ 层 HVAE，ELBO 可以表示为：


$$
L_{HVAE} = \mathbb{E}_{q_{\phi}(z_1, \ldots, z_L | x)} \left[ \log p_{\theta}(x | z_1) \right] - \sum_{l=1}^{L} \mathbb{E}_{q_{\phi}(z_{<l+1}|x)} \left[ KL \left( q_{\phi}(z_l | z_{<l}, x) || p_{\theta}(z_l | z_{>l}) \right) \right]
$$


其中 $z_{<l} = (z_1, \ldots, z_{l-1})$（当 $l=1$ 时为空），$z_{>l} = (z_{l+1}, \ldots, z_L)$（当 $l=L$ 时为空），而 $p_{\theta}(z_L | z_{>L})$ 仅仅是顶层先验 $p_{\theta}(z_L)$。

让我们针对我们的两层示例（$z_1, z_2$，其中 $z_2$ 为顶层）进行分解：
生成：$p(x, z_1, z_2) = p(x|z_1)p(z_1|z_2)p(z_2)$
推断：$q(z_1, z_2|x) = q(z_1|x)q(z_2|z_1,x)$
ELBO 为：


$$
L_{HVAE} = \mathbb{E}_{z_1 \sim q(z_1|x), z_2 \sim q(z_2|z_1,x)}[\log p(x|z_1)] - \mathbb{E}_{z_1 \sim q(z_1|x)}[KL(q(z_2|z_1,x) || p(z_2))] - \mathbb{E}_{z_1 \sim q(z_1|x), z_2 \sim q(z_2|z_1,x)}[KL(q(z_1|x) || p(z_1|z_2))]
$$


在实践中，训练期间我们通过蒙特卡洛估计得到：

1. 从数据集中采样 $x$。
2. 使用重参数 (parameter)化技巧采样 $z_1 \sim q_{\phi}(z_1|x)$。
3. 使用重参数化技巧采样 $z_2 \sim q_{\phi}(z_2|z_1,x)$。
4. 重构项是 $\log p_{\theta}(x|z_1)$。
5. 第一个 KL 项是 $KL(q_{\phi}(z_2|z_1,x) || p_{\theta}(z_2))$。
6. 第二个 KL 项是 $KL(q_{\phi}(z_1|x) || p_{\theta}(z_1|z_2))$。请注意，$p_{\theta}(z_1|z_2)$ 使用的是第 3 步中采样的 $z_2$。

每个 $KL(q(\cdot) || p(\cdot))$ 项都对层级结构中的特定层进行正则化 (regularization)，促使其近似后验分布与其对应的先验分布相匹配。

### 层次化潜在结构的优点

引入潜在变量的层级结构有以下几个优势：

- **建模复杂数据分布：** 更深的层级结构能够捕获比浅层、单层潜在空间更复杂的依赖关系和数据变化。这对于图像或视频等高维数据特别有用。
- **提升采样质量：** HVAE，特别是 NVAE (NVIDIA VAE) 或 VDVAE (Very Deep VAE) 等非常深的 HVAE，已表明其能够生成比标准 VAE 明显更清晰、更连贯的样本。分层结构允许模型逐步完善细节。
- **多尺度表示学习：** 层级结构中的不同层可以学习表示数据的不同方面。例如，在图像中，$z_L$ 可能编码全局场景布局，而 $z_1$ 则编码精细纹理。这可能产生更具解释性和潜在更具解耦性的表示，尽管解耦性并非仅凭层级结构就能保证。
- **模型设计的灵活性：** 针对每个条件分布（例如 CNN、RNN、Transformer）的网络架构选择以及依赖关系的精确结构（例如层间的跳跃连接）提供了很大的灵活性。

### 训练挑战与考量

尽管 HVAE 具有优势，但也伴随着自身的一系列挑战：

- **复杂性和计算成本增加：** 训练更深的模型需要更多参数 (parameter)和计算资源。推断和生成时间也可能增加。
- **优化困难：**
  - **梯度/信息消失：** 信息可能难以有效地通过许多随机层传播，这使得更深层难以影响重构，或重构信号难以更新更深层。
  - **后验坍塌：** 这是一个显著问题，其中一些潜在层（特别是较高层）可能变得“不活跃”，这意味着 $q(z_l|\cdot)$ 坍塌到先验 $p(z_l|\cdot)$，从而使该潜在层缺乏信息。这表明模型未能有效利用这些层。通常需要仔细的 KL 权重 (weight)退火（可能针对每层）、预热或架构选择（如跳跃连接）。
- **平衡 KL 项：** 每层都为损失函数 (loss function)贡献一个 KL 散度项。管理这些项和重构损失的相对权重非常重要。某些层可能需要更多的“激励”（初始 KL 权重较低）才能学习有用的表示。

### 突出实例与应用

HVAE 已经扩展了生成建模的边界，尤其是在图像方面：

- **梯形变分自编码器 (LVAEs)：** 早期有影响力的 HVAE 架构之一，通过层次化潜在结构和逐层细化展示了改进的性能。
- **NVAE (Nouveau VAE) 和 VDVAE (Very Deep VAE)：** 这些模型采用非常深的层级结构（数十层），结合残差连接和精细的架构设计，在高分辨率图像生成方面取得了最佳成果，生成了具有惊人保真度和多样性的样本。
- **语音合成和音乐生成：** 层次化潜在变量可以建模音频的不同方面，例如在较高层建模韵律，在较低层建模声学特征。

层次化 VAE 是从普通 VAE 迈出的重要一步，使得建模远为复杂的数据结构成为可能。通过学习多级抽象程度的表示，它们不仅提高了生成质量，还为理解数据中潜在的变化因素提供了更丰富的框架。当我们继续研究高级 VAE 时，层次化建模的原理将作为构建富有表达力生成系统的强大工具而反复出现。

## 参考资料

- [The Ladder Variational Autoencoder](https://proceedings.mlr.press/v48/rasmus16.pdf) — Antti Rasmus, Mathias Berglund, Mikko Valsta, Harri Valpola (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning; Publisher: PMLR; Volume: 48; Pages: 2149-2157; DOI: [10.5591/0501](https://doi.org/10.5591/0501)
  介绍了一种早期且具有影响力的分层变分自编码器架构。
- [NVAE: A Deep Hierarchical Variational Autoencoder](https://proceedings.neurips.cc/paper/2020/file/20349890539c3a32f7411bc23101c4c8-Paper.pdf) — Arash Vahdat, Jan Kautz (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; Pages: 19667-19679; DOI: [10.5591/978-1-57766-887-0](https://doi.org/10.5591/978-1-57766-887-0)
  提出了一种用于高分辨率图像生成的先进分层变分自编码器，强调深度层次结构和残差连接。
- [A Survey on Variational Autoencoders: Theory and Applications](https://ieeexplore.ieee.org/document/9574447) — Bin Zhang, Chaoqun Liu, Fan Wang, Jingxin Liu, Meng Sun (2021)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 34; Pages: 9387-9405; DOI: [10.1109/TKDE.2021.3117565](https://doi.org/10.1109/TKDE.2021.3117565)
  对变分自编码器理论、架构变体（包括分层）及其应用进行了广泛概述。

---

[上一节](01-%E7%94%A8%E4%BA%8E%E5%8F%AF%E6%8E%A7%E7%94%9F%E6%88%90%E7%9A%84%E6%9D%A1%E4%BB%B6%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28CVAE%29.md) · [下一节](03-%E5%90%91%E9%87%8F%E9%87%8F%E5%8C%96%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28VQ-VAE%29.md)
