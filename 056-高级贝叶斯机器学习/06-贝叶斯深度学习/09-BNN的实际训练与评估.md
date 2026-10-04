# BNN的实际训练与评估

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/practical-training-evaluation-bnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

贝叶斯神经网络 (neural network) (BNN) 的实际训练和性能评估是核心内容。BNN使用变分推断 (VI) 和马尔可夫链蒙特卡洛 (MCMC) 等推断方法。实施和评估BNN涉及对标准深度神经网络的考量，特别是关于超参数 (parameter) (hyperparameter)调整、收敛性监控和不确定性量化 (quantization)。

### 配置贝叶斯神经网络 (neural network)

在训练之前，仔细配置是必需的。这包括为权重 (weight)和偏差选择合适的先验分布，并定义网络架构。

- **先验选择：** 先验编码了我们在观察数据之前对参数 (parameter)的信念。常见的选择包括高斯先验，通常中心为零 ($\mathcal{N}(0, \sigma^2)$)。方差 $\sigma^2$ 充当正则化 (regularization)参数。大方差意味着较弱的正则化（权重更灵活），而小方差意味着更强的正则化（权重更接近零）。$\sigma^2$ 的选择可以通过经验贝叶斯（从数据估计，尽管可能复杂）或根据领域知识或交叉验证进行设置。层次化先验，当 $\sigma^2$ 本身具有先验时，也可以用于增加灵活性。
- **网络架构：** 架构（层、激活函数 (activation function)）的选择遵循与标准深度学习 (deep learning)相似的原则，但与贝叶斯方法有互动。例如，参数的数量直接影响后验分布的维度，从而影响推断的难度。

### 使用变分推断训练BNN

VI，特别是像反向传播 (backpropagation)贝叶斯 (Bayes by Backprop) 这样的方法，将推断重新构造为一个优化问题，即最大化证据下界 (ELBO)。

- **目标函数 (ELBO)：** 回顾ELBO：
  
  $$
  \mathcal{L}(\phi) = \mathbb{E}_{q_\phi(w)}[\log p(\mathcal{D}|w)] - \text{KL}(q_\phi(w) || p(w))
  $$
  
  这里，$q_\phi(w)$ 是由 $\phi$ 参数 (parameter)化的变分近似（通常是对角高斯），$p(\mathcal{D}|w)$ 是似然，$p(w)$ 是先验。第一项鼓励拟合数据，而第二项KL散度作为正则项，使近似后验接近先验。
- **优化：** 随机梯度上升方法（如Adam）通常用于最大化ELBO，相对于变分参数 $\phi$。梯度通常使用重参数化技巧进行估计，这允许通过采样过程进行反向传播。
- **收敛监控：** 在训练期间追踪ELBO值。它通常会增加并趋于平稳。单独监控似然项和KL散度项，以理解其中的权衡。有时，KL项可能在早期阶段占据主导，或者模型可能难以拟合数据（低似然）。由于随机梯度，波动是预期的，但稳定的趋势很重要。
- **KL退火：** 有时，对ELBO中的KL项加权有助于训练。从一个小权重 (weight)（例如接近0）开始，并逐渐增加到1（退火），可以使模型首先专注于拟合数据，然后先验的正则化 (regularization)效果再充分生效。这可以防止变分后验 $q_\phi(w)$ 过早地坍缩到先验上。

### 使用MCMC训练BNN

MCMC方法，例如随机梯度哈密尔顿蒙特卡洛 (SGHMC)，旨在直接从真实的后验分布 $p(w | \mathcal{D})$ 中生成样本。

- **采样过程：** MCMC算法不优化参数 (parameter)，而是模拟马尔可夫链，其平稳分布是目标后验。SGHMC通过使用小批量梯度来适应大型数据集的HMC，并引入摩擦项以抵消梯度噪声。
- **超参数 (hyperparameter)：** SGHMC涉及调整步长（学习率）$\epsilon$ 和摩擦系数 $\alpha$ 等参数。这些参数显著影响采样效率和稳定性。调整通常需要反复试验。周期性学习率等技术有时会有益。
- **预热和稀疏化：** MCMC链的早期样本（预热期）通常会被丢弃，因为链可能尚未达到平稳分布。为减少样本间的自相关性，常采用稀疏化（只保留每 $k$ 个样本），尽管其必要性存在争议，特别是在HMC/NUTS变体等高效采样器中。
- **收敛诊断：** 与VI的ELBO不同，MCMC收敛通过对生成样本进行诊断来评估。轨迹图（绘制迭代过程中采样到的参数值）应显示良好的混合性（围绕稳定均值的随机波动）。正式诊断方法，如Gelman-Rubin统计量 ($\hat{R}$)，需要运行多条链并比较链内和链间方差。$\hat{R}$ 值接近1表明收敛。

### 进行预测和估计不确定性

BNN的一个重要优点是它们提供不确定性估计的能力。预测是通过对权重 (weight)后验分布进行边缘化来做出的。

- **预测分布：** 对于新输入 $x^*$，预测分布为：
  
  $$
  p(y^* | x^*, \mathcal{D}) = \int p(y^* | x^*, w) p(w | \mathcal{D}) dw
  $$
  
  在实践中，这个积分通过使用从后验分布（通过MCMC获得）或变分分布（通过VI获得）中抽取的样本进行近似。
  
  $$
  p(y^* | x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p(y^* | x^*, w^{(s)})
  $$
  
  其中 $w^{(s)}$ 是从 $p(w | \mathcal{D})$ (MCMC) 或 $q_\phi(w)$ (VI) 中抽取的 $S$ 个样本。
- **预测：** 最终预测通常是该近似预测分布的平均值。对于分类任务，这涉及对多次前向传播中，使用不同采样权重获得的输出概率进行平均。
- **不确定性量化 (quantization)：** 预测分布的方差量化了不确定性。它可分解为：
  - **偶然不确定性：** 数据生成过程中的固有噪声。通过似然函数的方差进行建模（例如，回归任务中高斯似然的方差参数 (parameter)）。它无法通过更多数据来减少。
  - **认知不确定性：** 由模型参数引起的不确定性。反映在后验分布 $p(w | \mathcal{D})$ 的方差中。它可以通过更多数据来减少。
    在实践中，总预测方差是从 $S$ 个预测 $y^{*(s)}$ 的方差中估计的，如果明确建模，则可能与估计的偶然不确定性方差相结合。高方差表明低置信度。

### 评估BNN性能

评估应评估预测准确性以及不确定性估计的质量。

- **标准指标：** 准确率、F1分数（分类）、均方根误差 (RMSE)、平均绝对误差 (MAE)（回归）是根据平均预测计算的。这些指标衡量预测分布的中心趋势。
- **概率指标：**
  - **负对数似然 (NLL)：** 衡量整个预测分布 $p(y^* | x^*, \mathcal{D})$ 与测试数据的拟合程度。NLL越低越好。
  - **校准：** 评估预测概率是否与经验频率匹配。例如，如果模型对一组预测预测了80%的置信度，那么这些预测中大约80%应该是正确的。可靠性图（校准图）可视化了这一点。



[交互图表：可靠性图（校准图）](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/practical-training-evaluation-bnns#plot-1h832mk)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9
      ],
      "y": [
        0.08,
        0.25,
        0.55,
        0.65,
        0.92
      ],
      "type": "bar",
      "name": "模型置信度",
      "marker": {
        "color": "#339af0"
      }
    },
    {
      "x": [
        0,
        1
      ],
      "y": [
        0,
        1
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "理想校准",
      "line": {
        "color": "#495057",
        "dash": "dash"
      }
    }
  ],
  "layout": {
    "title": "可靠性图（校准图）",
    "xaxis": {
      "title": "预测置信度（bin中点）"
    },
    "yaxis": {
      "title": "实际准确率"
    },
    "bargap": 0.2,
    "width": 600,
    "height": 400
  }
}
```

</details>



> 该可靠性图显示了预测置信度水平（分组到不同的区间）与在这些区间内观察到的实际准确率之间的关系。对角线上方的条形表示置信度不足，而对角线下方的条形表示置信度过高。

- **不确定性质量：** 评估模型对于分布外 (OOD) 样本或错误分类的例子是否比正确分类的分布内样本更不确定。可以在这些组之间比较预测熵或方差等指标。

### 实际挑战与考虑

- **计算成本：** 训练BNN，特别是使用MCMC时，通常比训练标准DNN计算量更大。VI速度更快，但会引入近似误差。预测也需要多次前向传播。
- **超参数 (parameter) (hyperparameter)敏感性：** 性能可能对先验选择、变分族结构（针对VI）、MCMC采样器参数（步长、摩擦）以及优化超参数（学习率、批量大小）敏感。通常需要仔细调整和敏感性分析。
- **软件：** TensorFlow Probability (TFP)、Pyro (PyTorch)、PyMC和Stan等库提供构建和训练BNN的工具，可以实施各种推断算法和概率层。熟悉这些工具会很有帮助。
- **可解释性：** 尽管BNN提供了不确定性，但解释不确定性的*来源*（偶然不确定性与认知不确定性）以及特定先验的影响需要仔细分析。

"训练和评估BNN需要思维方式从点估计转向分布。虽然有挑战，但由此产生的模型提供了对预测置信度更丰富的理解，这在许多需要可靠性和风险评估的应用中具有重要价值。"

## 参考资料

- [Weight Uncertainty in Neural Networks](http://proceedings.mlr.press/v37/blundell15.pdf) — Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 37; Pages: 1618-1626
  介绍了Bayes by Backprop，一种广泛用于训练贝叶斯神经网络的变分推断方法，详细阐述了ELBO目标函数和重参数化技巧。
- [Stochastic Gradient Hamiltonian Monte Carlo](https://proceedings.mlr.press/v32/cheni14.html) — Tianqi Chen, Emily Fox, Carlos Guestrin (2014)
  Journal: Proceedings of the 31st International Conference on Machine Learning; Publisher: PMLR; Volume: 32; Pages: 1683-1691
  介绍了随机梯度哈密顿蒙特卡洛（SGHMC），这是一种使用小批量梯度训练贝叶斯神经网络的关键MCMC算法，如本节所述。
- [On Calibration of Modern Neural Networks](http://proceedings.mlr.press/v70/guo17a.html) — Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1321-1330
  研究了现代神经网络中的校准问题，并介绍了可靠性图，为评估贝叶斯神经网络的概率预测质量提供了背景信息。

---

[上一节](08-%E5%B0%86%20Dropout%20%E8%A7%86%E4%B8%BA%E8%BF%91%E4%BC%BC%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%8E%A8%E6%96%AD.md) · [下一节](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
