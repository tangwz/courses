# 分析RLHF的性能与稳定性

来源：[原文](https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/analyzing-rlhf-performance-stability)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对于包含已训练的奖励模型 $r_\theta(x, y)$ 和使用PPO微调 (fine-tuning)的语言模型策略 $\pi_\phi(y|x)$ 的模型，性能和稳定性分析是主要关注点。这种分析也扩展到训练过程本身。仅仅在训练期间获得高奖励分数是不够的；我们需要确保模型行为符合预期，并且训练过程是可靠的。

### 评估RLHF后的策略表现

评估RLHF的有效性涉及多种定量和定性方法：

1. **奖励分数提升：** 最直接的衡量方式是奖励模型 $r_\theta$ 为最终策略 $\pi_\phi$ 生成内容所分配的平均奖励，与初始监督微调 (fine-tuning)（SFT）策略相比。绘制PPO训练步骤中的奖励变化图有助于看到收敛情况。然而，要警惕奖励作弊：如果奖励模型本身有缺陷或容易被利用，高奖励分数并不总能保证真正的对齐 (alignment)。
2. **KL散度监测：** 在PPO训练中，我们使用KL散度项来惩罚偏离原始策略 $\pi_{SFT}$ 过大的情况： $KL(\pi_\phi(y|x) || \pi_{SFT}(y|x))$。监测这个KL值很重要。

   - **低KL：** 表示策略没有偏离SFT模型太远，可能保留了流畅性和能力，但对齐可能没有明显提升。
   - **高KL：** 表明策略已大幅改变以最大化奖励。如果对齐得到改善，这可能是好的，但过高的KL值可能导致生成质量下降、能力丢失或策略崩溃。
   - PPO目标中的KL系数 $\beta$ 直接控制着这种权衡。分析对 $\beta$ 的敏感度通常是必要的。
3. **在评估基准上的表现：** 在标准NLP基准（如GLUE、SuperGLUE）上运行RLHF调优后的模型 $\pi_\phi$，检查是否存在能力退化。更重要的是，在对齐和安全基准（例如HELM子集、TruthfulQA、Anthropic的HHH评估或定制内部基准）上进行评估，以专门衡量在所需特点（如有用性、诚实性和无害性）上的改进。将这些分数与基线SFT模型进行比较。
4. **人类偏好评估：** 最终，人类判断仍是黄金标准。使用与奖励建模相同的偏好收集界面，进行A/B测试，比较 $\pi_\phi$ 与 $\pi_{SFT}$（或其他模型变体）的生成内容。根据人类评估者的判断， $\pi_\phi$ 的高胜率有力地表明了成功的对齐。

### 分析训练稳定性

PPO优化过程本身可能很复杂，且容易出现不稳定，尤其是对于大型模型。分析训练动态对于诊断问题和确保可复现性很重要。

1. **奖励和KL轨迹：** 绘制每个PPO批次或回合的平均奖励和平均KL散度。稳定的训练通常表现为奖励稳步增加，而KL散度保持在可控范围内（如果使用自适应KL控制，通常会在目标KL值附近波动）。奖励的突然飙升或下降，或失控的KL散度，都表明存在不稳定。

   
   
   [交互图表：RLHF PPO训练动态](https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/analyzing-rlhf-performance-stability#plot-jszj2b)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "layout": {
       "title": "RLHF PPO训练动态",
       "xaxis": {
         "title": "训练步数"
       },
       "yaxis": {
         "title": "奖励分数",
         "range": [
           0,
           5
         ]
       },
       "yaxis2": {
         "title": "KL散度",
         "overlaying": "y",
         "side": "right",
         "range": [
           0,
           50
         ],
         "showgrid": false
       }
     },
     "data": [
       {
         "type": "scatter",
         "mode": "lines",
         "name": "奖励",
         "x": [
           0,
           100,
           200,
           300,
           400,
           500
         ],
         "y": [
           1.5,
           2.8,
           3.5,
           4.0,
           4.2,
           4.3
         ],
         "line": {
           "color": "#228be6"
         }
       },
       {
         "type": "scatter",
         "mode": "lines",
         "name": "KL散度",
         "x": [
           0,
           100,
           200,
           300,
           400,
           500
         ],
         "y": [
           0,
           15,
           22,
           25,
           26,
           27
         ],
         "yaxis": "y2",
         "line": {
           "color": "#fd7e14"
         }
       }
     ]
   }
   ```
   
   </details>
   
   

   > PPO训练曲线示例，显示奖励增加和KL散度稳定。
2. **熵：** 监测策略输出分布 $\pi_\phi(y|x)$ 的熵可以提供信息。熵衡量策略预测中的不确定性或随机性。一个变得过于确定性（低熵）的策略可能会狭隘地迎合奖励模型，并且泛化能力差。PPO通常包含一个熵奖励以鼓励多样性。熵的骤降可能指向过度优化或不稳定。
3. **价值函数损失：** 分析PPO训练中学到的价值函数 $V(x)$ 的损失。这个损失应该减少并稳定。大或波动的价值损失可能表明在估计未来奖励时存在问题，可能导致策略更新不稳定。
4. **超参数 (parameter) (hyperparameter)敏感度：** RLHF训练，尤其是PPO，对超参数很敏感，比如学习率、KL系数 $\beta$、批次大小（PPO批次大小、小批次大小）以及PPO特有参数（例如，裁剪比率 $\epsilon$、PPO回合数）。稳定的训练通常需要仔细调整。记录所用参数并可能分析微小变化的影响是一种好方法。

### 定性评估

除了指标和图表，定性分析也是不可或缺的。

- **人工生成内容检查：** 从最终策略 $\pi_\phi$ 中抽取大量不同提示（尤其是在训练和评估中使用的提示）的生成内容。检查以下几点：
  - 持续符合预期行为（例如，拒绝有害请求、提供有用的回答、在需要时引用来源）。
  - 一些不易察觉的失败模式，如过度奉承（过度同意）、回避，或生成虽不易察觉但仍可能获得高奖励分数的错误/有偏见内容。
  - 与SFT模型相比，创造力丧失或风格退化。
- **红队比较：** 比较RLHF前后红队提示（旨在引发不良行为）的有效性。成功的RLHF应该使模型对这些探测更加有抵抗力。

"有效的分析结合了这些定量指标、训练稳定性检查和定性评估。这种全面方法使得我们有信心，RLHF过程不仅增加了可衡量的奖励，而且以可靠的方式真正改善了模型的对齐 (alignment)和安全。未能进行此分析，可能导致部署一个基于表面指标看似对齐、但在实际条件下却隐藏漏洞或表现出不良行为的模型。"

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  描述了核心 PPO 算法，该算法是 RLHF 优化过程的中心，详细阐述了剪切代理目标、KL 散度惩罚和价值函数估计等概念。
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  介绍了一个用于大型语言模型的全面评估框架，涵盖了各种能力和风险，与“评估基准上的性能”部分直接相关。
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) — Stephanie Lin, Jacob Hilton, Owain Evans (2022)
  Journal: ACL 2022; DOI: [10.48550/arXiv.2109.07958](https://doi.org/10.48550/arXiv.2109.07958)
  介绍了 TruthfulQA，这是一个旨在衡量语言模型回答问题时真实性的基准，与评估对齐的“诚实性”方面高度相关。

---

[上一节](06-PPO%20%E5%AE%9E%E6%96%BD%E8%80%83%E9%87%8F.md) · [下一节](08-RLHF%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%E6%89%A9%E5%B1%95.md)
