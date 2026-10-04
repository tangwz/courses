---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-neural-networks-architectures-training"
lesson: "overfitting-generalization-qnn"
sourceId: 1069
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-5-quantum-neural-networks-architectures-training/overfitting-generalization-qnn"
title: "量子神经网络的过拟合与泛化"
description: "分析量子神经网络中的过拟合现象以及提升泛化表现的方法。"
order: 7
plots: ["plots/1069-0.json"]
sourceHash: "0070ae543f1373423524a86c4d83011e47c59beab4ae231e8ca54041d37d07cd"
sourceCorrections: []
---

与经典神经网络 (neural network)一样，量子神经网络（QNN）易受过拟合 (overfitting)影响。过拟合是指模型对训练数据学习得过于充分，不仅学习了数据固有的规律，还吸纳了训练集中存在的噪声和特殊特征。尽管这会在训练数据上带来出色的表现，但模型无法对新的、未见过的数据进行泛化，而这正是任何机器学习 (machine learning)模型的最终目标。因此，理解、识别和减轻过拟合在量子神经网络开发中与在经典机器学习中同等重要。

本章前面讨论的关于量子神经网络架构、参数 (parameter)化量子电路（PQC）和训练复杂性的挑战，直接影响过拟合的出现方式以及我们如何应对它。

### 量子神经网络 (neural network)为何会过拟合 (overfitting)

有多个因素会增加量子神经网络模型过拟合的风险：

1. **参数 (parameter)化量子电路的表达能力：** 如第4章所述，参数化量子电路（PQC）可以设计成具有不同程度的表达能力。高度表达能力的PQC，通常表现为拥有大量参数、深度电路或特定的纠缠结构，具有表示复杂函数的强大能力。尽管有时需要高表达能力，但这同样意味着PQC可能会完美拟合训练数据中的噪声，从而导致泛化性能不佳。在使PQC足以表达目标函数和避免其变得过于复杂之间存在一种巧妙的平衡。
2. **有限的训练数据：** 许多当前的量子机器学习 (machine learning)实验和应用所使用的数据集相对较小，与经典深度学习 (deep learning)中常用的数据集相比。样本越少，任何模型（包括量子神经网络）都越容易记忆训练集，而不是学习到可泛化的特征。
3. **噪声：** 来自量子硬件的噪声（在第7章中进一步说明）甚至训练期间有限测量次数导致的统计噪声，都可能被优化器无意中学到。量子神经网络可能会找到在*特定噪声状况下*表现良好的参数，但这在噪声特征改变或理想评估（例如在模拟器或不同硬件上）时无法泛化。
4. **优化：** 与变分量子算法（VQA）和量子神经网络相关的复杂非凸优化（包括贫瘠高原的可能性，第4章）可能导致过拟合。优化算法可能会收敛到尖锐的最小值点，这些最小值点对应的解决方案非常适合训练数据，但位于参数空间的某些区域，其中微小扰动会导致输出的巨大变化，表明泛化性能不佳。
5. **高维参数空间：** 量子神经网络，与深度经典网络类似，可以拥有大量可训练参数$\theta$。如果没有足够的数据或正则化 (regularization)，在此高维空间 (high-dimensional space)中进行优化会增加找到仅对训练样本有效的参数配置的风险。

### 量子神经网络 (neural network)中过拟合 (overfitting)的识别

检测过拟合的主要工具仍然是比较模型在训练集和独立验证集上的表现。

- **训练/验证集划分：** 在训练之前，将数据集划分为（至少）一个训练集和一个验证集。量子神经网络仅在训练数据上进行训练。在训练周期中或之后，定期评估模型在训练集和验证集上的表现（例如，准确率、损失函数 (loss function)值）。
- **学习曲线：** 绘制训练指标和验证指标随训练迭代次数或周期数的变化图。过拟合的典型迹象是：训练损失持续下降而验证损失开始上升，或者训练准确率持续提升而验证准确率趋于平稳或下降。



![量子神经网络学习曲线](plots/1069-0.json)



> 说明过拟合的学习曲线。训练损失（蓝色）持续下降，而验证损失（橙色）最初下降，但在大约50次迭代后开始上升，表明模型开始拟合训练数据中的噪声。

- **偏差-方差权衡：** 过拟合与高方差相关。具有高方差的模型对特定的训练数据过于敏感。相反，过于简单的模型可能会欠拟合 (underfitting)（高偏差），未能捕获训练数据和验证数据中固有的结构。目标是找到一种模型复杂性（受PQC结构、参数 (parameter)数量影响），使其平衡偏差和方差，以达到最低的验证误差。

### 提升泛化能力的策略

几种技术，改编自经典机器学习 (machine learning)或量子电路特有，可以帮助减轻过拟合 (overfitting)并提升量子神经网络 (neural network)的泛化能力：

1. **正则化 (regularization)：**

   - *参数 (parameter)范数惩罚：* 类似于经典L1或L2正则化，可以在代价函数中添加一个基于PQC参数$\theta$范数的惩罚项。然而，此类惩罚对量子态几何的影响不如经典线性模型中直接，并且这种方法尚未成为标准做法。
   - *噪声注入：* 在训练过程中故意添加噪声（例如，模拟去极化噪声或门错误）有时可以起到正则化器的作用，类似于经典神经网络中的Dropout，阻止模型过度依赖任何单一路径或参数。
   - *架构约束：* 像量子卷积神经网络 (CNN)（QCNN）这样的架构本身就带有结构约束（例如，局部性），与通用全连接PQC相比，这可以作为一种正则化形式。
2. **参数化量子电路设计：**

   - *控制表达能力：* 选择表达能力与问题预期复杂性相符的PQC ansatz。避免不必要的深层或参数过多的电路。设计硬件高效的ansatz（第7章）通常会得到更简单的电路，这可能间接有助于泛化。
   - *问题专用Ansatz：* 如果问题具有已知的对称性或结构，尝试将其融入PQC设计中。这可以引导模型趋向相关特征空间，并减小参数的搜索范围。
3. **提前停止：** 这是最常见且有效的正则化技术之一。在训练期间监控验证集上的表现，并在验证表现开始下降时停止训练过程，即使训练表现仍在提升。保存对应于所观察到的最佳验证表现的模型参数。
4. **数据增强：** 尽管不像图像或文本数据那样明显，但研究在编码前增强经典输入数据的方法，或发展量子数据增强技术，可以帮助模型接触更多变体并提升稳健性。细致的经典预处理仍然重要。
5. **优化策略：**

   - *优化器选择：* 一些优化器可能不太容易收敛到尖锐的最小值点。SPSA等随机方法或Adam等自适应优化器可能与标准梯度下降 (gradient descent)有所不同。
   - *量子自然梯度（QNG）：* 如第4章所述，量子自然梯度（QNG）考虑了量子态空间的几何结构。通过遵循根据量子信息度量的最陡下降路径，量子自然梯度可能会得到比不了解这种几何结构的优化器所找到的解决方案泛化能力更佳的解，尽管其实际实现可能要求较高。
6. **集成方法：** 独立训练多个量子神经网络（例如，不同的参数初始化，PQC结构的微小变化），并平均它们对未见过数据的预测。这可以减少方差，但计算成本更高。

### 量子语境下的泛化

量子神经网络 (neural network)的量子属性在泛化方面引入了一些需要考量的独特之处：

- **贫瘠高原：** 尽管这主要是一个优化挑战，但由于梯度消失（第4章）而无法有效训练深度PQC，通常会导致使用较浅的电路。这种限制可能*间接*有助于避免与过于复杂模型相关的一些极端过拟合 (overfitting)形式，尽管它主要限制了模型的容量。
- **纠缠与特征空间：** 参数 (parameter)化量子电路使用纠缠的方式（第1章，第1.3节）决定了量子特征空间（第2章）。PQC产生的纠缠量或结构与所得量子神经网络的泛化性能之间的关系是一个活跃的研究方向。并非更多的纠缠总是能带来更好或更差的泛化；纠缠结构与数据模式的*关联性*可能更重要。
- **测量效应：** 测量算子和后处理策略的选择影响从量子态中提取的信息，从而影响量子神经网络学习到的有效函数。即使使用相同的PQC和数据进行训练，不同的测量方案也可能导致模型具有不同的泛化特性。

总之，确保量子神经网络能够良好泛化到未见过的数据，对其实际用途非常重要。尽管许多技术借鉴自经典机器学习 (machine learning)，但参数化量子电路、量子噪声、测量和优化环境的独特特征需要认真考量，有时还需要量子特有的方法。监测验证集上的表现并应用适当的正则化 (regularization)或架构选择，是构建高效可靠的量子神经网络模型的必要步骤。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本经典教材，全面介绍了经典神经网络的过拟合、正则化和泛化原理，为理解量子神经网络奠定了基础。
- [Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://doi.org/10.1002/qute.201900070) — Sukin Sim, Peter D. Johnson, and Alán Aspuru-Guzik (2019)
  Journal: Advanced Quantum Technologies; Publisher: Wiley-VCH GmbH; Volume: 2; Pages: 1900070; DOI: [10.1002/qute.201900070](https://doi.org/10.1002/qute.201900070)
  讨论了参数化量子电路（PQCs）的设计如何影响其表达能力和纠缠能力，这直接关系到它们表示复杂函数的能力和过拟合的可能性。
- [Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  一篇开创性论文，指出了荒漠高原现象，这是训练深度量子神经网络的挑战，会影响其有效容量并间接影响其泛化能力。
- [The quantum natural gradient](https://quantum-journal.org/papers/q-2020-05-25-269/) — James Stokes, Josh Izaac, Nathan Killoran, Giuseppe Carleo (2020)
  Journal: Quantum; Publisher: Verein zur Förderung des Open Access Publizierens in den Quantenwissenschaften; Volume: 4; Pages: 269; DOI: [10.22331/q-2020-05-25-269](https://doi.org/10.22331/q-2020-05-25-269)
  介绍了量子自然梯度，这是一种考虑量子态空间几何的优化方法，可以改善变分量子算法的训练和泛化能力。
