---
course: "federated-learning"
chapter: "addressing-heterogeneity-personalization"
lesson: "sources-fl-heterogeneity"
sourceId: 3572
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-4-addressing-heterogeneity-personalization/sources-fl-heterogeneity"
title: "异构性的来源：统计性与系统性"
description: "详细说明联邦学习中遇到的异构性类型：统计性（数据分布）和系统性（硬件、网络）。"
order: 1
plots: ["plots/3572-0.json"]
sourceHash: "eee9b37a41650db9afde2473e6db37d8512da0b6de1de156f92d2182381ab3ed"
sourceCorrections: []
---

联邦平均（FedAvg）算法虽然为协作模型训练提供了一个框架，但它在理想化条件下运行效果最佳，而这些条件在实际中很少成立。联邦网络的特点是客户端之间存在显著差异，这种现象通常被称为**异构性**。了解异构性的不同方面，是构建有效联邦学习系统的基础。我们主要区分两种类型：统计异构性与系统异构性。

### 统计异构性（非IID数据）

统计异构性是指参与客户端之间底层数据分布的差异。在理想的独立同分布（IID）情况下，每个客户端的本地数据集 $D_k$ 都将从相同的全局数据分布 $P(x, y)$ 中获取。然而，在大多数实际使用中，客户端 $k$ 上的数据分布，记作 $P_k(x, y)$，与全局分布以及与其他客户端（$P_{k'}(x, y)$，其中 $k \neq k'$）上的分布显著不同。这通常被称为非IID数据。

**来源：**

- **用户行为与偏好：** 在移动键盘预测或推荐系统等应用中，用户具有不同的词汇、兴趣和交互模式。
- **地理位置：** 从不同区域收集的数据可能体现出当地语言、人口结构或环境条件（例如，传感器数据）。
- **时间：** 在不同时间收集的数据可能显示出时间变化或季节性。
- **设备特有性：** 不同设备可能以不同方式捕获数据（例如，相机传感器、麦克风质量）。
- **数据划分：** 在跨筒仓设置中，组织自然拥有代表不同人群或业务部分的数据（例如，专攻不同医疗领域的医院）。

**表现形式：** 非IID数据可以以几种方式呈现：

1. **特征分布偏差：** 客户端间的 $P_k(x)$ 不同，但 $P(y|x)$ 相似。（例如，手写数字识别中，某些客户端只书写特定数字）。
2. **标签分布偏差：** 客户端间的 $P_k(y)$ 不同，但 $P(x|y)$ 相似。（例如，人脸识别中，客户端主要拥有特定个体的照片）。
3. **条件分布偏差（意义转变）：** 客户端间的 $P_k(y|x)$ 不同，即使 $P_k(x)$ 相似。（例如，情感分析中，某些短语的含义因文化而异）。
4. **数量偏差：** 客户端持有显著不同的数据量。

下方的可视化内容显示了数字分类任务（例如，MNIST）中四个客户端的标签分布偏差。



![示例：标签分布偏差（非IID数据）](plots/3572-0.json)



> 四个客户端中，每个数字标签的样本分布。客户端1主要有'1'，客户端2有'6'和'8'，客户端3有'0'和'3'，客户端4有'2'和'4'。这与IID情况下（分布大致均匀）形成鲜明对比。

**影响：** 统计异构性带来一些重要挑战：

- **客户端漂移：** 在本地训练步骤中，在倾斜数据上训练的本地模型可能会与全局目标显著偏离。聚合时，这些偏离的更新会产生破坏性干扰，减缓甚至阻止全局模型的收敛。
- **全局模型精度降低：** 最终的全局模型可能在整体数据分布上表现不佳，对于其本地分布与平均值偏离最大的客户端而言，表现尤其差。
- **公平性问题：** 模型可能会表现出偏见，对多数数据模式表现良好，但对少数群体或特定客户端则表现不佳。
- **通信轮次增加：** 与IID设置相比，可能需要更多的通信轮次才能达到目标精度。

### 系统异构性

系统异构性是指联邦网络中客户端在硬件能力、网络条件和可用性方面的差异。

**来源：**

- **硬件差异：** 客户端（尤其在跨设备设置中）拥有不同的CPU、GPU、内存容量和电池电量。
- **网络条件：** 客户端通过具有不同带宽、延迟和稳定性的网络连接（例如，快速WiFi与缓慢、间歇性的蜂窝连接）。
- **可用性：** 客户端可能由于用户活动、网络问题或电池限制而不可预测地加入或退出训练过程。

下图显示了系统异构性。

> 联邦网络中的客户端通常具有不同的计算能力、内存和网络连接。有些客户端可能离线或响应非常缓慢。

**影响：**

- **滞后者：** 在同步联邦学习中，服务器在聚合之前等待一部分客户端的更新，缓慢的客户端（滞后者）会成为瓶颈，延迟整个训练过程。快速客户端则保持空闲，浪费资源。
- **参与者掉线：** 客户端在训练轮次中途掉线可能导致更新丢失，如果处理不当，还可能导致聚合偏差。
- **资源利用效率低下：** 差异性使得难以在所有客户端上统一优化本地训练轮次数量（$E$）等参数 (parameter)。固定的 $E$ 对于慢速设备来说可能计算量过大，而对于快速设备来说则可能过少。
- **客户端选择偏差：** 朴素的客户端选择可能倾向于更快、更可靠的客户端，如果与统计异构性相关，这可能会使模型偏向它们的数据分布。
- **异步联邦学习的挑战：** 虽然异步联邦学习通过在更新到达时处理它们来避免滞后者的瓶颈，但它引入了诸如管理更新过时性（使用旧的全局模型）和确保收敛稳定性等挑战。

### 相互关联与结果

统计异构性和系统异构性通常并存。例如，使用较旧、性能较低设备的用户（系统异构性）可能与使用高端设备的用户相比，表现出不同的应用使用模式（统计异构性）。这种相互影响使优化过程进一步复杂化。未能处理异构性会导致收敛速度变慢、最终模型精度降低、客户端间可能出现不公平，以及部署可靠联邦学习系统遇到困难。本章后续部分将检查专门为缓解这些问题而设计的技术，范围从可靠的聚合规则到个性化模型训练。

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) — Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: PMLR; Volume: 54; Pages: 1273-1282
  介绍了联邦平均（FedAvg）算法，为联邦学习奠定了基础。
- [Advances and Open Problems in Federated Learning](https://doi.org/10.1561/2200000083) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, Sen Zhao (2021)
  Journal: Foundations and Trends in Machine Learning; Publisher: now publishers; Volume: 14; Pages: 1-210; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  对联邦学习的全面综述，讨论了统计异质性和系统异质性作为核心挑战。
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a.html) — Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan (2020)
  Journal: Proceedings of the 37th International Conference on Machine Learning; Publisher: PMLR; Volume: 119; Pages: 5132-5143
  提出了SCAFFOLD方法，以应对联邦学习中由统计异质性引起的客户端漂移。
