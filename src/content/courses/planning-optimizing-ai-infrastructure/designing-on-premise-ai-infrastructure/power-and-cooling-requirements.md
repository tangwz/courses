---
course: "planning-optimizing-ai-infrastructure"
chapter: "designing-on-premise-ai-infrastructure"
lesson: "power-and-cooling-requirements"
sourceId: 6987
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-2-designing-on-premise-ai-infrastructure/power-and-cooling-requirements"
title: "功耗与散热要求"
description: "计算并规划高密度GPU服务器的大量功耗和散热需求。"
order: 6
plots: []
sourceHash: "639aab972ddab1f4565971815c68fa584d983cc78bb8007f665ac9803136fd16"
sourceCorrections: []
---

高性能CPU和GPU是AI基础设施的核心，但它们带来的物理限制常常被低估。这些组件不仅功能强大，而且耗电量也很大，几乎将消耗的每一瓦电能都转化为热量。未能妥善规划供电和散热是一个常见的失误，会导致热节流、系统不稳定，甚至硬件过早损坏。本地部署系统的可靠性，与其说建立在其芯片上，不如说更多地依赖于其电力和散热基础设施。

### 计算功耗

规划的第一步是估算服务器在重度机器学习 (machine learning)工作负载下的总功耗。组件的散热设计功耗（TDP）是此计算的一个良好起点。虽然TDP技术上衡量的是组件冷却系统设计耗散的最大热量，但它可作为组件在持续负载下功耗的可靠估算。

要计算服务器的总功耗需求，您必须将所有主要组件的TDP相加。

- **GPU：** 这些通常是功耗最高的组件。像NVIDIA A100这样的高端数据中心GPU的TDP可以达到400W。
- **CPU：** 现代服务器CPU也有显著的电力需求，通常每个处理器在150W到280W之间。
- **主板、内存和存储：** 这些组件单独贡献较少，但总和起来会增加。对于主板、内存、NVMe驱动器和系统风扇的组合，一个安全的估算值约为150-250W。

让我们计算一个常见AI服务器配置的功耗预算：

- 4块NVIDIA A100 GPU：$4 \times 400W = 1600W$
- 2块AMD EPYC CPU：$2 \times 200W = 400W$
- 主板、内存、存储等：$\approx 200W$

组件的总估算功耗为：


$$
1600W + 400W + 200W = 2200W
$$


这2200W代表了组件所需的直流（DC）电源。然而，服务器的电源供应单元（PSU）从墙壁插座获取交流（AC）电，并将其转换为直流电，这个过程并非100%高效。高质量的服务器PSU具有“80 Plus”评级，白金或钛金评级表示效率达到90%或更高。

如果我们的服务器的PSU效率为90%，则从墙壁插座实际消耗的功率将是：


$$
\frac{2200W}{0.90} \approx 2444W
$$


这是您规划电路时必须使用的数值。对于生产系统，务必使用冗余电源。如果一个电源出现故障，另一个可以承担全部负载，从而避免在长时间训练运行期间意外关机。

### 规划供电

一个常见的错误是假设标准墙壁插座能够支持一台强大的AI服务器。在北美，典型的家用电路是120伏（V），并由一个15安培（A）的断路器保护。您可以安全持续抽取的最大功率是电路最大值的80%，即：


$$
120V \times 15A \times 0.80 = 1440W
$$


我们示例中的服务器功耗为2444W，将立即导致此断路器跳闸。即使是20A的电路（$120V \times 20A \times 0.80 = 1920W$）也无法满足需求。

这就是为什么数据中心和专用服务器机房使用更高电压的电路，通常是208V或240V。一个208V、20A的电路可以提供：


$$
208V \times 20A \times 0.80 = 3328W
$$


这为我们2444W的服务器提供了安全余量。在建立本地设施时，您必须与电工合作安装合适的大功率插座。对于服务器机架，您将使用电源分配单元（PDU），它本质上是一个设计用于安装在机架中，并从高安培墙壁电路向多个服务器分配电源的电源板。

> 电源从墙壁插座流向服务器组件，以及由此产生的必须由散热系统处理的热量。

### 计算散热需求

热力学定律是无情的：服务器消耗的几乎所有电能都转化为热量。必须将这些热量从服务器和房间中移走，以防止组件过热。衡量热量的标准单位是英制热量单位（BTU）。

瓦特到BTU每小时的转换很简单：


$$
1W \approx 3.412 \text{ BTU/小时}
$$


使用从墙壁插座消耗的2444W，我们可以计算服务器的散热量：


$$
2444W \times 3.412 \text{ BTU/小时} \approx 8339 \text{ BTU/小时}
$$


从这个角度来看，一个小型个人取暖器大约产生5,000 BTU/小时的热量。我们的单台AI服务器产生的热量远超于此。标准的办公室暖通空调系统并未设计来处理这种集中式的热负荷。将这样的服务器放置在狭小、不通风的房间里，会迅速将环境温度提升到导致硬件性能受限或完全关闭的水平。

### 散热策略

有效散热需要两级方法：管理服务器内部的气流和管理房间本身的温度。

#### 服务器层面散热

服务器机箱设计有特定的气流模式，通常是前向后。冷空气从前方吸入，流经组件，热废气从后方排出。务必不要阻塞这些通风口。大功率服务器使用高速、高压风扇（以每分钟立方英尺，或CFM衡量），其噪音远高于台式电脑。

#### 房间层面散热

即使只是一台AI服务器，通常也需要一台专用的便携式空调或“分体式”系统。这些设备的制冷量以BTU/小时为单位。对于我们8339 BTU/小时的服务器，您需要一台制冷量至少为10,000 BTU/小时的空调，以提供安全余量。

对于机架中的多台服务器，数据中心最佳实践变得适用。**热通道/冷通道**布局是一种简单而有效的策略。机架排列成行，所有服务器正面（进气口）面向一个通道（冷通道），所有服务器背面（热排气口）面向另一个通道（热通道）。这可以防止服务器吸入邻近服务器排出的热空气，显著提高散热效率。

归根结底，电力和散热是您本地基础设施的根本。它们不仅是操作细节，更是核心设计要求，直接影响系统的性能、可靠性和总拥有成本。

## 参考资料

- [Thermal Guidelines for Data Processing Environments, 5th Edition](https://webstore.ansi.org/Standards/ASHRAE/ASHRAEDatacomBook15thEdition) — ASHRAE Technical Committee 9.9 (2021)
  Publisher: ASHRAE
  提供数据中心热管理行业的认可实践和指南，涵盖环境类别、运行条件以及IT设备的冷却策略。
- [80 PLUS® Performance Specification](https://www.80plus.org/) — CLEAResult ()
  Publisher: CLEAResult
  电源单元效率认证的官方项目，详细说明了计算交流功耗所需的方法和各种认证等级（例如，铜牌、银牌、金牌、白金牌、钛金牌）。
- [Uptime Institute Data Center Site Infrastructure Tier Standard: Topology, Version III](https://uptimeinstitute.com/tiers) — Uptime Institute (2023)
  Publisher: Uptime Institute
  定义了数据中心基础设施弹性的行业标准分级系统，提供了任务关键型AI工作负载所需的电源和冷却冗余要求。
- [Thermal Management of High Performance Servers: Design, Modeling and Experiment](https://link.springer.com/book/10.1007/978-981-10-6725-7) — Yong Han, Xianfan Xu (2018)
  Publisher: Springer; DOI: [10.1007/978-981-10-6725-7](https://doi.org/10.1007/978-981-10-6725-7)
  详细研究了高性能服务器特有的先进热管理技术和挑战，包括组件级和系统级冷却策略。
