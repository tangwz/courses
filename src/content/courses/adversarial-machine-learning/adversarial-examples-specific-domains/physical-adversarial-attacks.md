---
course: "adversarial-machine-learning"
chapter: "adversarial-examples-specific-domains"
lesson: "physical-adversarial-attacks"
sourceId: 4189
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-7-adversarial-examples-specific-domains/physical-adversarial-attacks"
title: "物理对抗性攻击"
description: "了解在现实物理环境中创建有效对抗样本的挑战和方法。"
order: 4
plots: []
sourceHash: "9e9e32ad6f05aa6f97cc8a90739112ed6390b04282d75a8943f04aa84c419c16"
sourceCorrections: []
---

我们目前讨论的大多数对抗性攻击都在数字环境中运行。攻击者直接操作像素值，通常假设对模型接收的输入有完全的了解。然而，许多人工智能系统通过摄像头等传感器与物理环境交互。从数字比特到物理对象的这种转变，带来了一系列新的重要挑战和机遇。物理对抗性攻击旨在生成在真实环境中部署、被传感器捕获并由目标模型处理后仍能起效的扰动。

### 数字到物理的差异

创建有效的物理攻击比制作数字攻击要困难得多。核心难点在于**数字与物理间的差异**：数字上优化的扰动在真实环境中可能会因诸多不可控因素而失效。

以图像分类器上的攻击为例。数字攻击优化扰动模式$\delta$，使得$f(x + \delta)$导致错误分类，其中$x$是数字图像，$f$是模型。实际上，攻击者会制作一个物理对象（例如，打印的贴纸、改装的眼镜）来体现这种扰动。然后将此对象放置在环境中，并由摄像头捕获。

$physical\_object \rightarrow environment \xrightarrow{\text{捕获}} sensor \rightarrow preprocessing \rightarrow x' \rightarrow f(x')$

在此过程中，有几个因素会降低或改变对抗性模式：

1. **环境条件：** 光照变化（强度、色温、阴影）、天气影响（雨、雾）和遮挡都可能显著改变物体的外观。
2. **传感器噪声和处理：** 摄像头传感器会引入噪声。内部摄像头软件可能会应用色彩校正、伽马校正或压缩（例如JPEG）等变换，这些通常是未知且不可微的。
3. **视角变化：** 物体将从不同角度、距离和透视被查看，导致几何失真（仿射、投影变换）。
4. **制作限制：** 物理复制并非完美。打印机有有限的色域（可打印颜色）、分辨率限制，且材料特性会影响反射和外观。应用扰动可能涉及贴纸、颜料或3D打印，每种都有其自身的限制。

### 物理攻击的技术

为了克服这些挑战，物理攻击必须对各种变换和环境条件都有效力。实现这一目标的一个重要方法是**变换期望 (EOT)**。

#### 变换期望 (EOT)

由Athalye等人（2018）提出，变换期望旨在生成在变换分布$T$下仍然有效的对抗性样本。EOT不是针对单个输入$x$优化扰动$\delta$，而是对其进行优化，以使一组变换$t \in T$下的预期损失最大化。

目标函数变为：

$\max_{\delta} \mathbb{E}_{t \sim T} [ \mathcal{L}(f(t(x + \delta)), y_{target}) ]$

这里，$\mathcal{L}$是损失函数 (loss function)（例如，针对目标类别$y_{target}$的交叉熵），$f$是模型，$x$是原始输入（或被扰动对象的表示），$\delta$是扰动，$t$表示从分布$T$中采样的变换。

分布$T$通常包含物理环境中预期的变换：

- 旋转和平移
- 缩放（放大缩小）
- 亮度和对比度调整
- 透视变化
- 添加噪声

通过在此分布上优化，生成的扰动$\delta$在物理部署时更有可能抵御这些变化。变换期望通常通过在迭代优化过程（如PGD）的每一步采样变换$t$来实现。

#### 物理攻击的案例

研究已经展示了在各种计算机视觉任务中成功的物理攻击：

1. **攻击图像分类器（例如，交通标志）：** Elovic等人（2017）展示了精心制作的贴纸贴在停车标志上，即使从不同角度和距离观察，也能导致分类器将其误判为限速标志或其他物体。这些攻击常使用变换期望等技术。

   > 该图展示了使用对抗性贴纸对停车标志分类器进行物理攻击的过程。
2. **欺骗人脸识别：** 对抗性模式可以集成到眼镜框中（Sharif等人，2016）或作为补丁佩戴（Thys等人，2019），以导致错误识别（规避识别）或冒充。这些常需要仔细优化，考虑面部特征点和3D结构。
3. **攻击物体检测器：** 对抗性补丁可以设计成使检测器无法识别特定物体（例如，使YOLO检测器无法检测到某人），或导致虚假物体检测。这些补丁通常需要相对较大且放置位置需仔细考量。

### 制作与评估物理攻击的难点

尽管变换期望等技术提高了适应性，但物理攻击仍然面临障碍：

- **优化难度：** 包含物理变换（特别是像传感器噪声或复杂光照等不可微的变换）使优化变得困难得多。常需要替代或近似方法。
- **制作准确性：** 精确复制优化的数字扰动具有挑战性。数字规格与打印输出之间的颜色差异（“色域”问题）很常见。
  "\* **评估成本与精力：** 严格的评估需要在不同条件下进行大量测试，这与数字评估相比耗时且资源密集。模拟器虽有帮助，但可能无法完全反映复杂性。"
- **检测与缓解：** 物理扰动通常在视觉上很显眼（例如，大而图案奇怪的贴纸），使它们可能被人或专用防御机制检测到。然而，更不显眼的物理攻击是当前一个活跃的研究方向。

物理对抗性攻击强调了评估模型安全的重要性，不仅在模拟中，而且在实际部署条件下。它们是理解机器学习 (machine learning)在与物理环境交互的系统中实际安全影响的重要一步。随着人工智能在自动驾驶和监控等方面的部署增加，理解和防御这些攻击变得愈发重要。

## 参考资料

- [Synthesizing Robust Adversarial Examples](http://proceedings.mlr.press/v80/athalye18a/athalye18a.pdf) — Anish Athalye, Logan Engstrom, Andrew Ilyas, Kevin Kwok (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research (PMLR); Volume: 80; Pages: 284-293; DOI: [10.5555/3326943.3327170](https://doi.org/10.5555/3326943.3327170)
  介绍了期望变换（EOT）方法，用于生成对物理变换（包括影响交通标志的变换）具有鲁棒性的对抗性样本。
- [Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition](https://dl.acm.org/doi/10.1145/2976749.2978392) — Mahmood Sharif, Sruti Moosavi-Dezfooli, Arash Taheri, L.R.K. Singh, Adrian G. Steyrl (2016)
  Journal: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS); Publisher: ACM New York, NY, USA; Pages: 1528-1540; DOI: [10.1145/2976749.2978392](https://doi.org/10.1145/2976749.2978392)
  展示了使用特殊设计的眼镜框对人脸识别系统进行物理对抗性攻击。
- [Fooling automated surveillance cameras: adversarial patches to attack person detection](https://arxiv.org/pdf/1904.08654.pdf) — Simen Thys, Wiebe Van Ranst, Toon Goedemé (2019)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW); Publisher: IEEE Computer Society Conference Publishing Services (CPS); Pages: 0-0; DOI: [10.1109/CVPRW.2019.00204](https://doi.org/10.1109/CVPRW.2019.00204)
  介绍了能够使人对YOLO等目标检测器不可见的物理对抗性补丁。
