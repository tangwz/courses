---
course: "intro-normalizing-flows"
chapter: "building-normalizing-flows"
lesson: "planar-and-radial-flows"
sourceId: 8260
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-2-building-normalizing-flows/planar-and-radial-flows"
title: "平面流与径向流"
description: "分析平面和径向标准化流的架构与数学公式。"
order: 3
plots: ["plots/8260-0.json"]
sourceHash: "618c47a9c1e45f35beb76aac1593333b3054b73e7958ace30c7ad279cabb9f2d"
sourceCorrections: []
---

堆叠多个可逆函数可以创建表达能力极强的概率分布。为了在实际中应用，需要特定的数学函数，这些函数既要可逆，又要具备易于计算的雅可比行列式。平面流（Planar flows）和径向流（Radial flows）是满足这些严苛要求的两种基础架构。这些模型为理解如何将简单的概率密度扭曲成复杂的几何形状提供了良好的起点。

### 平面流

平面流通过沿特定超平面应用变换来修改输入空间。你可以将其想象成拿一张平整的橡胶板，沿着一条直线进行拉伸或压缩。垂直于该线的所有部分基本保持不变，而沿线空间则会扩张或收缩。

设 $z \in \mathbb{R}^D$ 为我们的连续输入向量 (vector)。平面流变换 $f(z)$ 的数学定义为：

$f(z) = z + u h(w^T z + b)$

这里，$w \in \mathbb{R}^D$ 和 $u \in \mathbb{R}^D$ 是可学习的参数 (parameter)向量，用于定义变换的方向和比例。项 $b \in \mathbb{R}$ 是移动超平面的标量偏置 (bias)。函数 $h$ 是一个平滑、可微的非线性激活函数 (activation function)。双曲正切函数（记作 $\tanh$）最常用于 $h$，因为它的导数性质良好且有界。

为了在标准化流中使用此变换，我们必须计算其雅可比矩阵的行列式。对 $f(z)$ 关于输入向量 $z$ 求导得出：

$\frac{\partial f}{\partial z} = I + u h'(w^T z + b) w^T$

该结果是一个单位矩阵 $I$ 加上两个向量的外积，并由激活函数的导数进行缩放。我们可以利用矩阵行列式引理高效地计算这种特定矩阵结构的行列式。该引理指出 $\det(A + u v^T) = (1 + v^T A^{-1} u) \det(A)$。将这个线性代数恒等式应用于我们的雅可比矩阵，可以得到一个简单得多的标量方程：

$\det\left(\frac{\partial f}{\partial z}\right) = 1 + u^T w h'(w^T z + b)$

这个标量结果的计算成本极低。它的运算时间为 $O(D)$，而计算一般的 $D \times D$ 矩阵行列式通常需要 $O(D^3)$ 时间。这种效率允许我们在训练过程中堆叠数百个平面流层，而不会产生无法承受的计算开销。

> 单个平面流变换的计算图，显示了将原始输入加到缩放后激活值上的跳跃连接。

为使流动有效，变换必须严格可逆。这要求雅可比行列式永远不为零且正负号保持一致。在实践中，我们对参数进行约束以确保 $1 + u^T w h'(w^T z + b) > 0$。使用 $\tanh$ 激活函数时，其导数 $h'$ 严格限定在 $0$ 到 $1$ 之间。我们通过在正向传播过程中稍微修改向量 $u$ 来强制实现可逆性，以满足几何条件 $u^T w \geq -1$。

### 径向流

平面流沿直线超平面应用变换，而径向流则引入从特定中心点向外辐射的畸变。你可以把径向流想象成在空间的特定坐标上放置一个放大镜。空间要么从该点向外拉伸，要么向该点向内压缩。

对于输入向量 (vector) $z$，径向流变换定义为：

$f(z) = z + \beta h(\alpha, r)(z - z_0)$

这里，$z_0 \in \mathbb{R}^D$ 是可学习的参考点或流的中心。标量 $r = ||z - z_0||$ 代表输入与参考点之间的欧几里得距离。参数 (parameter) $\alpha \in \mathbb{R}^+$ 和 $\beta \in \mathbb{R}$ 决定了畸变的范围和强度。函数 $h(\alpha, r)$ 设计为随着距离 $r$ 的增加而衰减，通常定义为 $1 / (\alpha + r)$。

将这些组件结合起来，得到显式公式：

$f(z) = z + \beta \frac{1}{\alpha + ||z - z_0||} (z - z_0)$

径向流的雅可比行列式也受益于特殊的计算方法，避免了构建完整的 $D \times D$ 矩阵。使用类似的矩阵恒等式，行列式的计算结果为：

$\det\left(\frac{\partial f}{\partial z}\right) = \left( 1 + \beta h(\alpha, r) \right)^{D-1} \left( 1 + \beta h(\alpha, r) + \beta h'(\alpha, r) r \right)$

与平面流一样，这种行列式计算在线性时间 $O(D)$ 内完成。为了保证径向流的可逆性，我们在数学上限制参数 $\beta$，使其满足 $\beta \geq -\alpha$。



![Interactive chart](plots/8260-0.json)



> 应用可逆变换层前后的概率密度函数。

单个平面层或径向层在建模能力上相当有限。单个平面层仅沿一个方向压缩或扩张，而单个径向层仅围绕单个点进行扭曲。为了逼近极不规则的概率密度，我们按顺序应用这些变换。通过将 $K$ 个层串联在一起，整体变换变得具有高度非线性，完全能够将简单的各向同性基础分布映射为极其复杂的目标形态。
