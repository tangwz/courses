---
course: "intro-to-reinforcement-learning"
chapter: "foundations-reinforcement-learning"
lesson: "setting-up-python-environment-rl"
sourceId: 1571
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-1-foundations-reinforcement-learning/setting-up-python-environment-rl"
title: "为强化学习搭建Python环境"
description: "关于安装强化学习开发所需Python库（如Gymnasium、NumPy）的指南。"
order: 8
plots: []
sourceHash: "37620a7ff72918e446682412593c2fe85776843d0a5e7071ae33693c0985441d"
sourceCorrections: []
---

实现和试用强化学习 (reinforcement learning)算法需要准备合适的工作环境。搭建Python环境以用于强化学习，包括配置特定的工具并关注课程中会用到的主要库。配置得当的环境可确保您顺利运行代码示例并构建自己的强化学习智能体。

由于本课程假定您已熟悉Python及基本的机器学习 (machine learning)原理，我们认为您已安装Python（建议3.8或更高版本）和包管理器`pip`。如果尚未安装，请查阅Python官方文档获取安装说明。

### 使用虚拟环境管理依赖

在安装任何包之前，强烈建议使用虚拟环境。虚拟环境创建隔离的Python配置，避免项目依赖之间的冲突。这是Python开发中的常见做法。

您可以使用Python内置的`venv`模块创建虚拟环境：

1. **创建环境**（将`rl_env`替换为您喜欢的名称）：

   ```bash
   python -m venv rl_env
   ```
2. **激活环境**：

   - 在macOS和Linux上：

     ```bash
     source rl_env/bin/activate
     ```
   - 在Windows上：

     ```bash
     .\rl_env\Scripts\activate
     ```

激活后，您的终端提示符通常会改变，表示您正在虚拟环境中工作。此时安装的任何包都将仅限于此环境。

### 强化学习 (reinforcement learning)的主要库

尽管存在各种库，但NumPy（用于数值计算）和Gymnasium（用于标准化的强化学习环境）是我们工作中很重要的两个库。

#### NumPy：数值运算的根本

强化学习大量涉及数值数据：状态通常表示为向量 (vector)或矩阵，动作可能是数值型的，奖励也必定是数值型的。NumPy是Python中数值计算的根本库，提供高效的数组对象和数学函数。

- **NumPy的用途？** 它允许高效存储和操作数值数组，非常适合表示状态、动作值($Q(s, a)$)、状态值($V(s)$)以及管理批量经验数据。其向量化 (quantization)操作在数值任务上比标准Python列表快得多。
- **安装**：在虚拟环境激活后，使用pip安装NumPy：

  ```bash
  pip install numpy
  ```

您可以通过在Python解释器中导入它来快速验证安装：

```python
import numpy as np

# 示例：创建一个简单的NumPy数组
state = np.array([0.1, -0.5, 0.3, 0.8])
print(f"已创建NumPy数组：{state}")
print(f"数组的形状：{state.shape}")
```

#### Gymnasium：一个标准化的环境工具集

为了开发和比较强化学习算法，我们需要智能体可以互动的环境。Gymnasium（OpenAI Gym的一个分支和延续）为此类环境提供了标准API，从简单的玩具问题到更复杂的模拟，如经典控制任务和Atari游戏。

- **Gymnasium的用途？** 它提供了一个简单统一的接口（`reset`、`step`）用于与各种环境互动。这使得您可以专注于算法的逻辑，而不是每个环境实现的具体细节。使用标准化的环境也使得基准测试和比较不同算法变得更简单。
- **安装**：安装核心的Gymnasium包：

  ```bash
  pip install gymnasium
  ```

Gymnasium还提供了许多需要额外依赖的附加环境。例如，要安装对经典控制环境（如我们将经常使用的CartPole）和Atari游戏（需要ROM许可证）的支持，您可以使用：

```bash
# 安装经典控制及其他基本环境（通常默认包含）
pip install gymnasium[classic-control]

# 对于Atari游戏（需要接受ROM许可证）
# 关于Atari ROM的详情请查阅Gymnasium文档
pip install gymnasium[atari,accept-rom-license]
```

对于本课程，最初通常只需`gymnasium`基础包和`classic-control`即可。

#### （可选）Matplotlib：可视化学习进展

理解智能体如何学习通常需要将其性能可视化，例如绘制随时间变化的奖励或可视化价值函数。Matplotlib是Python中广泛使用的绘图库。

- **Matplotlib的用途？** 它提供了创建静态、动态和交互式可视化的工具。我们将使用它来绘制学习曲线及其他诊断图表。
- **安装**：

  ```bash
  pip install matplotlib
  ```

### 验证您的设置

让我们确保核心组件协同工作。创建一个简单的Python脚本（例如`verify_setup.py`），内容如下：

```python
import gymnasium as gym
import numpy as np

print(f"Gymnasium版本：{gym.__version__}")
print(f"NumPy版本：{np.__version__}")

try:
    # 创建一个简单环境
    env = gym.make("CartPole-v1", render_mode="rgb_array") # 如果需要图形输出，请使用“human”
    print("成功创建CartPole-v1环境。")

    # 重置环境以获得初始观测
    observation, info = env.reset(seed=42) # 使用种子以保证可重复性
    print(f"初始观测：{observation}")

    # 执行一个随机动作
    action = env.action_space.sample() # 采样一个随机动作（0或1）
    print(f"正在执行随机动作：{action}")

    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"下一观测：{observation}")
    print(f"获得的奖励：{reward}")
    print(f"回合结束：{terminated}")
    print(f"回合截断：{truncated}") # 截断表示达到时间限制

    # 关闭环境（清理很重要）
    env.close()
    print("环境互动成功。")

except Exception as e:
    print(f"验证过程中发生错误：{e}")
```

从已激活的虚拟环境运行此脚本：

```bash
python verify_setup.py
```

如果脚本运行无误，并打印出与代码注释类似的输出，那么您的基础强化学习 (reinforcement learning)环境就准备好了。您已成功安装NumPy用于数值运算，以及Gymnasium用于访问标准强化学习环境。您现在已准备好开始实现我们将在后续章节中介绍的算法和原理。

## 参考资料

- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2023)
  Python 官方文档，说明如何创建和管理独立的 Python 环境。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — NumPy Developers (2023)
  NumPy 官方指南，介绍其数组对象和核心数值计算功能。
- [Gymnasium Documentation](https://gymnasium.farama.org/) — The Gymnasium Contributors (2023)
  Gymnasium 官方文档，详细说明其标准化环境和用于 RL 代理交互的 API。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本广受认可的教材，介绍强化学习的核心概念和算法。
