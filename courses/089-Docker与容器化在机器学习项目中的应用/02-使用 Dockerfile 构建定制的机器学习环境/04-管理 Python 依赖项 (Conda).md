# 管理 Python 依赖项 (Conda)

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/managing-conda-dependencies)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 `pip` 和 `requirements.txt` 是管理 Python 包的常用工具，但机器学习 (machine learning)项目通常涉及更复杂的依赖项，包括 CUDA、MKL 或系统工具等非 Python 库。Conda 在应对此类情况时表现突出。Conda 是一个开源的包管理系统和环境管理系统，可在多种操作系统上运行。它擅长安装包及其依赖项，尤其是那些包含二进制组件的包，并能创建隔离环境以避免冲突。将 Conda 集成到 Docker 工作流程中，能提供一个有效的方法来可靠地管理这些复杂的机器学习环境。

### 为什么在 Docker 中为机器学习 (machine learning)依赖项使用 Conda？

在 Docker 镜像中使用 Conda 有多个优点，尤其适合机器学习：

1. **全面的依赖项管理：** Conda 像 `pip` 一样管理 Python 包，但它还能处理 `pip` 无法处理的非 Python 库（例如，编译器、CUDA Toolkit、cuDNN、OpenBLAS）和系统级依赖项。这对于常依赖优化过的 C/C++/Fortran 库的机器学习框架来说非常重要。
2. **环境隔离：** Conda 创建不同的环境，每个环境都有自己的包集合和 Python 解释器。尽管 Docker 容器本身就提供隔离，但若有需要，在容器*内部*使用 Conda 环境可以帮助管理机器学习工作流程的不同阶段或组件，尽管通常每个容器一个环境就足够了。
3. **先进的冲突解决能力：** Conda 采用比 `pip` 更复杂的依赖项解析器，在查找兼容的包集合方面表现更好，尤其是在处理科学 Python 栈中常见的复杂相互依赖时。
4. **使用 `environment.yml` 实现可重现性：** 类似于 `requirements.txt`，Conda 使用 `environment.yml` 文件声明依赖项，包括特定版本和 Conda 渠道，确保任何人都可以精确重现该环境。

### 选择 Conda 基础镜像

最简单的开始方式是使用 Anaconda 公司提供的官方基础镜像：

- `continuumio/miniconda3`：包含最小的 Conda 安装器和 Python 3。这通常是 Docker 镜像的首选，因为它能保持更小的尺寸。你只需安装所需的包。
- `continuumio/anaconda3`：包含完整的 Anaconda 发行版，包括 Conda、Python 3 和数百个预安装的科学计算包。这会导致镜像尺寸大得多，除非你特别需要其中大部分包，否则通常是过度的。

另外，你可以从一个标准的 Python 镜像（例如 `python:3.9-slim`）或一个操作系统镜像（例如 `ubuntu:22.04`）开始，然后在 Dockerfile 中自行安装 Miniconda。这能提供更多的控制，但需要额外的步骤。

对于大多数机器学习 (machine learning)应用，使用 `continuumio/miniconda3` 作为开始能在便利性和镜像尺寸之间取得不错的平衡。

### 使用 `environment.yml` 进行可重现构建

定义 Conda 环境的标准方法是通过 `environment.yml` 文件。此文件列出环境名称、Conda 应搜索包的渠道以及所需的依赖项。

下面是一个用于基本机器学习 (machine learning)设置的 `environment.yml` 示例：

```yaml
# environment.yml (环境文件)
name: ml_env
channels:
  - defaults
  - conda-forge # 通常提供更新或特定的包
dependencies:
  - python=3.9
  - numpy=1.23
  - pandas>=1.4
  - scikit-learn=1.1
  - matplotlib
  - pip # 如果需要安装 Conda 无法提供的包，请包含 pip
  - pip:
    - some-pip-only-package==0.1.0
```

**说明：**

- `name`：指定 Conda 环境的名称 (`ml_env`)。
- `channels`：定义 Conda 搜索包的顺序。`defaults` 指的是 Anaconda 的默认渠道。`conda-forge` 是一个受欢迎的社区维护渠道。顺序很重要。
- `dependencies`：列出要安装的包。
  - 你可以指定精确版本（`numpy=1.23`）、最低版本（`pandas>=1.4`），或者让 Conda 选择最新的兼容版本（`matplotlib`）。
  - 包含 `pip` 允许你随后使用 `pip install`。
  - 嵌套的 `pip:` 部分允许你列出在 Conda 包设置*之后*通过 `pip` 安装的包。这对于仅在 PyPI 上可用的包很有用。

### 在 Dockerfile 中构建环境

要在 Docker 构建中使用此 `environment.yml` 文件，你通常会将其复制到镜像中，然后运行 `conda env create`。

以下是一个演示此过程的 Dockerfile 片段：

```dockerfile
# 使用 Miniconda 基础镜像
FROM continuumio/miniconda3

# 定义环境名称和文件的参数
ARG CONDA_ENV_NAME=ml_env
ARG CONDA_ENV_FILE=environment.yml

# 设置工作目录
WORKDIR /app

# 将环境文件复制到镜像中
COPY ${CONDA_ENV_FILE} .

# 创建 Conda 环境
# 合并更新和创建以减少层数
# 添加清理命令以减少镜像大小
RUN conda update -n base -c defaults conda && \
    conda env create -f ${CONDA_ENV_FILE} && \
    conda clean -afy

# 使 RUN 命令使用新环境：
SHELL ["conda", "run", "-n", "${CONDA_ENV_NAME}", "/bin/bash", "-c"]

# 示例：验证安装（在 ml_env 中运行）
RUN echo "正在验证 Python 和 Scikit-learn 安装..." && \
    python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')" && \
    echo "Conda 环境设置完成。"

# 复制其余的应用代码
COPY . .

# 设置在 Conda 环境中运行的默认命令
# 注意：上面的 SHELL 指令确保此命令在激活的环境中运行
CMD ["python", "your_script.py"]
```

**说明：**

1. `FROM continuumio/miniconda3`：从最小的 Conda 镜像开始。
2. `ARG`：定义构建时参数 (parameter)以增加灵活性（可选但良好实践）。
3. `WORKDIR /app`：设置容器内的工作目录。
4. `COPY ${CONDA_ENV_FILE} .`：将 `environment.yml` 文件复制到 `/app` 目录。
5. `RUN conda update ... && conda env create ... && conda clean ...`：这是核心步骤。
   - 它首先更新 Conda 的基础安装（良好实践）。
   - `conda env create -f ${CONDA_ENV_FILE}` 读取 YAML 文件并将所有指定包安装到名为 `ml_env` 的环境中。
   - `conda clean -afy` 删除未使用的包和缓存，显著减少最终镜像尺寸。将这些步骤组合成一个 `RUN` 命令有助于最小化层数。
6. `SHELL ["conda", "run", "-n", "${CONDA_ENV_NAME}", "/bin/bash", "-c"]`：这是一个非常重要的指令。它改变了后续 `RUN`、`CMD` 和 `ENTRYPOINT` 指令使用的默认 shell。命令现在将在激活的 `ml_env` Conda 环境*内部*执行，这意味着你不需要为每个步骤手动激活它。
7. `RUN echo ... && python ...`：这个 `RUN` 命令现在使用上面定义的 shell 执行，因此 `python` 指的是 `ml_env` 中的 Python 解释器。
8. `COPY . .`：将项目的其余文件（例如 Python 脚本、数据文件）复制到工作目录。
9. `CMD ["python", "your_script.py"]`：定义容器启动时运行的默认命令。由于 `SHELL` 指令，此命令也会在激活的 `ml_env` 中运行。

### Conda 和 Pip：协同工作

如 `environment.yml` 示例所示，你可以在 Conda 环境中使用 `pip`。这对于通过 Conda 渠道无法获得的包通常是必要的。Conda 首先安装 `dependencies` 下列出的包，然后使用 `pip` 安装嵌套 `pip:` 部分下列出的包。这种方法通常效果良好，允许你将 Conda 用于复杂依赖项，而将 `pip` 用于 Python 特定的依赖项。

### 总结与最佳实践

将 Conda 与 Docker 结合使用，提供了一个管理复杂机器学习 (machine learning)依赖项的解决方案：

- **优选 `miniconda3`：** 从 `continuumio/miniconda3` 开始，以获得更小、更易于管理的镜像。
- **使用 `environment.yml`：** 以声明方式定义你的环境以实现可重现性。在必要时固定版本。
- **指定渠道：** 在 `environment.yml` 中明确指定 Conda 渠道（`defaults`、`conda-forge` 等）。
- **激活环境：** 使用 `SHELL` 指令或 `conda run -n <env_name>` 来确保你的命令在正确的 Conda 环境中执行。`SHELL` 方法对于后续的 `RUN`、`CMD` 和 `ENTRYPOINT` 指令通常更简洁。
- **清理：** 始终在与 `conda env create` 或 `conda install` *相同的* `RUN` 层中运行 `conda clean -afy`，通过删除缓存文件来最小化镜像尺寸。
- **组合命令：** 在单个 `RUN` 指令中使用 `&&` 链接相关命令（如 `conda update`、`conda env create`、`conda clean`），以减少镜像层数。

通过将 Conda 有效地集成到你的 Dockerfile 中，你创建了能够处理现代机器学习开发和部署所需的复杂软件栈的自包含、可重现环境。这解决了机器学习项目中常见的阻碍，确保了不同机器和机器学习生命周期阶段的一致性。

## 参考资料

- [Conda Documentation](https://docs.conda.io/en/latest/) — Anaconda, Inc. (2024)
  提供关于Conda包和环境管理的全面指导，包括`environment.yml`和通道。
- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/) — Docker Inc. (2024)
  Dockerfile指令（`FROM`、`RUN`、`COPY`、`CMD`、`SHELL`）的官方指南，对于构建自定义Docker镜像很重要。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  讨论机器学习的软件工程原则，涵盖依赖管理和可复现环境等主题，这些都受益于Conda和Docker。

---

[上一节](03-%E7%AE%A1%E7%90%86%20Python%20%E4%BE%9D%E8%B5%96%E9%A1%B9%20%28pip%29.md) · [下一节](05-%E4%BD%BF%E7%94%A8%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F.md)
