---
course: "docker-for-ml-projects"
chapter: "managing-ml-data-containers"
lesson: "bind-mounts-development"
sourceId: 5273
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/bind-mounts-development"
title: "用于开发的绑定挂载"
description: "在开发过程中，将本地目录挂载到容器中，以快速修改代码和访问数据。"
order: 2
plots: []
sourceHash: "588b136589af39207f6ddd1607000d42858bf40752eff6d3d1f6a733d1d1840e"
sourceCorrections: []
---

容器提供了独立环境，但这种隔离在您需要访问或修改位于容器*外部*的文件时会带来一个难题，尤其是在您的机器学习 (machine learning)项目活跃开发阶段。每次修改一行Python代码或更新配置文件时都重建Docker镜像效率低下，并且会明显减慢迭代周期。

这就是**绑定挂载**起作用的地方。绑定挂载在您的宿主机（您计算机的文件系统）上的目录或文件与容器内部的目录或文件之间建立一个直接的链接或映射。与Dockerfile中的`COPY`指令不同，`COPY`指令在*构建时*创建文件的静态快照，而绑定挂载提供了一个实时、动态的连接。对宿主系统上已挂载目录或文件的更改会立即反映在容器内部，反之（取决于权限），在容器内已挂载路径中所做的更改也可能影响宿主系统的文件。

可以把它想象成在您的宿主机和容器之间创建一个共享文件夹。这种直接访问在开发过程中非常有用，原因有以下几点：

1. **快速代码迭代：** 使用您宿主机上首选的编辑器编辑Python脚本、Jupyter笔记本或配置文件。更新后的代码会立即在容器内部可用，让您可以重新运行脚本或分析，而无需重建镜像。
2. **访问本地数据：** 对于本地存储的较小数据集或配置文件，您可以将包含这些数据的宿主目录直接挂载到容器中，以便您的机器学习脚本使用。
3. **实时调试：** 使用依赖于访问源代码文件的调试工具。通过绑定挂载，在容器内部运行的调试器可以访问宿主机上最新的源文件。

### 绑定挂载的实现

当您使用`docker run`命令启动容器时，就可以创建绑定挂载。主要有两种标志用于此目的：

1. **`-v` 或 `--volume` 标志（更简洁的语法）：**

   ```bash
   docker run -v /path/on/host:/path/in/container image_name [command]
   ```

   此语法将您计算机上的`/path/on/host`直接映射到容器内部的`/path/in/container`。
2. **`--mount` 标志（更明确的语法）：**

   ```bash
   docker run --mount type=bind,source=/path/on/host,target=/path/in/container image_name [command]
   ```

   这种语法因其清晰性而常被选用。它明确指出`type`是`bind`，`source`是宿主路径，`target`是容器路径。如果需要，您还可以添加其他选项，例如`readonly`。

我们来看一个常见的机器学习 (machine learning)开发场景。假设您的宿主机上有一个项目结构如下所示：

```
/home/user/my_ml_project/
├── src/
│   └── train.py
├── data/
│   └── features.csv
└── Dockerfile
```

您已经使用包含Python、scikit-learn和pandas的`Dockerfile`构建了一个名为`my-ml-dev-env`的镜像。现在，您想在容器内运行`train.py`脚本，使用`features.csv`数据，同时仍然能够在您的宿主机上编辑`train.py`。

您可以使用绑定挂载来实现这一点：

```bash
# 使用 --mount 语法
docker run --rm -it \
  --mount type=bind,source=/home/user/my_ml_project/src,target=/app/src \
  --mount type=bind,source=/home/user/my_ml_project/data,target=/app/data \
  my-ml-dev-env \
  python /app/src/train.py --data_path /app/data/features.csv
```

或者使用`-v`简写：

```bash
# 使用 -v 语法
docker run --rm -it \
  -v /home/user/my_ml_project/src:/app/src \
  -v /home/user/my_ml_project/data:/app/data \
  my-ml-dev-env \
  python /app/src/train.py --data_path /app/data/features.csv
```

在以上两个例子中：

- 我们将本地的`src`目录映射到容器内部的`/app/src`。
- 我们将本地的`data`目录映射到容器内部的`/app/data`。
- 然后，我们执行位于*容器映射路径*`/app/src`内的`train.py`脚本。
- 脚本通过*容器的映射路径*`/app/data`访问数据。

现在，如果您在宿主机上打开`/home/user/my_ml_project/src/train.py`并进行更改，这些更改将在您下次运行`docker run`命令时立即生效（或者，如果您在容器内运行交互式shell或服务器，则在访问文件时会立即反映这些更改）。

> 该图示说明了如何使用绑定挂载，将宿主机上的目录（`my_ml_project/src`、`my_ml_project/data`）直接映射到容器的文件系统（`/app/src`、`/app/data`）中。

### 绑定挂载的注意事项

虽然绑定挂载对开发来说功能强大，但请记住以下几点：

- **宿主依赖性：** 绑定挂载在容器与特定宿主机的文件结构之间创建了紧密耦合。源路径*必须*在您运行`docker run`命令的宿主机上存在。如果镜像严重依赖于挂载特定的宿主路径，这将降低镜像的可移植性。
- **权限：** 文件所有权和权限有时会引起问题。容器内部的用户可能没有必要的权限来读取或写入已挂载的宿主目录，或者在容器内部创建的文件在宿主机上可能具有意外的所有权。Docker会尝试管理这一点，但尤其是在不同的操作系统之间，可能会出现不一致。
- **性能：** 对于大型文件的I/O密集型操作，与Docker管理的卷相比，绑定挂载可能会产生轻微的性能开销，尽管对于典型的开发任务，这通常可以忽略不计。
- **平台差异：** 文件路径在Windows (`C:\Users\...`) 和 Linux/macOS (`/home/user/...`) 上指定方式不同。在跨不同开发环境共享`docker run`命令或脚本时，请注意这一点。

绑定挂载是简化机器学习 (machine learning)开发内部循环的不可或缺的工具：在Docker容器提供的一致环境中编辑代码、运行实验和访问本地资源。当您需要那种直接、实时连接时，它们正好弥合了宿主机与容器化环境之间的差距。然而，为了更稳定地管理持久数据，尤其是在非开发场景或在容器之间共享数据时，Docker提供了另一种机制：卷（Volumes），我们接下来将查看它。

## 参考资料

- [Use bind mounts](https://docs.docker.com/storage/bind-mounts/) — Docker Documentation (2024)
  Publisher: Docker Inc.
  阐述了Docker容器中绑定挂载的概念和实现方法。
- [Storage: Understand storage drivers, volumes, and bind mounts](https://docs.docker.com/storage/) — Docker Documentation (2024)
  概述了Docker的多种存储选项，比较了绑定挂载与其他持久化存储方法的区别。
- [Docker: Up & Running: Shipping Reliable Containers in Production](https://www.oreilly.com/library/view/docker-up-running/9781098135891/) — Karl Matthias, Sean P. Kane (2023)
  Publisher: O'Reilly Media; Pages: 416
  一本关于Docker基础、容器化和数据管理方法的指南，涵盖了开发和生产中数据处理的实践。
