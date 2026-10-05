# 在 Compose 中使用卷

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/compose-volumes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

容器默认是临时的。容器文件系统内写入的任何数据在容器移除时都会丢失。对于多容器应用，管理持久化数据（如数据集、训练好的模型、日志或数据库文件）变得十分必要。虽然 `docker run` 允许使用 `-v` 标志挂载卷，但 Docker Compose 提供了一种更有条理、更易于管理的方式来为服务定义和附加卷。

在 Docker Compose 中，您主要通过两个步骤管理卷：

1. **声明具名卷：** 您在 `docker-compose.yml` 文件的顶层 `volumes:` 部分下定义具名卷。这会告知 Compose 您的应用堆栈所需的卷。
2. **将卷挂载到服务：** 在每个服务定义中，您指定哪些已声明的卷应挂载到该容器中，以及它们在容器文件系统中应出现的位置，同样使用该服务特有的 `volumes:` 配置。

### 定义顶层卷

具名卷是持久化由 Docker 容器生成和使用的数据的首选机制。Docker 管理主机上的存储区域，您只需按名称引用该卷即可。

要声明一个具名卷，请在 `docker-compose.yml` 中添加一个顶层 `volumes:` 部分：

```yaml
version: '3.8' # 或更高版本

services:
  # ... 您的服务定义在此处 ...

volumes:
  postgres_data: # 声明一个名为 'postgres_data' 的具名卷
  ml_models:     # 声明另一个名为 'ml_models' 的具名卷
```

在此示例中，我们声明了两个具名卷：`postgres_data` 和 `ml_models`。如果这些卷在您的 Docker 主机上尚不存在，Compose 会在您首次运行 `docker-compose up` 时自动创建它们。您也可以在此处指定驱动程序选项（如果需要），但默认的本地驱动程序通常就足够了。

### 在服务中挂载卷

一旦声明了卷，就可以将其挂载到一个或多个服务中。在特定的服务定义下，使用 `volumes:` 键（注意：这与顶层键不同）。语法通常是 `volume-name:/path/in/container`。

我们来扩展前面的例子。假设您有一个用于存储实验元数据的 PostgreSQL 数据库服务，以及一个需要访问训练模型的推理 (inference) API 服务。

```yaml
version: '3.8'

services:
  db:
    image: postgres:14-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: ml_metadata
    volumes:
      - postgres_data:/var/lib/postgresql/data # 挂载 'postgres_data' 卷

  inference_api:
    build: ./inference_service # 假设此处有一个 Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ml_models:/app/models # 挂载 'ml_models' 卷
    depends_on:
      - db

volumes:
  postgres_data:
  ml_models:
```

情况如下：

- **`db` 服务：** 我们将 `postgres_data` 卷（在顶层声明）挂载到容器内标准的 PostgreSQL 数据目录 `/var/lib/postgresql/data`。现在，所有由 PostgreSQL 创建的数据库文件将在此 Docker 管理的卷中持久保存，即使容器重启或移除也能保留。
- **`inference_api` 服务：** 我们将 `ml_models` 卷挂载到容器内的 `/app/models`。如果一个训练服务（可能单独运行或定义在同一个 Compose 文件中）将模型保存到这个卷中，`inference_api` 服务就可以从 `/app/models` 加载它们。

这种设置确保了数据持久性，并在需要时允许容器之间共享数据（尽管在此特定示例中，每个卷仅由一个服务使用）。

> 示意图显示了 Docker Compose 中定义的两个服务，每个服务都挂载了一个由 Docker 管理的独立具名卷。

### 在 Compose 中使用卷的优点

通过 Docker Compose 管理卷带来了多项好处：

- **集中定义：** 应用堆栈所需的所有卷都在一个位置（`docker-compose.yml`）定义，使配置易于理解和管理。
- **自动创建：** 当应用启动时（`docker-compose up`），Compose 会自动处理具名卷的创建。
- **生命周期管理：** 即使容器停止和移除，卷也会保留。要连同 Compose 文件中定义的容器和网络一起明确移除卷，您可以使用 `docker-compose down -v` 命令。这简化了清理工作。
- **解耦：** 卷将数据的生命周期与容器的生命周期分离，这对于数据库、模型存储和数据集来说非常重要。

### 常见的机器学习 (machine learning)用例

在通过 Compose 管理的机器学习工作流中，卷常用于：

- **数据库：** 持久化用于实验跟踪、元数据存储或特征存储的数据库（如 PostgreSQL 或 MySQL）的数据。
- **模型存储：** 将训练服务生成的训练模型保存到卷中，然后推理 (inference)服务可以只读方式挂载此卷。
- **数据集：** 将大型数据集挂载到训练或处理容器中。虽然在开发过程中有时使用绑定挂载以便轻松访问本地文件，但具名卷对于 Docker 环境内管理的数据集可能很有用，特别是当它们由另一个容器服务（例如，数据下载/预处理服务）生成时。
- **共享产物：** 存储由一个服务生成并由另一个服务使用的中间产物（如预处理数据）。
- **日志：** 将不同服务生成的应用日志持久化到中心卷，以便后续分析或发送到日志系统。

通过在 `docker-compose.yml` 文件中定义和挂载卷，您可以获得一种可管理的方式来处理多容器机器学习应用的持久化数据，确保模型、数据集和配置等重要信息在单个容器生命周期结束后仍然存在。这种方法简化了设置，提高了可复现性，并与容器化良好实践保持一致。

## 参考资料

- [Compose file specification](https://docs.docker.com/compose/compose-file/) — Docker documentation team (2024)
  Publisher: Docker Inc.
  官方文档，详细说明了 `docker-compose.yml` 文件的结构和语法，特别是服务、网络和卷的定义。
- [Use volumes](https://docs.docker.com/storage/volumes/) — Docker documentation team (2024)
  官方文档，描述了 Docker 卷，包括命名卷、绑定挂载，以及它们在容器中实现持久化存储的用法。
- [Docker: Up & Running](https://www.oreilly.com/library/view/docker-up-running-3rd/9781098131784/) — Sean P. Kane, Karl Matthias (2023)
  Publisher: O'Reilly Media
  一本关于 Docker 的通用指南，涵盖了容器化、镜像管理、网络以及使用卷进行数据持久化等基本概念。

---

[上一节](03-%E5%AE%B9%E5%99%A8%E9%97%B4%E7%BD%91%E7%BB%9C%E8%BF%9E%E6%8E%A5.md) · [下一节](05-Compose%20%E4%B8%AD%E7%9A%84%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F.md)
