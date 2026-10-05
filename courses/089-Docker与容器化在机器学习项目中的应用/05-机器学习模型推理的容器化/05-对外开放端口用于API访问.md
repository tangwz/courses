# 对外开放端口用于API访问

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/exposing-ports-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要使容器化推理 (inference)API能够从外部访问，尤其是在宿主机或网络上的其他机器上，端口暴露至关重要。在Docker容器中运行Flask或FastAPI等Web服务器应用程序时，它会*在*容器的隔离网络环境中监听特定端口。默认情况下，此端口无法从宿主机访问。为了弥合这一间隔，需要将宿主机上的一个端口映射到容器内部的端口。

### 理解容器网络和端口

可以把Docker容器看作有自己的私有网络接口和IP地址。当你的Python Web服务器（例如用于FastAPI的Uvicorn或Flask的开发服务器）启动时，它会绑定到这个容器网络内的某个端口。例如，FastAPI与Uvicorn通常默认使用端口8000，而Flask默认使用5000。

如果你的服务器在容器内部只绑定到 `127.0.0.1`（localhost），它将只接受来自该容器*内部*的连接。为了允许来自Docker宿主网络的连接（并通过端口映射，从而从外部连接），你必须配置你的服务器监听 `0.0.0.0`。这告诉服务器接受容器内部所有可用网络接口上的连接。

以下是通常运行Uvicorn（用于FastAPI）使其在容器内所有接口上监听端口8000的方法：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

或者对于Flask的开发服务器：

```bash
flask run --host=0.0.0.0 --port=5000
```

### 使用 `EXPOSE` 记录端口

Dockerfile提供了`EXPOSE`指令。它的主要作用是记录容器内应用程序打算监听哪些端口。它作为镜像使用者（包括你自己在内）的元数据，并可被其他工具使用。

例如，如果你的FastAPI应用程序监听端口8000，你会在你的Dockerfile中添加这一行：

```dockerfile
# ... 其他指令 (FROM, COPY, RUN pip install ...)

# 记录应用程序使用端口8000
EXPOSE 8000

# 运行应用程序的命令，监听0.0.0.0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

很重要的一点是，要明白`EXPOSE` **并不会** 实际发布端口或使其可从宿主机访问。它纯粹是提供信息。实际的连接是在你运行容器时建立的。

### 使用 `docker run -p` 发布端口

为了让容器的端口可以从你的宿主机访问到，你在执行`docker run`命令时使用`-p`或`--publish`标志。这个标志将宿主机上的一个端口映射到容器内部的一个端口。

格式是：`-p <host_port>:<container_port>`

- `<host_port>`: 你的Docker宿主机（你的电脑）上的端口号。你将使用这个端口来访问服务。
- `<container_port>`: 容器内部应用程序正在监听的端口号（即在`EXPOSE`中指定并由你的应用程序服务器使用的那个端口）。

**示例：**

1. **将宿主机端口8080映射到容器端口8000：**

   ```bash
   docker run -d -p 8080:8000 --name my_inference_api my_inference_image:latest
   ```

   在这种情况下，你将通过宿主机上的`http://localhost:8080`访问你的API。Docker会将来自宿主机端口8080的流量转发到容器端口8000。
2. **将宿主机端口8000映射到容器端口8000：**

   ```bash
   docker run -d -p 8000:8000 --name my_inference_api my_inference_image:latest
   ```

   这里，宿主机端口和容器端口相同。访问将通过`http://localhost:8000`。这只有在端口8000没有被你的宿主机上其他应用程序占用时才有效。
3. **将容器端口8000映射到随机宿主机端口：**

   ```bash
   docker run -d -p 8000 --name my_inference_api my_inference_image:latest
   ```

   如果你只指定容器端口，Docker会自动在宿主机上分配一个随机的可用端口。你可以使用`docker ps`命令查找分配了哪个端口：

   ```bash
   docker ps
   ```

   查看输出中的`PORTS`列，可能会显示类似`0.0.0.0:32768->8000/tcp`的内容。这表示宿主机端口32768映射到容器端口8000。

下面的图表说明了如何将宿主机端口8080映射到容器端口8000。

> 此图表显示，导向宿主机端口8080的流量由Docker转发到运行中容器内部的端口8000，推理 (inference)API应用程序正在那里监听。

### 测试连接

一旦你的容器运行起来且端口正确映射，你可以使用`curl`等工具测试推理 (inference)API，或者简单地在网页浏览器中访问URL（如果你的API支持根路径的GET请求）。

使用上面的第一个示例（`-p 8080:8000`），你将向*宿主*端口发送请求：

```bash
# 示例：向/predict端点发送带有JSON数据的POST请求
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.5, 0.8]}'
```

如果映射成功并且你的API在容器内部运行正常，你应该会收到预测响应。

请记住选择一个未被系统上其他应用程序使用的宿主端口。在Dockerfile中使用`EXPOSE`会记录预期的容器端口，而使用`docker run -p <host_port>:<container_port>`则使你的容器化推理服务可以接受请求。

## 参考资料

- [Networking with containers](https://docs.docker.com/network/) — Docker Docs (2024)
  描述了 Docker 核心网络概念，包括桥接网络和容器隔离。
- [EXPOSE (Dockerfile reference)](https://docs.docker.com/reference/dockerfile/#expose) — Docker Docs (2024)
  解释了 Dockerfile 中的 `EXPOSE` 指令，阐明了其作为端口文档元数据的作用。
- [docker run (--publish, -p)](https://docs.docker.com/reference/cli/docker/container/run/#publish-p) — Docker Docs (2024)
  详细介绍了 `docker run` 命令的 `--publish` (`-p`) 标志，展示了如何将主机端口映射到容器端口以进行外部访问。
- [Deployment](https://fastapi.tiangolo.com/deployment/) — Sebastián Ramírez (tiangolo) (2024)
  概述了 FastAPI 应用的部署建议，特别提及了容器内服务器绑定到 `0.0.0.0` 的做法。

---

[上一节](04-%E5%87%8F%E5%B0%91%E6%8E%A8%E7%90%86%E6%97%B6%E7%9A%84%E4%BE%9D%E8%B5%96%E9%A1%B9.md) · [下一节](06-%E6%8E%A8%E7%90%86%E5%AE%B9%E5%99%A8%E7%9A%84%E5%81%A5%E5%BA%B7%E6%A3%80%E6%9F%A5.md)
