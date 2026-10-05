# 第 3 章：可扩展部署架构

来源：[原章节](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment)

[返回课程目录](../README.md)

在对扩散模型进行推理优化之后，下一步是构建其能够高效进行大规模运行的运行环境。本章将专注于构建必要的支撑体系。我们将介绍如何使用 Docker 等容器打包模型及其依赖项、使用 Kubernetes 等编排工具管理部署和扩展，以及配置云资源，包括 GPU 等专用硬件和无服务器计算选项。

主要议题包括管理容器内的 GPU 资源、根据推理需求实现自动扩展，以及处理大型模型和生成数据时的存储考量。您将学习设计并实现能够处理可变负载同时管理计算资源的系统，专门处理 GPU 密集型扩散模型的需求。实践练习将指导您在 Kubernetes 集群上部署容器化模型。

## 小节

- 1. [扩散模型Docker容器化](01-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8BDocker%E5%AE%B9%E5%99%A8%E5%8C%96.md)
- 2. [容器中的 GPU 资源管理](02-%E5%AE%B9%E5%99%A8%E4%B8%AD%E7%9A%84%20GPU%20%E8%B5%84%E6%BA%90%E7%AE%A1%E7%90%86.md)
- 3. [使用 Kubernetes 进行编排](03-%E4%BD%BF%E7%94%A8%20Kubernetes%20%E8%BF%9B%E8%A1%8C%E7%BC%96%E6%8E%92.md)
- 4. [管理 Kubernetes 中的 GPU 节点](04-%E7%AE%A1%E7%90%86%20Kubernetes%20%E4%B8%AD%E7%9A%84%20GPU%20%E8%8A%82%E7%82%B9.md)
- 5. [推理工作负载的自动扩缩容策略](05-%E6%8E%A8%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E7%9A%84%E8%87%AA%E5%8A%A8%E6%89%A9%E7%BC%A9%E5%AE%B9%E7%AD%96%E7%95%A5.md)
- 6. [无服务器GPU推理选项](06-%E6%97%A0%E6%9C%8D%E5%8A%A1%E5%99%A8GPU%E6%8E%A8%E7%90%86%E9%80%89%E9%A1%B9.md)
- 7. [模型与数据的存储考量](07-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%9A%84%E5%AD%98%E5%82%A8%E8%80%83%E9%87%8F.md)
- 8. [动手实践：在 Kubernetes 上部署](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8%20Kubernetes%20%E4%B8%8A%E9%83%A8%E7%BD%B2.md)
