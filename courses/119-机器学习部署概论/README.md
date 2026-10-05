# 机器学习部署概论

来源：[机器学习部署概论](https://apxml.com/zh/courses/basics-ml-deployment)

学习部署机器学习 (machine learning)模型的基本知识与方法。本课程涵盖模型生产环境准备、构建简易预测服务及理解基本部署模式，助您将训练好的模型变得可用、实用。

预计学时：7 小时

先修要求：Python及机器学习入门

## 课程目录

### 1. [模型部署入门](01-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%85%A5%E9%97%A8/README.md)

- 1. [什么是机器学习部署？](01-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%85%A5%E9%97%A8/01-%E4%BB%80%E4%B9%88%E6%98%AF%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%83%A8%E7%BD%B2%EF%BC%9F.md)
- 2. [为什么要部署机器学习模型？](01-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%85%A5%E9%97%A8/02-%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E9%83%A8%E7%BD%B2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%EF%BC%9F.md)
- 3. [机器学习工作流程概述](01-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%85%A5%E9%97%A8/03-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%E6%A6%82%E8%BF%B0.md)
- 4. [部署策略的类型 (简介)](01-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%85%A5%E9%97%A8/04-%E9%83%A8%E7%BD%B2%E7%AD%96%E7%95%A5%E7%9A%84%E7%B1%BB%E5%9E%8B%20%28%E7%AE%80%E4%BB%8B%29.md)
- 5. [模型部署中的挑战](01-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%85%A5%E9%97%A8/05-%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E4%B8%AD%E7%9A%84%E6%8C%91%E6%88%98.md)
- [章节测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-1-getting-started-model-deployment/quiz)

### 2. [为模型部署做准备](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/README.md)

- 1. [保存训练好的模型](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/01-%E4%BF%9D%E5%AD%98%E8%AE%AD%E7%BB%83%E5%A5%BD%E7%9A%84%E6%A8%A1%E5%9E%8B.md)
- 2. [模型序列化简介](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/02-%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E7%AE%80%E4%BB%8B.md)
- 3. [使用 Pickle 保存模型](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/03-%E4%BD%BF%E7%94%A8%20Pickle%20%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B.md)
- 4. [使用 Joblib 保存模型](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/04-%E4%BD%BF%E7%94%A8%20Joblib%20%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B.md)
- 5. [处理模型依赖](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/05-%E5%A4%84%E7%90%86%E6%A8%A1%E5%9E%8B%E4%BE%9D%E8%B5%96.md)
- 6. [保存预处理步骤](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/06-%E4%BF%9D%E5%AD%98%E9%A2%84%E5%A4%84%E7%90%86%E6%AD%A5%E9%AA%A4.md)
- 7. [动手实践：保存和加载简单模型](02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E7%AE%80%E5%8D%95%E6%A8%A1%E5%9E%8B.md)
- [章节测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/quiz)

### 3. [使用 Flask 创建预测服务](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/README.md)

- 1. [什么是API？](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/01-%E4%BB%80%E4%B9%88%E6%98%AFAPI%EF%BC%9F.md)
- 2. [Web框架简介](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/02-Web%E6%A1%86%E6%9E%B6%E7%AE%80%E4%BB%8B.md)
- 3. [设置 Flask](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/03-%E8%AE%BE%E7%BD%AE%20Flask.md)
- 4. [构建一个基础的 Flask 应用](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/04-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80%E7%9A%84%20Flask%20%E5%BA%94%E7%94%A8.md)
- 5. [在 Flask 中加载已保存的模型](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/05-%E5%9C%A8%20Flask%20%E4%B8%AD%E5%8A%A0%E8%BD%BD%E5%B7%B2%E4%BF%9D%E5%AD%98%E7%9A%84%E6%A8%A1%E5%9E%8B.md)
- 6. [定义预测端点](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/06-%E5%AE%9A%E4%B9%89%E9%A2%84%E6%B5%8B%E7%AB%AF%E7%82%B9.md)
- 7. [处理输入数据 (JSON)](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/07-%E5%A4%84%E7%90%86%E8%BE%93%E5%85%A5%E6%95%B0%E6%8D%AE%20%28JSON%29.md)
- 8. [返回预测结果](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/08-%E8%BF%94%E5%9B%9E%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)
- 9. [在本地测试你的 API](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/09-%E5%9C%A8%E6%9C%AC%E5%9C%B0%E6%B5%8B%E8%AF%95%E4%BD%A0%E7%9A%84%20API.md)
- 10. [动手实践：构建一个简单的Flask预测API](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1/10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84Flask%E9%A2%84%E6%B5%8BAPI.md)
- [章节测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/quiz)

### 4. [Docker 容器化简介](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/README.md)

- 1. [什么是容器化？](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%AE%B9%E5%99%A8%E5%8C%96%EF%BC%9F.md)
- 2. [Docker 入门](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/02-Docker%20%E5%85%A5%E9%97%A8.md)
- 3. [Docker 核心理念：镜像与容器](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/03-Docker%20%E6%A0%B8%E5%BF%83%E7%90%86%E5%BF%B5%EF%BC%9A%E9%95%9C%E5%83%8F%E4%B8%8E%E5%AE%B9%E5%99%A8.md)
- 4. [安装 Docker](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/04-%E5%AE%89%E8%A3%85%20Docker.md)
- 5. [编写一个简单的Dockerfile](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/05-%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84Dockerfile.md)
- 6. [为Flask应用构建Docker镜像](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/06-%E4%B8%BAFlask%E5%BA%94%E7%94%A8%E6%9E%84%E5%BB%BADocker%E9%95%9C%E5%83%8F.md)
- 7. [在 Docker 容器中运行应用](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/07-%E5%9C%A8%20Docker%20%E5%AE%B9%E5%99%A8%E4%B8%AD%E8%BF%90%E8%A1%8C%E5%BA%94%E7%94%A8.md)
- 8. [动手实践：容器化预测服务](04-Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E4%BB%8B/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%B9%E5%99%A8%E5%8C%96%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1.md)
- [章节测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker/quiz)

## 学习目标

- **模型部署理解**：了解模型部署的必要性及其在机器学习生命周期中的作用。
- **模型准备**：学习如何使用常用序列化技术保存和加载训练好的机器学习模型。
- **接口开发要点**：了解接口的用途，以及它如何用于模型服务。
- **简易网络服务构建**：使用 Flask 构建基本网络服务，从训练模型提供预测。
- **容器化入门**：掌握使用 Docker 进行应用打包的容器化要点。
