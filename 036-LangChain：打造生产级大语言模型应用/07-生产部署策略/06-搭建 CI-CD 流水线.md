# 搭建 CI/CD 流水线

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-7-deployment-strategies-production/cicd-pipelines-langchain)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自动化部署过程是实现生产环境中可靠且可重复发布的基础。手动部署容器化应用，特别是像使用 LangChain 构建的复杂系统，会带来很大的人为错误风险、环境间的不一致性以及缓慢的发布周期。持续集成和持续部署 (CI/CD) 流水线提供了必要的自动化框架来应对这些问题。

CI/CD 流水线自动化了将代码更改从开发者机器部署到生产环境的步骤。它们强制保持一致性，运行自动化检查，并加快发布过程，让团队能够更快、更有信心地交付更新。

### 核心理念：CI 和 CD

- **持续集成 (CI)：** 这种做法是指开发者频繁地将代码更改合并到中央仓库，然后运行自动化构建和测试。对于 LangChain 应用来说，一个常见的 CI 过程包括：

  1. **触发：** 开发者将代码更改推送到版本控制系统（如 Git）。
  2. **拉取代码：** CI 服务器拉取最新代码。
  3. **环境配置：** 安装必要的依赖（Python 包、系统库）。
  4. **代码质量检查：** 运行代码检查工具（如 Flake8, Black）和静态分析工具。
  5. **测试：** 执行自动化测试（单元测试、集成测试）。这可能包括对提示结构或链逻辑进行基础测试，并可能模拟 LLM 交互。
  6. **构建产物：** 创建可部署的单元，通常是用于 LangChain 应用的 Docker 容器镜像。
- **持续部署 (CD)：** 这种做法通过自动部署所有通过 CI 阶段的代码更改到生产环境，从而扩展了 CI。一个紧密相关的理念是**持续交付**，它自动将更改部署到预演或预生产环境，在部署到生产环境之前需要手动审批。CD 过程通常包括：

  1. **触发：** CI 流水线成功完成。
  2. **产物存储：** 将构建好的 Docker 镜像推送到容器注册表（例如 Docker Hub、AWS ECR、Google Artifact Registry）。
  3. **部署：** 自动将新的容器镜像部署到目标环境（Kubernetes、Serverless 平台、虚拟机）。
  4. **部署后检查：** 可选地对新部署的应用运行冒烟测试或健康检查。

### 选择 CI/CD 平台

有多个平台提供 CI/CD 功能。流行的选择包括：

- **GitHub Actions：** 与 GitHub 仓库紧密结合，提供慷慨的免费额度以及丰富的可复用操作市场。配置通过仓库内的 YAML 文件完成（`.github/workflows/`）。
- **GitLab CI/CD：** 集成到 GitLab 平台中，也通过 YAML 配置（`.gitlab-ci.yml`）。提供 Auto DevOps 和内置容器注册表等功能。
- **Jenkins：** 一个高度可扩展的开源自动化服务器。提供极大的灵活性，但通常需要更多的设置和维护。
- **AWS CodePipeline：** AWS 上的云原生 CI/CD 服务，与 CodeBuild、CodeDeploy、ECR 和 Lambda 等其他 AWS 服务很好地集成。
- **Google Cloud Build：** 谷歌云的托管 CI/CD 服务，与 Cloud Source Repositories、Artifact Registry、Cloud Run、GKE 和 Cloud Functions 集成。
- **CircleCI：** 一个流行的云端 CI/CD 平台，以其速度和配置选项著称。

最佳选择视您现有的基础设施、团队专业知识、预算和具体功能需求而定。对于托管在 GitHub 上的项目，GitHub Actions 通常提供最高效的体验。同样地，GitLab 用户通常会觉得 GitLab CI/CD 非常方便。

### 使用 GitHub Actions 搭建 CI 流水线

我们来演示如何使用 GitHub Actions 为一个容器化的 LangChain 应用搭建一个基础的 CI 流水线。创建一个名为 `.github/workflows/ci.yml` 的文件：

```yaml
name: LangChain CI

on:
  push:
    branches: [ main ] # 在推送到 main 分支时触发
  pull_request:
    branches: [ main ] # 在针对 main 的拉取请求时触发

jobs:
  build-and-test:
    runs-on: ubuntu-latest # 使用标准 Linux 环境

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11' # 指定您项目的 Python 版本

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        # 如果有开发依赖（例如用于测试），则安装它们
        # pip install -r requirements-dev.txt

    - name: Lint with Flake8 (Example)
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Run tests with Pytest
      env:
        # 重要提示：请使用 GitHub Secrets 存储实际的 API 密钥！
        # 这些是占位符，真实的密钥应安全存储。
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }} # 示例密钥
        # 添加其他必要的测试环境变量
      run: |
        pip install pytest pytest-mock # 安装测试框架
        pytest tests/ # 假设您的测试在 'tests' 目录中

    - name: Build Docker image
      run: |
        # 适当标记镜像，例如使用 Git SHA
        docker build . --file Dockerfile --tag my-langchain-app:${{ github.sha }}
```

**管理密钥：** 请注意测试步骤中的 `env` 块。**切勿**将 API 密钥或其他敏感凭据直接提交到您的代码或 CI 配置文件中。所有主要的 CI/CD 平台都提供安全存储密钥（如示例中的 `secrets.OPENAI_API_KEY_TEST`）的机制，这些密钥仅在流水线执行期间作为环境变量注入。

### 搭建 CD 流水线

在此 CI 流水线的基础上，我们可以添加一个部署作业。本示例说明了如何将 Docker 镜像推送到注册表并概述了部署步骤。

```yaml
# （之前的 CI 作业 'build-and-test' 定义在此处）

  deploy:
    needs: build-and-test # 确保 CI 通过后才部署
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push' # 仅在推送到 main 分支时部署

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Docker Hub (Example Registry)
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v6
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.DOCKERHUB_USERNAME }}/my-langchain-app:latest
          ${{ secrets.DOCKERHUB_USERNAME }}/my-langchain-app:${{ github.sha }}

    - name: Deploy to Kubernetes (Example)
      # 此步骤需要设置 kubectl 上下文或使用特定的 action
      # 例如：azure/k8s-deploy@v4, google-github-actions/deploy-gke@v2
      run: |
        echo "Setting up kubectl context..." # 设置 kubectl 上下文...
        # 使用密钥配置 kubectl 访问（例如 KUBECONFIG 数据）
        echo "Applying Kubernetes manifests..." # 应用 Kubernetes 配置...
        # kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
        # kubectl set image deployment/my-langchain-app my-langchain-app=${{ secrets.DOCKERHUB_USERNAME }}/my-langchain-app:${{ github.sha }}
        echo "Deployment step would execute here." # 部署步骤将在此处执行。

    - name: Deploy to AWS Lambda (Example)
      # 此步骤需要 AWS 凭证，并可能需要 AWS CLI 或 SAM CLI
      # 使用 aws-actions/configure-aws-credentials@v4 等 action
      run: |
        echo "Configuring AWS credentials..." # 配置 AWS 凭证...
        # 使用密钥配置 AWS CLI 访问
        echo "Deploying function update..." # 部署函数更新...
        # aws lambda update-function-code --function-name my-langchain-function --image-uri ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com/my-langchain-repo:${{ github.sha }}
        echo "Deployment step would execute here." # 部署步骤将在此处执行。

    # 同样地添加其他部署目标（Serverless Framework, Google Cloud Functions 等）
```

**部署步骤：** 实际的部署命令很大程度上视您选择的基础设施（Kubernetes、AWS Lambda、Google Cloud Functions、Azure Functions、虚拟机）而定。您通常会使用：

- 平台特有的 CLI 工具（`kubectl`、`aws`、`gcloud`、`az`）。
- 在流水线中执行的基础设施即代码工具（Terraform、Pulumi）。
- 平台特有的 GitHub Actions（例如 `google-github-actions/deploy-cloud-run`、`aws-actions/amazon-ecs-deploy-task-definition`）。

请记得使用密钥安全地配置目标平台的访问凭证。

### CI/CD 中的测试

在 CI/CD 流水线中测试 LLM 应用会带来独特的难题：

- **单元测试与集成测试：** 侧重于测试您应用的逻辑、数据处理、工具使用和链构建，*在可能的情况下不*进行实际的 LLM 调用。使用模拟库（`pytest-mock`）来模拟 LLM 响应和工具输出。测试提示模板是否具有正确的格式和变量注入。
- **LLM 评估：** 对 LLM 响应质量（准确性、相关性、安全性）的全面评估通常太慢，而且对于每次提交都在标准 CI 流水线中运行来说可能成本过高。可以考虑：
  - 在 CI 中运行一小部分快速的评估用例作为基本健全性检查。
  - 将完整的评估集成到 CD 过程中，可能在提升到生产环境之前针对预演环境运行。
  - 使用 LangSmith（在第 5 章中讨论）等平台进行更详细、可能异步或手动触发的评估工作流。
- **成本：** 测试期间频繁的 LLM 调用可能会产生很高的成本。模拟对于保持 CI 流水线高效运行必不可少。

### 流程可视化

一个常见的 CI/CD 流水线遵循顺序流程，通常会根据不同环境进行分支。

> CI/CD 流水线的简化示意图，包含可选的预演环境部署和生产环境前的手动审批步骤。

### CI/CD 流水线最佳实践

- **保持快速：** 利用 CI/CD 平台提供的 Docker 层缓存和依赖缓存来优化构建时间。更快的反馈循环可以提升开发者生产力。
- **安全密钥：** 始终使用 CI/CD 平台的密钥管理功能来处理 API 密钥、密码和证书。
- **环境一致：** 尽量保持测试和预演环境与生产环境相似，以便尽早发现特定于环境的问题。
- **原子性：** 将部署步骤设计得尽可能具有原子性。使用健康检查和就绪探针（尤其是在 Kubernetes 中），以确保新版本在转移流量之前是健康的。
- **监控流水线：** 记录流水线成功率、持续时间和常见故障点，以找出需要改进的地方。
- **基础设施即代码 (IaC)：** 使用 Terraform 或 Pulumi 等工具定义您的部署基础设施（服务器、数据库、负载均衡器），并将其集成到流水线中，以实现环境的一致配置。
- **版本控制一切：** 您的应用程序代码、Dockerfile、CI/CD 配置（`.github/workflows/`、`.gitlab-ci.yml`）、Kubernetes manifest 和 IaC 定义都应存储在版本控制中。

在生产环境中运行 LangChain 应用时，搭建 CI/CD 流水线是一项投入，它会在稳定性、速度和减少运维开销方面带来回报。它将部署从一项有风险的手动任务转变为可靠的自动化过程。

## 参考资料

- [Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation](https://www.oreilly.com/library/view/continuous-delivery/9780321601919/) — Jez Humble, David Farley (2010)
  Publisher: Addison-Wesley Professional
  关于持续集成和持续部署的奠基性著作，涵盖软件可靠交付的原则、实践和自动化策略。
- [GitHub Actions Documentation](https://docs.github.com/en/actions) — GitHub (2024)
  GitHub Actions 的官方文档，提供设置自动化 CI/CD 工作流的全面指南和参考，如本节所示。
- [Docker Documentation: Build and run containers](https://docs.docker.com/build/) — Docker, Inc. (2024)
  Docker 的官方文档，提供构建、管理和推送容器镜像的指南，这些是容器化应用CI/CD的核心。

---

[上一节](05-%E7%AE%A1%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E5%92%8C%E6%95%8F%E6%84%9F%E4%BF%A1%E6%81%AF.md) · [下一节](07-%E8%93%9D%E7%BB%BF%E9%83%A8%E7%BD%B2%E4%B8%8E%E9%87%91%E4%B8%9D%E9%9B%80%E9%83%A8%E7%BD%B2%E7%AD%96%E7%95%A5.md)
