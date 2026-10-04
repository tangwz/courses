# 静态课程书架

一个 Astro + TypeScript + React 项目。首页展示全部课程，每门课程共用课程概览和三栏阅读器。Markdown 正文、公式和代码在构建时生成 HTML；React 处理学习进度和阅读设置；Plotly 渲染课程原始图表；Pagefind 提供中文全文搜索。

阅读页以 [Modern LLM Notebook](https://walkinglabs.github.io/modern-llm-notebook/#01-tokenizer-basics) 为界面基准：左侧固定课程目录、中间白色正文卡片、右侧当前小节大纲，使用浅色背景、蓝色强调色和紧凑导航。目录和大纲可收起；窄屏以抽屉和浮层显示。所有课程通过 `ReaderLayout.astro` 与 `reader.css` 共用这套设计，课程内容不维护独立界面。首页保留课程书架设计。

侧栏宽度为 256px，书籍标题、课程目录标题和底部工具固定，章节与小节在中间独立滚动。章节标题链接到章节概览，底部提供课程概览、学习进度、主题和阅读设置。桌面正文取消横向导航栏，采用流式卡片、卡片内标题及 48px 内边距；阅读宽度设置控制卡片内的正文宽度，卡片本身随窗口展开。卡片内标题位于正文节点之外，不参与全文索引。

阅读器本地加载 Plus Jakarta Sans 和 JetBrains Mono 的 Latin 字体子集，字体文件位于 `src/assets/fonts/`，总计约 57KB；中文使用系统字体。开放字体许可证位于 `public/licenses/`，并随静态产物交付，访问时不依赖远程字体服务。

## 运行和交付

默认使用 Node.js 24（见 `.nvmrc`），最低要求为 22.12。课程内容已经生成，日常开发与构建不需要 Python，也不需要重新访问原站。

```bash
npm ci
npm run dev
npm run check
npm run format:check
npm test
NODE_OPTIONS=--max-old-space-size=8192 npm run build
npm run verify
npm run preview
```

开发地址和预览地址由终端打印。开发模式搜索小节标题和摘要；构建后预览支持全文搜索。`dist/` 是完整静态产物，可以部署到任意静态 HTTP 服务。需要 HTTP 服务加载搜索索引与图表 JSON。

如果部署在子路径下，构建和校验时指定同一个前缀：

```bash
SITE_BASE=/library/ npm run build
SITE_BASE=/library/ npm run verify
npm run preview
```

无需服务器业务接口。主题、阅读设置、学习进度和阅读位置保存在 localStorage。当前版本不提供笔记、收藏和文字高亮，也没有账号和跨设备同步；此前保存在浏览器 IndexedDB 中的笔记和收藏数据不会被清除。

## 内容目录

```text
src/content/courses/
  <course-slug>/
    index.md
    project.md
    plots/
      <section-id>-<plot-index>.json
    <chapter-slug>/
      index.md
      <lesson-slug>.md
```

课程元数据、学习目标、章节顺序与正文统一从 Content Collections 读取，并通过 schema 校验。课程、章节、正文和项目有独立 collection。每个课程自己拥有图表文件，图表不进入 Markdown collection。

正文通过普通 Markdown 图片语法引用图表；`plots/` 始终相对于课程根目录解析：

```markdown
![Absolute errors](plots/3996-0.json)
```

课程 Markdown 插件将它转换为图表容器。构建时生成同课程下的 JSON 地址，浏览器只加载当前页的图表；Plotly 库在图表接近视口时才加载。源码 JSON 保持原样，运行时适配 Plotly 3 标题格式、旧 trace 类型和时间线数据。

图表脚本加载失败后可以直接重试；搜索初始化失败后，新查询会重新加载全文搜索资源。全文搜索不可用时降级为标题和摘要搜索；两种索引都无法加载时显示错误并清除旧结果。灰度图像转换为热图时默认固定 0–255 的强度范围，源数据显式指定的范围会保留。

原归档保留在根目录的编号课程文件夹中；原始证据在 `.crawl/cache/`。转换不会移动或删除它们。封面采用本地 SVG/CSS 排版，不依赖远程图片。正文里的原图片仍保留外链；原站签名链接可能过期，图片下载不属于本次范围。原站测验保留在线入口，未抓取登录后的题目。

课程来源链接和参考资料保留。课程之间的已知内容链接被改为本地路径。普通 Markdown 的公式使用 KaTeX；表格中的绝对值符号转换为等价的 `\vert`，避免被识别为列分隔符。原文美元符号跨表格列的错误会被修正，并记录在 `sourceCorrections` 元数据及验证报告中。

## 从原始归档重新转换

只有更新抓取数据后才需要转换。该命令会重新生成 `src/content/courses/`，因此手动修改课程内容前应备份，或先将修改同步到原始归档。`dev` 和 `build` 不会自动重新转换课程。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt
npm run content:prepare
```

也可以通过 `COURSE_PYTHON` 指定已有的 Python 环境：

```bash
COURSE_PYTHON=/path/to/python npm run content:prepare
```

首个验证课程是 `basics-model-evaluation-metrics`，包含公式、表格、代码和图表。单课程转换只生成该课程，不会删除其他课程：

```bash
npm run content:prepare -- --course basics-model-evaluation-metrics
```

转换后重新执行全量转换可以恢复完整内容清单。

## 主要代码与验证

- `src/layouts/ReaderLayout.astro`：静态正文、目录、大纲及上下篇导航。
- `src/components/ReaderControls.tsx`：学习进度、主题和阅读设置。
- `src/components/SearchPanel.tsx`：全文搜索及课程范围筛选。
- `src/lib/plots.ts`：图表适配与按需加载。
- `scripts/prepare_content.py`：从已验证归档生成课程内容。
- `scripts/verify-site.mjs`：全量页面、内部链接、代码、公式、表格和图表数据校验。
- `scripts/audit_rendered_content.py`：独立对照归档，检查每一节生成 HTML 的正文文字。

```bash
.venv/bin/python scripts/audit_rendered_content.py
```

结果位于 `reports/`。`prototype-verification.json` 和 `prototype-content-verification.json` 保存首个课程的验证范围；`site-verification.json` 和 `rendered-content-verification.json` 保存全量验证结果；`reader-refinement-verification.json` 保存阅读器布局及交互检查结果；`reader-feature-removal-verification.json` 保存移除笔记与收藏后的交互检查结果。浏览器截图和检查产物位于 `output/playwright/`。

新增课程时按课程目录维护 Markdown 与图表，更新元数据及顺序，然后执行类型检查、构建和全量验证。若新课程来自原站，先通过原抓取脚本更新缓存和编号归档，再转换。

GitHub Actions 在 PR 和主分支更新时执行格式检查、类型检查、回归测试、完整归档校验、全量构建、页面校验和正文独立对照。工作流使用 `.nvmrc` 中的 Node.js 版本，依赖通过锁文件安装；Python 校验依赖固定在 `scripts/requirements.txt`。构建无需访问原站，浏览器仍可能请求正文保留的外链图片。
