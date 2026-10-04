# 抓取脚本

课程目录来自 ApX 的公开分页接口；课程、章节和正文来自公开中文网页。脚本通过独立 Playwright Chrome 会话读取页面，不需要登录。课程页面请求默认间隔 0.75 秒，最多同时读取两个页面；图片不会单独下载。

`.crawl/cache/catalog.json` 保存抓取时的完整课程列表与顺序；`courses/` 保存课程目录；`sections/` 保存已经获取的小节正文。重复执行会复用缓存，并补齐未完成项。正文会保留代码块、公式、表格、复选框与参考资料；交互图表保留原网页入口及可折叠的原始 JSON 数据；课程项目单独保存为 `PROJECT.md`，章节测验保留在线入口。

先准备依赖并打开浏览器；等课程列表正常显示后再启动脚本：

```bash
python3 -m venv /tmp/apxml-venv
/tmp/apxml-venv/bin/pip install -r scripts/requirements.txt
npx --yes --package @playwright/cli playwright-cli --session apxml open https://apxml.com/zh/courses --browser chrome --headed
```

只生成完整目录、课程说明、章节介绍和小节入口：

```bash
/tmp/apxml-venv/bin/python scripts/crawl_apxml.py --phase curricula
```

继续抓取全部小节正文：

```bash
/tmp/apxml-venv/bin/python scripts/crawl_apxml.py --phase sections
```

目录与正文一起处理：

```bash
/tmp/apxml-venv/bin/python scripts/crawl_apxml.py --phase all
```

需要指定 Playwright CLI 安装位置时使用 `--cli /absolute/path/to/playwright-cli.js`；需要降低抓取频率时使用 `--delay 2`。每轮结束会更新根目录的 `CRAWL_STATUS.md` 和 `.crawl/report.json`。

校验课程概览、章节介绍、学习目标、项目内容、文件覆盖和本地 Markdown 导航：

```bash
/tmp/apxml-venv/bin/python scripts/verify_archive.py
```

完成抓取后，可完全离线地从缓存重新生成所有 Markdown，并核对全部正文与参考资料：

```bash
/tmp/apxml-venv/bin/python scripts/crawl_apxml.py --phase render
/tmp/apxml-venv/bin/python scripts/verify_archive.py --require-full
```

抓取脚本的错误恢复测试使用临时目录和模拟响应，不访问原站，也不会修改课程归档：

```bash
/tmp/apxml-venv/bin/python -m unittest discover -s tests -p '*_test.py'
```

独立检查正文文本、公式、表格单元格、图表数据和参考链接，并核对目录顺序及上一节/下一节导航：

```bash
/tmp/apxml-venv/bin/python scripts/audit_content_apxml.py
/tmp/apxml-venv/bin/python scripts/audit_navigation_apxml.py
```

结果分别保存在 `.crawl/content-verification.json` 和 `.crawl/navigation-verification.json`。原始 HTML 的文字统计可能受混合 Markdown 排版影响，文本差异需要结合公式及代码检查判断。

需要重新核对线上内容时，打开独立会话后执行：

```bash
npx --yes --package @playwright/cli playwright-cli --session apxml-audit open https://apxml.com/zh/courses --browser chrome --headed
/tmp/apxml-venv/bin/python scripts/audit_live_apxml.py
npx --yes --package @playwright/cli playwright-cli --session apxml-audit close
```

线上复核会重新读取课程列表和全部课程大纲，并从每门课选一节内容较丰富的正文核对。它将新快照保存到 `.crawl/audit-live/`，不会直接覆盖原始缓存；其结果不等同于重新下载全部正文。
