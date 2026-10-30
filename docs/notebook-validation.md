# 全部 Notebook 浏览器验证

脚本 `tests/notebook-full.cjs` 会从 `machine_learning/` 动态发现所有 `.ipynb`，逐本在 JupyterLite 的 Python (Pyodide) 内核中运行所有非空代码单元格，并将执行状态、错误回溯和耗时写入 `notebook-validation/shard-0.json`。文件名标明“切勿运行”的 Notebook 会在报告中记为 `excluded`。

在仓库根目录用 PowerShell 运行线上站点的有界面验证：

```powershell
npm install --prefix .browser-test --no-save --ignore-scripts playwright@1.56.1
.\.browser-test\node_modules\.bin\playwright.cmd install chromium
$env:NOTEBOOK_SITE_URL = 'https://scluzlep.github.io/machine_learning/lab/'
$env:NOTEBOOK_HEADED = '1'
node tests/notebook-full.cjs
```

不设置 `NOTEBOOK_SITE_URL` 时，脚本会启动本地 HTTP 服务并验证 `dist/` 中的构建结果。`NOTEBOOK_HEADED=1` 用于显示浏览器；不设置时默认无界面。可用 `NOTEBOOK_TIMEOUT_MS` 修改每本的最长执行时间。单本失败后脚本继续验证其余 Notebook，最终以非零退出码报告失败。GitHub Actions 将清单分为四片执行，JSON 结果保存在对应的运行 artifact 中。

