# 机器学习课程实验

把机器学习课程的 Jupyter Notebook 整理到一起，提供一个可以直接在浏览器中打开的实验环境。

**在线使用：** https://scluzlep.github.io/machine_learning/lab/

## 课程内容

项目包含线性回归、逻辑回归、模型训练与评估、神经网络等实验。课程按照学习顺序分成七个目录，保留了 Notebook、练习数据、配图和实验辅助代码。

进入网站后，可以像使用 JupyterLab 一样打开 Notebook、修改代码、运行单元格，或者创建自己的 Python 文件。

## 本地运行

使用 JupyterLite 构建静态网站：

```bash
python -m pip install -r site-requirements.txt
jupyter lite build --contents machine_learning --output-dir dist
python -m http.server 8000 --directory dist
```

在浏览器打开 http://localhost:8000/lab/ 。

## 项目说明

- **纯静态部署**：使用 JupyterLite + Pyodide，在浏览器中运行 Python。
- **中文界面**：集成 JupyterLab 简体中文语言包，默认使用中文。
- **课程实验**：包含 NumPy、Matplotlib、scikit-learn 以及神经网络相关的 Notebook。
- **交互绘图**：集成 ipywidgets / ipympl 前端组件，支持课程中的交互绘图。
- **环境记录**：`machine_learning/environment/` 保存了原始 Python、Conda 依赖版本，便于查阅和复现。

## 浏览器环境测试与 TensorFlow.js

在文件浏览器中打开 **`浏览器环境自检.ipynb`**，可以依次检测 NumPy、SciPy、pandas、scikit-learn、Matplotlib、ipywidgets、ipympl 和 TensorFlow.js。

神经网络章节的原版 **`2. CoffeeRoasting_TF.ipynb`** 保持原样。浏览器内核会自动将课程使用的部分 TensorFlow / Keras 调用交给 TensorFlow.js 执行，无需在教材中添加 `await`。

运行时优先使用 **WebGPU**，其次尝试 WebGL，最后回退 CPU。可以在 Notebook 中运行 `from browser_ml.tfjs_bridge import backend_info; backend_info()` 查看当前后端及回退原因。兼容层位于 `machine_learning/browser_ml/tfjs_compat.py`，底层执行模块位于 `machine_learning/browser_ml/tfjs_bridge.py`。

另附 **`2a. CoffeeRoasting_TFJS.ipynb`**，用于直接查看 TensorFlow.js 桥接接口的精简示例。

执行层自动处理 TensorFlow.js 的异步训练；教材仍使用标准 Python 写法。仓库的 GitHub Actions 会使用 Chromium 运行 Pyodide、科学计算包、原样 Keras 语句以及 Notebook 内核检测。

## 可选的 PyTorch 兼容模式：Borch（实验性）

站点使用 [Borch](https://github.com/playidea-lab/borch) `pyborch==1.14.1`，
为传统教材提供 `import torch` 形式的**部分** PyTorch API；
这不是 PyTorch 官方实现，也不支持 CUDA、分布式训练等全部功能。

在 JupyterLite 的 Python Notebook 中直接执行：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

print(torch.__borch_backend__)  # webgpu 或 numpy
x = torch.tensor([1., 2., 3.], requires_grad=True)
(x * x).sum().backward()
print(x.grad)
```

自定义内核仅在单元格显式导入 `torch` 时才加载已打包的
`pyborch` wheel，原始教学 Notebook 不做修改。能够使用 WebGPU + JSPI
时优先使用 `borch_webgpu`；否则回退到基于 NumPy/Pyodide/WASM
的 `borch` CPU 计算实现，并给出显式警告。

**旧电脑/Windows 7 注意：** Chrome/Chromium 120～130 默认没有 JSPI，
因此通常会选择 CPU；CPU 不依赖 WebGPU，也不依赖 JSPI，
但仍需要浏览器支持当前站点的 Pyodide/WASM 功能。
Win7 + 第三方 Chromium 修改版（如部分夸克构建）未经验证。
不保证 GPU 和 CPU 后端拥有完全相同的 API。核显只会影响 WebGPU 路径，
CPU 模式主要取决于处理器、内存和浏览器 WebAssembly 实现。

`torch.nn`、`torch.nn.functional`、`torch.optim`、
`torch.optim.lr_scheduler`、`torch.utils.data` 等已做
Python 模块导入桥接；不存在的上游 API 不会被伪装成完整实现。
不能把这种模式称为原生 PyTorch，也不能直接取代远程真 PyTorch 内核。

当前属于**待浏览器实测的实验分支**，尚需在 Chrome WebGPU、
禁用 WebGPU 的 Chromium，以及真实 Win7 夸克环境分别跑单元测试、
梯度检查、训练循环、内存增长与 Notebook 导入测试。

## 自动部署

提交到 `main` 分支后，GitHub Actions 会构建 JupyterLite 并发布到 GitHub Pages。也可以在 Actions 页面手动运行 **Deploy JupyterLite**。

需要维护或查看构建流程，可以参考 [Pages 工作流](.github/workflows/pages.yml)。

## 目录结构

```text
machine_learning/        课程 Notebook、数据与辅助代码
  1. Linear Regression/
  2. Logistic regression/
  3. ML In Practice/
  4. Neural Network_part_1/
  5. Neural Network_part_2/
  6. Neural Network In Practice 1/
  7. Neural Network In Practice 2/
  environment/           原环境依赖版本记录
site-requirements.txt    静态网站构建依赖
.github/workflows/       GitHub Pages 自动部署
```
