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
- **交互绘图**：集成 ipywidgets / ipympl 前端组件，持续完善课程中的交互实验。
- **环境记录**：`machine_learning/environment/` 保存了原始 Python、Conda 依赖版本，便于查阅和复现。

## 浏览器环境测试与 TensorFlow.js

在文件浏览器中打开 **`浏览器环境自检.ipynb`**，可以依次检测 NumPy、SciPy、pandas、scikit-learn、Matplotlib、ipywidgets、ipympl 和 TensorFlow.js。

神经网络章节的原版 **`2. CoffeeRoasting_TF.ipynb`** 已接入 TensorFlow.js：保留 60 个原有教学单元格，只调整 3 个代码单元格；数据处理、画图和大部分 Keras 调用保持原样。

运行时优先使用 **WebGPU**，其次尝试 WebGL，最后回退 CPU。可以在 Notebook 中运行 `from browser_ml.tfjs_bridge import backend_info; backend_info()` 查看当前后端及回退原因。兼容层位于 `machine_learning/browser_ml/tfjs_compat.py`，底层执行模块位于 `machine_learning/browser_ml/tfjs_bridge.py`。

另附 **`2a. CoffeeRoasting_TFJS.ipynb`**，用于直接查看 TensorFlow.js 桥接接口的精简示例。

训练时使用 `await model.fit(...)`，计算直接在浏览器内完成。仓库的 GitHub Actions 也会使用 Chromium 运行 Pyodide、科学计算包和 TensorFlow.js 的自动测试。

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
