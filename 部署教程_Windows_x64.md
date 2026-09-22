# YOLO26 AI 视觉 Studio 部署教程（Windows 10 / 11 x64）

本教程适用于把本项目部署到**其它 Windows 电脑**（开发机之外的机器）。
程序是 PyQt6 图形界面 + Ultralytics YOLO 推理 + 离线图像增强 + DINOv3 异常检测 + **RF-DETR 训练/验证/导出（可选）**。

> 最后更新：2026-09-16（新增第 15 节 RF-DETR、部署后自检脚本 `smoke_test.py`、权重预取脚本 `dl_weight.py`）

> 如果你要部署到 NVIDIA Jetson / Ubuntu ARM64，请参考同目录的 `部署教程_Ubuntu22_ARM64.md`。

---

## 0. 适用平台

| 项目 | 说明 |
| --- | --- |
| 系统 | Windows 10 / 11（64 位） |
| Python | **3.12.x（64 位）** —— 由 Miniconda3 创建，必须是 x64，ARM 版 Windows 不在支持范围 |
| GPU（可选） | NVIDIA 显卡（如 RTX 3050/3060/4060 等）；无独显也能跑（自动用 CPU） |
| 网络 | 需要联网下载依赖（已配置国内镜像，见第 2 节） |

**两种部署形态：**
- **A. 源码运行（推荐）**：拷贝项目文件夹 + 装 Python 依赖，直接 `python main.py`。灵活、好调试、好升级。
- **B. 打包成 EXE（可选）**：用 PyInstaller 打成单目录程序（见第 12 节）。适合给不懂 Python 的同事用，但体积大、CUDA 打包易踩坑。

---

## 1. 目标机准备：安装 Miniconda3（x64）

> 本项目**统一用 Miniconda3 管理运行环境**，不再单独装 Python。Miniconda 自带 `conda` 包管理器，能干净地隔离出 Python 3.12 环境，且对 PyTorch(CUDA) 这类体积大、依赖复杂的包更友好。

1. 到 **清华大学开源镜像站** 下载 Miniconda3 安装包（比官网快很多）：
   https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/ 
   选文件名形如 `Miniconda3-latest-Windows-x86_64.exe`（**必须是 x86_64 / 64 位**）。
   （需要的话也可从官网 https://www.anaconda.com/download 下载，但国内建议用镜像站。）
2. 双击运行安装：
   - 安装路径建议保持默认 `C:\Users\<你的用户名>\miniconda3`（或自定义一个**纯英文、无空格**的路径均可）。
   - **「Advanced Options」页**：勾选 `Create shortcuts`（建议勾），`Register Miniconda3 as the system Python 3.12` 可选；**不要**勾「Add Miniconda3 to my PATH」（避免污染系统 PATH，改用开始菜单的「Anaconda Prompt (miniconda3)」即可）。
3. 安装完成后，打开 **「Anaconda Prompt (miniconda3)」**（开始菜单搜 Anaconda 即可），验证：

```powershell
conda --version      # 应显示 conda 23.x 或更高
python --version     # 显示 base 环境的 Python（版本不重要，下一步会新建环境）
```

> 之后本教程所有命令都在 **Anaconda Prompt** 里执行即可，不必关心 PATH。
> **不要使用系统自带的 3.13+ Python** 去建环境（部分依赖尚未兼容）。

---

## 2. 配置国内镜像（conda + pip 双配置，关键）

conda 和 pip 是两套独立的源，建议都配成国内镜像，否则下载极慢。

### 2.1 配置 conda 镜像（清华源）

在 **Anaconda Prompt** 里执行（全局生效，之后 `conda install` 走清华源）：

```powershell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### 2.2 配置 pip 镜像（清华源）

```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

> pip 其它可选国内源（任选其一，把 2.2 的 URL 替换即可）：
> - 阿里云：`https://mirrors.aliyun.com/pypi/simple/`
> - 中科大：`https://pypi.mirrors.ustc.edu.cn/simple/`
> - 腾讯云：`https://mirrors.cloud.tencent.com/pypi/simple/`
> - 华为云：`https://repo.huaweicloud.com/repository/pypi/simple/`

---

## 3. 用 Miniconda3 创建并激活运行环境

```powershell
# 进入你把项目放好的目录，例如 D:\apps\
cd D:\apps\ultralytics-26_2

# 用 conda 新建一个独立环境（环境名 yolo，Python 3.12）
conda create -n yolo python=3.12 -y

# 激活环境（激活后命令行前面会出现 (yolo) 提示）
conda activate yolo
```

激活后，后续所有 `pip` / `python` 都作用在这个隔离的 `yolo` 环境里，不会影响系统或其它项目。

> 常用命令：
> - 查看所有环境：`conda env list`
> - 退出当前环境：`conda deactivate`
> - 删除环境（如需重来）：`conda remove -n yolo --all -y`
> - **注意**：以后每次打开新终端跑程序前，都要先 `conda activate yolo` 再执行 `python main.py`。

---

## 4. 安装 PyTorch（GPU / CPU 二选一）

> ⚠️ **重要**：`torch` 必须用带 CUDA 的专用 wheel（`+cu126`），**不能**从普通 pip 源装（那里只有 CPU 版，会导致 `torch.cuda.is_available()` 为 False）。
> 下面的命令请在**已激活 `yolo` 环境**（`(yolo)` 提示符）下执行，此时 `pip` 指向该环境内的 pip。

### 4.1 有 NVIDIA 显卡 → 装 CUDA 12.6 版（项目开发机同款）

```powershell
pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

**国内加速（可选）**：上面的官方源在国内可能偏慢，可加阿里云 PyTorch 镜像作为下载源（实测明显更快）：

```powershell
pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126 -f https://mirrors.aliyun.com/pytorch-wheels/cu126
```

### 4.2 无独立显卡（只有核显 / 纯 CPU）→ 装 CPU 版

```powershell
pip install torch==2.9.1 torchvision==0.24.1
```

> CPU 版照样能跑全部功能，只是推理速度慢；程序会自动回退到 CPU（代码里有设备兜底）。

### 4.3 验证 GPU 是否可用

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出类似 `2.9.1+cu126 True` ，最后一项必须是 **True**（GPU 版）。若为 False，见第 11 节排查表。

> **不必安装 CUDA Toolkit**：PyTorch 的 CUDA wheel 自带 CUDA 运行时，目标机**只需装好 NVIDIA 显卡驱动**（驱动版本需支持 CUDA 12.6，即驱动 ≥ 560；用 GeForce Experience 或官网更新到最新即可）。cuDNN 也由 wheel 自带，无需单独装。

---

## 5. 安装其余 Python 依赖

项目已提供依赖清单 `requirements-Windows.txt`（位于项目根目录），一键安装（确保已 `conda activate yolo`）：

```powershell
pip install -r requirements-Windows.txt
```

该清单包含：PyQt6、opencv-python、mss、Pillow、numpy、scipy、matplotlib、PyYAML、psutil、requests、filelock、networkx、sympy、jinja2、ultralytics-thop、onnx、onnxruntime 等。

> 关于 `ultralytics`：本项目根目录自带本地 `ultralytics/`（版本 8.4.132），运行时优先使用本地这份，**不需要** `pip install ultralytics`（装了也会被本地源码覆盖）。上面安装的只是它依赖的底层库。

---

## 6. 部署项目文件

把开发机上的项目文件夹整体拷贝到目标机（U 盘 / 内网共享 / `scp` / `git clone` 均可）。**必须包含的**：

| 文件 / 目录 | 作用 |
| --- | --- |
| `main.py` `workers.py` `config.py` `styles.py` `val_report.py` `build.py` | 主程序与各模块 |
| `ultralytics/` | **本地 ultralytics 源码，必须带！** 否则程序起不来 |
| `rfdetr_adapter.py` | RF-DETR 数据集适配器（**要用 RF-DETR 就必须带**，见第 15 节） |
| `Python/` | 海康威视相机 SDK（`MvImport`，Windows 专用，可选） |
| `*.yaml` | 数据集 / 模型配置文件（如 `6171.yaml`） |
| `*.pt` `*.onnx` | 你的模型权重文件 |
| `weights/` | 若程序用到里面的模型 |
| `requirements-Windows.txt` | 依赖清单 |
| `smoke_test.py` `dl_weight.py` | 部署后自检脚本 / RF-DETR 权重预取脚本（可选但建议带） |
| `requirements-rfdetr.txt` `constraints-torch.txt` | RF-DETR 可选依赖与 torch 约束文件（见第 15 节） |
| `部署教程_Windows_x64.md` | 本教程 |

**可以跳过、不必拷贝**（占空间且可重新生成）：
- `runs/`（训练/验证输出）、`__pycache__/`、`.git/`、`.workbuddy/`、`.trae/`
- `models/`、`out/`（异常检测生成的特征库 / 验证结果，可重新生成）
- `.rfdetr_models/`（RF-DETR 预训练权重缓存，349MB+，目标机会自动下载或用 `dl_weight.py` 预取）
- 根目录大量 `.pt`/`.onnx` 历史模型（只保留你实际要用的）

---

## 7. 海康威视相机支持（Windows 专用，可选）

程序接入了海康 MVS 相机 SDK（`Python/MvImport`）。代码中该导入已被 `try/except` 包裹，**缺失时只打印一行 Warning 并跳过，不影响程序启动和其余功能**。

- 若要用海康相机：目标机先装好 **海康威视 MVS**（Machine Vision Software）运行环境（提供 `MvCameraControl.dll` 等），并把相机接入网络。
- 若不用海康相机：用 **USB 摄像头**（实时检测里“摄像头索引”选 0/1）或 **RTSP 视频流** 即可，无需任何额外安装。

---

## 8. 运行程序

在项目根目录、且 conda 环境已激活的前提下：

```powershell
# 确保还在项目根目录且 (yolo) 已激活
conda activate yolo
python main.py
```

> 必须从**项目根目录**运行 `main.py`，这样程序才能找到本地 `ultralytics/` 和配置文件。

程序启动后是一个 PyQt6 图形界面，左侧是功能标签页，共 **10 个**：

| # | 标签页 | 用途 |
| --- | --- | --- |
| 1 | 实时检测 | 摄像头 / RTSP 实时推理 |
| 2 | 图片 | 单张 / 批量图片推理 |
| 3 | 视频 | 视频文件推理 |
| 4 | 训练 | YOLO 训练（含离线图像增强） |
| 5 | 验证 | 模型指标验证 |
| 6 | 导出 | ONNX 等格式导出 |
| 7 | **RF-DETR** | RF-DETR 训练 / 验证 / 导出（未装依赖时整页禁用并提示，见第 15 节） |
| 8 | Benchmark | 性能基准测试 |
| 9 | 异常检测 | DINOv3 特征建库 / 验证 / 导出 |
| 10 | 设置 | 运行设备等全局配置 |

窗口左上角和任务栏会显示 🎯 图标。

---

## 9. 命令行快速验证（不启动界面）

先确认模型 + GPU 推理链路正常：

```powershell
python -c "
from ultralytics import YOLO
import torch
print('GPU available:', torch.cuda.is_available())
m = YOLO('6171.pt')                 # 换成你实际的模型文件
r = m.predict('任意一张测试图片.jpg', device=0)   # device=0 用 GPU；无 GPU 改成 device='cpu'
print('detections:', len(r[0].boxes))
"
```

能看到检测框数量输出，说明环境部署成功，可以启动界面正式使用了。

### 9.1 一键自检（推荐，比上面那条命令更全面）

项目根目录自带 `smoke_test.py`，它会 offscreen 构建**真实的 MainWindow**，
调用与你点击按钮**完全相同**的代码路径，跑一遍「UI 结构 → 训练 → 导出 → 验证 → RF-DETR」：

```powershell
conda activate yolo
cd D:\apps\ultralytics-26_2

python smoke_test.py                        # 默认数据集 4940_has_labled.yaml + yolo26n.pt，1 epoch
python smoke_test.py 6171.yaml yolo26n.pt   # 指定你自己的数据集与基础模型
python smoke_test.py 6171.yaml yolo26n.pt 3 # 第三个参数是 epoch 数（默认 1）
```

通过时会打印 `冒烟结果: 通过`，并给出训练产物目录、导出的 ONNX 与验证指标行数。

| 检查项 | 说明 |
| --- | --- |
| UI 结构 | 10 个标签页、导航索引无错位、± 按钮齐备 |
| 训练 | 1 epoch，产物 `runs/detect/train*/weights/best.pt` |
| 导出 | ONNX，并用 ONNXRuntime 实际加载验证 |
| 验证 | 结果表非空（P / R / mAP 等指标） |
| RF-DETR | 未装依赖会跳过（属正常），不影响其余结论 |

> 单次约 2~4 分钟（视显卡）。产物都在 `runs/` 下，已被 `.gitignore` 忽略，不污染仓库。
> **注意**：自检只验证链路通，1 epoch 的 mAP 数值没有参考意义。

**参考基准**（开发机 RTX 3050 Laptop 4GB，全程 2 分 12 秒）：

| 阶段 | 实测 |
| --- | --- |
| 训练（1 epoch，511 张） | 86.6 秒，显存峰值 0.715G / 4G |
| 导出 ONNX | 2.2 秒，9.4 MB，ONNXRuntime 可加载 |
| 验证 | 23.2 秒，结果表 7 行（P / R / F1 / mAP@.5 / mAP@.5:.95 / mAP@.75 / Fitness） |

---

## 10. 训练 / 增强 / 异常检测功能速览（部署后怎么用）

- **训练**：填模型 + 数据集 yaml → 设置轮数/批次/图像尺寸（已横排）→ 可选「图像增强」分组（镜像/旋转/亮度/对比度/饱和度，原地不覆盖原图，生成 `<数据集>_aug/` 副本）→ 开始训练。右下角实时显示用时 / 预计剩余 / 预计完成。
- **导出**：把 `.pt` 导出为 ONNX（可勾选 simplify），产物在模型同目录。
- **异常检测**：用 DINOv3 提取特征建库（`.fbin`/`.npz`），验证时用 ONNX 推理。依赖 `onnx` + `onnxruntime`，CPU 即可。
- **RF-DETR**（需按第 15 节装依赖）：独立标签页，分「训练 / 验证 / 导出」三组。直接吃你现有的 YOLO 数据集（自动生成 `data.yaml`，不复制图片），导出 ONNX。详见第 15 节。

---

## 11. 常见问题排查

| 现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版 torch | 按第 4.1 节重装 `+cu126` 版，重装前先 `pip uninstall torch torchvision` |
| 导入 torch 报 DLL / 找不到模块 | NVIDIA 驱动过旧 | 更新显卡驱动到支持 CUDA 12.6 的版本（驱动 ≥ 560），**无需装 CUDA Toolkit** |
| `No module named 'ultralytics'` | 没拷贝本地 `ultralytics/` 文件夹 | 把项目根目录的 `ultralytics/` 一起拷过来 |
| `ImportError: attempted relative import` 或找不到本地模块 | 没在项目根目录运行 | `cd` 到项目根目录再 `python main.py` |
| 界面起不来 / 黑屏 | PyQt6 缺系统运行库（少见，Windows 通常自带） | 安装 [Visual C++ 运行库](https://learn.microsoft.com/zh-CN/cpp/windows/latest-supported-vc-redist) |
| 海康相机打不开 | 未装 MVS 或导入失败 | 见第 7 节；不影响其它功能 |
| `pip install` 极慢 / 超时 | 没配国内源 | 重新执行第 2 节 `pip config set global.index-url` |
| 实时检测卡顿 | 模型偏大 / 用 CPU 推理 | 改用轻量模型，或在程序内“运行设备”选 GPU；推理时调小 `imgsz` |
| 数字框 −/+ 按钮点不动 | 旧版本 bug | 用最新代码（已加显式 ± 按钮并隐藏原生箭头） |
| 「RF-DETR」标签页整页禁用 | 没装 rfdetr 依赖（设计如此，不影响其它功能） | 按第 15 节安装；不需要则忽略 |
| `import cv2` 版本变成 5.0 / 报 cv2 相关错 | 装 RF-DETR 时 `roboflow` 拉入了 `opencv-python-headless`，顶掉了原 `cv2` | `pip uninstall -y opencv-python-headless` 后 `pip install --force-reinstall --no-deps opencv-python==4.12.0.88`（见 15.2） |
| `import torch` 报 `ModuleNotFoundError: colorama / certifi / charset_normalizer` | site-packages 里有**空目录残骸**（程序运行时升级 pip 导致，会被当成命名空间包抢占导入） | 删掉空目录后重装：`pip install --force-reinstall --no-deps colorama certifi charset-normalizer attrs cycler` |
| RF-DETR 首次训练卡在下载权重（几十分钟） | 权重源站在境外，约 230KB/s；且默认写 C 盘 | 先用 `python dl_weight.py nano` 多线程预取（见 15.2），或设 `RF_HOME` 指向非 C 盘 |
| `smoke_test.py` 断言失败 | 环境或代码有问题 | 看失败那一步的打印；90% 是漏拷 `ultralytics/` 或数据集 yaml 路径不对 |

---

## 12. 可选：打包成 EXE（PyInstaller，进阶）

适合分发给不懂 Python 的同事。项目根目录有 `YOLO26_Studio.spec`（PyInstaller 配置），但它是早期写的，路径需要按当前项目名调整。

```powershell
# 在 conda 环境里安装打包工具
conda activate yolo
pip install pyinstaller

# 修改 YOLO26_Studio.spec 里的路径为当前项目名后，执行：
pyinstaller YOLO26_Studio.spec

# 产物在 dist/YOLO26_Studio/，把整个文件夹拷到目标机即可双击运行
```

⚠️ **注意**：
- 打包 **torch + CUDA** 体积很大（数 GB），首次打包慢。
- 需把 `ultralytics/` 本地源码、模型 `.pt`、`.yaml` 等一并打进 `datas`；CUDA 运行时也要随包携带。
- 若只给同架构 Windows 用，建议在目标机直接跑源码（第 8 节）更省事、更好维护。

---

## 13. 一键回顾（最小可行步骤）

```powershell
# 1) 装好 Miniconda3 x64（见第 1 节），打开 Anaconda Prompt
# 2) 配置国内镜像（conda + pip，见第 2 节）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 3) 用 conda 建并激活环境
cd D:\apps\ultralytics-26_2
conda create -n yolo python=3.12 -y
conda activate yolo
# 4) 装 GPU 版 torch（国内加速）
pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126 -f https://mirrors.aliyun.com/pytorch-wheels/cu126
# 5) 装其余依赖
pip install -r requirements-Windows.txt
# 6) 验证
python -c "import torch; print(torch.cuda.is_available())"
# 7) 一键自检（可选但推荐，跑完会打印「冒烟结果: 通过」）
python smoke_test.py
# 8) 运行
python main.py

# —— 可选：装 RF-DETR（见第 15 节）——
pip install -c constraints-torch.txt -r requirements-rfdetr.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python==4.12.0.88   # 把 cv2 拿回来
python dl_weight.py nano        # 预取权重到项目内，避免下到 C 盘
```

完成上面的 8 步，程序就能在其它 Windows 电脑上跑起来了。

---

## 14. 从旧版本升级到最新版（其它已部署电脑）

如果你之前已经按本教程（或旧版 venv 教程）在别的电脑上部署过，现在只要**同步最新代码 + 拉齐依赖**，不用重装系统、也不用重做数据集。

### 14.1 先确认你那台电脑是哪种部署形态

| 形态 | 判断方法 | 升级方式 |
| --- | --- | --- |
| **A. git 克隆部署** | 项目根目录有 `.git/` 且当初是 `git clone` 下来的 | 直接 `git pull`（最省事，见 14.2） |
| **B. 文件夹拷贝部署** | 当初是 U 盘 / 内网共享把文件夹整份拷过去的 | 手动同步程序文件（见 14.3，要小心别动数据） |

### 14.2 方式 A：git 克隆部署 → 一条命令升级

```powershell
cd <项目根目录>
git stash          # 仅当本地改过代码才需要；纯部署机一般跳过
git pull           # 拉最新代码
conda activate yolo   # 或你旧版的 venv： .\venv\Scripts\activate
pip install -r requirements-Windows.txt   # 拉齐依赖（保险起见跑一次）
python main.py
```

> `git pull` **只更新被 git 跟踪的文件**，你本地的数据集（`3631/`、`6171*/`、`*.pt`、`*.yaml` 等）是未跟踪文件，不会被碰。

### 14.3 方式 B：文件夹拷贝部署 → 只覆盖「程序文件」，保留「数据文件」

⚠️ **最关键的一步**：别把整个旧文件夹删了重拷，否则你的数据集、训练好的模型会丢。只把下面两张表里的「程序文件」从开发机最新版拷过去覆盖即可。

**✅ 程序文件（可安全覆盖，来自开发机最新版）：**

| 文件 / 目录 | 说明 |
| --- | --- |
| `main.py` `workers.py` `config.py` `styles.py` `val_report.py` `build.py` | 主程序与各模块 |
| `ultralytics/` | **本地 ultralytics 源码，必须一起更新！** 它不是 pip 装的，漏更新会出现「函数找不到 / 行为不一致」 |
| `requirements-Windows.txt` `部署教程_Windows_x64.md` | 依赖清单与教程 |
| **`rfdetr_adapter.py`** | RF-DETR 数据集适配器（**本次新增，漏拷会导致 RF-DETR 页报 ImportError**） |
| `smoke_test.py` `dl_weight.py` | 自检脚本 / 权重预取脚本（**本次新增**，建议一并拷） |
| `requirements-rfdetr.txt` `constraints-torch.txt` | RF-DETR 依赖清单与 torch 约束文件（**本次新增**，装 RF-DETR 时才需要） |
| 任何新增的 `.py` 模块 | 若有新文件也一并拷（如新增的增强/校验模块） |

**🚫 数据文件（**绝对不要覆盖或删除**）：**

| 文件 / 目录 | 说明 |
| --- | --- |
| `3631/` `6171/` `6171_*/` `6171_aug/` 等 | 你自己的图片 + 标注数据集 |
| `*.pt` `*.onnx` | 模型权重（你训练/使用的） |
| `*.yaml`（数据集类，如 `6171.yaml`） | 你自己的数据集配置 |
| `weights/` `runs/` `out/` `models/` | 训练/验证输出与特征库 |
| `Python/` `Hikvison/` | 海康 SDK（可选） |

操作建议：在开发机上把上述「程序文件」单独打包（或用 `robocopy` 只同步这几个路径），拷到目标机对应目录覆盖；其余原样保留。

### 14.4 运行环境要换 Miniconda3 吗？

**不用强求。** 分两种情况：

- **旧版用的是 venv 且还能正常跑**（Python 3.12 + 已装好 `torch+cu126` + 其余依赖）：直接沿用旧环境即可，跳到 14.3 同步代码后 `python main.py` 就能用，**不必重装**。
- **想统一到本教程最新标准（Miniconda3）**：按第 1~5 节新建 `yolo` 环境，旧 venv 可丢弃（`Remove-Item -Recurse venv` 或留着不管）。

> 无论沿用旧环境还是新建 conda 环境，**都建议跑一次** `pip install -r requirements-Windows.txt` 把依赖拉齐（YOLO 主链路当前无新增包，这一步基本是空跑，但能避免遗漏）。
> 若那台电脑也要用 **RF-DETR**，额外按第 15 节装可选依赖。

### 14.5 升级后验证

```powershell
conda activate yolo        # 或旧 venv
python -c "import torch; print(torch.cuda.is_available())"   # 应为 True（GPU 版）
python main.py             # 界面能起、功能正常即可
```

### 14.6 升级常见坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 升级后报 `AttributeError: ... 找不到某个函数/类` | 只更新了 `main.py`，漏了 `ultralytics/` 本地源码 | 把开发机最新的 `ultralytics/` 整个拷过去覆盖 |
| 数据集 / 模型不见了 | 整文件夹删除重拷，覆盖了数据 | 以后只按 14.3 覆盖程序文件；数据用备份恢复 |
| `pip install -r` 报 python 版本不兼容 | 旧环境是 Python 3.11 或更早 | 按第 1~3 节建 Python 3.12 的 conda 环境重装 |
| 界面能起但增强功能异常 | 漏拷新增 `.py` 模块 | 把开发机最新版所有 `.py` 都同步过去 |
| `ModuleNotFoundError: rfdetr_adapter` / RF-DETR 页点不动 | 漏拷新增的 `rfdetr_adapter.py`，或没装 rfdetr 依赖 | 拷入 `rfdetr_adapter.py`；要用就按第 15 节装依赖，不用则忽略该页 |
| 升级后 `import cv2` 版本异常 / 报 cv2 错 | 装 RF-DETR 时 cv2 被 headless 版顶掉 | 见第 11 节排查表对应条目（卸载 headless + 重装 opencv-python 4.12.0.88） |
| `git pull` 冲突 | 目标机本地改过代码 | 先 `git stash` 再 `pull`，或 `git checkout -- .` 放弃本地改动后 `pull` |

### 14.7 升级后一键自检

```powershell
conda activate yolo
cd <项目根目录>
python smoke_test.py            # 打印「冒烟结果: 通过」即代表升级无误
```

---

## 15. 可选：安装 RF-DETR（新增「RF-DETR」标签页）

RF-DETR 是**可选功能**，不装也能正常使用 YOLO 的全部功能（界面会多一个
「RF-DETR」标签页，但未装依赖时整页禁用并给出安装提示）。

> **不想用界面训练？** 另有专门的一篇《命令行训练教程.md》，
> 覆盖 YOLO 与 RF-DETR 的纯命令行训练、参数速查、显存预算与排错，
> 显存不够或要跑长任务时推荐走那条路。

### 15.1 安装（务必带约束文件）

```powershell
conda activate yolo
cd D:\apps\ultralytics-26_2

# -c 约束文件用于防止 pip 把 torch 换成 CPU 版（关键！）
pip install -c constraints-torch.txt -r requirements-rfdetr.txt
```

### 15.2 装完必做的两件事

**第 1 件：把被顶掉的 cv2 拿回来**

`roboflow` 会拉入 `opencv-python-headless`，它和项目用的 `opencv-python` 抢同一个
`cv2` 模块，装完会被覆盖成 5.0（大版本跳跃，有破坏性变更）。

```powershell
python -c "import cv2; print(cv2.__version__)"
```

若显示的不是 `4.12.0.88`，执行：

```powershell
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python==4.12.0.88
```

**第 2 件：预取预训练权重到非 C 盘**

首次训练会自动下载权重，默认落在 `C:\Users\<用户名>\.roboflow\models\`
（Nano 权重 349MB，更大的变体更多），且源站在境外、速度约 230KB/s。

程序已内置重定向（默认指向 `<项目根>\.rfdetr_models`，不会写 C 盘）。
建议部署后先手动预取（8 线程分块，实测 2.5MB/s 起步，349MB 约 2 分钟）：

```powershell
python dl_weight.py nano           # 下载到 <项目根>\.rfdetr_models\rf-detr-nano.pth
python dl_weight.py small          # Small
python dl_weight.py medium         # Medium
python dl_weight.py nano D:\cache  # 第二参数可指定其它目录
```

支持 `nano / small / medium / base / large` 五个变体（与界面开放的一致）。
脚本支持断点续传，中断后重跑即可继续；**已存在且大小正确的权重会自动跳过**，不会重复下载。

各变体实测数据（体积为实测值，分辨率为该变体默认训练分辨率）：

| 变体 | 权重体积 | 默认分辨率 | 适用显存 |
| --- | --- | --- | --- |
| Nano | 349 MB | 384 | **4 GB 够用**（实测峰值 1.02 GB） |
| Small | 368 MB | 512 | 6 GB 起 |
| Medium | 386 MB | 576 | 8 GB 起 |
| Base | ⚠️ 官方源当前返回 403，暂不可下载 | 640 | — |
| Large | 1499 MB | 704 | 16 GB 起（4GB 卡不必下） |

> Base 的官方地址 `rf-detr-base.pth` 目前返回 HTTP 403（官方源问题，非本脚本问题）。
> `python dl_weight.py base` 会明确报「源不可用」而不是写出损坏文件。
> 需要 Base 的话，可改用同族的 `rf-detr-base-2.pth`（478 MB，可访问），
> 但文件名与官方注册的不一致，需自行放进缓存目录并注意 MD5 校验会失败。

若要在命令行单独跑训练，先设环境变量：

```powershell
$env:RF_HOME = "D:\apps\ultralytics-26_2\.rfdetr_models"   # PowerShell
set RF_HOME=D:\apps\ultralytics-26_2\.rfdetr_models        # CMD
```

### 15.3 显存要求（实测）

官方 FAQ 说微调建议 ≥8GB，**实测远没有那么高**：

| 显卡 | 显存 | 实测峰值 | 结论 |
| --- | --- | --- | --- |
| RTX 3050 Laptop | 4 GB | **约 1.5 GB**（batch=2，Nano） | 够用，Nano 训练 1 epoch 约 135 秒 |

默认参数已按低显存配置（Nano + batch 由程序按当前可用显存自动选为 2 + 关闭 EMA），
4GB 显卡可直接开训。想换更大变体（Small/Medium）再按需调。

**物理 batch 与显存峰值**（RTX 3050 Laptop 4GB + Nano + 3631 数据集，实测）

显存开销主要由**物理 batch（一次送进显存的图片数）**决定；梯度累积只是多累加几步再更新，
几乎不增加峰值。所以想加大等效 batch，**加梯度累积远比加 batch 划算**。

| 物理 batch | 梯度累积 | 等效 batch | 显存峰值 | 结论 |
| --- | --- | --- | --- | --- |
| 4 | 4 | 16 | **2.47 GB** | 太紧，4GB 卡上极易崩，不要手动设 |
| 2 | 8 | 16 | **1.50 GB** | **推荐**：GUI 的 auto 与手动「2」都落在这档，等效 batch 一样但省一半显存 |
| 2 | 4 | 8 | 约 1.5 GB | 等效 batch 较小，小数据集够用，想更大就把梯度累积加到 8 |

> ⚠️ **不要用 RF-DETR 自带的 `batch_size="auto"` 探测**：它会额外建 shadow 参数副本 + 跑 AdamW 的 `step()`，
> 再用合成 batch 从大到小试到 OOM。在已被 Windows 桌面 + YOLO 推理模型占掉约 1.3GB 的 4GB 卡上，
> 探测本身就会把显存顶爆，连 `torch.cuda.empty_cache()` 都抛 `CUDA error: out of memory`，
> 进程被**硬崩**（GUI 里表现为无 traceback 的 `Unhandled Python exception`）。
> 本程序 GUI 里的「auto（batch=0）」**不会**透传这个危险值，而是由 `rfdetr_adapter.suggest_rf_batch()`
> 按当前可用显存算一个具体整数（4GB 卡固定为 2）再交给 RF-DETR，从而跳过探测分支。
> 命令行脚本 `train_rfdetr.py` 也**不接受** `auto`，请直接给数字（如 `--batch 2`）。

#### 训练前，你的 4GB 到底被谁占了？（实测）

打开任务管理器看到的「GPU 显存」里，**大头根本不是训练程序**。实测顺序扣除如下：

| 占用方 | 显存 | 说明 |
| --- | --- | --- |
| **Windows 桌面本身** | **约 1000 MB** | Explorer、Edge、输入法、豆包、ToDesk、WPS 等都在用 GPU 合成画面（nvidia-smi 里那一长串 `C+G` 进程）。**笔记本独显_shared 架构下这部分无法省掉** |
| Python 的 CUDA 上下文 | **约 70 MB** | 只要 `import torch` 并建一个 CUDA 张量就有，无法省 |
| YOLO 推理（载入模型 + 跑一次 640） | **约 100 MB** | 开了实时相机 / 图片 / 视频检测才会产生 |
| **真正留给训练的** | **约 2.8 GB** | 这就是 batch=4 会崩、batch=2 能跑的原因 |

**界面本身只占一百多 MB，不用为它担心**；真正吃紧的是桌面那 1GB 固定开销。要最大化训练预算：

1. **训练前别开着实时相机/视频检测**：模型停了也没用 —— PyTorch 的缓存分配器不会把显存还给驱动。
   程序已在每次训练/验证前自动执行 `torch.cuda.empty_cache()` 回收缓存，日志里会打印
   `[RF-DETR] 已回收 xx MB 显存缓存`；若有推理线程还在跑，会提示
   `[RF-DETR] 注意：仍有任务在跑（…）`，**此时应先把它们停下来**。
2. **关掉占 GPU 的桌面程序**（浏览器硬件加速、视频/直播类软件），能省几百 MB。
3. **想完全隔离**：用本教程配套的命令行方式训练（第 16 章），完全不启动界面。

### 15.4 数据集要求

RF-DETR 原生只认「`<数据集根>/train/images` + `<数据集根>/train/labels`」这一种层级，
但本程序的适配器（`rfdetr_adapter.py`）会**自动在两种常见写法间选择磁盘上真实存在的目录**，
因此以下两种布局都能直接用，**无需手动改目录**：

| 你的数据集布局 | 能否直接用 | 说明 |
| --- | --- | --- |
| `<根>/train/images` + `<根>/train/labels` | ✅ | split-first，RF-DETR 原生布局 |
| `<根>/images/train` + `<根>/labels/train` | ✅ | images-first，适配器自动改映射到 `train/images` |
| `4940_has_labled` | ✅ | 程序自动生成 `data.yaml` |
| `3631` | ✅ | 适配器检测到 `images/train` 后自动改用 `train/images`，并在日志里提示一行 |

程序会在数据集根目录生成 `data.yaml`（绝对路径），**不复制任何图片、不改动你的原始 yaml**。
若 yaml 里写的路径（如 `images/train`）与磁盘实际布局不一致，适配器会自动选真实存在的目录并记一行日志。

### 15.5 学习率（lr）怎么设

**一句话**：权重更新的公式是 `θ ← θ − lr · ∇L(θ)`。梯度告诉你往哪个方向改，**学习率决定这一步迈多大**。
好比蒙眼下山：步幅太大会在山谷两侧来回横跳、甚至冲出去（loss 振荡不收敛）；
太小则半天走不到底（loss 几乎不动）。合适的值是「平稳下降，最后自然走平」。

**RF-DETR 的默认值**（取自你安装的 `rfdetr/config.py` 的 `TrainConfig`）：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `lr` | **1e-4** | 主学习率，管解码器 / head 等大部分参数；界面上的「学习率」框就是它 |
| `lr_encoder` | 1.5e-4 | 主干 backbone 单独的学习率（界面未暴露，保持默认即可） |
| `optimizer` | `adamw` | AdamW 优化器 |
| `weight_decay` | 1e-4 | 权重衰减 |
| `clip_max_norm` | 0.1 | 梯度裁剪，防止单步更新过猛 |
| `warmup_epochs` | 0.0 | 默认**没有** warmup |
| `lr_scheduler` / `lr_drop` | `step` / 100 | 默认第 100 轮才衰减 |
| `lr_vit_layer_decay` | 0.8 | backbone 逐层递减 |
| `lr_component_decay` | 0.7 | 模块级递减 |

界面里的学习率默认是 **1e-4**，与官方默认一致，**一般不用动**。

**建议取值**：

| 场景 | 建议 lr |
| --- | --- |
| RF-DETR 微调（常规） | **1e-4**（直接用默认值） |
| 数据集很小 / 怕破坏预训练特征 | 5e-5 ~ 1e-4 |
| loss 振荡、出现突刺或 NaN | 降到 5e-5，甚至 3e-5 |
| 收敛太慢、epoch 又给足 | 可试 2e-4 |
| 从头训（RF-DETR 很少这么用） | 2e-4 ~ 4e-4 |

**怎么判断合不合适**：直接看日志区的 loss 曲线 —— 忽高忽低来回跳 = 太大，降到 5e-5；
几乎画成一条平线 = 太小，试 2e-4；平稳下降最后走平 = 刚好。

**三个容易踩的细节**：

1. **backbone 的学习率是独立的**：`lr` 管大部分参数，主干另有 `lr_encoder`（默认 1.5e-4）。
   界面只暴露 `lr`，通常不必改。
2. **默认没有 lr 衰减**：`lr_drop=100` 而 `epochs=100`，等于跑满 100 轮才衰减一次；
   界面默认 50 轮，实务上是**全程不衰减**。想让后期收敛更稳，需要调小 `lr_drop`（界面暂未暴露）。
3. **lr 不会随 batch 自动缩放**：RF-DETR 源码注释写得很明确 —— auto-batch 时 effective batch
   会随显存变化，而 `lr stays put`。手动改大 batch 时，程序**不会**帮你同步放大 lr。

### 15.6 已知限制

- 导出的 ONNX **不能在现有「实时检测 / 图片 / 视频」页直接加载**。
  RF-DETR 输出是 NMS-free 格式（`dets [1,300,4]` + `labels [1,300,2]`），
  与 YOLO 完全不同，需要在外部 ONNXRuntime 自行后处理。
- 只开放 Nano / Small / Medium / Base / Large（Apache 2.0）。
  XLarge / 2XLarge 是 PML 1.0 许可，商用需自行确认条款。

### 15.7 界面上怎么用

打开「RF-DETR」标签页，共三组，按顺序走：

**① 训练**
| 控件 | 建议值 | 说明 |
| --- | --- | --- |
| 模型变体 | Nano（≤4GB）/ Small（6GB）/ Medium（8GB） | 默认 Nano；4GB 卡建议就别动 |
| 数据集 YAML | 选你的 `xxx.yaml` | `train/images` 或 `images/train` 两种布局都行，见 15.4 |
| 训练轮数 | 50（默认） | |
| 批次大小 | **auto**（0） | 程序按当前可用显存自动选一个**安全整数**（4GB 卡固定为 2），**不**调用 RF-DETR 自带探测；也可手动填 2 |
| 梯度累积 | 4 | 与 batch 共同决定 effective batch |
| 学习率 | 1e-4 | |
| 输出目录 | `<项目根>\rfdetr_output` | 项目目录；每次训练自动在其下建 `train1`、`train2`… 递增子目录，产物与导出 ONNX 都落在当前 `trainN` 里 |

点「开始训练」后，日志区会滚动训练日志，结束后**自动把最佳权重填进下面两组的 checkpoint 框**。

**② 验证**：选 `test` 或 `val` → 点「验证」，指标以表格展示（mAP@50:95、mAP@50、mAR、F1 等）。

**③ 导出**：选 opset（默认 17）→ 点「导出 ONNX」，产物在输出目录下（`rfdetr-<变体>.onnx`）。

### 15.8 RF-DETR 常见问题

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 页面顶部红色提示条「未检测到 rfdetr」 | 依赖未安装或导入失败 | 按 15.1 重装；重启程序 |
| 训练报「找不到训练集图片目录」 | yaml 的 train/val 指向的路径在磁盘上确实不存在 | 检查 yaml 里 train 字段是否写对（适配器已自动兼容 `train/images` 与 `images/train` 两种写法，只有真的找不到目录才会报此错） |
| 日志出现 `incorrect MD5 hash` 并重新下载权重 | 权重文件损坏（多线程下载时分片写入异常） | 已修：`dl_weight.py` 现在会校验 MD5；若仍损坏，删掉文件重跑脚本 |
| `dl_weight.py base` 报「源不可用」 | 官方 `rf-detr-base.pth` 链接返回 403 | 换用 Nano / Small / Medium，或改用 `rf-detr-base-2.pth` |
| 训练 OOM（显存不足） | 变体太大 / 物理 batch 手动设太大 | 换 Nano；batch 保持 `auto`（程序会自动选 2），或手动设 **2** 并把梯度累积加倍以保持等效 batch（见 15.3 实测）。**切勿手动设 4 及以上** |
| 点「开始训练」后立刻崩，日志最后只有 `Unhandled Python exception` 且没有堆栈 | **GUI 进程内直接跑 CUDA 训练**导致 C 层 abort（CUDA 上下文与 Qt 事件循环 / 常驻推理模型在同一进程内的线程/显存冲突）；`try/except` 与 `faulthandler` 都抓不到，故无堆栈。与 batch/分辨率/AMP/多尺度/数据无关（同一份训练在 CLI、后台线程、甚至带 YOLO 常驻显存下都 100% 跑通，唯独 GUI 进程内崩） | **已修复**：GUI 训练改为启动**子进程**跑 `train_rfdetr.py`（用 `sys.executable`，隔离出干净的 CUDA 上下文与显存池），日志/进度/停止照常转发。你无需任何操作；若仍看到此提示，先确认依赖为最新版（workers.py 的 `RFDETRTrainWorker` 已是子进程版） |
| 权重下载卡住 | 源站在境外 | 用 `python dl_weight.py nano` 预取（见 15.2） |
| 导出的 ONNX 在实时检测页加载失败 | RF-DETR 输出格式与 YOLO 不同（NMS-free，非缺陷） | 在外部 ONNXRuntime 自行后处理：`dets[1,300,4]` + `labels[1,300,2]` |
