# YOLO26 AI视觉 Studio 部署教程（Ubuntu 22.04 ARM64 / NVIDIA Jetson）

本教程适用于在 **NVIDIA Jetson 开发板**（如 Jetson Orin Nano / Orin NX / AGX Orin）上部署本项目。

> 最后更新：2026-09-16（补充自检脚本、RF-DETR 说明）
> Windows x64 部署请看 `部署教程_Windows_x64.md`。本教程只覆盖 ARM64 特有的部分
> （Jetson 专用 torch、Qt 依赖等），通用逻辑与 Windows 版一致。

## 0. 适用平台与版本对应关系

| 项目 | 说明 |
| --- | --- |
| 硬件 | NVIDIA Jetson（带 GPU，aarch64 架构） |
| 系统 | Ubuntu 22.04 LTS（ARM64） |
| JetPack | 6.x（JetPack 5.x 是 Ubuntu 20.04，与教程命令不完全兼容） |
| CUDA | 12.x（JetPack 6 自带） |
| Python | 3.10（JetPack 6 默认） |

> 注意：Jetson 是 ARM64 架构，**不能直接使用 PyPI 上的官方 PyTorch**（那个是 x86_64 的），必须安装 NVIDIA 为 Jetson 编译的专用 wheel，否则 `torch.cuda.is_available()` 会返回 False。

## 1. 环境确认

先确认设备状态，所有命令都在 Jetson 的终端里执行。

```bash
# 查看架构（应显示 aarch64）
uname -m

# 查看系统版本（应为 Ubuntu 22.04）
lsb_release -a

# 查看 JetPack / L4T 版本
cat /etc/nv_tegra_release

# 查看 Python 版本（应为 3.10）
python3 --version
```

## 2. 系统更新与基础依赖

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y python3-pip python3-venv git build-essential cmake \
    libjpeg-dev libpng-dev libtiff-dev zlib1g-dev \
    libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```

## 3. 安装 Qt6 图形界面依赖（PyQt6 必需）

程序是 PyQt6 图形界面程序，必须先装好 Qt 的 xcb 平台插件依赖。**`libxcb-cursor0` 缺失是“界面起不来”最常见的原因**。

```bash
sudo apt install -y \
    libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-shape0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 \
    libx11-xcb1 libegl1 libgl1 libglib2.0-0 libfontconfig1 libdbus-1-3 \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-libav
```

## 4. 创建 Python 虚拟环境

```bash
python3 -m venv ~/venv_yolo
source ~/venv_yolo/bin/activate
pip install --upgrade pip
```

## 5. 安装 PyTorch（Jetson 专用，GPU 加速）

### 方式 A（推荐）：Jetson AI Lab 预编译源，一次装好 torch + torchvision

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
```

> `jp6/cu126` 对应 JetPack 6.x + CUDA 12.6。若你的 JetPack 版本不同，到 <https://pypi.jetson-ai-lab.io> 查看对应的路径（如 `jp6/cu128` 等）。

### 方式 B：NVIDIA 官方 wheel

以 JetPack 6.1 / PyTorch 2.5 为例（torchvision 需要从源码编译，较麻烦，一般用方式 A）：

```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### 验证 GPU 可用

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

输出应类似：`2.5.0a0+... 12.6 True`，最后一项必须是 `True`。

## 6. 补齐 cuDNN / cuSPARSELt（如 import torch 报缺库）

JetPack 6 通常自带 cuDNN，但如果 `import torch` 报 `libcudnn.so.9` 缺失：

```bash
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo ldconfig
```

如果报 `libcusparseLt.so.0` 缺失：

```bash
pip install nvidia-cusparselt-cu12
# 找到其 lib 路径并加入动态库搜索路径（建议写入 ~/.bashrc）
export LD_LIBRARY_PATH=$HOME/venv_yolo/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
```

## 7. 安装其余 Python 依赖

```bash
# NumPy 必须用 1.x（Jetson 版 PyTorch 基于 NumPy 1.x 编译，2.x 会冲突）
pip install "numpy<2"

# 图形界面 + 屏幕采集
pip install PyQt6 mss

# OpenCV：程序界面用 PyQt6 渲染，装 headless 版可避免与系统 GTK 冲突
pip install opencv-python-headless

# 安装 ultralytics（会一并装好其全部依赖）
pip install ultralytics
```

> **重要**：本项目根目录自带一份 `ultralytics` 源码，程序运行时会优先使用本地这份（`main.py` / `workers.py` 已把项目根目录加入 `sys.path`）。上面安装 ultralytics 只是为了把它的依赖（torch、numpy、opencv、Pillow、matplotlib、pyyaml、scipy、psutil 等）装齐。若想完全使用 pip 版本，可删除项目根目录下的 `ultralytics` 文件夹。

## 8. 部署项目

1. 把项目文件夹（含 `main.py`、`workers.py`、`config.py`、`styles.py`、`build.py`、`weights`、`3631.yaml` 等）整体拷贝到 Jetson，例如 `/home/nvidia/ultralytics-26_2`。
2. 拷贝方式：U 盘、`scp`、`rsync` 或 git 均可。
3. 项目里的 `Python/MvImport`（海康威视相机 SDK）是 **Windows 专用**。代码中该导入已被 `try/except` 包裹，Linux 下无法导入时会打印一行 Warning 后跳过，**不影响程序启动和其余功能**。
   - 在 Jetson 上请改用 **USB 摄像头**（实时检测里“摄像头索引”选 0）或 RTSP 视频流。

## 9. 运行程序

程序是图形界面程序，**必须在桌面环境（X11 / Wayland）下运行**。Jetson 开发套件默认带 Ubuntu 桌面。

```bash
cd /home/nvidia/ultralytics-26_2
~/venv_yolo/bin/python main.py
```

### 无显示器时（远程 VNC）

若 Jetson 无外接屏幕，可通过 VNC 连接桌面：

```bash
sudo apt install -y tigervnc-standalone-server
# 启动虚拟桌面并开启 VNC（以 1280x800 为例）
Xvfb :1 -screen 0 1280x800x24 &
DISPLAY=:1 x11vnc -forever -passwd 123456 &
# 在电脑上用 VNC 客户端连接 <Jetson-IP>:5900
```

连接后运行程序时指定 DISPLAY：

```bash
cd /home/nvidia/ultralytics-26_2
DISPLAY=:1 ~/venv_yolo/bin/python main.py
```

### 使用 GPU 推理

程序内“运行设备”选择 `0`（GPU）即可获得 CUDA 加速。首次推理会加载模型到显存，稍慢属正常。

## 10. 常见问题排查

| 现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `could not load the Qt platform plugin "xcb"` | 缺少 Qt xcb 依赖 | 安装第 3 节的全部依赖，重点确认 `libxcb-cursor0` |
| `import torch` 报 `libcudnn.so.9` 缺失 | 缺 cuDNN | `sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && sudo ldconfig` |
| `import torch` 报 `libcusparseLt.so.0` 缺失 | 缺 cuSPARSELt | `pip install nvidia-cusparselt-cu12` 并配置 `LD_LIBRARY_PATH` |
| `torch.cuda.is_available()` 返回 False | 装成了 CPU 版 torch | `pip uninstall torch torchvision` 后按第 5 节重装 Jetson 专用 wheel |
| NumPy 2.x 报错 / torch 无法 import | 版本冲突 | `pip install "numpy<2" --force-reinstall` |
| 界面中文显示成方块 | 缺中文字体 | `sudo apt install -y fonts-wqy-zenhei fonts-wqy-microhei` |
| 摄像头打不开 | 权限或索引不对 | USB 相机索引调 0/1 尝试；权限问题可加 `sudo`；CSI 相机需系统自带带 GStreamer 的 OpenCV |
| 推理卡顿 | 模型偏大 | 选用 `yolo11n` 等轻量模型，或在导出/推理时调小 `imgsz` |

## 11. 命令行验证

不启动界面，先验证模型 + GPU 推理链路是否正常：

```bash
cd /home/nvidia/ultralytics-26_2
~/venv_yolo/bin/python -c "
from ultralytics import YOLO
import torch
print('GPU available:', torch.cuda.is_available())
m = YOLO('weights/yolo11n.pt')   # 换成你实际的模型文件
r = m.predict('bus.jpg', device=0)   # 放一张测试图片，或换成你的图片
print('detections:', len(r[0].boxes))
"
```

能看到检测框数量输出，说明环境部署成功，可以启动界面正式使用了。

## 12. 一键自检（可选，推荐）

项目根目录的 `smoke_test.py` 在 Windows / Linux 上通用（offscreen 模式，不需要显示器），
会跑一遍「UI 结构 → 训练 → 导出 → 验证」全链路：

```bash
cd /home/nvidia/ultralytics-26_2
~/venv_yolo/bin/python smoke_test.py                        # 默认数据集与模型
~/venv_yolo/bin/python smoke_test.py 6171.yaml yolo26n.pt   # 指定数据集 / 基础模型
```

打印 `冒烟结果: 通过` 即代表环境、代码、数据三者都配对。
（Jetson 上训练较慢，1 epoch 可能要几分钟，属正常。）

## 13. 关于 RF-DETR（ARM64 上**未验证**）

Windows 版教程的第 15 节描述了 RF-DETR 训练 / 验证 / 导出功能。在 Jetson ARM64 上：

| 项 | 状态 |
| --- | --- |
| YOLO 全部功能（训练 / 验证 / 导出 / 增强 / 异常检测） | ✅ 完全支持 |
| RF-DETR 标签页 | ⚠️ **未安装依赖时会整页禁用并提示**，不影响其它功能 |
| 在 ARM64 上装 `rfdetr` | ❓ 未实测。`rfdetr` 依赖 `roboflow` / `pytorch-lightning` / `transformers`，在 aarch64 上可能需自行编译，且会拉入 `opencv-python-headless` 顶替系统 `cv2`（Jetson 的 cv2 是带 GStreamer 的定制版，被覆盖后 **CSI 相机可能失效**） |

**建议**：Jetson 上**不要装 RF-DETR**，保持 YOLO 链路即可。
如果确实要试，务必先备份当前 `cv2`，并在独立的虚拟环境里安装，不要污染 `venv_yolo`。
