# YOLO26 Studio

基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 8.4.132 的桌面级视觉训练 / 推理一体化工具。主界面采用 PyQt6，原生支持 **YOLO26 检测与深度估计**，并集成 **RF-DETR** 目标检测训练与 **DINOv3** 异常检测模块。

## 功能概览

- **YOLO 训练 / 验证 / 导出**
  - 支持 YOLO26 系列（n / s / m / l / x）检测与深度估计模型
  - 训练页：epochs / batch / 学习率 / 图像增强 等参数可视化配置，实时进度与日志
  - 模型验证结果界面展示，支持验证报告导出
  - 导出 ONNX / INT8 校准数据
- **RF-DETR 训练**
  - 变体：Nano / Small / Medium / Base / Large
  - 训练在独立子进程中执行，隔离 GUI 进程的 CUDA / 线程冲突，避免界面闪退
  - 自动按可用显存选定安全 batch（4GB 显存默认 2），绕过 RF-DETR 自带探测
- **DINOv3 异常检测**
  - 特征库建库 / 验证 / 导出 ONNX
  - 特征库文件名自动追加库容量 + 时间戳，避免多次建库互相覆盖
- **其它**
  - 离线图像增强（训练数据增广）
  - 多语言界面、基准测速、训练用时 / ETA / 预计完成时间实时刷新

## 环境要求

- Python 3.12+（推荐用 conda 环境，如 `yolo`）
- PyTorch 2.9+（CUDA 12.6 版本，需 NVIDIA GPU）
- Windows 10/11 或 Linux（Ubuntu 22.04 ARM64 部署见 `部署教程_Ubuntu22_ARM64.md`）
- 依赖清单：`requirements-Windows.txt`、`requirements-rfdetr.txt`、`constraints-torch.txt`

## 快速开始

```bash
# 1. 创建并激活环境
conda create -n yolo python=3.12
conda activate yolo

# 2. 安装依赖（Windows）
pip install -r requirements-Windows.txt
pip install -r requirements-rfdetr.txt

# 3. 启动界面
python main.py
```

## 数据集配置

训练使用标准 Ultralytics `data.yaml`，示例：

```yaml
path: ./3631
train: train/images
val: val/images
names:
  0: ng_1
  1: ok_1
```

> ⚠️ **重要格式坑**：`names` 下每项的冒号后**必须留空格**（`0: ng_1`）。
> 写成 `0:ng_1`（无空格）会被 YAML 整段读成纯文本字符串，导致训练后段
> `class_id not in self.names` 崩溃。

## 命令行训练（参考）

- YOLO：`python train_yolo.py --data 3631.yaml --epochs 50 --batch 16`
- RF-DETR：`python train_rfdetr.py --data 3631.yaml --variant nano --epochs 50 --batch 2 --resolution 384`

GUI 内的 RF-DETR 训练即调用 `train_rfdetr.py` 作为子进程，参数一致。

## 目录结构（要点）

| 文件 / 目录 | 说明 |
|---|---|
| `main.py` | PyQt6 主界面入口 |
| `workers.py` | 训练 / 验证 / 导出后台 worker（YOLO 进程内；RF-DETR 子进程隔离） |
| `rfdetr_adapter.py` | RF-DETR 数据集适配与训练封装 |
| `train_rfdetr.py` / `train_yolo.py` | 命令行训练脚本 |
| `ultralytics/` | 内置 ultralytics 8.4.132（含 YOLO26 支持） |
| `部署教程_Windows_x64.md` / `部署教程_Ubuntu22_ARM64.md` | 部署说明 |
| `RF-DETR训练说明.md` | RF-DETR 训练专项说明 |

## 说明

- 权重文件（`*.pt` / `*.pth`）、`runs/`、`rfdetr_output/`、`.rfdetr_models/` 已由 `.gitignore` 忽略，不纳入版本控制。
- GUI 内训练默认不传 `device`，由 RF-DETR 自动选择 CUDA；显存不足时请在界面降低 batch / 分辨率。

## License

本仓库基于 Ultralytics（AGPL-3.0）二次开发。商业用途请遵循 [Ultralytics 许可条款](https://www.ultralytics.com/license)。
