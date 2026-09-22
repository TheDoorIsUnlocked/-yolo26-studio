# RF-DETR 训练 / 验证 / 导出 集成方案

> 目标：在现有 YOLO26 AI 视觉 Studio 中增加 RF-DETR 模型的**训练、验证、导出**能力。
> 状态：**待评审**（方案先行，Sir 确认后再编码）
> 调研日期：2026-09-15

---

## 0. 结论（先说行不行）

**可行，风险可控，建议做。** 三个关键事实：

1. **RF-DETR 原生支持 YOLO 格式数据集**（自动检测 `dataset_dir/data.yaml` + `train/images/`），**不需要**把数据集转成 COCO JSON，也不用复制图片。
2. **依赖无冲突**：`rfdetr` 只要求 `torch>=2.2.0`（无上限 pin），当前环境 `torch 2.9.1+cu126` 满足；唯一可能冲突的 `transformers` 当前**未安装**，仓库代码也没用到，可安全装 5.1+。
3. **建议做成独立标签页**，不改动现有 YOLO 训练/验证/导出链路，避免再次出现"越改越差"的回归。

---

## 1. RF-DETR 关键事实（官方文档核实）

| 项 | 事实 |
| --- | --- |
| PyPI 包 | `rfdetr`，当前最新 **1.10.1**（2026-09-15） |
| Python | `>=3.10`（本机 3.12.12 ✅） |
| 核心依赖 | `torch>=2.2.0`、`torchvision>=0.17.0`、`transformers>=5.1,<6`、`supervision>=0.29`、`pydantic>=2,<3` |
| 训练额外依赖 | `pip install "rfdetr[train]"` → pytorch_lightning(≠2.6.2/3, ≥2.6)、torchmetrics[detection]、faster-coco-eval、pycocotools、scipy、torch-hungarian、roboflow、peft |
| ONNX 导出依赖 | `pip install "rfdetr[onnx]"` → onnx、onnxsim、onnx_graphsurgeon、polygraphy、onnxruntime |
| 模型变体 | `RFDETRNano / Small / Medium / Base / Large`（Apache 2.0）；`XLarge / 2XLarge` 需 `rfdetr[plus]`（**PML 1.0 许可**）；另有 Seg（实例分割）、KeypointPreview |
| 训练 API | `model.train(dataset_dir=, epochs=, batch_size=, grad_accum_steps=, lr=, output_dir=, resume=, dataset_file=, tensorboard=, wandb=)`；`batch_size` 支持 `"auto"` |
| 验证 API | `model.evaluate(dataset_dir=, split="test"\|"val")` → 返回 dict，含 `mAP_50_95`、`mAR`、macro-F1 sweep |
| 导出 API | `model.export(format="onnx"\|"tflite"\|"tensorrt"\|"executorch"\|"coreml", output_dir=, opset_version=17, shape=(h,w), dynamic_batch=, fp16=, batch_size=, output_name=)` |
| 权重格式 | 预训练/微调产物为 **`.pth`**；加载用 `RFDETRMedium(pretrain_weights="<path>.pth")` |
| 数据集格式 | **自动检测**：COCO（`train/_annotations.coco.json`）或 YOLO（**根目录 `data.yaml` 或 `data.yml`** + `train/images/`） |
| 硬件建议 | 微调建议 **≥8GB 显存**；Nano/Small 可在 6GB 上用小 batch 跑 |

**⚠️ 许可提示（工业商用需留意）**：Nano ~ Large 是 **Apache 2.0**，可商用；**XLarge / 2XLarge 是 PML 1.0**，商用前需确认条款。建议本期**只开放 Nano ~ Large**。

---

## 2. 项目现状

| 文件 | 规模 | 说明 |
| --- | --- | --- |
| `main.py` | 2477 行 | `MainWindow`，标签页：实时 / 图片 / 视频 / 训练 / 验证 / 导出 / 基准 / **异常检测** |
| `workers.py` | 1473 行 | `VideoThread`、`ImageWorker`、`VideoFileWorker`、`AugmentWorker`、`TrainWorker`、`ValWorker` …（QThread + pyqtSignal） |
| `config.py` | 198 行 | `Config.TRANS` 中英文案表（CN/EN） |
| `styles.py` | 220 行 | 样式，含 `NumButton` 实底样式、隐藏原生箭头 |

**现有环境**：Python 3.12.12 / torch 2.9.1+cu126 / torchvision 0.24.1+cu126 / numpy 2.1.3 / onnx 1.22.0 / onnxruntime 1.29.0；`transformers`、`pytorch-lightning`、`pycocotools` 均未安装。

**现有数据集布局（不一致，需适配）**：

| 数据集 | 布局 | 是否直接满足 RF-DETR |
| --- | --- | --- |
| `4940_has_labled` | `train/images`、`train/labels`、`val/…`、`test/…` | ✅ 满足（`train/images`） |
| `3631` | `images/train`、`images/val` | ❌ 不满足（目录层级相反） |

**共同问题**：两者的 yaml 都在**项目根**（`4940_has_labled.yaml`），而 RF-DETR 要求 yaml 在**数据集根目录**且名为 `data.yaml`。

---

## 3. 架构方案

沿用仓库里已有的**"异常检测"标签页模式**——一个标签页内用 `QGroupBox` 分「训练 / 验证 / 导出」三组，与 DINOv3 异常检测的「建库 / 验证 / 导出」结构保持一致，视觉与交互统一。

```
MainWindow
└── [新增] RF-DETR 标签页
    ├── 训练组：变体选择 / epochs / batch_size / grad_accum / lr / output_dir / resume
    ├── 验证组：数据集 / split(test|val) → mAP 指标展示
    └── 导出组：格式 / opset / 输入尺寸 / 动态 batch → 输出 .onnx
```

数据流（适配器是关键，避免动数据集）：

```
选中 YOLO yaml  →  data.yaml 适配器  →  RF-DETR 训练  →  checkpoint .pth  ┬→ 验证 evaluate() → mAP
 (项目根)          (写入数据集根)        (model.train)                      └→ 导出 export()   → .onnx
                    绝对路径 + names
```

**新增/修改文件清单**：

| 文件 | 改动 |
| --- | --- |
| `rfdetr_adapter.py`（新增） | 读选中 yaml → 解析 `path/train/val/test/names` → 在数据集根生成 `data.yaml`（**绝对路径**，复用之前"YOLO 相对路径按 CWD 解析"踩坑教训）→ 校验 `train/images/` 存在，缺失则明确报错 |
| `workers.py` | 新增 `RFDETRTrainWorker` / `RFDETREvalWorker` / `RFDETRExportWorker`（QThread，沿用 `log_signal` / `progress_signal` / `finished_signal` 约定） |
| `main.py` | 新增 `create_rfdetr_tab()` 并注册；复用 `_num_row()`（± 显式按钮）与日志区 |
| `config.py` | 新增 RF-DETR 相关中英文案 key |
| `styles.py` | 预计无需改动（复用现有样式） |
| `requirements-rfdetr.txt`（新增） | 单独列出 `rfdetr[train,onnx]` 等，不污染主清单 |
| `部署教程_Windows_x64.md` | 补充 RF-DETR 可选依赖安装段 |

---

## 4. 分阶段实施计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| **P1 依赖与探测** | 写 `requirements-rfdetr.txt`；运行时 `import rfdetr` 探测；未安装时 RF-DETR 页显示安装提示并禁用按钮，**其余功能完全不受影响**（沿用海康 SDK 缺失只 warn 跳过的模式） | 装/不装都不影响现有功能 |
| **P2 数据集适配** | `rfdetr_adapter.py`：生成 `data.yaml`（绝对路径）；`images/train` 布局检测与友好报错；单元自测（用 `4940_has_labled` / `3631` 各验一遍） | 适配器 + 自测通过 |
| **P3 训练** | RF-DETR 训练组 UI + `RFDETRTrainWorker`；stdout 日志透传到日志区；复用现有 QTimer 做 Duration/ETA；进度按 epoch 粗粒度上报 | 小 epoch 跑通，产出 `.pth` |
| **P4 验证** | `evaluate()` 走 QThread；结果以表格展示 mAP_50_95 / mAR / F1 | 指标可见 |
| **P5 导出** | `export(format=...)` 走 QThread；默认 ONNX；产出落 `output_dir` | 产出 `.onnx`，可被 onnxruntime 加载 |
| **P6 文档与部署** | 教程补充 RF-DETR 段；梳理**部署机需更新文件清单** | 教程 + 清单 |

---

## 5. 关键决策点 —— ✅ 已确认（2026-09-15）

| # | 决策 | **最终选择** | 理由 |
| --- | --- | --- | --- |
| D1 | **集成形态** | ✅ **A. 独立 RF-DETR 标签页**（三组：训练 / 验证 / 导出） | 零回归风险，与异常检测页模式一致；不动已跑通的 YOLO 链路 |
| D2 | **数据集适配** | ✅ **A. 运行时自动生成 `data.yaml`** | 零图片复制，不改动原数据集，`4940_has_labled` 直接可用 |
| D3 | **导出范围** | ✅ **A. 仅 ONNX** | 最通用、依赖最少、能接现有部署链路；TensorRT / TFLite 等后续按需加 |
| D4 | **依赖策略** | ✅ **A. 可选安装 + 优雅降级**（单独 `requirements-rfdetr.txt`） | 不强制所有部署机安装；未装时其余功能不受影响 |

**由此锁定的范围收窄**：
- 只做**目标检测**变体 `Nano / Small / Medium / Base / Large`（Apache 2.0）。
- 导出只出 **`.onnx`**（opset 17，可选动态 batch / 指定输入尺寸）。
- 不做 Seg、关键点、多卡 DDP、W&B / TensorBoard 深度集成。

## 5.1 开工前需核实（P1 第一步）

| 项 | 为什么 | 怎么核实 |
| --- | --- | --- |
| **目标机 GPU 显存 ≥ 8GB** | RF-DETR 微调官方建议 ≥8GB；不足则只能用 Nano/Small + 小 batch + `grad_accum_steps` 补偿 | `nvidia-smi` 看显存；或告诉我显卡型号 |
| **磁盘空间** | `rfdetr[train,onnx]` + transformers/PL/roboflow 约数百 MB～数 GB；conda 环境在 E 盘 | 确认 E 盘余量 |
| **装前备份** | 万一依赖冲突可回滚 | `pip freeze > requirements-backup-YYYYMMDD.txt` 后再装 |

---

## 6. 风险与**本期不做**的事

| 项 | 说明 |
| --- | --- |
| **显存** | ~~RF-DETR 微调建议 ≥8GB 显存~~ → **实测已推翻**：见 6.2，RTX 3050 Laptop 4GB 上 Nano 训练峰值仅 **1.02GB**。仍保留低显存默认值以兼容更弱的机器 |
| **训练进度上报** | RF-DETR 基于 PyTorch Lightning，进度来自 tqdm/stdout。正则解析脆弱 → 采用**"日志原样透传 + 计时 + epoch 粗粒度进度"**，不做精确 loss 曲线（避免格式一变就崩） |
| **依赖体积** | `rfdetr[train]` 会装 roboflow、pytorch-lightning、transformers 等，环境会明显变大；装前先 `pip freeze > 备份` |
| **许可** | XLarge/2XLarge 为 PML 1.0，本期不开放 |
| **❌ 本期不做推理接入** | 导出的 RF-DETR ONNX **输出格式与 YOLO 不同**，现有「实时检测 / 图片 / 视频」页**不能直接加载**。导出后请在外部（ONNXRuntime）使用；若需接入实时检测，另立需求（涉及后处理与 NMS-free 解码） |
| **❌ 本期不做** | 实例分割（Seg 变体）、关键点、多卡 DDP、W&B/TensorBoard 深度集成 |

### 6.2 真机实测结论（2026-09-15，RTX 3050 Laptop / 4GB）

> 官方 FAQ 说微调建议 ≥8GB，实测**远没有那么夸张**，4GB 完全够。

| 项 | 实测值 |
| --- | --- |
| 变体 | RF-DETR **Nano** |
| 配置 | `batch_size=1`、`grad_accum_steps=8`、`use_ema=False`、`num_workers=0`、epochs=1 |
| 数据集 | `4940_has_labled`（511 训练 / 48 验证 / 34 测试，1280×960，1 类） |
| 训练耗时 | **116.9s**（1 epoch） |
| **峰值显存** | **1.02 GB / 4.0 GB**（远低于门槛） |
| 精度 | mAP@50:95 **0.9109**、mAP@50 **0.9968**、mAR@500 0.9469 |
| 产物 | `checkpoint_best_regular.pth`、`checkpoint_best_total.pth`、`last.ckpt`、`metrics.csv`、`training_config.json` |

**由此确认的默认值**（已落到 UI）：
- 变体默认 **Nano**；`batch_size` 默认 **auto**（`QSpinBox` 用 0 + `setSpecialValueText` 表示，让 RF-DETR 自己探测显存）
- `grad_accum_steps=4`、`lr=1e-4`、`epochs=50`
- `use_ema` **默认关闭**（4GB 卡省显存）
- `num_workers=0`（Windows 下规避多进程 dataloader 问题）
- 检测头会**自动从 90 类（COCO）重建为数据集实际类别数**，无需手工指定 `num_classes`

### 6.1 实测踩坑记录（2026-09-15，部署到其它电脑会重演）

| # | 坑 | 现象 | 修复 |
| --- | --- | --- | --- |
| 1 | **`roboflow` 顶掉 `cv2`** | `roboflow` 依赖 `opencv-python-headless`，与项目 `opencv-python 4.12.0.88` 抢同一个 `cv2` 模块。装完 rfdetr 后 `cv2.__version__` 变成 **5.0.0**（大版本跳跃，有破坏性 API 变更） | `pip uninstall -y opencv-python-headless` 再 `pip install --force-reinstall --no-deps opencv-python==4.12.0.88`。已固化进 `requirements-rfdetr.txt` 注释 |
| 2 | **`site-packages` 残骸导致导入崩** | 空目录（无 `__init__.py`、无 dist-info）会被 Python 当成命名空间包抢占导入。本机 `colorama`、`certifi` 均为空目录 → `import torch` / `import requests` 直接 `ImportError` | 删除空目录 + `pip install --force-reinstall --no-deps <pkg>`。已清理：`colorama certifi attrs cycler aiohappyeyeballs annotated-doc astunparse click anyio av` + 8 个空 dist-info |
| 3 | **`~orch` / `~umpy` 目录** | pip 在 Windows 上替换被占用文件时的备份前缀残留，说明**曾有人在程序运行时 pip 升级 numpy/torch**。当前 import 正常，暂不处理 | 提醒：以后升级依赖前**务必先关闭 GUI 程序** |

> 坑 2 的排查要点：报错形如 `module 'colorama' has no attribute 'init'` 或
> `cannot import name 'where' from 'certifi' (unknown location)` —— `unknown location`
> 就是空目录/命名空间包的典型特征，不要去改代码，去清 site-packages。

---

## 7. 验收标准

| 项 | 标准 |
| --- | --- |
| 降级 | 未装 `rfdetr` 时，程序正常启动，其余标签页功能不受影响，RF-DETR 页给出明确安装提示 |
| 适配 | 对 `4940_has_labled` 生成可用 `data.yaml`；对 `3631` 给出清晰报错而非静默失败 |
| 训练 | 用小 epoch（如 2）在 `4940_has_labled` 上跑通，`output_dir` 下产出 checkpoint |
| 验证 | `evaluate()` 返回 mAP 并在界面展示 |
| 导出 | 产出 `.onnx`，且能被 `onnxruntime` 成功加载（冒烟：`InferenceSession` 能建 + 跑一张图） |
| 回归 | 现有 YOLO 训练 / 验证 / 导出 / 实时检测全部保持可用 |
