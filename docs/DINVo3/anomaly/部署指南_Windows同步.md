# DINOv3 异常检测 + YOLO 训练 多平台部署指南

> 适用:注塑件外观检测(缺胶/毛边/黑点),缺陷样本极少、以良品建模为主的方案。
> Windows 端做"训练/建库/验证/导出",Jetson 端做"生产推理"。

## 1. 方案架构(先弄清几个概念)

| 组件 | 是什么 | 需要训练吗 | 文件格式 |
|---|---|---|---|
| YOLO 模型 | 已有缺陷分类(认识的缺陷) | ✅ ultralytics 正常训练 | .pt → .onnx → .engine |
| DINOv3 骨干 | Meta 预训练基础模型,提特征用 | ❌ **不训练**,直接用官方权重 | .pth → .onnx → .engine |
| 良品特征库 | 每个产品×每路相机一个,"正常长什么样"的记忆 | ❌ 不训练,良品照片建库 | .npz(离线验证)/ .fbin(服务端) |

**关键认知:DINOv3 没有"训练"这一步。** 它是 Meta 发布的预训练模型
(`dinov3_vits16_pretrain_lvd1689m-*.pth`,83MB,最小档 ViT),拿来直接提特征。
我们要"生成"的只是特征库——把良品照片过一遍模型,把特征向量存下来。

## 2. Windows 端(开发/建库/验证)

### 2.1 环境
```powershell
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # 有 N 卡装 CUDA 版
pip install transformers==4.46.3 opencv-python scikit-learn numpy onnx
```

### 2.2 权重与代码
- DINOv2:首次运行自动从 hf-mirror 下载,无需手动操作;
- DINOv3:
  1. 到 HuggingFace `facebook/dinov3-vits16-pretrain-lvd1689m` 页面接受许可后下载
     `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`;
  2. 到 GitHub `facebookresearch/dinov3` 下载仓库 zip 解压(模型定义代码)。

### 2.3 建库与验证(把 `anomaly/` 目录整个拷到 Windows 即可)
```powershell
python anomaly\build_bank.py --good-dir 良品照片目录 --out models\anomaly_bank.npz
python anomaly\validate.py --bank models\anomaly_bank.npz --good-dir 良品验证 --ng-dir 缺陷照片 --out out\anomaly
```
看三样东西:**AUROC**(0.95+ 方法有效)、**零误报阈值**(生产起步值)、
**heatmap/**(缺陷定位是否准确)。

### 2.4 导出 ONNX(跨平台通用)
```powershell
python anomaly\export_dino_onnx.py --model dinov3_vits16 ^
    --weights DINVo3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth ^
    --out models\dinov3_vits16_anomaly.onnx
```
> **.engine 不能跨设备**:Windows/RTX 上构建的 TensorRT 引擎在 Jetson 上不可用。
> ONNX 是通用的,拿到 Jetson 上用 trtexec 重建即可。

## 3. Jetson 端(生产推理)

### 3.1 部署清单(拷到 Jetson 的 yolo_c++)
```
models/dinov3_vits16_anomaly.onnx         # ONNX 源
models/anomaly_bank.fbin                  # 特征库(每产品×每相机一个)
```

### 3.2 构建 TensorRT FP16 引擎(Jetson 上执行,几分钟)
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/dinov3_vits16_anomaly.onnx \
    --fp16 --saveEngine=models/dinov3_vits16_anomaly_fp16.engine \
    --memPoolSize=workspace:1024
```

### 3.3 启用:cameras.conf 每路加 bank=/athresh=
```
camera:0 continuous 3072x2048 debounce=10 delay=0 3631_fp16 conf=0.25 bank=models/anomaly_bank.fbin athresh=0.15
```
启动参数可选:`--anomaly-confirm 3`(连续 N 帧超阈值判 NG)。

### 3.4 按产品绑定特征库(可选)
`deploy/products.conf` 每行末尾追加两列(cam0/cam1 特征库路径):
```
P001 | 产品A | yolo11n | 0.5 | yolo26n | 0.5 | 1 | 0 | 0 | 6 | 6 | models/bank_P001_cam0.fbin | models/bank_P001_cam1.fbin
```
- 留空 = 切产品时保持当前库;填 `-` = 切到该产品时禁用异常检测;
- GUI"产品管理"对话框里有对应输入框;
- 切产品自动换库,验证时会校验"建库骨干与当前骨干一致"。

### 3.5 界面
每个相机窗格:标题栏"异常检测"开关、参数行"阈值"滑条 + 实时异常分数
(红色 = 超阈值)。心跳按路上报 `anomaly_ready/en/score/th/ng`。

## 4. YOLO 训练(常规路线,照常)

在 ultralytics-26 用 `train.py` 正常训练,导出链:
`.pt → ultralytics export format=onnx → Jetson trtexec --fp16 → .engine`。
异常检测与 YOLO 并行运行、互不影响:YOLO 管"认识的缺陷",异常检测兜底
"不认识的异常",两者任一触发都走 NG 报警输出。

## 5. 常见问题

| 现象 | 原因/处理 |
|---|---|
| trtexec 解析 ONNX 失败(If 节点) | 用本仓库 export_dino_onnx.py(RoPE 已冻结为常量) |
| 换骨干后验证报"不一致" | 特征库与骨干绑定,重建库 |
| 生产分数与离线分数有偏移 | fp16 + 插值差异,以产线实测重新定阈值 |
| 良品分数普遍偏高 | 成像条件与建库时不一致;检查光照/曝光/位置,必要时重建库 |
| Windows 上 torch.hub 超时 | 脚本自动回退 HF 镜像;也可手动下载权重 |
