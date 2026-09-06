# 异常检测使用说明(DINOv2/v3 良品建模)

适用场景:缺陷样本极少(甚至为零)的注塑件外观检测(缺胶/毛边/黑点)。
只需大量**良品**照片建特征库,缺陷照片仅用于调阈值与验证召回。

## 环境要求

- conda 环境 `yolo26`(含 torch+CUDA、transformers 4.46.3、torchmetrics 等)
- DINOv3 需本地权重:`DINVo3/*.pth` 与仓库代码 `dinov3-main/`(均不入库)
- DINOv2 首次使用自动从 HF 镜像下载,之后离线可用

## 第一步:采集良品照片

用 yolo_c++ 程序自带"拍照采集"功能(定时或硬触发),保存全分辨率原图:

```
out/records/<产品名>/capture/cam0/   # 相机1
out/records/<产品名>/capture/cam1/   # 相机2
```

**关键要求**(直接影响建库质量):
- 只存良品画面;
- 采集期间固定曝光/增益/光照/产品位置,与之后生产检测时保持一致;
- 目标 200~500 张/产品/相机;成像条件改了就要重建库。

## 第二步:建库(GUI)

YOLO26 Studio → 侧边导航"异常检测" → 左侧"建库(良品特征库)":

1. 选择**模型骨干**:DINOv2 ViT-S(自动下载)或 DINOv3 ViT-S/16(本地权重);
2. 良品目录:填上面采集的目录;
3. 特征库容量:默认 16384(良品多可加大,更准、稍慢);
4. 特征库文件:输出路径(默认 `models/anomaly_bank.npz`);
5. 点"开始建库",进度在下方 System log。

等价命令行:
```bash
~/miniconda3/envs/yolo26/bin/python anomaly/build_bank.py \
    --good-dir out/records/<产品名>/capture/cam0 \
    --out models/anomaly_bank.npz
```

## 第三步:离线验证

右侧"离线验证":特征库文件 + 良品验证目录 + 缺陷目录 → "开始验证"。

结果解读(同时写入输出目录):
- **AUROC**:越接近 1.0 越好;0.95+ 说明方法对这类缺陷有效;
- **建议阈值**:Youden 最优(漏检/误报综合最平衡),生产上建议直接用
  **零误报阈值**(良品最高分×1.05)起步,宁可漏检先不误杀;
- **heatmap/** 目录:缺陷位置热力图,确认定位是否落在真实缺陷上;
- scores.csv:每张图分数,人工核对。

等价命令行:
```bash
~/miniconda3/envs/yolo26/bin/python anomaly/validate.py \
    --bank models/anomaly_bank.npz \
    --good-dir 良品验证目录 --ng-dir 缺陷照片目录 --out out/anomaly
```

## 注意事项

- **建库与验证/生产必须用同一骨干**,切换骨干后特征库要重建
  (验证时会自动拦截不匹配的组合);
- 特征库可随时追加良品重建(覆盖原文件);
- 每个产品型号/每种成像配置各建一个库;
- 缺陷照片不用多,十几张就能把阈值定下来;之后新缺陷样本继续喂给
  验证,持续监控召回。
