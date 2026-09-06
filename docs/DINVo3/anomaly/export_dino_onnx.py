#!/usr/bin/env python
"""DINOv3/DINOv2 → ONNX 导出(异常检测服务端推理用)。

导出的模型:输入 1×3×S×S float32 RGB(ImageNet 归一化;S=patch*37,
dinov3=592 / dinov2=518),输出 1×1369×384 的 L2 归一化 patch tokens
(C++ 端直接与特征库做余弦比对)。

用法:
  .../python anomaly/export_dino_onnx.py \
      --model dinov3_vits16 \
      --weights /home/ubuntu/ultralytics-26/DINVo3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
      --out models/dinov3_vits16_anomaly.onnx
然后建 TensorRT FP16 引擎:
  /usr/src/tensorrt/bin/trtexec --onnx=models/dinov3_vits16_anomaly.onnx \
      --fp16 --saveEngine=models/dinov3_vits16_anomaly_fp16.engine
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dino_anomaly as da  # noqa: E402


class HubPatchWrapper(nn.Module):
    """dinov3 hub 模型 → L2 归一化 patch tokens(与离线建库一致)。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return F.normalize(
            self.inner.forward_features(x)["x_norm_patchtokens"], dim=-1)


def export_onnx(model: str = "dinov3_vits16", weights: str = "",
                repo_dir: str = "",
                out: str = "models/dinov3_vits16_anomaly.onnx",
                device: str = "cuda", log=print):
    """导出骨干为 ONNX:输入 1×3×S×S,输出 1×P×D 的 L2 归一化 patch tokens。

    返回 (onnx 路径, 输入边长 S, 输出形状 tuple)。GUI 与 CLI 共用本函数。"""
    device = da.norm_device(device)
    if model.startswith("dinov3"):
        # 直接取模型本体(与 load_model 共用同一加载路径);
        # 这同时会把 dinov3 仓库目录加入 sys.path,供下面 import rope 模块使用
        inner, patch_size, device = da.load_backbone_model(
            model, weights, repo_dir, device)
        wrapper = HubPatchWrapper(inner)
    else:
        # dinov2:HF transformer 模型,last_hidden_state 去 CLS
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoModel
        hf_name = da._HF_NAME_MAP.get(model, model)
        inner_hf = AutoModel.from_pretrained(hf_name)

        class HfWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return F.normalize(self.m(x).last_hidden_state[:, 1:, :],
                                   dim=-1)

        patch_size = int(inner_hf.config.patch_size)
        wrapper = HfWrapper(inner_hf)

    wrapper.eval().to(device)
    # 输入尺寸固定:把 RoPE 预计算成常量并冻结 forward,否则 dinov3 的
    # rope 里对张量 Size 的动态比较会导出 If 节点,TRT 8.6 解析失败
    # (If 输出 shape 不一致)。
    img_size = patch_size * 37
    hw = img_size // patch_size
    from dinov3.layers.rope_position_encoding import RopePositionEmbedding
    n_frozen = 0
    for mod in wrapper.modules():
        if isinstance(mod, RopePositionEmbedding):
            with torch.no_grad():
                sin_c, cos_c = mod(H=hw, W=hw)
            sin_c = sin_c.detach().clone()
            cos_c = cos_c.detach().clone()

            def frozen_forward(*_a, _s=sin_c, _c=cos_c, **_k):
                return (_s, _c)

            mod.forward = frozen_forward
            n_frozen += 1
    log(f"RoPE 已冻结 {n_frozen} 处(输入边长 {img_size})")
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        ref = wrapper(dummy)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_p), opset_version=17,
        input_names=["images"],
        output_names=["patch_tokens"],
        dynamic_axes=None,  # 固定 batch=1,服务端单路顺序推理
    )
    return str(out_p), img_size, tuple(ref.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="dinov3_vits16",
                    help="dinov3_vits16(本地权重) 或 dinov2_vits14(自动下载)")
    ap.add_argument("--weights", default="",
                    help="dinov3 本地权重 .pth(模型名以 dinov3 开头时必填)")
    ap.add_argument("--repo-dir", default="",
                    help="dinov3 仓库代码目录(默认自动找 anomaly 同级)")
    ap.add_argument("--out", default="models/dinov3_vits16_anomaly.onnx")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    path, img_size, shape = export_onnx(
        args.model, args.weights, args.repo_dir, args.out, args.device)
    print(f"ONNX saved: {path}")
    print(f"  input: 1x3x{img_size}x{img_size}  output: {shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
