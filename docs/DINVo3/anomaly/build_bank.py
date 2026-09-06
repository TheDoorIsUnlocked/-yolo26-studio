#!/usr/bin/env python
"""良品特征库构建:用良品照片目录生成 DINO 异常检测特征库。

用法:
  ~/miniconda3/envs/yolo26/bin/python tools/anomaly/build_bank.py \
      --good-dir data/good_product --out models/anomaly_bank.npz

产出 models/anomaly_bank.npz,供 validate.py 与后续服务端推理使用。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dino_anomaly as da


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--good-dir", required=True, help="良品照片目录(递归)")
    ap.add_argument("--out", default="models/anomaly_bank.npz",
                    help="特征库输出路径(.npz)")
    ap.add_argument("--model", default="dinov3_vits16",
                    help="骨干名(dinov3_vits16 等 dinov3 骨干,或 dinov2_vits14)")
    ap.add_argument("--hub-repo", default="facebookresearch/dinov2",
                    help="仅 dinov2 走 torch.hub 时使用")
    ap.add_argument("--weights", default="",
                    help="dinov3 本地权重 .pth(模型名以 dinov3 开头时必填)")
    ap.add_argument("--repo-dir", default="",
                    help="dinov3 仓库代码目录(默认自动找 anomaly 同级的 dinov3-main)")
    ap.add_argument("--bank-size", type=int, default=16384,
                    help="coreset 后特征库向量数(越大越准越慢)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = da.list_images(args.good_dir)
    if not paths:
        print(f"error: no images under {args.good_dir}")
        return 2
    print(f"building bank from {len(paths)} images "
          f"(model={args.model}, bank_size={args.bank_size})")
    model, patch_size, device = da.load_model(
        args.model, args.hub_repo, args.device,
        weights_path=args.weights, repo_dir=args.repo_dir)
    # .fbin(服务端 C++ 格式: int32 M,D + float32 M*D) 由 build_bank 内部一并产出,
    # 这里不再重复写(原实现引用了未定义的 out 变量,必然 NameError)
    info = da.build_bank(model, patch_size, paths, args.out, device,
                         args.bank_size, model_name=args.model)
    fbin = Path(args.out).with_suffix('.fbin')
    print("done:", info, f"(fbin: {fbin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
